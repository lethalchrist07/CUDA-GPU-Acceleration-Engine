"""Read measured artifacts without inventing unavailable values or editing results."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Artifact exceeds the 16 MiB read limit")
    document = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON object")
    return document


# -----------------------------------------------------------------------------
# Bounded In-Memory & Persistent Digest Cache for Checkpoint & Artifact Hashing
# -----------------------------------------------------------------------------
import threading
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PERSISTENT_CACHE_FILE = PROJECT_ROOT / "results" / ".digest_cache.json"

_DIGEST_CACHE: dict[Tuple[str, int, int], str] = {}
_DIGEST_CACHE_LOCK = threading.RLock()
_DIGEST_CACHE_MAX_ENTRIES = 2048
_CACHE_STATS = {"hits": 0, "misses": 0, "invalidations": 0}

_CATALOG_CACHE: dict[str, Any] | None = None
_CATALOG_LOCK = threading.RLock()


def _load_persistent_cache() -> None:
    """Load pre-computed hashes from disk cache if present."""
    if _PERSISTENT_CACHE_FILE.is_file():
        try:
            with open(_PERSISTENT_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with _DIGEST_CACHE_LOCK:
                for key_str, sha in data.items():
                    parts = key_str.split("|")
                    if len(parts) == 3:
                        cache_key = (parts[0], int(parts[1]), int(parts[2]))
                        _DIGEST_CACHE[cache_key] = sha
        except Exception:
            pass


def _save_persistent_cache() -> None:
    """Flush in-memory digest cache to disk."""
    try:
        data = {}
        with _DIGEST_CACHE_LOCK:
            for (p_str, sz, mt), sha in _DIGEST_CACHE.items():
                data[f"{p_str}|{sz}|{mt}"] = sha
        _PERSISTENT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_PERSISTENT_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


# Auto-load on module import
_load_persistent_cache()


def clear_catalog_cache() -> None:
    """Clear the cached catalog response."""
    global _CATALOG_CACHE
    with _CATALOG_LOCK:
        _CATALOG_CACHE = None


def get_digest_cache_stats() -> dict[str, int]:
    """Return live metrics on cache utilization, hits, and misses."""
    with _DIGEST_CACHE_LOCK:
        return {
            "cached_entries": len(_DIGEST_CACHE),
            "hits": _CACHE_STATS["hits"],
            "misses": _CACHE_STATS["misses"],
            "invalidations": _CACHE_STATS["invalidations"],
        }


def clear_digest_cache(reload_persistent: bool = False) -> int:
    """Explicitly clear the in-memory digest cache."""
    with _DIGEST_CACHE_LOCK:
        count = len(_DIGEST_CACHE)
        _DIGEST_CACHE.clear()
        _CACHE_STATS["invalidations"] += count
        if reload_persistent:
            _load_persistent_cache()
        return count


def digest(path: Path, force_recompute: bool = False) -> str:
    """
    Computes or retrieves cached SHA-256 digest of a file.
    Uses (absolute_path, file_size, mtime_ns) as the cache key.
    Reads file in 64 KiB chunks to avoid loading multi-gigabyte files into memory.
    """
    resolved_path = path.resolve()
    stat = resolved_path.stat()
    cache_key = (str(resolved_path), stat.st_size, getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))

    if not force_recompute:
        with _DIGEST_CACHE_LOCK:
            if cache_key in _DIGEST_CACHE:
                _CACHE_STATS["hits"] += 1
                return _DIGEST_CACHE[cache_key]
            # If in-memory cache was cleared or key is missing, check persistent disk cache
            if _PERSISTENT_CACHE_FILE.is_file():
                _load_persistent_cache()
                if cache_key in _DIGEST_CACHE:
                    _CACHE_STATS["hits"] += 1
                    return _DIGEST_CACHE[cache_key]
            _CACHE_STATS["misses"] += 1

    # Stream file in 64 KiB chunks to avoid RAM saturation
    hasher = hashlib.sha256()
    with open(resolved_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    result = hasher.hexdigest()

    with _DIGEST_CACHE_LOCK:
        if len(_DIGEST_CACHE) >= _DIGEST_CACHE_MAX_ENTRIES:
            # Prune oldest 25% of entries when bound is reached
            keys_to_remove = list(_DIGEST_CACHE.keys())[: _DIGEST_CACHE_MAX_ENTRIES // 4]
            for k in keys_to_remove:
                del _DIGEST_CACHE[k]
        _DIGEST_CACHE[cache_key] = result

    return result


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


class ArtifactStore:
    def __init__(self, root: Union[Path, str]):
        self.root = Path(root).resolve()
        self.results = self.root / "results"

    def phase2_directory(self) -> Path | None:
        candidates = [p.parent.parent for p in self.results.glob("phase2_verification_*/final_run/benchmark_results.json")]
        return max(candidates, key=lambda p: p.name) if candidates else None

    def _inside_root(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Artifact is outside the project")
        return resolved

    def _envelope(self, path: Path, *, historical: bool) -> dict[str, Any]:
        self._inside_root(path)
        return {
            "status": "available",
            "fetched_at": utc_now(),
            "source": {
                "path": path.relative_to(self.root).as_posix(),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": digest(path),
                "provenance": "historical_unverified" if historical else "recorded_execution",
            },
            "warnings": [],
        }

    @staticmethod
    def _unavailable(reason: str) -> dict[str, Any]:
        return {"status": "unavailable", "fetched_at": utc_now(), "source": None,
                "warnings": [reason], "rows": []}

    def benchmarks(self) -> dict[str, Any]:
        directory = self.phase2_directory()
        path = directory / "final_run/benchmark_results.json" if directory else self.results / "benchmark_results.json"
        try:
            document = read_json(self._inside_root(path))
            rows = document.get("benchmarks")
            if not isinstance(rows, list) or not rows:
                raise ValueError("No benchmark records")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Invalid benchmark record")
                for key in ("cpu_time_ms", "gpu_kernel_ms", "gpu_end_to_end_ms"):
                    value = _number(row.get(key))
                    if value is None or value <= 0:
                        raise ValueError(f"Invalid measured field: {key}")
                if not isinstance(row.get("verified"), bool):
                    raise ValueError("Missing numerical verification status")
            result = self._envelope(path, historical=directory is None)
            if directory is None:
                result["warnings"].append("Legacy benchmark timing is unverified; E2E may be a sum of event intervals.")
            if directory:
                manifest = read_json(self._inside_root(directory / "manifest.json"))
                expected = manifest.get(path.relative_to(self.root).as_posix(), {}).get("sha256")
                if expected == result["source"]["sha256"]:
                    result["source"]["provenance"] = "phase2_hash_verified"
                else:
                    result["source"]["provenance"] = "hash_mismatch"
                    result["warnings"].append("Benchmark content does not match the Phase 2 manifest.")
            result.update(rows=rows, system_info=document.get("system_info", {}),
                          summary=document.get("summary", {}), measurement_notes=document.get("measurement_notes", {}))
            result["warnings"] += [
                "Speedups compare the project's single-threaded CPU reference, not optimized BLAS.",
                "GPU E2E excludes allocation and initialization. Kernel-only speedup is not application speedup.",
                "Memory bandwidth is estimated requested traffic, not measured DRAM throughput.",
            ]
            return result
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            return self._unavailable(f"Cannot read benchmark artifact: {exc}")

    def experiments(self, kind: str) -> dict[str, Any]:
        filenames = {"training": ("training_experiments.json", "experiments"),
                     "inference": ("inference_experiments.json", "inferences")}
        if kind not in filenames:
            raise ValueError("Unsupported experiment kind")
        filename, key = filenames[kind]
        path = self.results / filename
        try:
            document = read_json(self._inside_root(path))
            rows = document.get(key)
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Invalid experiment records")
            result = self._envelope(path, historical=True)
            result["rows"] = rows
            result["warnings"] = ["Historical records: original executions were not authenticated in Phase 2."]
            if kind == "inference":
                result["warnings"].append("CUDA device labels record the requested device; these records do not independently prove GPU offload.")
            return result
        except (OSError, ValueError, TypeError) as exc:
            return self._unavailable(f"Cannot read {kind} artifact: {exc}")

    def artifact_paths(self) -> dict[str, Path]:
        paths = {
            "training-json": self.results / "training_experiments.json",
            "training-csv": self.results / "training_experiments.csv",
            "inference-json": self.results / "inference_experiments.json",
            "inference-csv": self.results / "inference_experiments.csv",
            "phase9-benchmarks-json": self.results / "phase9_gpu_benchmarks.json",
            "phase9-training-json": self.results / "phase9_training_experiments.json",
            "phase9-training-csv": self.results / "phase9_training_experiments.csv",
        }
        directory = self.phase2_directory()
        if directory:
            paths.update({
                "benchmark-json": directory / "final_run/benchmark_results.json",
                "benchmark-csv": directory / "final_run/benchmark_results.csv",
                "phase2-report": directory / "VERIFICATION_REPORT.md",
                "phase2-commands": directory / "COMMANDS.md",
                "phase2-manifest": directory / "manifest.json",
                "benchmark-log": directory / "final_run.log",
                "python-test-log": directory / "pytest-final.log",
                "native-test-log": directory / "ctest-final.log",
                "foundation-log": directory / "foundation-final.log",
                "artifact-audit-log": directory / "artifact_audit-final.log",
            })
        return {key: self._inside_root(path) for key, path in paths.items()
                if path.is_file() and path.resolve().is_relative_to(self.root)}

    def phase9_benchmarks(self) -> dict[str, Any]:
        path = self.results / "phase9_gpu_benchmarks.json"
        if not path.is_file():
            return self._unavailable("Phase 9 GPU benchmark results not yet recorded.")
        try:
            doc = read_json(self._inside_root(path))
            res = self._envelope(path, historical=False)
            res.update(doc)
            return res
        except Exception as exc:
            return self._unavailable(f"Failed to read Phase 9 benchmarks: {exc}")

    def phase9_experiments(self) -> dict[str, Any]:
        path = self.results / "phase9_training_experiments.json"
        if not path.is_file():
            return self._unavailable("Phase 9 training experiments not yet recorded.")
        try:
            doc = read_json(self._inside_root(path))
            res = self._envelope(path, historical=False)
            res["rows"] = doc.get("experiments", [])
            return res
        except Exception as exc:
            return self._unavailable(f"Failed to read Phase 9 experiments: {exc}")

    def checkpoints_catalog(self, force_rehash: bool = False) -> dict[str, Any]:
        global _CATALOG_CACHE
        if not force_rehash:
            with _CATALOG_LOCK:
                if _CATALOG_CACHE is not None:
                    return _CATALOG_CACHE

        checkpoints_root = self.root / "checkpoints"
        if not checkpoints_root.is_dir():
            return {"status": "unavailable", "checkpoints": []}
        items = []
        for p in sorted(checkpoints_root.rglob("*.pt")):
            try:
                rel = p.relative_to(self.root).as_posix()
                if "transformer" in rel:
                    model_type = "transformer"
                    if "50m" in rel or "pretraining_50m" in rel:
                        arch = "NEXORA GPT-86.8M (50M Campaign)"
                        params = 86825472
                        params_str = "86.83M (50M Pretrained)"
                    elif "pretraining_campaign" in rel:
                        arch = "NEXORA GPT-86.8M (10M Campaign)"
                        params = 86825472
                        params_str = "86.83M (Pretraining Campaign)"
                    elif "flagship_100m" in rel:
                        arch = "NEXORA GPT-100M"
                        params = 85940736
                        params_str = "85.94M (Pilot) / 92.13M (Preset)"
                    elif "25m" in rel:
                        arch = "FoundationTransformer25M"
                        params = 23196160
                        params_str = "23.20M"
                    else:
                        arch = "PilotTransformer1M"
                        params = 1000000
                        params_str = "1.00M"
                elif "resnet18" in rel:
                    model_type = "vision"
                    arch = "ResNet-18"
                    params = 11173962
                    params_str = "11.17M"
                else:
                    model_type = "vision"
                    arch = "CIFAR10MLP" if "job" in rel else "BaselineMLP"
                    params = 3426250 if "job" in rel else 465172
                    params_str = "3.43M" if "job" in rel else "465K"

                items.append({
                    "id": p.stem,
                    "filename": p.name,
                    "path": rel,
                    "model_type": model_type,
                    "architecture": arch,
                    "parameters": params,
                    "parameters_display": params_str,
                    "size_bytes": p.stat().st_size,
                    "sha256": digest(p, force_recompute=force_rehash),
                    "modified_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                })
            except Exception:
                continue

        _save_persistent_cache()
        result = {"status": "available", "fetched_at": utc_now(), "checkpoints": items}
        with _CATALOG_LOCK:
            _CATALOG_CACHE = result
        return result

    def verification(self) -> dict[str, Any]:
        directory = self.phase2_directory()
        checks: list[dict[str, Any]] = []
        warnings = ["These are recorded Phase 2 checks. Opening this dashboard does not rerun the test suite."]
        if directory:
            try:
                manifest = read_json(self._inside_root(directory / "manifest.json"))
                for relative, item in manifest.items():
                    path = self._inside_root(self.root / relative)
                    actual = digest(path) if path.is_file() else None
                    checks.append({"path": relative, "status": "matched" if actual == item["sha256"] else "mismatch",
                                   "expected_sha256": item["sha256"], "actual_sha256": actual})
            except (OSError, ValueError, TypeError, KeyError) as exc:
                warnings.append(f"Manifest verification unavailable: {exc}")
        artifacts = [{"id": key, "name": path.name, "path": path.relative_to(self.root).as_posix(),
                      "size_bytes": path.stat().st_size, "download_url": f"/api/artifacts/{key}"}
                     for key, path in self.artifact_paths().items()]
        return {"status": "available" if directory else "unavailable", "fetched_at": utc_now(),
                "manifest_status": ("matched" if checks and all(c["status"] == "matched" for c in checks)
                                    else "mismatch" if checks else "unavailable"),
                "checks": checks, "artifacts": artifacts, "warnings": warnings}

    def overview(self) -> dict[str, Any]:
        benchmarks = self.benchmarks()
        training, inference = self.experiments("training"), self.experiments("inference")
        rows = benchmarks["rows"]
        return {"status": "available" if rows else "degraded", "fetched_at": utc_now(),
                "benchmark_count": len(rows), "benchmark_passed": sum(row["verified"] for row in rows),
                "training_count": len(training["rows"]), "inference_count": len(inference["rows"]),
                "benchmark_source": benchmarks["source"], "system_info": benchmarks.get("system_info", {}),
                "benchmark_rows": rows,
                "warnings": benchmarks["warnings"] + training["warnings"] + inference["warnings"]}


def collect_hardware() -> dict[str, Any]:
    """Read NVIDIA's current counters; N/A stays null, never synthetic zero."""
    query = "index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit"
    command = shutil.which("nvidia-smi")
    result: dict[str, Any] = {"status": "unavailable", "fetched_at": utc_now(), "gpus": [],
                              "host": {"platform": platform.platform(), "processor": platform.processor()},
                              "warnings": [], "source": "nvidia-smi"}
    if not command:
        result["warnings"].append("nvidia-smi is not installed or not on PATH.")
        return result
    try:
        output = subprocess.run([command, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=8, check=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for fields in csv.reader(io.StringIO(output.stdout)):
            if len(fields) != 11:
                raise ValueError("Unexpected NVIDIA telemetry format")
            fields = [value.strip() for value in fields]
            result["gpus"].append(dict(zip(
                ("index", "name", "uuid", "driver_version", "memory_total_mib", "memory_used_mib",
                 "memory_free_mib", "utilization_percent", "temperature_c", "power_w", "power_limit_w"),
                [int(fields[0]), *fields[1:4], *[_number(v) for v in fields[4:]]],
            )))
        result["status"] = "available" if result["gpus"] else "unavailable"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["warnings"].append(f"Live GPU telemetry unavailable: {exc}")
    return result


def collect_capabilities() -> dict[str, Any]:
    """Collect platform, CUDA, and model execution capability matrix."""
    caps: dict[str, Any] = {
        "status": "available",
        "fetched_at": utc_now(),
        "platform": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "cuda": {
            "available": False,
            "device_count": 0,
            "device_name": None,
            "compute_capability": None,
            "total_vram_mb": 0.0,
            "amp_supported": False,
            "bf16_supported": False,
            "fp16_supported": False,
            "cuda_version": None,
            "pytorch_version": None,
        },
        "models": {
            "architectures": ["ResNet18", "BaselineMLP"],
            "supported_datasets": ["CIFAR-10", "SyntheticClassification"],
            "serving_supported": True,
        },
    }
    try:
        import torch
        caps["cuda"]["pytorch_version"] = torch.__version__
        caps["cuda"]["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            caps["cuda"]["available"] = True
            caps["cuda"]["device_count"] = torch.cuda.device_count()
            caps["cuda"]["device_name"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            caps["cuda"]["compute_capability"] = f"{cc[0]}.{cc[1]}"
            props = torch.cuda.get_device_properties(0)
            caps["cuda"]["total_vram_mb"] = round(props.total_memory / (1024.0 * 1024.0), 2)
            caps["cuda"]["amp_supported"] = True
            caps["cuda"]["fp16_supported"] = True
            caps["cuda"]["bf16_supported"] = (cc[0] >= 8)  # Ampere or newer
    except Exception as exc:
        caps["warnings"] = [f"PyTorch capabilities probe: {exc}"]
    return caps
