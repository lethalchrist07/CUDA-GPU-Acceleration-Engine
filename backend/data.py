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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
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

    def checkpoints_catalog(self) -> dict[str, Any]:
        checkpoints_root = self.root / "checkpoints"
        if not checkpoints_root.is_dir():
            return {"status": "unavailable", "checkpoints": []}
        items = []
        for p in sorted(checkpoints_root.rglob("*.pt")):
            try:
                rel = p.relative_to(self.root).as_posix()
                items.append({
                    "id": p.stem,
                    "filename": p.name,
                    "path": rel,
                    "size_bytes": p.stat().st_size,
                    "sha256": digest(p),
                    "modified_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                })
            except Exception:
                continue
        return {"status": "available", "fetched_at": utc_now(), "checkpoints": items}

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
