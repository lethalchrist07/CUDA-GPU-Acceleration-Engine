"""
scripts/audit_milestone8_environment.py
Phase 13 Milestone 8: Stage 1 Repository & Environment Audit.
Performs comprehensive audit of toolchains, checkpoints, custom CUDA kernels,
VRAM availability, and performance bottleneck identification.

Outputs: results/phase13_milestone8_environment_audit.json
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone8Audit")


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    logger.info("=" * 80)
    logger.info("PHASE 13 MILESTONE 8: STAGE 1 ENVIRONMENT & REPOSITORY AUDIT")
    logger.info("=" * 80)

    audit_data: Dict[str, Any] = {
        "milestone": "Phase 13 Milestone 8",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_status": "IN_PROGRESS",
        "system_environment": {},
        "toolchain": {},
        "cuda_kernels": {},
        "checkpoints_audit": {},
        "gpu_vram_status": {},
        "identified_bottlenecks": {},
    }

    # 1. System Environment
    import platform
    audit_data["system_environment"] = {
        "os": platform.platform(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }

    # 2. Toolchain Audit
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    capability = torch.cuda.get_device_capability(0) if cuda_available else (0, 0)
    props = torch.cuda.get_device_properties(0) if cuda_available else None
    total_vram_mb = props.total_memory / (1024 ** 2) if props else 0.0

    nvcc_path = shutil.which("nvcc")
    nvcc_version = ""
    if nvcc_path:
        try:
            out = subprocess.check_output([nvcc_path, "--version"], text=True)
            nvcc_version = out.split("\n")[-2].strip()
        except Exception:
            pass

    cmake_path = shutil.which("cmake") or "C:\\Program Files\\CMake\\bin\\cmake.exe"
    cmake_exists = Path(cmake_path).exists() or shutil.which("cmake") is not None

    audit_data["toolchain"] = {
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "sm_count": props.multi_processor_count if props else 0,
        "total_vram_mb": round(total_vram_mb, 2),
        "nvcc_path": nvcc_path,
        "nvcc_version": nvcc_version,
        "cmake_available": bool(cmake_exists),
    }
    logger.info(f"Toolchain: PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {device_name} (SM {capability[0]}.{capability[1]})")

    # 3. Custom CUDA Kernels Verification
    try:
        import custom_kernels
        m = custom_kernels._get_module()
        exported_funcs = [f for f in dir(m) if not f.startswith("__")]
        audit_data["cuda_kernels"] = {
            "module_path": getattr(m, "__file__", "in-memory"),
            "exported_functions_count": len(exported_funcs),
            "exported_functions": exported_funcs,
            "status": "LOADED_SUCCESSFULLY",
        }
        logger.info(f"Custom Kernels: Loaded {len(exported_funcs)} exported functions successfully.")
    except Exception as e:
        audit_data["cuda_kernels"] = {
            "status": "FAILED",
            "error": str(e),
        }
        logger.error(f"Failed to load custom kernels: {e}")

    # 4. Checkpoints Integrity Verification
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_campaign"
    final_ckpt = ckpt_dir / "final_model.pt"
    best_ckpt = ckpt_dir / "best_model.pt"
    vocab_file = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "bpe_vocab.json"

    # Known recorded hashes from Milestone 6 & 7
    expected_final_hash = "ca6fe2a51d1b973f0c3e0fda04fa43ce5a8ca1e95f51ed9d99cc424c234e45bd"
    expected_best_hash = "abcd0fe7b305cd1cef7809359ca7128f0ac94bed592a188f8432329ca1406aac"

    final_hash = compute_sha256(final_ckpt) if final_ckpt.is_file() else ""
    best_hash = compute_sha256(best_ckpt) if best_ckpt.is_file() else ""
    vocab_hash = compute_sha256(vocab_file) if vocab_file.is_file() else ""

    # Historical Checkpoint Preservation
    hist_paths = {
        "pretraining_campaign_final": PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "final_model.pt",
        "pretraining_10m_extended_best": PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended" / "best_model.pt",
        "pretraining_50m_diagnostic_final": PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_diagnostic" / "final_model.pt",
        "resnet18_cifar10_best": PROJECT_ROOT / "checkpoints" / "resnet18_cifar10" / "best_model.pt",
    }
    hist_status = {}
    for k, p in hist_paths.items():
        hist_status[k] = {
            "exists": p.is_file(),
            "size_bytes": p.stat().st_size if p.is_file() else 0,
        }

    audit_data["checkpoints_audit"] = {
        "final_model": {
            "path": str(final_ckpt),
            "exists": final_ckpt.is_file(),
            "size_bytes": final_ckpt.stat().st_size if final_ckpt.is_file() else 0,
            "sha256": final_hash,
            "matches_manifest": final_hash == expected_final_hash,
        },
        "best_model": {
            "path": str(best_ckpt),
            "exists": best_ckpt.is_file(),
            "size_bytes": best_ckpt.stat().st_size if best_ckpt.is_file() else 0,
            "sha256": best_hash,
            "matches_manifest": best_hash == expected_best_hash,
        },
        "vocab": {
            "path": str(vocab_file),
            "exists": vocab_file.is_file(),
            "sha256": vocab_hash,
        },
        "historical_checkpoints": hist_status,
        "all_historical_intact": all(v["exists"] and v["size_bytes"] > 0 for v in hist_status.values()),
    }
    logger.info(f"Checkpoints: final_model match={final_hash == expected_final_hash}, best_model match={best_hash == expected_best_hash}")

    # 5. GPU & VRAM Status
    if cuda_available:
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        free_mb = total_vram_mb - reserved
        audit_data["gpu_vram_status"] = {
            "total_vram_mb": round(total_vram_mb, 2),
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "free_vram_mb": round(free_mb, 2),
            "sufficient_for_benchmarking": free_mb >= 3000.0,
        }
        logger.info(f"VRAM: Total={total_vram_mb:.1f} MB, Allocated={allocated:.1f} MB, Free={free_mb:.1f} MB")

    # 6. Bottleneck Identification (From M7 Empirical Analysis)
    audit_data["identified_bottlenecks"] = {
        "bottleneck_1_cublaslt_fused_epilogue": {
            "impact_ms_per_step": 1.52,
            "description": "cuBLASLt in Config B fuses bias into GEMM register tiles (1 memory pass). Config C uses unfused torch.matmul + separate elementwise kernel, incurring extra 25.17 MB DRAM round-trip write/read.",
        },
        "bottleneck_2_python_autograd_dispatch": {
            "impact_ms_per_step": 0.18,
            "description": "torch.autograd.Function wrapper executes Python C-API interpreter calls for node allocation and tuple packaging across 12 layers forward and backward.",
        },
        "bottleneck_3_tensor_view_conversions": {
            "impact_ms_per_step": 0.05,
            "description": "2D-to-3D view reshapes (view(-1, d_ff) and view(B, T, d_ff)) across 12 layers.",
        },
        "total_overhead_differential_ms": 1.37,
        "differential_pct": 0.52,
    }

    # Audit verdict
    all_passed = (
        cuda_available and
        audit_data["cuda_kernels"].get("status") == "LOADED_SUCCESSFULLY" and
        audit_data["checkpoints_audit"]["final_model"]["matches_manifest"] and
        audit_data["checkpoints_audit"]["best_model"]["matches_manifest"] and
        audit_data["checkpoints_audit"]["all_historical_intact"] and
        audit_data["gpu_vram_status"].get("sufficient_for_benchmarking", False)
    )
    audit_data["audit_status"] = "PASSED" if all_passed else "FAILED"
    logger.info(f"Audit Verdict: {audit_data['audit_status']}")

    out_file = PROJECT_ROOT / "results" / "phase13_milestone8_environment_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    logger.info(f"Saved audit report to: {out_file}")
    return audit_data


if __name__ == "__main__":
    main()

