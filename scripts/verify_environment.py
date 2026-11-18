"""
scripts/verify_environment.py
Production Readiness Environment & Hardware Audit Script.
Validates Python runtime, PyTorch CUDA bindings, GPU device capabilities,
compiler availability, and filesystem readiness.
"""

from __future__ import annotations

import sys
import platform
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_environment() -> int:
    print("=" * 70)
    print("CUDA GPU ACCELERATION ENGINE — ENVIRONMENT AUDIT")
    print("=" * 70)

    errors = []
    warnings = []

    # 1. Python Runtime
    py_ver = sys.version_info
    print(f"[*] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro} ({platform.platform()})")
    if py_ver < (3, 10):
        errors.append(f"Python 3.10+ required; found {py_ver.major}.{py_ver.minor}")
    elif py_ver >= (3, 13):
        warnings.append(f"Python {py_ver.major}.{py_ver.minor} may have limited pre-built PyTorch CUDA wheels.")

    # 2. PyTorch & CUDA Check
    try:
        import torch
        print(f"[*] PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"[*] CUDA Available: {cuda_avail}")

        if cuda_avail:
            device_count = torch.cuda.device_count()
            dev_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            print(f"    - Device Count: {device_count}")
            print(f"    - Device 0: {dev_name}")
            print(f"    - Compute Capability: {cc[0]}.{cc[1]}")
            print(f"    - Dedicated VRAM: {vram_mb:.2f} MB")

            if cc[0] < 7:
                warnings.append(f"Compute capability {cc[0]}.{cc[1]} is older than Volta/Ampere. Tensor cores / SDPA may be limited.")
        else:
            warnings.append("CUDA is NOT available. Engine will operate in CPU Emulation mode.")
    except ImportError as e:
        errors.append(f"PyTorch is not installed: {e}")

    # 3. FastAPI & Backend Dependencies
    deps = ["fastapi", "uvicorn", "pydantic", "yaml", "httpx", "pytest"]
    for d in deps:
        try:
            mod = __import__(d)
            ver = getattr(mod, "__version__", "installed")
            print(f"[*] Dependency '{d}': {ver}")
        except ImportError:
            errors.append(f"Required dependency '{d}' is missing. Run `pip install -r requirements.txt`.")

    # 4. Critical Filesystem Directories
    required_dirs = [
        "backend",
        "engine",
        "custom_kernels",
        "checkpoints",
        "data",
        "results",
    ]
    for rd in required_dirs:
        p = PROJECT_ROOT / rd
        if p.is_dir():
            print(f"[*] Directory '{rd}': Present")
        else:
            errors.append(f"Required directory '{rd}' missing at {p}")

    # 5. Checkpoints Sanity
    res_ckpt = PROJECT_ROOT / "checkpoints/resnet18_cifar10/best_model.pt"
    gpt_ckpt = PROJECT_ROOT / "checkpoints/transformer/pretraining_50m_campaign/best_model.pt"
    if res_ckpt.is_file():
        print(f"[*] ResNet18 Checkpoint: Verified ({res_ckpt.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        warnings.append(f"ResNet-18 checkpoint missing at {res_ckpt}")

    if gpt_ckpt.is_file():
        print(f"[*] GPT-86.8M Pretrained Checkpoint: Verified ({gpt_ckpt.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        warnings.append(f"50M Campaign checkpoint missing at {gpt_ckpt}")

    # 6. C++ Build Toolchain (optional for runtime, required for compilation)
    cl_path = shutil.which("cl.exe")
    nvcc_path = shutil.which("nvcc.exe")
    if cl_path:
        print(f"[*] MSVC Host Compiler: {cl_path}")
    else:
        print("[*] MSVC Host Compiler (cl.exe): Not in PATH (using pre-compiled or JIT extensions)")

    if nvcc_path:
        print(f"[*] NVIDIA CUDA Compiler (nvcc): {nvcc_path}")
    else:
        print("[*] NVIDIA CUDA Compiler (nvcc): Not in PATH")

    # Summary
    print("-" * 70)
    if warnings:
        print(f"[!] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    if errors:
        print(f"[X] CRITICAL ERRORS ({len(errors)}):")
        for err in errors:
            print(f"    - {err}")
        print("\nEnvironment verification FAILED.")
        return 1

    print("[+] Environment verification PASSED. System is production ready.")
    return 0

if __name__ == "__main__":
    sys.exit(check_environment())

