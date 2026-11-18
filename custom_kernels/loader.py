"""
custom_kernels/loader.py
Portable CUDA Kernel Loader & Toolchain Environment Manager
Phase 10 — Stage 2 / November 15 Remediation (BLK-02)

Provides portable, automatic detection of:
- CUDA Toolkit installations via CUDA_PATH, CUDA_PATH_V*, nvcc in PATH, and standard Windows paths.
- MSVC compiler toolsets via vswhere.exe, VSINSTALLDIR, and standard Visual Studio paths.
- Dynamic GPU architecture targeting (Ampere RTX 3050/3060 sm_86, Ada Lovelace RTX 4050 sm_89, Blackwell RTX 5090 sm_100/sm_120).
- Pure PyTorch CPU / CUDA fallback module when native compilation toolchains are absent.
- Guaranteed import-time safety (never crashes module import when tools are missing).
"""

from __future__ import annotations

import os
import sys
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger("cuda_engine.custom_kernels.loader")

_CACHED_MODULE: Optional[Any] = None
_TOOLCHAIN_STATUS: Optional[Dict[str, Any]] = None


# ============================================================================
# 1. Portable CUDA Toolkit Detection
# ============================================================================

def is_valid_cuda_dir(path: Path) -> bool:
    """Validate that directory contains nvcc executable and cuda.h header."""
    if not path.is_dir():
        return False
    nvcc_exe = path / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
    cuda_h = path / "include" / "cuda.h"
    return nvcc_exe.is_file() and cuda_h.is_file()


def find_cuda_toolkit() -> Optional[Path]:
    """
    Dynamically discover the best available CUDA Toolkit installation without
    assuming a single fixed path or CUDA version.
    """
    # 1. Environment: CUDA_HOME takes highest precedence
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and is_valid_cuda_dir(Path(cuda_home)):
        return Path(cuda_home)

    # 2. Environment: CUDA_PATH
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and is_valid_cuda_dir(Path(cuda_path)):
        return Path(cuda_path)

    candidates: List[Tuple[str, Path]] = []

    # 3. Environment: CUDA_PATH_V* (e.g. CUDA_PATH_V12_4, CUDA_PATH_V13_4)
    v_vars: List[Tuple[str, str]] = []
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH_V"):
            v_vars.append((k, v))
    # Sort descending by variable name to prefer newer versions if available
    for k, v in sorted(v_vars, reverse=True):
        p = Path(v)
        if is_valid_cuda_dir(p):
            candidates.append((k, p))

    # 4. nvcc in PATH
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        cand = Path(nvcc_path).resolve().parent.parent
        if is_valid_cuda_dir(cand):
            candidates.append(("nvcc_in_path", cand))

    # 5. Standard Windows installation locations
    if sys.platform == "win32":
        standard_roots = [
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
            Path(r"C:\Program Files\NVIDIA\CUDA"),
        ]
        # Also check other drive letters if available
        for drive in ("D", "E", "F"):
            standard_roots.append(Path(f"{drive}:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"))

        for sroot in standard_roots:
            if sroot.is_dir():
                for sub in sorted(sroot.glob("v*"), reverse=True):
                    if is_valid_cuda_dir(sub):
                        candidates.append((f"standard_{sub.name}", sub))

    # 6. Temp staging directory fallbacks
    temp_dir = os.environ.get("TEMP", "")
    if temp_dir:
        for staging_name in ("cuda_staging", "cuda_12.4_staging", "cuda_toolkit"):
            p = Path(temp_dir) / staging_name
            if is_valid_cuda_dir(p):
                candidates.append((f"staging_{staging_name}", p))

    if not candidates:
        logger.debug("No valid CUDA Toolkit installation found on system.")
        return None

    # Prioritize candidate matching PyTorch's compiled CUDA version if available
    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda:
        torch_cuda_dot = str(torch_cuda).lower()
        torch_cuda_us = torch_cuda_dot.replace(".", "_")
        for src, cand in candidates:
            cand_str = str(cand).lower()
            if (f"v{torch_cuda_dot}" in cand_str or torch_cuda_dot in cand_str or
                f"v{torch_cuda_us}" in cand_str or torch_cuda_us in cand_str):
                logger.debug("Selected CUDA Toolkit %s matching PyTorch %s from %s", cand, torch_cuda, src)
                return cand

    # Default to highest priority candidate
    src, cand = candidates[0]
    logger.debug("Selected CUDA Toolkit %s from %s", cand, src)
    return cand


# ============================================================================
# 2. Portable MSVC Compiler & Toolset Detection
# ============================================================================

def query_vswhere() -> List[Path]:
    """Query vswhere.exe for Visual Studio installations with C++ Desktop tools."""
    if sys.platform != "win32":
        return []

    vswhere_exe = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
                  "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere_exe.is_file():
        return []

    try:
        cmd = [
            str(vswhere_exe),
            "-latest",
            "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-format", "json",
            "-utf8",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            installs = json.loads(res.stdout)
            paths = []
            for inst in installs:
                p_str = inst.get("installationPath")
                if p_str:
                    paths.append(Path(p_str))
            return paths
    except Exception as e:
        logger.debug("vswhere.exe query encountered non-fatal error: %s", e)

    return []


def find_msvc_environment(cuda_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path], Optional[Dict[str, str]]]:
    """
    Locate Visual Studio / MSVC environment without assuming a fixed directory.
    Returns (msvc_bin_dir, vcvars64_bat_path, vcvars_environment_dict).
    """
    if sys.platform != "win32":
        # On Linux/macOS, check for gcc/clang in PATH
        cc_bin = shutil.which("gcc") or shutil.which("clang")
        return (Path(cc_bin).parent if cc_bin else None, None, None)

    vs_candidates: List[Path] = []

    # 1. vswhere.exe results
    vs_candidates.extend(query_vswhere())

    # 2. VSINSTALLDIR env var
    vs_install_dir = os.environ.get("VSINSTALLDIR")
    if vs_install_dir:
        vs_candidates.append(Path(vs_install_dir))

    # 3. Known standard Visual Studio 2022 / BuildTools paths
    standard_vs_paths = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"),
    ]
    vs_candidates.extend(standard_vs_paths)

    for vs_root in vs_candidates:
        if not vs_root.is_dir():
            continue

        vcvars64 = vs_root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if not vcvars64.is_file():
            # Check legacy VS2017/2015 path
            vcvars64 = vs_root / "VC" / "bin" / "amd64" / "vcvars64.bat"
            if not vcvars64.is_file():
                continue

        msvc_base = vs_root / "VC" / "Tools" / "MSVC"
        if not msvc_base.is_dir():
            continue

        toolset_dirs = [d.name for d in msvc_base.iterdir() if d.is_dir()]
        if not toolset_dirs:
            continue

        # Compatibility check: If CUDA < 12.5 (e.g. 12.4), nvcc strictly requires MSVC 14.3x (VS 2022 v143).
        # Newer MSVC toolsets (14.4x / 14.5x) trigger fatal unsupported compiler errors in CUDA 12.4 nvcc.
        selected_toolset = None
        v143_toolsets = [t for t in toolset_dirs if t.startswith("14.3")]
        if v143_toolsets:
            selected_toolset = sorted(v143_toolsets)[-1]
        else:
            selected_toolset = sorted(toolset_dirs)[-1]

        msvc_tool_dir = msvc_base / selected_toolset
        msvc_bin = msvc_tool_dir / "bin" / "Hostx64" / "x64"
        if not (msvc_bin / "cl.exe").is_file():
            msvc_bin = msvc_tool_dir / "bin" / "Hostx86" / "x64"
            if not (msvc_bin / "cl.exe").is_file():
                continue

        # Extract environment from vcvars64.bat
        env_dict = None
        try:
            version_arg = f"-vcvars_ver={selected_toolset[:5]}" if selected_toolset else ""
            cmd = f'call "{vcvars64}" {version_arg} >nul && "{sys.executable}" -c "import os, json; print(json.dumps(dict(os.environ)))"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and res.stdout.strip():
                env_dict = json.loads(res.stdout.strip())
        except Exception as e:
            logger.debug("Failed querying vcvars64.bat with version arg: %s", e)

        if not env_dict:
            try:
                # Retry without version arg
                cmd = f'call "{vcvars64}" >nul && "{sys.executable}" -c "import os, json; print(json.dumps(dict(os.environ)))"'
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
                if res.returncode == 0 and res.stdout.strip():
                    env_dict = json.loads(res.stdout.strip())
            except Exception as e:
                logger.debug("Failed querying vcvars64.bat default: %s", e)

        return (msvc_bin, vcvars64, env_dict)

    # Fallback: check if cl.exe is already present in PATH
    cl_path = shutil.which("cl")
    if cl_path:
        return (Path(cl_path).parent, None, None)

    return (None, None, None)


# ============================================================================
# 3. Dynamic GPU Architecture & Hardware Targeting
# ============================================================================

def detect_gpu_architecture() -> Dict[str, Any]:
    """
    Detect target NVIDIA GPU compute architecture dynamically:
    - RTX 3050 / 3060: Ampere (sm_86, compute_86)
    - RTX 4050 / 4060: Ada Lovelace (sm_89, compute_89)
    - RTX 5090: Blackwell (sm_100 / sm_120)
    """
    # Safe defaults for RTX 3050 Laptop GPU baseline
    arch_str = "8.6"
    arch_num = "86"
    family = "Ampere (RTX 3050 Default)"

    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            arch_str = f"{major}.{minor}"
            arch_num = f"{major}{minor}"
            if major == 8 and minor == 6:
                family = "Ampere (RTX 3050/3060)"
            elif major == 8 and minor == 9:
                family = "Ada Lovelace (RTX 4050/4060)"
            elif major >= 10:
                family = f"Blackwell (sm_{major}{minor})"
            else:
                family = f"Compute Capability {major}.{minor}"
        except Exception as e:
            logger.debug("Could not query GPU capability; defaulting to Ampere 8.6: %s", e)

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        f"-gencode=arch=compute_{arch_num},code=compute_{arch_num}",
        f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}",
    ]

    return {
        "arch_list": arch_str,
        "arch_code": f"compute_{arch_num}",
        "sm_code": f"sm_{arch_num}",
        "family": family,
        "extra_cuda_cflags": extra_cuda_cflags,
    }


# ============================================================================
# 4. Safe Toolchain Configuration
# ============================================================================

def configure_process_toolchain() -> Dict[str, Any]:
    """
    Detect and apply process-local CUDA and MSVC environment variables safely.
    NEVER raises unhandled exceptions at import time.
    Returns status dictionary.
    """
    global _TOOLCHAIN_STATUS
    if _TOOLCHAIN_STATUS is not None:
        return _TOOLCHAIN_STATUS

    diagnostics: List[str] = []
    cuda_dir = find_cuda_toolkit()
    msvc_bin, vcvars_bat, env_dict = find_msvc_environment(cuda_dir)
    arch_info = detect_gpu_architecture()

    is_ready = False

    if cuda_dir is None:
        diagnostics.append("CUDA Toolkit not detected on system.")
    else:
        diagnostics.append(f"Located CUDA Toolkit at: {cuda_dir}")

    if msvc_bin is None:
        diagnostics.append("MSVC compiler environment (cl.exe / vcvars64.bat) not detected.")
    else:
        diagnostics.append(f"Located MSVC compiler at: {msvc_bin}")

    if cuda_dir is not None and msvc_bin is not None:
        try:
            # Apply vcvars environment variables if acquired
            if env_dict:
                for k, v in env_dict.items():
                    os.environ[k] = v

            cuda_bin = str(cuda_dir / "bin")
            msvc_bin_str = str(msvc_bin)

            os.environ["CUDA_HOME"] = str(cuda_dir)
            os.environ["CUDA_PATH"] = str(cuda_dir)
            os.environ["VCToolsInstallDir"] = str(msvc_bin.parent.parent) + "\\"
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_info["arch_list"]

            # Filter conflicting MSVC PATH entries if specific toolset was pinned
            clean_paths = [
                p for p in os.environ.get("PATH", "").split(os.pathsep)
                if not ("VC\\Tools\\MSVC" in p and msvc_bin_str not in p)
            ]
            os.environ["PATH"] = f"{cuda_bin};{msvc_bin_str};" + os.pathsep.join(clean_paths)
            is_ready = True
            diagnostics.append("Successfully configured process toolchain environment.")
        except Exception as e:
            diagnostics.append(f"Failed to apply process toolchain environment: {e}")

    _TOOLCHAIN_STATUS = {
        "is_compilation_ready": is_ready,
        "cuda_dir": str(cuda_dir) if cuda_dir else None,
        "msvc_bin": str(msvc_bin) if msvc_bin else None,
        "vcvars_bat": str(vcvars_bat) if vcvars_bat else None,
        "architecture": arch_info,
        "diagnostics": diagnostics,
    }
    return _TOOLCHAIN_STATUS


def get_toolchain_status() -> Dict[str, Any]:
    """Retrieve current toolchain status dictionary."""
    return configure_process_toolchain()


# ============================================================================
# 5. Pure PyTorch Fallback CUDA & CPU Module
# ============================================================================

class FallbackCUDAKernelModule:
    """
    Pure PyTorch fallback implementation of all 18 custom kernel operations.
    Seamlessly executes on CUDA (when GPU is available) or CPU (when GPU is absent)
    using verified PyTorch reference operations with zero loss of numerical accuracy.
    """

    def __init__(self):
        self.is_native: bool = False
        self.is_fallback: bool = True
        self._logged_ops: Set[str] = set()
        self.call_counts: Dict[str, int] = {}

    def _check_device(self, *tensors: Optional[torch.Tensor], name: str = "Input"):
        if torch.cuda.is_available():
            for t in tensors:
                if t is not None and not t.is_cuda:
                    raise RuntimeError(f"{name} must be a CUDA tensor")
        else:
            for t in tensors:
                if t is not None and t.device.type != "cpu":
                    raise RuntimeError(f"{name} must be on CPU in CPU-only mode")

    def _log_fallback(self, op: str, device: torch.device):
        self.call_counts[op] = self.call_counts.get(op, 0) + 1
        if op not in self._logged_ops:
            self._logged_ops.add(op)
            logger.info("Custom CUDA kernel '%s' running in PyTorch fallback mode on %s", op, device)

    # Functional forward operators
    def vector_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._check_device(a, b, name="vector_add: inputs")
        self._log_fallback("vector_add", a.device)
        return a + b

    def vector_add_backward(self, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_device(grad_output, name="vector_add_backward: grad_output")
        self._log_fallback("vector_add_backward", grad_output.device)
        return grad_output.clone(), grad_output.clone()

    def vector_scale(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        self._check_device(x, name="vector_scale: x")
        self._log_fallback("vector_scale", x.device)
        return x * float(alpha)

    def vector_scale_backward(self, grad_output: torch.Tensor, alpha: float) -> torch.Tensor:
        self._check_device(grad_output, name="vector_scale_backward: grad_output")
        self._log_fallback("vector_scale_backward", grad_output.device)
        return grad_output * float(alpha)

    def relu(self, x: torch.Tensor) -> torch.Tensor:
        self._check_device(x, name="relu: x")
        self._log_fallback("relu", x.device)
        return torch.relu(x)

    def relu_backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._check_device(grad_output, x, name="relu_backward: inputs")
        self._log_fallback("relu_backward", grad_output.device)
        return grad_output * (x > 0).to(grad_output.dtype)

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        self._check_device(x, name="sigmoid: x")
        self._log_fallback("sigmoid", x.device)
        return torch.sigmoid(x)

    def sigmoid_backward(self, grad_output: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        self._check_device(grad_output, out, name="sigmoid_backward: inputs")
        self._log_fallback("sigmoid_backward", grad_output.device)
        return grad_output * out * (1.0 - out)

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        self._check_device(x, name="gelu: x")
        self._log_fallback("gelu", x.device)
        return F.gelu(x, approximate="tanh")

    def gelu_backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._check_device(grad_output, x, name="gelu_backward: inputs")
        self._log_fallback("gelu_backward", grad_output.device)
        x_det = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = F.gelu(x_det, approximate="tanh")
            grad = torch.autograd.grad(y, x_det, grad_output, retain_graph=False)[0]
        return grad

    def fused_add_relu(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._check_device(a, b, name="fused_add_relu: inputs")
        self._log_fallback("fused_add_relu", a.device)
        return torch.relu(a + b)

    def fused_add_relu_backward(self, grad_output: torch.Tensor, out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_device(grad_output, out, name="fused_add_relu_backward: inputs")
        self._log_fallback("fused_add_relu_backward", grad_output.device)
        mask = (out > 0).to(grad_output.dtype)
        g = grad_output * mask
        return g.clone(), g.clone()

    def fused_bias_activation(self, x: torch.Tensor, bias: torch.Tensor, activation: str = "relu") -> torch.Tensor:
        self._check_device(x, bias, name="fused_bias_activation: inputs")
        self._log_fallback("fused_bias_activation", x.device)
        out = x + bias
        act = activation.lower()
        if act == "relu":
            return torch.relu(out)
        elif act == "gelu":
            return F.gelu(out, approximate="tanh")
        elif act == "none":
            return out
        else:
            raise ValueError(f"Unsupported activation: '{activation}'")

    def fused_bias_activation_backward(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        bias: torch.Tensor,
        activation: str = "relu",
        need_x: bool = True,
        need_bias: bool = True,
    ) -> List[torch.Tensor]:
        self._check_device(grad_output, x, bias, name="fused_bias_activation_backward: inputs")
        self._log_fallback("fused_bias_activation_backward", grad_output.device)
        act = activation.lower()
        z = x + bias

        if act == "relu":
            dz = grad_output * (z > 0).to(grad_output.dtype)
        elif act == "gelu":
            z_det = z.detach().requires_grad_(True)
            with torch.enable_grad():
                y = F.gelu(z_det, approximate="tanh")
                dz = torch.autograd.grad(y, z_det, grad_output, retain_graph=False)[0]
        elif act == "none":
            dz = grad_output
        else:
            raise ValueError(f"Unsupported activation: '{activation}'")

        grad_x = dz if need_x else torch.empty(0, device=x.device, dtype=x.dtype)
        grad_bias = dz.sum(dim=0) if need_bias else torch.empty(0, device=bias.device, dtype=bias.dtype)
        return [grad_x, grad_bias]

    def matmul_tiled(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._check_device(a, b, name="matmul_tiled: inputs")
        self._log_fallback("matmul_tiled", a.device)
        return torch.matmul(a, b)

    def matmul_wmma(self, a: torch.Tensor, b: torch.Tensor, output_fp32: bool = False) -> torch.Tensor:
        self._check_device(a, b, name="matmul_wmma: inputs")
        self._log_fallback("matmul_wmma", a.device)
        out = torch.matmul(a, b)
        if output_fp32:
            return out.to(torch.float32)
        return out.to(torch.float16)

    def reduce_sum(self, x: torch.Tensor, output_fp32: bool = False) -> torch.Tensor:
        self._check_device(x, name="reduce_sum: x")
        self._log_fallback("reduce_sum", x.device)
        s = x.sum()
        if output_fp32:
            return s.to(torch.float32)
        return s

    def reduce_max(self, x: torch.Tensor) -> torch.Tensor:
        self._check_device(x, name="reduce_max: x")
        self._log_fallback("reduce_max", x.device)
        return x.max()


# ============================================================================
# 6. Build / Load Execution Layer
# ============================================================================

def reset_cached_module() -> None:
    """Clear cached kernel module instance (primarily for testing and environment mocking)."""
    global _CACHED_MODULE, _TOOLCHAIN_STATUS
    _CACHED_MODULE = None
    _TOOLCHAIN_STATUS = None


def load_custom_kernels_module(allow_fallback: bool = True, force_fallback: bool = False) -> Any:
    """
    Build or load custom_kernels C++ extension using dynamically discovered toolchain.
    When native toolchains are missing or compilation fails, safely falls back to pure
    PyTorch implementations when allow_fallback is True.
    """
    global _CACHED_MODULE
    if _CACHED_MODULE is not None and not force_fallback:
        return _CACHED_MODULE

    # Check environment variable override for testing fallback
    if force_fallback or os.environ.get("NEXORA_FORCE_FALLBACK", "0") == "1":
        logger.info("NEXORA_FORCE_FALLBACK active: Using pure PyTorch fallback module.")
        _CACHED_MODULE = FallbackCUDAKernelModule()
        return _CACHED_MODULE

    status = configure_process_toolchain()
    if not status["is_compilation_ready"]:
        if allow_fallback:
            logger.warning(
                "Native CUDA compiler toolchain is unavailable (%s). Activating PyTorch fallback module.",
                "; ".join(status["diagnostics"]),
            )
            _CACHED_MODULE = FallbackCUDAKernelModule()
            return _CACHED_MODULE
        else:
            raise RuntimeError(
                f"Native CUDA compilation toolchain is unavailable: {status['diagnostics']}"
            )

    try:
        from torch.utils.cpp_extension import load

        pkg_root = Path(__file__).resolve().parent
        csrc_dir = pkg_root / "csrc"
        sources = [
            str(csrc_dir / "bindings.cpp"),
            str(csrc_dir / "cuda" / "elementwise.cu"),
            str(csrc_dir / "cuda" / "activations.cu"),
            str(csrc_dir / "cuda" / "reductions.cu"),
            str(csrc_dir / "cuda" / "matmul_tiled.cu"),
            str(csrc_dir / "cuda" / "matmul_wmma.cu"),
            str(csrc_dir / "cuda" / "backward.cu"),
        ]

        include_dirs = [str(csrc_dir / "include")]
        msvc_bin = status["msvc_bin"]
        arch_info = status["architecture"]

        extra_cuda_cflags = list(arch_info["extra_cuda_cflags"])
        if msvc_bin:
            extra_cuda_cflags.append(f"-ccbin={msvc_bin}")

        mod = load(
            name="custom_kernels_cuda",
            sources=sources,
            extra_include_paths=include_dirs,
            extra_cflags=["/O2", "/MD", "/std:c++17"],
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )

        mod.is_native = True
        mod.is_fallback = False
        _CACHED_MODULE = mod
        return _CACHED_MODULE

    except Exception as e:
        logger.warning(
            "Native custom CUDA kernel compilation/load failed: %s. Falling back to PyTorch implementation.",
            e,
        )
        if allow_fallback:
            _CACHED_MODULE = FallbackCUDAKernelModule()
            return _CACHED_MODULE
        else:
            raise
