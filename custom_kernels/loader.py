import os
import sys
import json
import subprocess
from pathlib import Path

_CACHED_MODULE = None

def get_vcvars_environment() -> dict:
    """Extract environment variables from vcvars64.bat using MSVC 14.38 toolset."""
    vcvars_bat = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if not os.path.exists(vcvars_bat):
        raise FileNotFoundError(f"vcvars64.bat not found at {vcvars_bat}")

    cmd = f'call "{vcvars_bat}" -vcvars_ver=14.38 >nul && "{sys.executable}" -c "import os, json; print(json.dumps(dict(os.environ)))"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to query MSVC v14.38 environment: {res.stderr}")

    return json.loads(res.stdout.strip())

def configure_process_toolchain():
    """Apply process-local CUDA 12.4 and MSVC 14.38 environment variables without touching system/user scope."""
    cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    if not os.path.exists(cuda_dir):
        # Fallback to temp staging if permanent not found
        fallback = os.path.join(os.environ.get("TEMP", ""), "cuda_12.4_staging")
        if os.path.exists(fallback):
            cuda_dir = fallback
        else:
            raise FileNotFoundError(f"CUDA 12.4 toolkit not found at {cuda_dir}")

    msvc_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.38.33130"
    msvc_bin = os.path.join(msvc_dir, "bin", "Hostx64", "x64")
    cuda_bin = os.path.join(cuda_dir, "bin")

    # Inject full vcvars environment for MSVC 14.38
    try:
        env = get_vcvars_environment()
        for k, v in env.items():
            os.environ[k] = v
    except Exception as e:
        pass

    os.environ["CUDA_HOME"] = cuda_dir
    os.environ["CUDA_PATH"] = cuda_dir
    os.environ["VCToolsInstallDir"] = msvc_dir + "\\"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

    # Filter out other MSVC toolsets (e.g. 14.51/14.44) so Ninja strictly invokes 14.38 (19.38)
    clean_paths = [
        p for p in os.environ.get("PATH", "").split(os.pathsep)
        if not ("VC\\Tools\\MSVC" in p and "14.38.33130" not in p)
    ]
    os.environ["PATH"] = f"{cuda_bin};{msvc_bin};" + os.pathsep.join(clean_paths)

# Run process-local configuration on import
configure_process_toolchain()

def load_custom_kernels_module():
    """Build or load custom_kernels C++ extension."""
    global _CACHED_MODULE
    if _CACHED_MODULE is not None:
        return _CACHED_MODULE

    configure_process_toolchain()

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
    ]

    include_dirs = [str(csrc_dir / "include")]
    msvc_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.38.33130"
    msvc_bin = os.path.join(msvc_dir, "bin", "Hostx64", "x64")

    _CACHED_MODULE = load(
        name="custom_kernels_cuda",
        sources=sources,
        extra_include_paths=include_dirs,
        extra_cflags=["/O2", "/MD", "/std:c++17"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-gencode=arch=compute_86,code=compute_86",
            "-gencode=arch=compute_86,code=sm_86",
            f"-ccbin={msvc_bin}",
        ],
        verbose=False,
    )
    return _CACHED_MODULE
