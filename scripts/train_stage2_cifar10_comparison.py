"""
Phase 10 Stage 2 Milestone 4: Three-Backend CIFAR-10 Comparative Benchmark
Compares side-by-side training across:
1. CPU_NATIVE — PyTorch Host CPU execution
2. GPU_NATIVE — PyTorch CUDA (cuBLAS/cuDNN) execution
3. GPU_CUSTOM — Custom CUDA autograd engine execution

Measures:
- Total training wall-clock time & per-epoch duration
- Training-step latency statistics (median, mean, min, max, std)
- Training throughput (samples/second)
- Convergence trajectory (loss, validation accuracy, test accuracy)
- Peak allocated and reserved GPU memory (VRAM)
- Host process RSS memory (psutil)
- Exact custom CUDA kernel dispatch counts
- Full provenance (Git commit, PyTorch version, CUDA version, GPU telemetry)
"""
import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import ctypes
from ctypes import wintypes

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import subprocess
import torch
import torch.nn as nn

from engine.data.cifar10 import create_cifar10_dataloaders
from engine.models.cifar10_mlp import create_cifar10_model, get_canonical_initial_weights
from engine.training.cifar10_three_backend_trainer import ThreeBackendTrainer
from engine.training.dispatch_tracker import KernelDispatchTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone4Benchmark")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_gpu_telemetry() -> Dict[str, Any]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "temperature_c": float(parts[0]),
            "graphics_clock_mhz": float(parts[1]),
            "memory_clock_mhz": float(parts[2]),
            "power_draw_w": float(parts[3]),
        }
    except Exception as e:
        return {"error": str(e)}


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_host_memory_mb() -> float:
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def run_single_backend_experiment(
    backend: str,
    canonical_weights: Dict[str, torch.Tensor],
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(f"EXECUTING EXPERIMENT: BACKEND = {backend}")
    logger.info("=" * 80)

    # Force garbage collection and clean CUDA memory before run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    pre_rss_mb = get_host_memory_mb()
    pre_gpu_telemetry = get_gpu_telemetry() if torch.cuda.is_available() else {}

    model = create_cifar10_model(backend=backend, canonical_state_dict=canonical_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    tracker = KernelDispatchTracker()
    trainer = ThreeBackendTrainer(
        backend=backend,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        tracker=tracker,
    )

    t0 = time.perf_counter()
    results = trainer.fit(epochs=epochs)
    t1 = time.perf_counter()

    post_rss_mb = get_host_memory_mb()
    post_gpu_telemetry = get_gpu_telemetry() if torch.cuda.is_available() else {}

    results["measured_total_wall_time_sec"] = round(t1 - t0, 3)
    results["pre_host_rss_mb"] = round(pre_rss_mb, 2)
    results["post_host_rss_mb"] = round(post_rss_mb, 2)
    results["pre_gpu_telemetry"] = pre_gpu_telemetry
    results["post_gpu_telemetry"] = post_gpu_telemetry
    results["trainable_parameters"] = model.count_parameters()

    # Log summary
    logger.info(
        f"[{backend}] COMPLETED: Wall Time={results['total_wall_time_sec']}s, "
        f"Throughput={results['samples_per_sec']} samples/s, "
        f"Test Acc={results['test_accuracy']}%, Peak VRAM={results['peak_allocated_vram_mb']} MB"
    )
    return results


def run_comparative_benchmark(
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 0.01,
    max_train_samples: int = 10000,
    max_val_samples: int = 2000,
    max_test_samples: int = 2000,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 10 STAGE 2 MILESTONE 4: REAL CIFAR-10 THREE-BACKEND BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Toolchain: CUDA Toolkit 12.4.1 + MSVC 19.38 + PyTorch {torch.__version__}")
    print(f"Workload: {max_train_samples} train samples, {epochs} epochs, batch size={batch_size}")
    print("=" * 80)

    checkpoint_dir = Path("checkpoints/phase10_stage2_milestone4")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/phase10_stage2_milestone4")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare identical dataloaders
    logger.info("Preparing real CIFAR-10 dataloaders...")
    train_loader, val_loader, test_loader, class_names = create_cifar10_dataloaders(
        data_dir="data",
        batch_size=batch_size,
        val_split=0.1,
        seed=42,
        augment=True,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. Generate canonical initial weights
    logger.info("Generating canonical initial weights...")
    canonical_weights = get_canonical_initial_weights(seed=42)

    experiment_metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "workload_config": {
            "dataset": "CIFAR-10",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "max_test_samples": max_test_samples,
            "precision": "float32",
            "architecture": "4-Layer Deep MLP (3072 -> 1024 -> 256 -> 64 -> 10)",
            "model_parameters": 3426250,
        },
    }

    benchmark_results: Dict[str, Any] = {
        "metadata": experiment_metadata,
        "backends": {},
        "comparison_metrics": {},
    }

    # Execute backends sequentially: CPU_NATIVE -> GPU_NATIVE -> GPU_CUSTOM
    backends_to_run = ["CPU_NATIVE", "GPU_NATIVE", "GPU_CUSTOM"]

    for b in backends_to_run:
        b_res = run_single_backend_experiment(
            backend=b,
            canonical_weights=canonical_weights,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            checkpoint_dir=checkpoint_dir,
        )
        benchmark_results["backends"][b] = b_res

    # 3. Compute comparative metrics
    cpu_res = benchmark_results["backends"]["CPU_NATIVE"]
    gpu_nat_res = benchmark_results["backends"]["GPU_NATIVE"]
    gpu_cust_res = benchmark_results["backends"]["GPU_CUSTOM"]

    gpu_nat_vs_cpu_speedup = round(cpu_res["total_wall_time_sec"] / gpu_nat_res["total_wall_time_sec"], 3)
    gpu_cust_vs_cpu_speedup = round(cpu_res["total_wall_time_sec"] / gpu_cust_res["total_wall_time_sec"], 3)
    gpu_cust_vs_gpu_nat_speedup = round(gpu_nat_res["total_wall_time_sec"] / gpu_cust_res["total_wall_time_sec"], 3)

    benchmark_results["comparison_metrics"] = {
        "gpu_native_speedup_vs_cpu": gpu_nat_vs_cpu_speedup,
        "gpu_custom_speedup_vs_cpu": gpu_cust_vs_cpu_speedup,
        "gpu_custom_speedup_vs_gpu_native": gpu_cust_vs_gpu_nat_speedup,
        "is_custom_faster_than_cpu": gpu_cust_vs_cpu_speedup > 1.0,
        "is_custom_faster_than_gpu_native": gpu_cust_vs_gpu_nat_speedup > 1.0,
        "step_latency_median_ms": {
            "CPU_NATIVE": cpu_res["overall_step_stats"]["median_ms"],
            "GPU_NATIVE": gpu_nat_res["overall_step_stats"]["median_ms"],
            "GPU_CUSTOM": gpu_cust_res["overall_step_stats"]["median_ms"],
        },
        "test_accuracy_pct": {
            "CPU_NATIVE": cpu_res["test_accuracy"],
            "GPU_NATIVE": gpu_nat_res["test_accuracy"],
            "GPU_CUSTOM": gpu_cust_res["test_accuracy"],
        },
    }

    # 4. Save results to JSON artifact
    out_file = results_dir / "cifar10_three_backend_results.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info(f"Results successfully saved to {out_file.resolve()}")
    return benchmark_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    args = parser.parse_args()

    run_comparative_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
    )
