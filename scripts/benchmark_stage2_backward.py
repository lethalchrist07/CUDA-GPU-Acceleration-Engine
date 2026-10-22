"""
Phase 10 Stage 2 Milestone 3: Autograd & Backward Kernel Benchmark Suite
Device: NVIDIA GeForce RTX 3050 Laptop GPU (sm_86, 6 GB VRAM)
Toolchain: CUDA 12.4.1 + MSVC 19.38 + PyTorch 2.6.0+cu124

Measures:
1. Microbenchmarks:
   - Forward execution time (CUDA events & e2e)
   - Backward execution time (CUDA events & e2e)
   - Combined Forward + Backward latency
   - Peak allocated and reserved memory
   - Comparison against native PyTorch autograd reference
2. Full Neural Network Training Step:
   - Forward time
   - Backward time (loss.backward())
   - Optimizer step time
   - Full training step time
   - Peak VRAM footprint
   - Comparison: Native PyTorch vs Custom CUDA Autograd
"""
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime
import sys
from pathlib import Path

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, Any, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_kernels as ck
import custom_kernels.autograd as ck_auto


WARMUP_ITERS = 15
BENCH_ITERS = 30


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


def compute_stats(timings_ms: List[float]) -> Dict[str, float]:
    timings_ms.sort()
    n = len(timings_ms)
    mean_val = sum(timings_ms) / n
    median_val = timings_ms[n // 2] if n % 2 != 0 else (timings_ms[n // 2 - 1] + timings_ms[n // 2]) / 2.0
    variance = sum((x - mean_val) ** 2 for x in timings_ms) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(variance)
    return {
        "min_ms": round(min(timings_ms), 5),
        "median_ms": round(median_val, 5),
        "max_ms": round(max(timings_ms), 5),
        "mean_ms": round(mean_val, 5),
        "std_ms": round(std_val, 5),
        "variability_pct": round((std_val / mean_val * 100.0) if mean_val > 0 else 0.0, 2),
    }


def benchmark_autograd_op(
    fwd_fn,
    inputs: List[torch.Tensor],
    iters: int = BENCH_ITERS,
    warmup: int = WARMUP_ITERS,
) -> Dict[str, Any]:
    """Measures separate forward, backward, and combined forward+backward timings."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Pre-generate grad_outputs to avoid timing generation
    with torch.no_grad():
        sample_out = fwd_fn(*inputs)
        grad_out = torch.randn_like(sample_out)
    del sample_out
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        for inp in inputs:
            if inp.grad is not None:
                inp.grad = None
        out = fwd_fn(*inputs)
        out.backward(grad_out, retain_graph=False)
    torch.cuda.synchronize()

    # Measure Forward-only latency via CUDA events
    fwd_events_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    fwd_events_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    fwd_timings: List[float] = []

    outputs = []
    for i in range(iters):
        fwd_events_start[i].record()
        out = fwd_fn(*inputs)
        fwd_events_end[i].record()
        outputs.append(out)
    torch.cuda.synchronize()

    for i in range(iters):
        fwd_timings.append(fwd_events_start[i].elapsed_time(fwd_events_end[i]))

    # Measure Backward-only latency via CUDA events
    bwd_events_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_events_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_timings: List[float] = []

    for i in range(iters):
        for inp in inputs:
            if inp.grad is not None:
                inp.grad = None
        bwd_events_start[i].record()
        outputs[i].backward(grad_out, retain_graph=False)
        bwd_events_end[i].record()
    torch.cuda.synchronize()

    for i in range(iters):
        bwd_timings.append(bwd_events_start[i].elapsed_time(bwd_events_end[i]))
    del outputs

    # Measure Combined End-to-End latency
    combined_timings: List[float] = []
    for _ in range(iters):
        for inp in inputs:
            if inp.grad is not None:
                inp.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fwd_fn(*inputs)
        out.backward(grad_out, retain_graph=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        combined_timings.append((t1 - t0) * 1000.0)

    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    return {
        "forward_kernel_ms": compute_stats(fwd_timings),
        "backward_kernel_ms": compute_stats(bwd_timings),
        "combined_e2e_ms": compute_stats(combined_timings),
        "peak_allocated_mb": round(peak_alloc_mb, 3),
        "peak_reserved_mb": round(peak_res_mb, 3),
    }


def benchmark_training_step(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    iters: int = BENCH_ITERS,
    warmup: int = WARMUP_ITERS,
) -> Dict[str, Any]:
    """Measures a full neural network training step (fwd, loss, bwd, opt.step)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    fwd_times: List[float] = []
    bwd_times: List[float] = []
    step_times: List[float] = []

    e_fwd_start = torch.cuda.Event(enable_timing=True)
    e_fwd_end = torch.cuda.Event(enable_timing=True)
    e_bwd_start = torch.cuda.Event(enable_timing=True)
    e_bwd_end = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward
        e_fwd_start.record()
        out = model(x)
        loss = F.mse_loss(out, target)
        e_fwd_end.record()

        # Backward
        e_bwd_start.record()
        loss.backward()
        e_bwd_end.record()

        # Optimizer step
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        fwd_times.append(e_fwd_start.elapsed_time(e_fwd_end))
        bwd_times.append(e_bwd_start.elapsed_time(e_bwd_end))
        step_times.append((t1 - t0) * 1000.0)

    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    return {
        "forward_ms": compute_stats(fwd_times),
        "backward_ms": compute_stats(bwd_times),
        "full_step_ms": compute_stats(step_times),
        "peak_allocated_mb": round(peak_alloc_mb, 3),
        "peak_reserved_mb": round(peak_res_mb, 3),
    }


def run_all_benchmarks() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 10 STAGE 2 MILESTONE 3: AUTOGRAD & BACKWARD KERNEL BENCHMARKS")
    print("Device: NVIDIA GeForce RTX 3050 Laptop GPU (sm_86, 6 GB VRAM)")
    print("Toolchain: CUDA Toolkit 12.4.1 + MSVC 19.38 + PyTorch 2.6.0+cu124")
    print("=" * 80)

    pre_telemetry = get_gpu_telemetry()
    print(f"Pre-benchmark GPU telemetry: {pre_telemetry}\n")

    timestamp = datetime.now().isoformat()
    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": timestamp,
            "gpu": torch.cuda.get_device_name(0),
            "sm_capability": "8.6",
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "pre_telemetry": pre_telemetry,
        },
        "microbenchmarks": {},
        "training_step_benchmarks": {},
    }

    # -------------------------------------------------------------------------
    # 1. Microbenchmarks across operations and dtypes
    # -------------------------------------------------------------------------
    test_sizes = [65536, 1048576]
    dtypes = [("float32", torch.float32), ("float16", torch.float16)]

    for dtype_name, dtype in dtypes:
        for size in test_sizes:
            op_key = f"{dtype_name}_{size}"
            print(f"Benchmarking elementwise ops: {dtype_name}, size={size:,}...")

            # A. Vector Add
            a = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
            b = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
            res_ck = benchmark_autograd_op(ck_auto.vector_add, [a, b])
            res_pt = benchmark_autograd_op(lambda x, y: x + y, [a, b])
            results["microbenchmarks"][f"vector_add_{op_key}"] = {
                "custom_cuda": res_ck,
                "pytorch_native": res_pt,
                "fwd_speedup": round(res_pt["forward_kernel_ms"]["median_ms"] / res_ck["forward_kernel_ms"]["median_ms"], 3),
                "bwd_speedup": round(res_pt["backward_kernel_ms"]["median_ms"] / res_ck["backward_kernel_ms"]["median_ms"], 3),
            }

            # B. ReLU
            x = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
            res_ck = benchmark_autograd_op(ck_auto.relu, [x])
            res_pt = benchmark_autograd_op(F.relu, [x])
            results["microbenchmarks"][f"relu_{op_key}"] = {
                "custom_cuda": res_ck,
                "pytorch_native": res_pt,
                "fwd_speedup": round(res_pt["forward_kernel_ms"]["median_ms"] / res_ck["forward_kernel_ms"]["median_ms"], 3),
                "bwd_speedup": round(res_pt["backward_kernel_ms"]["median_ms"] / res_ck["backward_kernel_ms"]["median_ms"], 3),
            }

            # C. Sigmoid
            x = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
            res_ck = benchmark_autograd_op(ck_auto.sigmoid, [x])
            res_pt = benchmark_autograd_op(torch.sigmoid, [x])
            results["microbenchmarks"][f"sigmoid_{op_key}"] = {
                "custom_cuda": res_ck,
                "pytorch_native": res_pt,
                "fwd_speedup": round(res_pt["forward_kernel_ms"]["median_ms"] / res_ck["forward_kernel_ms"]["median_ms"], 3),
                "bwd_speedup": round(res_pt["backward_kernel_ms"]["median_ms"] / res_ck["backward_kernel_ms"]["median_ms"], 3),
            }

            # D. GELU
            x = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
            res_ck = benchmark_autograd_op(ck_auto.gelu, [x])
            res_pt = benchmark_autograd_op(lambda t: F.gelu(t, approximate="tanh"), [x])
            results["microbenchmarks"][f"gelu_{op_key}"] = {
                "custom_cuda": res_ck,
                "pytorch_native": res_pt,
                "fwd_speedup": round(res_pt["forward_kernel_ms"]["median_ms"] / res_ck["forward_kernel_ms"]["median_ms"], 3),
                "bwd_speedup": round(res_pt["backward_kernel_ms"]["median_ms"] / res_ck["backward_kernel_ms"]["median_ms"], 3),
            }

            # E. Fused Add + ReLU
            a = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
            b = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
            res_ck = benchmark_autograd_op(ck_auto.fused_add_relu, [a, b])
            res_pt = benchmark_autograd_op(lambda x, y: F.relu(x + y), [a, b])
            results["microbenchmarks"][f"fused_add_relu_{op_key}"] = {
                "custom_cuda": res_ck,
                "pytorch_native": res_pt,
                "fwd_speedup": round(res_pt["forward_kernel_ms"]["median_ms"] / res_ck["forward_kernel_ms"]["median_ms"], 3),
                "bwd_speedup": round(res_pt["backward_kernel_ms"]["median_ms"] / res_ck["backward_kernel_ms"]["median_ms"], 3),
            }

    # -------------------------------------------------------------------------
    # 2. Fused Bias + Activation Microbenchmarks
    # -------------------------------------------------------------------------
    for rows, cols in [(256, 1024), (512, 2048)]:
        print(f"Benchmarking fused_bias_activation: shape=({rows}, {cols})...")
        x = torch.randn(rows, cols, dtype=torch.float32, device="cuda", requires_grad=True)
        bias = torch.randn(cols, dtype=torch.float32, device="cuda", requires_grad=True)

        res_ck = benchmark_autograd_op(lambda m, b: ck_auto.fused_bias_activation(m, b, "gelu"), [x, bias])
        res_pt = benchmark_autograd_op(lambda m, b: F.gelu(m + b, approximate="tanh"), [x, bias])
        results["microbenchmarks"][f"fused_bias_gelu_fp32_{rows}x{cols}"] = {
            "custom_cuda": res_ck,
            "pytorch_native": res_pt,
            "fwd_speedup": round(res_pt["forward_kernel_ms"]["median_ms"] / res_ck["forward_kernel_ms"]["median_ms"], 3),
            "bwd_speedup": round(res_pt["backward_kernel_ms"]["median_ms"] / res_ck["backward_kernel_ms"]["median_ms"], 3),
        }

    # -------------------------------------------------------------------------
    # 3. Matrix Multiplication Autograd Microbenchmarks
    # -------------------------------------------------------------------------
    for dim in [256, 512, 1024]:
        print(f"Benchmarking matmul autograd: {dim}x{dim}...")
        # FP16 Tensor Core WMMA
        a_fp16 = torch.randn(dim, dim, dtype=torch.float16, device="cuda", requires_grad=True)
        b_fp16 = torch.randn(dim, dim, dtype=torch.float16, device="cuda", requires_grad=True)
        res_wmma = benchmark_autograd_op(ck_auto.matmul_wmma, [a_fp16, b_fp16])
        res_cublas_fp16 = benchmark_autograd_op(lambda x, y: x @ y, [a_fp16, b_fp16])

        # FP32 Tiled SGEMM
        a_fp32 = torch.randn(dim, dim, dtype=torch.float32, device="cuda", requires_grad=True)
        b_fp32 = torch.randn(dim, dim, dtype=torch.float32, device="cuda", requires_grad=True)
        res_tiled = benchmark_autograd_op(ck_auto.matmul, [a_fp32, b_fp32])
        res_cublas_fp32 = benchmark_autograd_op(lambda x, y: x @ y, [a_fp32, b_fp32])

        results["microbenchmarks"][f"matmul_fp16_wmma_{dim}x{dim}"] = {
            "custom_wmma": res_wmma,
            "pytorch_cublas": res_cublas_fp16,
            "fwd_speedup_vs_cublas": round(res_cublas_fp16["forward_kernel_ms"]["median_ms"] / res_wmma["forward_kernel_ms"]["median_ms"], 3),
            "bwd_speedup_vs_cublas": round(res_cublas_fp16["backward_kernel_ms"]["median_ms"] / res_wmma["backward_kernel_ms"]["median_ms"], 3),
        }
        results["microbenchmarks"][f"matmul_fp32_tiled_{dim}x{dim}"] = {
            "custom_tiled": res_tiled,
            "pytorch_cublas": res_cublas_fp32,
            "fwd_speedup_vs_cublas": round(res_cublas_fp32["forward_kernel_ms"]["median_ms"] / res_tiled["forward_kernel_ms"]["median_ms"], 3),
            "bwd_speedup_vs_cublas": round(res_cublas_fp32["backward_kernel_ms"]["median_ms"] / res_tiled["backward_kernel_ms"]["median_ms"], 3),
        }

    # -------------------------------------------------------------------------
    # 4. Full Neural Network Training Step Benchmark
    # -------------------------------------------------------------------------
    print("\nBenchmarking Full Neural Network Training Step (3-Layer MLP)...")
    batch_size = 64
    in_dim = 256
    h1 = 512
    h2 = 256
    out_dim = 64

    # FP32 Training Step
    x_fp32 = torch.randn(batch_size, in_dim, device="cuda", dtype=torch.float32)
    target_fp32 = torch.randn(batch_size, out_dim, device="cuda", dtype=torch.float32)

    class PyTorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, out_dim)
        def forward(self, x):
            return self.fc3(F.relu(self.fc2(F.gelu(self.fc1(x), approximate="tanh"))))

    class CustomCUDAMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = ck_auto.CustomLinear(in_dim, h1, activation="gelu")
            self.fc2 = ck_auto.CustomLinear(h1, h2, activation="relu")
            self.fc3 = ck_auto.CustomLinear(h2, out_dim, activation="none")
        def forward(self, x):
            return self.fc3(self.fc2(self.fc1(x)))

    mlp_pt_fp32 = PyTorchMLP().cuda()
    mlp_ck_fp32 = CustomCUDAMLP().cuda()

    bench_train_pt_fp32 = benchmark_training_step(mlp_pt_fp32, x_fp32, target_fp32)
    bench_train_ck_fp32 = benchmark_training_step(mlp_ck_fp32, x_fp32, target_fp32)

    results["training_step_benchmarks"]["mlp_fp32"] = {
        "pytorch_native": bench_train_pt_fp32,
        "custom_cuda": bench_train_ck_fp32,
        "step_speedup": round(bench_train_pt_fp32["full_step_ms"]["median_ms"] / bench_train_ck_fp32["full_step_ms"]["median_ms"], 3),
    }

    # FP16 Training Step
    x_fp16 = x_fp32.half()
    target_fp16 = target_fp32.half()
    mlp_pt_fp16 = PyTorchMLP().cuda().half()
    mlp_ck_fp16 = CustomCUDAMLP().cuda().half()

    bench_train_pt_fp16 = benchmark_training_step(mlp_pt_fp16, x_fp16, target_fp16)
    bench_train_ck_fp16 = benchmark_training_step(mlp_ck_fp16, x_fp16, target_fp16)

    results["training_step_benchmarks"]["mlp_fp16_wmma"] = {
        "pytorch_native": bench_train_pt_fp16,
        "custom_cuda": bench_train_ck_fp16,
        "step_speedup": round(bench_train_pt_fp16["full_step_ms"]["median_ms"] / bench_train_ck_fp16["full_step_ms"]["median_ms"], 3),
    }

    post_telemetry = get_gpu_telemetry()
    results["metadata"]["post_telemetry"] = post_telemetry
    print(f"\nPost-benchmark GPU telemetry: {post_telemetry}")

    # Save results
    out_dir = Path("results/custom_kernels")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage2_milestone3_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark completed and saved to {out_path.resolve()}")
    return results


if __name__ == "__main__":
    run_all_benchmarks()
