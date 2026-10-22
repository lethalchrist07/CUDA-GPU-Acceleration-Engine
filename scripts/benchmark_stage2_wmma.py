"""
Phase 10 Stage 2 Milestone 2: Ampere Tensor Core WMMA Matrix Multiplication Benchmark
Compares:
1. Stage 1 Tiled FP32 SGEMM (CUDA SIMD baseline)
2. Custom FP16 Tensor Core WMMA (sm_86 Ampere, FP32 accumulator, FP16 output)
3. Custom FP16 Tensor Core WMMA (sm_86 Ampere, FP32 accumulator, FP32 output)
4. PyTorch FP16 torch.matmul (cuBLAS Tensor Core baseline)
5. PyTorch FP32 torch.matmul (cuBLAS FP32 baseline)

Measures:
- Kernel-only latency (CUDA events)
- End-to-end latency (wall clock)
- Median, minimum, maximum, stddev
- Effective TFLOPS
- Speedup vs Stage 1 SGEMM and vs PyTorch FP16
- Peak VRAM allocation
"""
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import math

import torch

import custom_kernels as ck

# Benchmark configuration
BENCHMARK_SHAPES = [
    # Square matrices
    {"name": "square_128", "M": 128, "K": 128, "N": 128},
    {"name": "square_256", "M": 256, "K": 256, "N": 256},
    {"name": "square_512", "M": 512, "K": 512, "N": 512},
    {"name": "square_1024", "M": 1024, "K": 1024, "N": 1024},
    # Rectangular matrices
    {"name": "rect_512x256x1024", "M": 512, "K": 256, "N": 1024},
    {"name": "rect_256x1024x512", "M": 256, "K": 1024, "N": 512},
    {"name": "rect_128x512x64", "M": 128, "K": 512, "N": 64},
    # Irregular / Non-multiples of 16
    {"name": "irregular_100", "M": 100, "K": 100, "N": 100},
    {"name": "irregular_300x500x200", "M": 300, "K": 500, "N": 200},
    {"name": "irregular_65", "M": 65, "K": 65, "N": 65},
]

WARMUP_ITERS = 20
BENCH_ITERS = 50

def get_gpu_telemetry() -> Dict[str, Any]:
    """Query current GPU state via nvidia-smi."""
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
    """Compute summary statistics for a list of timing samples."""
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

def benchmark_op(fn, a, b, iters: int = BENCH_ITERS, warmup: int = WARMUP_ITERS) -> Dict[str, Any]:
    """Benchmark a single matrix multiplication operator."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        _ = fn(a, b)
    torch.cuda.synchronize()

    # Kernel-only timings via CUDA events
    kernel_timings: List[float] = []
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        _ = fn(a, b)
        end_events[i].record()
    torch.cuda.synchronize()

    for i in range(iters):
        kernel_timings.append(start_events[i].elapsed_time(end_events[i]))

    # End-to-end wall clock timings
    e2e_timings: List[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_timings.append((t1 - t0) * 1000.0)

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "kernel_stats": compute_stats(kernel_timings),
        "e2e_stats": compute_stats(e2e_timings),
        "peak_vram_mb": round(peak_mem_mb, 3),
    }

def run_benchmarks() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 10 STAGE 2 MILESTONE 2: AMPERE TENSOR CORE WMMA BENCHMARK")
    print("Device: NVIDIA GeForce RTX 3050 Laptop GPU (sm_86, 6 GB VRAM)")
    print("Toolchain: CUDA Toolkit 12.4.1 + MSVC 19.38 + PyTorch 2.6.0+cu124")
    print("=" * 80)

    pre_telemetry = get_gpu_telemetry()
    print(f"Pre-benchmark GPU telemetry: {pre_telemetry}")

    results = []

    for shape_cfg in BENCHMARK_SHAPES:
        name = shape_cfg["name"]
        M = shape_cfg["M"]
        K = shape_cfg["K"]
        N = shape_cfg["N"]

        flops = 2.0 * M * N * K
        print(f"\nBenchmarking {name}: M={M}, K={K}, N={N} (Total FLOPs: {flops:,.0f})")

        # FP16 tensors
        a_fp16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b_fp16 = torch.randn(K, N, dtype=torch.float16, device="cuda")

        # FP32 tensors
        a_fp32 = a_fp16.float()
        b_fp32 = b_fp16.float()

        shape_result = {
            "name": name,
            "M": M,
            "K": K,
            "N": N,
            "flops": flops,
            "implementations": {},
        }

        # 1. Stage 1 Tiled SGEMM (FP32)
        print("  Running Stage 1 Tiled SGEMM (FP32)...", end="", flush=True)
        res_s1 = benchmark_op(lambda x, y: ck.matmul(x, y), a_fp32, b_fp32)
        t_s1 = res_s1["kernel_stats"]["median_ms"]
        tflops_s1 = (flops / (t_s1 * 1e-3)) / 1e12 if t_s1 > 0 else 0.0
        res_s1["effective_tflops"] = round(tflops_s1, 4)
        shape_result["implementations"]["stage1_sgemm_fp32"] = res_s1
        print(f" {t_s1:.4f} ms | {tflops_s1:.3f} TFLOPS")

        # 2. Custom FP16 Tensor Core WMMA (FP16 out)
        print("  Running Custom WMMA (FP16 out)...", end="", flush=True)
        res_wmma_fp16 = benchmark_op(lambda x, y: ck.matmul_wmma(x, y, output_fp32=False), a_fp16, b_fp16)
        t_wmma_fp16 = res_wmma_fp16["kernel_stats"]["median_ms"]
        tflops_wmma_fp16 = (flops / (t_wmma_fp16 * 1e-3)) / 1e12 if t_wmma_fp16 > 0 else 0.0
        res_wmma_fp16["effective_tflops"] = round(tflops_wmma_fp16, 4)
        shape_result["implementations"]["custom_wmma_fp16"] = res_wmma_fp16
        print(f" {t_wmma_fp16:.4f} ms | {tflops_wmma_fp16:.3f} TFLOPS")

        # 3. Custom FP16 Tensor Core WMMA (FP32 out)
        print("  Running Custom WMMA (FP32 out)...", end="", flush=True)
        res_wmma_fp32 = benchmark_op(lambda x, y: ck.matmul_wmma(x, y, output_fp32=True), a_fp16, b_fp16)
        t_wmma_fp32 = res_wmma_fp32["kernel_stats"]["median_ms"]
        tflops_wmma_fp32 = (flops / (t_wmma_fp32 * 1e-3)) / 1e12 if t_wmma_fp32 > 0 else 0.0
        res_wmma_fp32["effective_tflops"] = round(tflops_wmma_fp32, 4)
        shape_result["implementations"]["custom_wmma_fp32_out"] = res_wmma_fp32
        print(f" {t_wmma_fp32:.4f} ms | {tflops_wmma_fp32:.3f} TFLOPS")

        # 4. PyTorch FP16 torch.matmul
        print("  Running PyTorch FP16 torch.matmul...", end="", flush=True)
        res_pt_fp16 = benchmark_op(lambda x, y: torch.matmul(x, y), a_fp16, b_fp16)
        t_pt_fp16 = res_pt_fp16["kernel_stats"]["median_ms"]
        tflops_pt_fp16 = (flops / (t_pt_fp16 * 1e-3)) / 1e12 if t_pt_fp16 > 0 else 0.0
        res_pt_fp16["effective_tflops"] = round(tflops_pt_fp16, 4)
        shape_result["implementations"]["pytorch_fp16"] = res_pt_fp16
        print(f" {t_pt_fp16:.4f} ms | {tflops_pt_fp16:.3f} TFLOPS")

        # 5. PyTorch FP32 torch.matmul
        print("  Running PyTorch FP32 torch.matmul...", end="", flush=True)
        res_pt_fp32 = benchmark_op(lambda x, y: torch.matmul(x, y), a_fp32, b_fp32)
        t_pt_fp32 = res_pt_fp32["kernel_stats"]["median_ms"]
        tflops_pt_fp32 = (flops / (t_pt_fp32 * 1e-3)) / 1e12 if t_pt_fp32 > 0 else 0.0
        res_pt_fp32["effective_tflops"] = round(tflops_pt_fp32, 4)
        shape_result["implementations"]["pytorch_fp32"] = res_pt_fp32
        print(f" {t_pt_fp32:.4f} ms | {tflops_pt_fp32:.3f} TFLOPS")

        # Compute speedup metrics
        speedup_vs_stage1 = round(t_s1 / t_wmma_fp16, 3) if t_wmma_fp16 > 0 else 0.0
        speedup_vs_pt_fp16 = round(t_pt_fp16 / t_wmma_fp16, 3) if t_wmma_fp16 > 0 else 0.0
        wmma_fraction_of_cublas = round((tflops_wmma_fp16 / tflops_pt_fp16) * 100.0, 2) if tflops_pt_fp16 > 0 else 0.0

        shape_result["comparison"] = {
            "speedup_wmma_vs_stage1_sgemm": speedup_vs_stage1,
            "speedup_wmma_vs_pytorch_fp16": speedup_vs_pt_fp16,
            "wmma_pct_of_cublas_fp16": wmma_fraction_of_cublas,
        }
        print(f"  --> Speedup vs Stage 1 SGEMM: {speedup_vs_stage1:.2f}x | % of PyTorch cuBLAS FP16: {wmma_fraction_of_cublas:.1f}%")

        results.append(shape_result)

        del a_fp16, b_fp16, a_fp32, b_fp32
        gc.collect()
        torch.cuda.empty_cache()

    post_telemetry = get_gpu_telemetry()
    print(f"\nPost-benchmark GPU telemetry: {post_telemetry}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "pre_telemetry": pre_telemetry,
        "post_telemetry": post_telemetry,
        "shapes": results,
    }

    out_path = Path("results/custom_kernels/stage2_milestone2_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBenchmark results successfully saved to {out_path.resolve()}")
    return summary

if __name__ == "__main__":
    run_benchmarks()
