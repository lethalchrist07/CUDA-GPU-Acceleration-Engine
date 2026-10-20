"""
Phase 10 Stage 2 Milestone 1: Calibrated Mixed-Precision Benchmark Harness
Applies independent audit corrections:
1. Calibrated Decimal Memory Bandwidth (GB/s = 1e9 bytes/sec) alongside binary GiB/s.
2. Direct native PyTorch reduction reference with FP32 accumulator: torch.sum(x, dtype=torch.float32).
3. Preserves original benchmark artifact and records calibrated results separately.
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
import torch.nn.functional as F

import custom_kernels as ck

BENCHMARK_SIZES = [100_000, 1_000_000, 10_000_000]
WARMUP_ITERS = 10
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
    return {
        "min_ms": round(min(timings_ms), 5),
        "median_ms": round(median_val, 5),
        "max_ms": round(max(timings_ms), 5),
        "mean_ms": round(mean_val, 5),
        "std_ms": round(math.sqrt(variance), 5),
    }

def benchmark_op(custom_fn, ref_fn, num_bytes_calc, iters=BENCH_ITERS, warmups=WARMUP_ITERS):
    """Benchmark a single kernel pair with CUDA events and host timer."""
    # Warmup
    for _ in range(warmups):
        custom_fn()
        ref_fn()
    torch.cuda.synchronize()

    # Benchmark Custom Kernel
    custom_kernel_ms = []
    custom_e2e_ms = []

    gc.collect()
    torch.cuda.empty_cache()
    gc.disable()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        t0 = time.perf_counter()
        start_event.record()
        custom_fn()
        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        custom_kernel_ms.append(start_event.elapsed_time(end_event))
        custom_e2e_ms.append((t1 - t0) * 1000.0)

    gc.enable()

    # Benchmark PyTorch Reference
    ref_kernel_ms = []
    ref_e2e_ms = []

    gc.collect()
    torch.cuda.empty_cache()
    gc.disable()

    for _ in range(iters):
        t0 = time.perf_counter()
        start_event.record()
        ref_fn()
        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ref_kernel_ms.append(start_event.elapsed_time(end_event))
        ref_e2e_ms.append((t1 - t0) * 1000.0)

    gc.enable()

    custom_stats = compute_stats(custom_kernel_ms)
    custom_e2e_stats = compute_stats(custom_e2e_ms)
    ref_stats = compute_stats(ref_kernel_ms)
    ref_e2e_stats = compute_stats(ref_e2e_ms)

    bytes_transferred = num_bytes_calc()
    # Decimal GB/s (10^9 bytes/sec) - standard hardware convention
    effective_bandwidth_gbps = (
        (bytes_transferred / (custom_stats["median_ms"] / 1000.0)) / 1e9
        if custom_stats["median_ms"] > 0
        else 0.0
    )
    # Binary GiB/s (1024^3 bytes/sec)
    effective_bandwidth_gibps = (
        (bytes_transferred / (custom_stats["median_ms"] / 1000.0)) / (1024**3)
        if custom_stats["median_ms"] > 0
        else 0.0
    )

    speedup_kernel = (
        round(ref_stats["median_ms"] / custom_stats["median_ms"], 3)
        if custom_stats["median_ms"] > 0
        else 0.0
    )
    speedup_e2e = (
        round(ref_e2e_stats["median_ms"] / custom_e2e_stats["median_ms"], 3)
        if custom_e2e_stats["median_ms"] > 0
        else 0.0
    )

    return {
        "custom_kernel": custom_stats,
        "custom_e2e": custom_e2e_stats,
        "ref_kernel": ref_stats,
        "ref_e2e": ref_e2e_stats,
        "bytes_transferred": bytes_transferred,
        "bandwidth_gbps": round(effective_bandwidth_gbps, 2),
        "bandwidth_gibps": round(effective_bandwidth_gibps, 2),
        "speedup_kernel": speedup_kernel,
        "speedup_e2e": speedup_e2e,
    }

def run_calibrated_benchmarks():
    print("=" * 80)
    print("PHASE 10 STAGE 2 MILESTONE 1: CALIBRATED MIXED-PRECISION BENCHMARK SUITE")
    print("Target: RTX 3050 6GB Laptop GPU (sm_86) | CUDA 12.4.1 | MSVC 19.38")
    print("Standard Decimal Bandwidth (GB/s = 1e9 B/s) | Direct PyTorch FP32 Accumulator Baselines")
    print("=" * 80)

    telemetry_start = get_gpu_telemetry()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"Initial Free VRAM: {free_bytes / (1024**3):.2f} GB / {total_bytes / (1024**3):.2f} GB")
    print(f"GPU Telemetry Before: {telemetry_start}")

    torch.cuda.reset_peak_memory_stats()

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "vram_total_gb": round(total_bytes / (1024**3), 2),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
            "telemetry_start": telemetry_start,
            "bandwidth_definition": "Decimal GB/s (10^9 B/s) with GiB/s (1024^3 B/s) recorded separately",
            "reduction_baseline": "torch.sum(x, dtype=torch.float32) for native single-pass accumulation",
        },
        "benchmarks": {},
    }

    precisions = [
        ("fp32", torch.float32, 4),
        ("fp16", torch.float16, 2),
        ("bf16", torch.bfloat16, 2),
    ]

    for prec_name, dtype, elem_size in precisions:
        print(f"\n>>> Benchmarking Precision: {prec_name.upper()} ({elem_size} bytes/elem)")
        results["benchmarks"][prec_name] = {}

        for N in BENCHMARK_SIZES:
            print(f"  --- Size: {N:,} elements ---")
            results["benchmarks"][prec_name][str(N)] = {}

            # 1. Vector Add (2 reads + 1 write = 3 * elem_size * N)
            a = torch.randn(N, dtype=dtype, device="cuda")
            b = torch.randn(N, dtype=dtype, device="cuda")
            res_add = benchmark_op(
                lambda: ck.vector_add(a, b),
                lambda: a + b,
                lambda: 3 * elem_size * N,
            )
            results["benchmarks"][prec_name][str(N)]["vector_add"] = res_add
            print(f"    vector_add:        Custom {res_add['custom_kernel']['median_ms']:.4f} ms ({res_add['bandwidth_gbps']} GB/s) | PyTorch {res_add['ref_kernel']['median_ms']:.4f} ms | Speedup: {res_add['speedup_kernel']}x")

            # 2. Vector Scale (1 read + 1 write = 2 * elem_size * N)
            x = torch.randn(N, dtype=dtype, device="cuda")
            res_scale = benchmark_op(
                lambda: ck.scale(x, 1.5),
                lambda: x * 1.5,
                lambda: 2 * elem_size * N,
            )
            results["benchmarks"][prec_name][str(N)]["vector_scale"] = res_scale
            print(f"    vector_scale:      Custom {res_scale['custom_kernel']['median_ms']:.4f} ms ({res_scale['bandwidth_gbps']} GB/s) | PyTorch {res_scale['ref_kernel']['median_ms']:.4f} ms | Speedup: {res_scale['speedup_kernel']}x")

            # 3. Fused Add + ReLU (2 reads + 1 write = 3 * elem_size * N)
            res_fused = benchmark_op(
                lambda: ck.fused_add_relu(a, b),
                lambda: F.relu(a + b),
                lambda: 3 * elem_size * N,
            )
            results["benchmarks"][prec_name][str(N)]["fused_add_relu"] = res_fused
            print(f"    fused_add_relu:    Custom {res_fused['custom_kernel']['median_ms']:.4f} ms ({res_fused['bandwidth_gbps']} GB/s) | PyTorch {res_fused['ref_kernel']['median_ms']:.4f} ms | Speedup: {res_fused['speedup_kernel']}x")

            # 4. ReLU (1 read + 1 write = 2 * elem_size * N)
            res_relu = benchmark_op(
                lambda: ck.relu(x),
                lambda: F.relu(x),
                lambda: 2 * elem_size * N,
            )
            results["benchmarks"][prec_name][str(N)]["relu"] = res_relu
            print(f"    relu:              Custom {res_relu['custom_kernel']['median_ms']:.4f} ms ({res_relu['bandwidth_gbps']} GB/s) | PyTorch {res_relu['ref_kernel']['median_ms']:.4f} ms | Speedup: {res_relu['speedup_kernel']}x")

            # 5. GELU (1 read + 1 write = 2 * elem_size * N)
            res_gelu = benchmark_op(
                lambda: ck.gelu(x),
                lambda: F.gelu(x, approximate="tanh"),
                lambda: 2 * elem_size * N,
            )
            results["benchmarks"][prec_name][str(N)]["gelu"] = res_gelu
            print(f"    gelu:              Custom {res_gelu['custom_kernel']['median_ms']:.4f} ms ({res_gelu['bandwidth_gbps']} GB/s) | PyTorch {res_gelu['ref_kernel']['median_ms']:.4f} ms | Speedup: {res_gelu['speedup_kernel']}x")

            # 6. Sigmoid (1 read + 1 write = 2 * elem_size * N)
            res_sig = benchmark_op(
                lambda: ck.sigmoid(x),
                lambda: torch.sigmoid(x),
                lambda: 2 * elem_size * N,
            )
            results["benchmarks"][prec_name][str(N)]["sigmoid"] = res_sig
            print(f"    sigmoid:           Custom {res_sig['custom_kernel']['median_ms']:.4f} ms ({res_sig['bandwidth_gbps']} GB/s) | PyTorch {res_sig['ref_kernel']['median_ms']:.4f} ms | Speedup: {res_sig['speedup_kernel']}x")

            # 7. Reduce Sum: Direct PyTorch single-pass FP32 accumulator baseline
            ref_sum_fn = (
                (lambda: torch.sum(x, dtype=torch.float32))
                if dtype in (torch.float16, torch.bfloat16)
                else (lambda: torch.sum(x))
            )
            res_rsum = benchmark_op(
                lambda: ck.reduce_sum(x, output_fp32=True),
                ref_sum_fn,
                lambda: elem_size * N,
            )
            results["benchmarks"][prec_name][str(N)]["reduce_sum"] = res_rsum
            print(f"    reduce_sum:        Custom {res_rsum['custom_kernel']['median_ms']:.4f} ms ({res_rsum['bandwidth_gbps']} GB/s) | PyTorch {res_rsum['ref_kernel']['median_ms']:.4f} ms | Speedup: {res_rsum['speedup_kernel']}x")

            # 8. Reduce Max
            res_rmax = benchmark_op(
                lambda: ck.reduce_max(x),
                lambda: torch.max(x),
                lambda: elem_size * N,
            )
            results["benchmarks"][prec_name][str(N)]["reduce_max"] = res_rmax
            print(f"    reduce_max:        Custom {res_rmax['custom_kernel']['median_ms']:.4f} ms ({res_rmax['bandwidth_gbps']} GB/s) | PyTorch {res_rmax['ref_kernel']['median_ms']:.4f} ms | Speedup: {res_rmax['speedup_kernel']}x")

            del a, b, x
            torch.cuda.empty_cache()

    telemetry_end = get_gpu_telemetry()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final_free, _ = torch.cuda.mem_get_info()

    results["metadata"]["telemetry_end"] = telemetry_end
    results["metadata"]["peak_allocated_mb"] = round(peak_allocated / (1024**2), 2)
    results["metadata"]["peak_reserved_mb"] = round(peak_reserved / (1024**2), 2)
    results["metadata"]["final_free_vram_gb"] = round(final_free / (1024**3), 2)

    print("\n" + "=" * 80)
    print(f"Peak VRAM Allocated: {peak_allocated / (1024**2):.2f} MB")
    print(f"Peak VRAM Reserved:  {peak_reserved / (1024**2):.2f} MB")
    print(f"Final Free VRAM:     {final_free / (1024**3):.2f} GB")
    print(f"GPU Telemetry After: {telemetry_end}")
    print("=" * 80)

    out_path = Path("results/custom_kernels/stage2_milestone1_benchmark_results_calibrated.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Calibrated benchmark results saved to {out_path.resolve()}")
    return results

if __name__ == "__main__":
    run_calibrated_benchmarks()
