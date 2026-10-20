"""
Comprehensive Benchmark Suite for Custom CUDA Kernel Library (Phase 10 Stage 1)
Hardware Target: NVIDIA GeForce RTX 3050 6GB Laptop GPU (sm_86, Ampere)
Toolchain: CUDA 12.4.1 + MSVC 19.38

MANDATORY BENCHMARK INTEGRITY REQUIREMENTS:
1. Pure Kernel vs End-to-End Timing:
   - Kernel timing via torch.cuda.Event(enable_timing=True) on active stream.
   - End-to-end timing via time.perf_counter() with synchronized host.
2. Robust Statistical Sampling:
   - 10 warmup iterations to prime instruction cache and GPU clocks.
   - 50 to 100 timed iterations.
   - Computes mean, median, standard deviation, and min/max.
3. Accurate Hardware Specifications:
   - Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
   - SM Count: 20 SMs
   - Bus Width: 96 bits
   - Memory Clock: 7001 MHz
   - Peak Theoretical Bandwidth: 168.02 GB/s
4. GPU Memory Safety:
   - Live memory accounting ensuring active allocations stay strictly below 1.5 GB.
5. Honest Performance Metrics:
   - Effective bandwidth (GB/s) and compute throughput (GFLOPS) vs PyTorch native baselines.
"""

import os
import sys
import time
import json
import gc
import statistics
from typing import Dict, Any, List, Callable
import torch
import torch.nn.functional as F

# Ensure custom_kernels is discoverable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import custom_kernels as ck

# Verified Hardware Constants for RTX 3050 Laptop GPU 6GB
PEAK_BANDWIDTH_GB_S = 168.02  # 96-bit bus @ 7001 MHz GDDR6
SM_COUNT = 20
VRAM_BUDGET_BYTES = 1.5 * 1024 * 1024 * 1024  # 1.5 GB safe limit


def verify_vram_safety(required_bytes: int):
    """Enforces conservative VRAM limits on the 6 GB RTX 3050 GPU."""
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if required_bytes + allocated > VRAM_BUDGET_BYTES:
        raise MemoryError(
            f"VRAM budget exceeded: requested {required_bytes / 1e6:.1f} MB + "
            f"allocated {allocated / 1e6:.1f} MB exceeds {VRAM_BUDGET_BYTES / 1e6:.1f} MB limit."
        )


def benchmark_op(
    name: str,
    custom_fn: Callable[[], Any],
    ref_fn: Callable[[], Any],
    bytes_transferred: int,
    flop_count: int = 0,
    warmup_iters: int = 10,
    bench_iters: int = 50,
) -> Dict[str, Any]:
    """
    Benchmarks custom kernel vs PyTorch native reference.
    Measures pure CUDA event kernel time and host end-to-end time separately with GC isolation.
    """
    # Warmup Custom
    for _ in range(warmup_iters):
        custom_fn()
    torch.cuda.synchronize()

    # Time Custom Kernel with GC isolation
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    custom_e2e_times_ms = []

    gc.collect()
    torch.cuda.empty_cache()
    gc.disable()
    try:
        for i in range(bench_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start_events[i].record()
            custom_fn()
            end_events[i].record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            custom_e2e_times_ms.append((t1 - t0) * 1000.0)
    finally:
        gc.enable()

    custom_kernel_times_ms = [
        start_events[i].elapsed_time(end_events[i]) for i in range(bench_iters)
    ]
    custom_kernel_ms = statistics.median(custom_kernel_times_ms)
    custom_e2e_ms = statistics.median(custom_e2e_times_ms)

    # Warmup Reference
    for _ in range(warmup_iters):
        ref_fn()
    torch.cuda.synchronize()

    # Time Reference Kernel with GC isolation
    ref_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    ref_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    ref_e2e_times_ms = []

    gc.collect()
    torch.cuda.empty_cache()
    gc.disable()
    try:
        for i in range(bench_iters):
            torch.cuda.synchronize()
            ref_t0 = time.perf_counter()
            ref_start_events[i].record()
            ref_fn()
            ref_end_events[i].record()
            torch.cuda.synchronize()
            ref_t1 = time.perf_counter()
            ref_e2e_times_ms.append((ref_t1 - ref_t0) * 1000.0)
    finally:
        gc.enable()

    ref_kernel_times_ms = [
        ref_start_events[i].elapsed_time(ref_end_events[i]) for i in range(bench_iters)
    ]
    ref_kernel_ms = statistics.median(ref_kernel_times_ms)
    ref_e2e_ms = statistics.median(ref_e2e_times_ms)

    # Statistical variability
    custom_kernel_std = statistics.stdev(custom_kernel_times_ms) if len(custom_kernel_times_ms) > 1 else 0.0
    ref_kernel_std = statistics.stdev(ref_kernel_times_ms) if len(ref_kernel_times_ms) > 1 else 0.0

    # Metrics calculation
    custom_kernel_sec = custom_kernel_ms / 1000.0
    effective_bw_gb_s = (bytes_transferred / (custom_kernel_sec * 1e9)) if bytes_transferred > 0 else 0.0
    bw_efficiency_pct = (effective_bw_gb_s / PEAK_BANDWIDTH_GB_S) * 100.0

    tflops = (flop_count / (custom_kernel_sec * 1e12)) if flop_count > 0 else 0.0
    speedup_kernel = ref_kernel_ms / custom_kernel_ms if custom_kernel_ms > 0 else 0.0
    speedup_e2e = ref_e2e_ms / custom_e2e_ms if custom_e2e_ms > 0 else 0.0

    return {
        "operation": name,
        "custom_kernel_ms": round(custom_kernel_ms, 4),
        "custom_kernel_min_ms": round(min(custom_kernel_times_ms), 4),
        "custom_kernel_max_ms": round(max(custom_kernel_times_ms), 4),
        "custom_kernel_std_ms": round(custom_kernel_std, 4),
        "custom_e2e_ms": round(custom_e2e_ms, 4),
        "custom_e2e_min_ms": round(min(custom_e2e_times_ms), 4),
        "custom_e2e_max_ms": round(max(custom_e2e_times_ms), 4),
        "ref_kernel_ms": round(ref_kernel_ms, 4),
        "ref_kernel_min_ms": round(min(ref_kernel_times_ms), 4),
        "ref_kernel_max_ms": round(max(ref_kernel_times_ms), 4),
        "ref_kernel_std_ms": round(ref_kernel_std, 4),
        "ref_e2e_ms": round(ref_e2e_ms, 4),
        "speedup_kernel": round(speedup_kernel, 2),
        "speedup_e2e": round(speedup_e2e, 2),
        "effective_bw_gb_s": round(effective_bw_gb_s, 2),
        "bw_efficiency_pct": round(bw_efficiency_pct, 1),
        "throughput_tflops": round(tflops, 3) if flop_count > 0 else None,
    }


def get_gpu_environment() -> Dict[str, Any]:
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,clocks.mem,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "temperature_c": float(parts[0]),
            "graphics_clock_mhz": float(parts[1]),
            "memory_clock_mhz": float(parts[2]),
            "power_draw_w": float(parts[3]),
        }
    except Exception:
        return {}


def run_all_benchmarks():
    print("=" * 80)
    print("PHASE 10 STAGE 1: CUSTOM CUDA KERNEL BENCHMARK SUITE")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Theoretical Peak Bandwidth: {PEAK_BANDWIDTH_GB_S} GB/s (96-bit GDDR6 @ 7001 MHz)")
    print(f"VRAM Safety Ceiling: {VRAM_BUDGET_BYTES / (1024**2):.0f} MB")
    print("=" * 80)

    results = []

    # 1. Vector Addition (N = 10,000,000 float32 elements ~ 40 MB per tensor, total ~120 MB)
    N_VEC = 10_000_000
    verify_vram_safety(N_VEC * 4 * 3)
    a_vec = torch.randn(N_VEC, device="cuda", dtype=torch.float32)
    b_vec = torch.randn(N_VEC, device="cuda", dtype=torch.float32)
    # 2 reads + 1 write = 3 * N * 4 bytes
    res_add = benchmark_op(
        "vector_add (10M)",
        lambda: ck.vector_add(a_vec, b_vec),
        lambda: torch.add(a_vec, b_vec),
        bytes_transferred=N_VEC * 4 * 3,
    )
    results.append(res_add)
    print(f"[{res_add['operation']}] Custom: {res_add['custom_kernel_ms']} ms | Ref: {res_add['ref_kernel_ms']} ms | Bandwidth: {res_add['effective_bw_gb_s']} GB/s ({res_add['bw_efficiency_pct']}% peak)")

    # 2. Vector Scale (N = 10,000,000 float32 elements)
    # 1 read + 1 write = 2 * N * 4 bytes
    res_scale = benchmark_op(
        "vector_scale (10M)",
        lambda: ck.scale(a_vec, 2.5),
        lambda: a_vec * 2.5,
        bytes_transferred=N_VEC * 4 * 2,
    )
    results.append(res_scale)
    print(f"[{res_scale['operation']}] Custom: {res_scale['custom_kernel_ms']} ms | Ref: {res_scale['ref_kernel_ms']} ms | Bandwidth: {res_scale['effective_bw_gb_s']} GB/s ({res_scale['bw_efficiency_pct']}% peak)")

    # 3. Fused Add + ReLU (N = 10,000,000 float32 elements)
    # Fused reads A, B and writes Out = 3 * N * 4 bytes
    # Unfused PyTorch reads A, B -> writes Temp -> reads Temp -> writes Out = 5 * N * 4 bytes
    res_fused_add_relu = benchmark_op(
        "fused_add_relu (10M)",
        lambda: ck.fused_add_relu(a_vec, b_vec),
        lambda: torch.relu(a_vec + b_vec),
        bytes_transferred=N_VEC * 4 * 3,
    )
    results.append(res_fused_add_relu)
    print(f"[{res_fused_add_relu['operation']}] Custom: {res_fused_add_relu['custom_kernel_ms']} ms | Ref: {res_fused_add_relu['ref_kernel_ms']} ms | Speedup: {res_fused_add_relu['speedup_kernel']}x | Bandwidth: {res_fused_add_relu['effective_bw_gb_s']} GB/s")

    # 4. Fused Bias + Activation (Matrix 4096 x 2048, Bias 2048)
    rows, cols = 4096, 2048
    N_MAT = rows * cols
    verify_vram_safety(N_MAT * 4 * 2 + cols * 4)
    x_mat = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    bias = torch.randn(cols, device="cuda", dtype=torch.float32)

    res_fused_bias_relu = benchmark_op(
        "fused_bias_relu (4096x2048)",
        lambda: ck.fused_bias_activation(x_mat, bias, "relu"),
        lambda: torch.relu(x_mat + bias),
        bytes_transferred=N_MAT * 4 * 2 + cols * 4,
    )
    results.append(res_fused_bias_relu)
    print(f"[{res_fused_bias_relu['operation']}] Custom: {res_fused_bias_relu['custom_kernel_ms']} ms | Ref: {res_fused_bias_relu['ref_kernel_ms']} ms | Speedup: {res_fused_bias_relu['speedup_kernel']}x")

    res_fused_bias_gelu = benchmark_op(
        "fused_bias_gelu (4096x2048)",
        lambda: ck.fused_bias_activation(x_mat, bias, "gelu"),
        lambda: F.gelu(x_mat + bias, approximate="tanh"),
        bytes_transferred=N_MAT * 4 * 2 + cols * 4,
    )
    results.append(res_fused_bias_gelu)
    print(f"[{res_fused_bias_gelu['operation']}] Custom: {res_fused_bias_gelu['custom_kernel_ms']} ms | Ref: {res_fused_bias_gelu['ref_kernel_ms']} ms | Speedup: {res_fused_bias_gelu['speedup_kernel']}x")

    # 5. Activations (N = 10,000,000)
    res_relu = benchmark_op(
        "relu (10M)",
        lambda: ck.relu(a_vec),
        lambda: torch.relu(a_vec),
        bytes_transferred=N_VEC * 4 * 2,
    )
    results.append(res_relu)
    print(f"[{res_relu['operation']}] Custom: {res_relu['custom_kernel_ms']} ms | Ref: {res_relu['ref_kernel_ms']} ms | Bandwidth: {res_relu['effective_bw_gb_s']} GB/s")

    res_gelu = benchmark_op(
        "gelu_tanh (10M)",
        lambda: ck.gelu(a_vec),
        lambda: F.gelu(a_vec, approximate="tanh"),
        bytes_transferred=N_VEC * 4 * 2,
    )
    results.append(res_gelu)
    print(f"[{res_gelu['operation']}] Custom: {res_gelu['custom_kernel_ms']} ms | Ref: {res_gelu['ref_kernel_ms']} ms | Bandwidth: {res_gelu['effective_bw_gb_s']} GB/s")

    res_sig = benchmark_op(
        "sigmoid (10M)",
        lambda: ck.sigmoid(a_vec),
        lambda: torch.sigmoid(a_vec),
        bytes_transferred=N_VEC * 4 * 2,
    )
    results.append(res_sig)
    print(f"[{res_sig['operation']}] Custom: {res_sig['custom_kernel_ms']} ms | Ref: {res_sig['ref_kernel_ms']} ms | Bandwidth: {res_sig['effective_bw_gb_s']} GB/s")

    # Clean up large vectors before reductions & matmul
    del a_vec, b_vec, x_mat, bias
    torch.cuda.empty_cache()

    # 6. Reductions (N = 10,000,000)
    N_RED = 10_000_000
    x_red = torch.randn(N_RED, device="cuda", dtype=torch.float32)
    # Read N elements = N * 4 bytes
    res_sum = benchmark_op(
        "reduce_sum (10M)",
        lambda: ck.reduce_sum(x_red),
        lambda: torch.sum(x_red),
        bytes_transferred=N_RED * 4,
    )
    results.append(res_sum)
    print(f"[{res_sum['operation']}] Custom: {res_sum['custom_kernel_ms']} ms | Ref: {res_sum['ref_kernel_ms']} ms | Bandwidth: {res_sum['effective_bw_gb_s']} GB/s")

    res_max = benchmark_op(
        "reduce_max (10M)",
        lambda: ck.reduce_max(x_red),
        lambda: torch.max(x_red),
        bytes_transferred=N_RED * 4,
    )
    results.append(res_max)
    print(f"[{res_max['operation']}] Custom: {res_max['custom_kernel_ms']} ms | Ref: {res_max['ref_kernel_ms']} ms | Bandwidth: {res_max['effective_bw_gb_s']} GB/s")

    del x_red
    torch.cuda.empty_cache()

    # 7. Tiled Matmul (M=1024, K=1024, N=1024)
    # FLOPs = 2 * M * N * K
    M, K, N = 1024, 1024, 1024
    verify_vram_safety((M * K + K * N + M * N) * 4)
    A_mm = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_mm = torch.randn(K, N, device="cuda", dtype=torch.float32)
    flops = 2 * M * N * K
    res_mm = benchmark_op(
        "matmul_tiled (1024x1024)",
        lambda: ck.matmul(A_mm, B_mm),
        lambda: torch.matmul(A_mm, B_mm),
        bytes_transferred=(M * K + K * N + M * N) * 4,
        flop_count=flops,
    )
    results.append(res_mm)
    print(f"[{res_mm['operation']}] Custom: {res_mm['custom_kernel_ms']} ms ({res_mm['throughput_tflops']} TFLOPS) | Ref cuBLAS: {res_mm['ref_kernel_ms']} ms | Speedup vs cuBLAS: {res_mm['speedup_kernel']}x")

    del A_mm, B_mm
    torch.cuda.empty_cache()

    # Save output JSON (preserving original stage1_benchmark_results.json)
    out_dir = os.path.join(PROJECT_ROOT, "results", "custom_kernels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stage1_benchmark_results_audited.json")
    with open(out_path, "w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "peak_theoretical_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
            "sm_count": SM_COUNT,
            "gpu_environment": get_gpu_environment(),
            "results": results
        }, f, indent=2)
    print(f"\n[OK] Audited benchmark results written to {out_path}")

    # Generate Markdown Table
    print("\n" + "=" * 80)
    print("STAGE 1 BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    headers = ["Operation", "Custom Kernel (ms)", "Custom E2E (ms)", "PyTorch Ref (ms)", "Speedup (Kernel)", "Eff. Bandwidth (GB/s)", "% Peak BW"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---'] * len(headers))} |")
    for r in results:
        print(f"| {r['operation']} | {r['custom_kernel_ms']} | {r['custom_e2e_ms']} | {r['ref_kernel_ms']} | {r['speedup_kernel']}x | {r['effective_bw_gb_s']} | {r['bw_efficiency_pct']}% |")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
