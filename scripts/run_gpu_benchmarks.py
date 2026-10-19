"""
Phase 9 GPU Performance Engineering & Benchmark Suite.
Measures:
1. Training Throughput & Duration (CPU vs GPU FP32 vs GPU AMP across batch sizes)
2. Inference Latency & Throughput (p50, p95, p99, mean ms, samples/sec)
3. Peak VRAM Allocation & Reservation
4. Accurate Speedup calculations strictly against measured CPU reference
All measurements include warmups, explicit CUDA synchronizations, and verified boundaries.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.data.cifar10 import create_cifar10_dataloaders
from engine.device.device_manager import DeviceManager
from engine.device.hardware_monitor import HardwareMonitor
from engine.models.resnet import ResNet18, create_resnet18

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GPUBenchmarkSuite")


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    num_warmup: int = 10,
    num_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Measure inference latency distribution (p50, p95, p99, mean) and throughput.
    """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    # Reset peak memory stats before measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    latencies_ms: List[float] = []

    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_start = time.perf_counter()

            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    lat_np = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_np))
    p50_lat = float(np.percentile(lat_np, 50))
    p95_lat = float(np.percentile(lat_np, 95))
    p99_lat = float(np.percentile(lat_np, 99))
    min_lat = float(np.min(lat_np))
    max_lat = float(np.max(lat_np))

    # Samples per second: (batch_size * num_iterations) / total_seconds
    total_time_sec = float(np.sum(lat_np)) / 1000.0
    throughput = (batch_size * num_iterations) / total_time_sec if total_time_sec > 0 else 0.0

    peak_vram_alloc = 0.0
    peak_vram_res = 0.0
    if device.type == "cuda":
        peak_vram_alloc = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2)
        peak_vram_res = round(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0), 2)

    return {
        "device": str(device),
        "use_amp": use_amp,
        "batch_size": batch_size,
        "mean_latency_ms": round(mean_lat, 3),
        "p50_latency_ms": round(p50_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "p99_latency_ms": round(p99_lat, 3),
        "min_latency_ms": round(min_lat, 3),
        "max_latency_ms": round(max_lat, 3),
        "throughput_samples_per_sec": round(throughput, 1),
        "peak_vram_allocated_mb": peak_vram_alloc,
        "peak_vram_reserved_mb": peak_vram_res,
        "num_iterations": num_iterations,
    }


def benchmark_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    num_warmup: int = 5,
    num_steps: int = 30,
) -> Dict[str, Any]:
    """
    Measure training throughput (step forward + backward + optimizer update) with CUDA sync.
    """
    model.train()
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None

    # Warmup
    for _ in range(num_warmup):
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    step_times_ms: List[float] = []

    for _ in range(num_steps):
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_end = time.perf_counter()
        step_times_ms.append((t_end - t_start) * 1000.0)

    step_np = np.array(step_times_ms)
    mean_step_ms = float(np.mean(step_np))
    p50_step_ms = float(np.percentile(step_np, 50))
    p95_step_ms = float(np.percentile(step_np, 95))

    total_time_sec = float(np.sum(step_np)) / 1000.0
    throughput = (batch_size * num_steps) / total_time_sec if total_time_sec > 0 else 0.0

    peak_vram_alloc = 0.0
    peak_vram_res = 0.0
    if device.type == "cuda":
        peak_vram_alloc = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2)
        peak_vram_res = round(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0), 2)

    return {
        "device": str(device),
        "use_amp": use_amp,
        "batch_size": batch_size,
        "mean_step_time_ms": round(mean_step_ms, 3),
        "p50_step_time_ms": round(p50_step_ms, 3),
        "p95_step_time_ms": round(p95_step_ms, 3),
        "throughput_samples_per_sec": round(throughput, 1),
        "peak_vram_allocated_mb": peak_vram_alloc,
        "peak_vram_reserved_mb": peak_vram_res,
        "num_steps": num_steps,
    }


def run_full_gpu_benchmarks():
    print("=" * 100)
    print("      CUDA AI Acceleration Engine - Phase 9 GPU Performance Benchmark Battery")
    print("=" * 100)

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"[*] CUDA Functional: {cuda_available} | GPU: {gpu_name}")
    print(f"[*] PyTorch Version: {torch.__version__} | CUDA Version: {torch.version.cuda}")

    monitor = HardwareMonitor(device_id=0)
    init_telem = monitor.get_telemetry()
    print(f"[*] Baseline Telemetry: Temp={init_telem['temperature_c']}°C, Util={init_telem['utilization_gpu_pct']}%, VRAM Free={init_telem['vram_free_mb']} MB\n")

    torch.manual_seed(42)
    base_model = create_resnet18(num_classes=10, cifar_stem=True)
    initial_weights = copy.deepcopy(base_model.state_dict())

    # =========================================================================
    # PART 1: TRAINING STEP BENCHMARKS (CPU vs GPU FP32 vs GPU AMP vs BATCHES)
    # =========================================================================
    print("-" * 100)
    print(">>> PART 1: RESNET-18 TRAINING BENCHMARKS (Forward + Backward + Optimizer)")
    print("-" * 100)

    training_results: List[Dict[str, Any]] = []

    # 1. CPU Reference (Batch Size 64)
    print("[1/6] Measuring CPU Reference Training (Batch Size 64)...")
    cpu_device = torch.device("cpu")
    cpu_model = create_resnet18(num_classes=10, cifar_stem=True)
    cpu_model.load_state_dict(copy.deepcopy(initial_weights))
    cpu_opt = torch.optim.AdamW(cpu_model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    cpu_train_res = benchmark_training_step(
        model=cpu_model,
        optimizer=cpu_opt,
        criterion=crit,
        device=cpu_device,
        use_amp=False,
        batch_size=64,
        num_warmup=2,
        num_steps=10,  # Controlled steps for CPU so test doesn't stall
    )
    cpu_train_res["config"] = "CPU FP32 (b=64)"
    cpu_train_res["speedup_vs_cpu"] = 1.0
    training_results.append(cpu_train_res)
    cpu_ref_throughput = cpu_train_res["throughput_samples_per_sec"]
    print(f"      CPU Throughput: {cpu_ref_throughput:.1f} samples/sec | Step: {cpu_train_res['mean_step_time_ms']:.1f} ms")

    if cuda_available:
        gpu_device = torch.device("cuda:0")

        # 2. GPU FP32 (Batch Size 64)
        print("[2/6] Measuring GPU FP32 Training (Batch Size 64)...")
        gpu_fp32_model = create_resnet18(num_classes=10, cifar_stem=True).to(gpu_device)
        gpu_fp32_model.load_state_dict(copy.deepcopy(initial_weights))
        gpu_fp32_opt = torch.optim.AdamW(gpu_fp32_model.parameters(), lr=0.001)

        gpu_fp32_train_res = benchmark_training_step(
            model=gpu_fp32_model,
            optimizer=gpu_fp32_opt,
            criterion=crit,
            device=gpu_device,
            use_amp=False,
            batch_size=64,
            num_warmup=5,
            num_steps=30,
        )
        gpu_fp32_train_res["config"] = "GPU FP32 (b=64)"
        gpu_fp32_train_res["speedup_vs_cpu"] = round(gpu_fp32_train_res["throughput_samples_per_sec"] / cpu_ref_throughput, 2)
        training_results.append(gpu_fp32_train_res)
        print(f"      GPU FP32 Throughput: {gpu_fp32_train_res['throughput_samples_per_sec']:.1f} samp/s ({gpu_fp32_train_res['speedup_vs_cpu']}x vs CPU)")

        # 3. GPU AMP (Batch Size 64)
        print("[3/6] Measuring GPU AMP Training (Batch Size 64)...")
        gpu_amp_model = create_resnet18(num_classes=10, cifar_stem=True).to(gpu_device)
        gpu_amp_model.load_state_dict(copy.deepcopy(initial_weights))
        gpu_amp_opt = torch.optim.AdamW(gpu_amp_model.parameters(), lr=0.001)

        gpu_amp_train_res = benchmark_training_step(
            model=gpu_amp_model,
            optimizer=gpu_amp_opt,
            criterion=crit,
            device=gpu_device,
            use_amp=True,
            batch_size=64,
            num_warmup=5,
            num_steps=30,
        )
        gpu_amp_train_res["config"] = "GPU AMP (b=64)"
        gpu_amp_train_res["speedup_vs_cpu"] = round(gpu_amp_train_res["throughput_samples_per_sec"] / cpu_ref_throughput, 2)
        training_results.append(gpu_amp_train_res)
        print(f"      GPU AMP Throughput: {gpu_amp_train_res['throughput_samples_per_sec']:.1f} samp/s ({gpu_amp_train_res['speedup_vs_cpu']}x vs CPU)")

        # 4, 5, 6. Batch Size Scaling with GPU AMP: 32, 128, 256
        for b_sz in [32, 128, 256]:
            print(f"[*] Measuring GPU AMP Training (Batch Size {b_sz})...")
            m = create_resnet18(num_classes=10, cifar_stem=True).to(gpu_device)
            m.load_state_dict(copy.deepcopy(initial_weights))
            opt = torch.optim.AdamW(m.parameters(), lr=0.001)

            res = benchmark_training_step(
                model=m,
                optimizer=opt,
                criterion=crit,
                device=gpu_device,
                use_amp=True,
                batch_size=b_sz,
                num_warmup=5,
                num_steps=30,
            )
            res["config"] = f"GPU AMP (b={b_sz})"
            res["speedup_vs_cpu"] = round(res["throughput_samples_per_sec"] / cpu_ref_throughput, 2)
            training_results.append(res)
            print(f"      b={b_sz} Throughput: {res['throughput_samples_per_sec']:.1f} samp/s ({res['speedup_vs_cpu']}x vs CPU) | VRAM: {res['peak_vram_allocated_mb']} MB")

    # =========================================================================
    # PART 2: INFERENCE LATENCY & THROUGHPUT BENCHMARKS
    # =========================================================================
    print("\n" + "-" * 100)
    print(">>> PART 2: RESNET-18 INFERENCE BENCHMARKS (Latency & Throughput)")
    print("-" * 100)

    inference_results: List[Dict[str, Any]] = []

    # CPU Latency (b=1 and b=32)
    print("[1/5] Measuring CPU Inference Latency (Batch Size 1)...")
    cpu_inf_b1 = benchmark_inference(cpu_model, cpu_device, use_amp=False, batch_size=1, num_warmup=5, num_iterations=20)
    cpu_inf_b1["config"] = "CPU FP32 (b=1)"
    cpu_inf_b1["speedup_vs_cpu"] = 1.0
    inference_results.append(cpu_inf_b1)
    cpu_b1_latency = cpu_inf_b1["mean_latency_ms"]
    print(f"      CPU b=1 Latency: {cpu_b1_latency:.2f} ms")

    if cuda_available:
        # GPU FP32 (b=1)
        print("[2/5] Measuring GPU FP32 Inference Latency (Batch Size 1)...")
        gpu_inf_b1 = benchmark_inference(gpu_fp32_model, gpu_device, use_amp=False, batch_size=1, num_warmup=10, num_iterations=50)
        gpu_inf_b1["config"] = "GPU FP32 (b=1)"
        gpu_inf_b1["speedup_vs_cpu"] = round(cpu_b1_latency / gpu_inf_b1["mean_latency_ms"], 2)
        inference_results.append(gpu_inf_b1)
        print(f"      GPU FP32 b=1 Latency: {gpu_inf_b1['mean_latency_ms']:.2f} ms ({gpu_inf_b1['speedup_vs_cpu']}x speedup)")

        # GPU AMP (b=1)
        print("[3/5] Measuring GPU AMP Inference Latency (Batch Size 1)...")
        gpu_amp_inf_b1 = benchmark_inference(gpu_amp_model, gpu_device, use_amp=True, batch_size=1, num_warmup=10, num_iterations=50)
        gpu_amp_inf_b1["config"] = "GPU AMP (b=1)"
        gpu_amp_inf_b1["speedup_vs_cpu"] = round(cpu_b1_latency / gpu_amp_inf_b1["mean_latency_ms"], 2)
        inference_results.append(gpu_amp_inf_b1)
        print(f"      GPU AMP b=1 Latency: {gpu_amp_inf_b1['mean_latency_ms']:.2f} ms ({gpu_amp_inf_b1['speedup_vs_cpu']}x speedup)")

        # GPU AMP Batch Throughput (b=32, b=128)
        for b_sz in [32, 128]:
            print(f"[*] Measuring GPU AMP Inference Throughput (Batch Size {b_sz})...")
            inf_res = benchmark_inference(gpu_amp_model, gpu_device, use_amp=True, batch_size=b_sz, num_warmup=10, num_iterations=50)
            inf_res["config"] = f"GPU AMP (b={b_sz})"
            inf_res["speedup_vs_cpu"] = round((inf_res["throughput_samples_per_sec"] / cpu_inf_b1["throughput_samples_per_sec"]), 2)
            inference_results.append(inf_res)
            print(f"      b={b_sz} Throughput: {inf_res['throughput_samples_per_sec']:.1f} inferences/s | Mean: {inf_res['mean_latency_ms']:.2f} ms")

    # =========================================================================
    # SUMMARY TABLES & EXPORT
    # =========================================================================
    print("\n" + "=" * 115)
    print("                     PHASE 9 TRAINING PERFORMANCE BENCHMARK RESULTS")
    print("=" * 115)
    print(f"{'Configuration':<22} | {'Throughput (samp/s)':<22} | {'Mean Step (ms)':<16} | {'VRAM Alloc (MB)':<16} | {'Speedup':<8}")
    print("-" * 115)
    for r in training_results:
        print(f"{r['config']:<22} | {r['throughput_samples_per_sec']:<22.1f} | {r['mean_step_time_ms']:<16.2f} | {r['peak_vram_allocated_mb']:<16.1f} | {r['speedup_vs_cpu']}x")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("                     PHASE 9 INFERENCE LATENCY & THROUGHPUT BENCHMARK RESULTS")
    print("=" * 115)
    print(f"{'Configuration':<22} | {'Throughput (inf/s)':<20} | {'Mean Lat (ms)':<14} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10} | {'Speedup':<8}")
    print("-" * 115)
    for r in inference_results:
        print(f"{r['config']:<22} | {r['throughput_samples_per_sec']:<20.1f} | {r['mean_latency_ms']:<14.2f} | {r['p50_latency_ms']:<10.2f} | {r['p95_latency_ms']:<10.2f} | {r['p99_latency_ms']:<10.2f} | {r['speedup_vs_cpu']}x")
    print("=" * 115)

    # Post-benchmark hardware check
    final_telem = monitor.get_telemetry()
    print(f"\n[*] Final Hardware Status: Temp={final_telem['temperature_c']}°C, Util={final_telem['utilization_gpu_pct']}%, Power={final_telem['power_draw_w']}W")

    # Export to JSON
    export_payload = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device_name": gpu_name,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "total_vram_mb": init_telem["vram_total_mb"],
        },
        "training_benchmarks": training_results,
        "inference_benchmarks": inference_results,
        "initial_telemetry": init_telem,
        "final_telemetry": final_telem,
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "phase9_gpu_benchmarks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[+] Successfully exported benchmark metrics to: {json_path}")


if __name__ == "__main__":
    run_full_gpu_benchmarks()
