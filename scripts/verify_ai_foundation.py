import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from engine.device.device_manager import DeviceManager
from engine.models.baseline_mlp import BaselineMLP
from engine.data.synthetic import create_synthetic_dataloaders
from engine.evaluation.metrics import compare_tensors
from engine.checkpointing.checkpoint_manager import CheckpointManager


def main():
    print("=" * 80)
    print("      CUDA AI Acceleration Engine - Phase 2 AI Platform Verification")
    print("=" * 80)

    results = []

    # 1. PyTorch & CUDA Environment
    print("\n[1/6] Probing PyTorch and CUDA Hardware Integration...")
    cuda_ok = DeviceManager.is_cuda_available()
    dev_name = DeviceManager.get_device_name(0)
    compute_cap = DeviceManager.get_compute_capability(0)
    mem_info = DeviceManager.get_memory_info(0)

    print(f"  PyTorch Version:         {torch.__version__}")
    print(f"  CUDA Available:          {cuda_ok}")
    print(f"  Device Name:             {dev_name}")
    print(f"  Compute Capability:      {compute_cap[0]}.{compute_cap[1]}")
    print(f"  Total VRAM:              {mem_info['total_mb']:.1f} MB")
    print(f"  Free VRAM:               {mem_info['free_mb']:.1f} MB")

    if cuda_ok and "RTX 3050" in dev_name and compute_cap == (8, 6):
        results.append(("CUDA & Hardware Detection", "PASS", f"{dev_name} (sm_{compute_cap[0]}{compute_cap[1]})"))
    else:
        results.append(("CUDA & Hardware Detection", "FAIL", f"Unexpected device: {dev_name}"))

    # 2. CPU Model Execution
    print("\n[2/6] Executing BaselineMLP Forward Pass on CPU...")
    model_cpu = BaselineMLP(input_dim=64, hidden_dims=[128, 64], output_dim=10, dropout=0.0)
    model_cpu.eval()
    x_cpu = torch.randn(32, 64, device="cpu")

    # Warmup + Timed CPU runs
    for _ in range(5):
        _ = model_cpu(x_cpu)

    cpu_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            out_cpu = model_cpu(x_cpu)
        t1 = time.perf_counter()
        cpu_times.append((t1 - t0) * 1000.0)
    cpu_median_ms = sorted(cpu_times)[len(cpu_times) // 2]

    cpu_valid = out_cpu.shape == (32, 10) and torch.isfinite(out_cpu).all().item()
    print(f"  CPU Forward Pass Time:   {cpu_median_ms:.4f} ms (median across 20 runs)")
    print(f"  Output Tensor Shape:     {tuple(out_cpu.shape)}")
    results.append(("CPU Model Execution", "PASS" if cpu_valid else "FAIL", f"{cpu_median_ms:.4f} ms"))

    # 3. GPU Model Execution
    print("\n[3/6] Executing BaselineMLP Forward Pass on GPU...")
    gpu_valid = False
    gpu_median_ms = 0.0
    out_gpu = None

    if cuda_ok:
        device = torch.device("cuda:0")
        model_gpu = BaselineMLP(input_dim=64, hidden_dims=[128, 64], output_dim=10, dropout=0.0)
        # Copy exact weights from CPU model for numerical parity comparison
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_gpu.to(device)
        model_gpu.eval()
        x_gpu = x_cpu.clone().to(device)

        # Warmup GPU runs
        for _ in range(10):
            _ = model_gpu(x_gpu)
        torch.cuda.synchronize()

        # Timed GPU runs using CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        gpu_times = []

        for _ in range(20):
            start_event.record()
            with torch.no_grad():
                out_gpu = model_gpu(x_gpu)
            end_event.record()
            torch.cuda.synchronize()
            gpu_times.append(start_event.elapsed_time(end_event))

        gpu_median_ms = sorted(gpu_times)[len(gpu_times) // 2]
        gpu_valid = out_gpu.shape == (32, 10) and torch.isfinite(out_gpu).all().item()
        print(f"  GPU Forward Pass Time:   {gpu_median_ms:.4f} ms (median across 20 runs)")
        print(f"  Output Tensor Shape:     {tuple(out_gpu.shape)}")
        speedup = cpu_median_ms / gpu_median_ms if gpu_median_ms > 0 else 0.0
        print(f"  Forward Pass Speedup:    {speedup:.2f}x")
        results.append(("GPU Model Execution", "PASS" if gpu_valid else "FAIL", f"{gpu_median_ms:.4f} ms ({speedup:.2f}x)"))
    else:
        results.append(("GPU Model Execution", "FAIL", "CUDA not available"))

    # 4. CPU vs GPU Numerical Parity Check
    print("\n[4/6] Validating Numerical Parity Between CPU and GPU Tensors...")
    if out_gpu is not None:
        parity = compare_tensors(out_cpu, out_gpu, tolerance=1e-5)
        print(f"  Max Absolute Difference: {parity['max_abs_diff']:.6e}")
        print(f"  Root Mean Square Error:  {parity['rmse']:.6e}")
        print(f"  Tolerance Threshold:     {parity['tolerance']:.6e}")
        print(f"  Mismatched Elements:     {parity['mismatched_elements']} / {parity['total_elements']}")
        status = "PASS" if parity["is_close"] else "FAIL"
        results.append(("CPU vs GPU Parity", status, f"Max Err: {parity['max_abs_diff']:.2e}"))
    else:
        results.append(("CPU vs GPU Parity", "FAIL", "No GPU tensor to compare"))

    # 5. Synthetic Data Pipeline
    print("\n[5/6] Testing Synthetic Dataset & DataLoader Pipeline...")
    train_loader, val_loader = create_synthetic_dataloaders(
        num_samples=500, input_dim=64, num_classes=10, batch_size=32, val_split=0.2
    )
    batch_x, batch_y = next(iter(train_loader))
    data_ok = batch_x.shape == (32, 64) and batch_y.shape == (32,)
    print(f"  Batch Feature Shape:     {tuple(batch_x.shape)}")
    print(f"  Batch Label Shape:       {tuple(batch_y.shape)}")
    results.append(("Data Pipeline", "PASS" if data_ok else "FAIL", "Batch (32, 64)"))

    # 6. Checkpoint Serialization & Deserialization
    print("\n[6/6] Testing Atomic Checkpoint Roundtrip...")
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "test_checkpoint.pt")

    opt = torch.optim.Adam(model_cpu.parameters(), lr=1e-3)
    CheckpointManager.save_checkpoint(
        model=model_cpu,
        filepath=ckpt_path,
        optimizer=opt,
        epoch=1,
        metrics={"loss": 0.35},
        metadata={"created_by": "phase2_verification"},
    )

    fresh_model = BaselineMLP(input_dim=64, hidden_dims=[128, 64], output_dim=10, dropout=0.0)
    meta = CheckpointManager.load_checkpoint(ckpt_path, fresh_model, map_location=torch.device("cpu"))

    with torch.no_grad():
        out_fresh = fresh_model(x_cpu)
    diff = torch.abs(out_cpu - out_fresh).max().item()
    ckpt_ok = diff < 1e-6 and meta["epoch"] == 1
    print(f"  Checkpoint File:         {ckpt_path}")
    print(f"  Restoration Max Diff:    {diff:.2e}")
    results.append(("Checkpointing", "PASS" if ckpt_ok else "FAIL", f"Diff: {diff:.2e}"))

    # Summary Table
    print("\n" + "=" * 80)
    print("                     PHASE 2 VERIFICATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Verification Item':<30} | {'Status':<10} | {'Details':<35}")
    print("-" * 80)

    all_passed = True
    for item, stat, detail in results:
        if stat != "PASS":
            all_passed = False
        print(f"{item:<30} | {stat:<10} | {detail:<35}")
    print("=" * 80)

    if all_passed:
        print("  PHASE 2 STATUS: [PASS] - AI Platform Foundation Successfully Verified!")
    else:
        print("  PHASE 2 STATUS: [FAIL] - Verification Discrepancies Detected!")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
