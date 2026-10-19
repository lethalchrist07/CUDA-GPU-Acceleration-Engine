"""
Phase 9 ResNet-18 & CIFAR-10 AI Training CLI & Benchmarking Pipeline.
Supports FP32, AMP, CPU reference, learning-rate scheduling, early stopping,
hardware protection, checkpoint resume, and JSON/CSV experiment logging.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.data.cifar10 import create_cifar10_dataloaders
from engine.device.device_manager import DeviceManager
from engine.device.hardware_monitor import HardwareMonitor
from engine.models.resnet import ResNet18, create_resnet18
from engine.training.experiment_tracker import ExperimentTracker
from engine.training.resnet_trainer import ResNetTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ResNetTrainCLI")


def build_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer '{name}', defaulting to AdamW.")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int) -> Optional[Any]:
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.2)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    elif name == "none":
        return None
    return None


def run_single_experiment(
    mode: str,
    device: torch.device,
    use_amp: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer_name: str,
    scheduler_name: str,
    seed: int,
    checkpoint_dir: str,
    resume_from: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    initial_weights: Optional[Dict[str, Any]] = None,
    monitor: Optional[HardwareMonitor] = None,
    early_stopping_patience: int = 0,
) -> Dict[str, Any]:
    """Execute a single training run with the given hyperparameter configuration."""
    print("=" * 85)
    print(f"  RUNNING: Device={device} | AMP={use_amp} | BatchSize={batch_size} | Epochs={epochs}")
    print("=" * 85)

    # Set deterministic random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prepare DataLoaders
    train_loader, val_loader, test_loader, class_names = create_cifar10_dataloaders(
        data_dir=str(PROJECT_ROOT / "data"),
        batch_size=batch_size,
        val_split=0.1,
        seed=seed,
        augment=True,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        pin_memory=(device.type == "cuda"),
    )

    # Instantiate ResNet-18
    model = create_resnet18(num_classes=len(class_names), cifar_stem=True)
    if initial_weights is not None:
        model.load_state_dict(copy.deepcopy(initial_weights))

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model, lr=lr)
    scheduler = build_scheduler(scheduler_name, optimizer, epochs=epochs)

    trainer = ResNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
        hardware_monitor=monitor,
        early_stopping_patience=early_stopping_patience,
    )

    history = trainer.fit(epochs=epochs, resume_from=resume_from, save_checkpoints=True)

    # Compute aggregate metrics
    avg_ep_time = (
        sum(history["epoch_times_ms"]) / len(history["epoch_times_ms"])
        if history["epoch_times_ms"]
        else 0.0
    )
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    final_train_acc = history["train_accuracy"][-1] if history["train_accuracy"] else 0.0
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
    best_val_acc = history["best_val_accuracy"]
    test_metrics = history.get("test_metrics") or {}
    test_acc = test_metrics.get("accuracy", 0.0)

    # Peak VRAM
    peak_vram_alloc = history.get("peak_gpu_memory_allocated_mb", 0.0)
    peak_vram_res = history.get("peak_gpu_memory_reserved_mb", 0.0)

    # Average throughput across epochs
    throughputs = history.get("train_throughput_samples_sec", [])
    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0

    exp_id = f"{'gpu_amp' if use_amp else device.type}_resnet18_b{batch_size}_ep{epochs}_{int(time.time())}"
    device_name = (
        torch.cuda.get_device_name(device.index or 0)
        if device.type == "cuda"
        else "Intel Core i5-13450HX"
    )

    record = {
        "experiment_id": exp_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "ResNet18",
        "parameters": model.count_parameters()["total_parameters"],
        "device": str(device),
        "device_name": device_name,
        "epochs": len(history["epoch_times_ms"]),
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": optimizer_name,
        "seed": seed,
        "use_amp": use_amp,
        "total_train_time_ms": round(history["total_time_ms"], 2),
        "avg_epoch_time_ms": round(avg_ep_time, 2),
        "final_train_loss": round(final_train_loss, 4),
        "final_train_accuracy": round(final_train_acc * 100, 2),
        "final_val_loss": round(final_val_loss, 4),
        "best_val_accuracy": round(best_val_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "peak_gpu_memory_mb": peak_vram_alloc,
        "peak_vram_reserved_mb": peak_vram_res,
        "avg_throughput_samples_sec": round(avg_throughput, 1),
        "speedup_vs_cpu": 1.0,
        "early_stopped": history.get("early_stopped", False),
    }

    return record


def main():
    parser = argparse.ArgumentParser(description="ResNet-18 CIFAR-10 Training & Performance Engine")
    parser.add_argument("--mode", type=str, default="pilot",
                        choices=["pilot", "train", "benchmark", "cpu", "gpu", "amp", "resume"],
                        help="Execution mode")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "plateau", "none"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/resnet18_cifar10")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--pilot-train-samples", type=int, default=2000)
    parser.add_argument("--pilot-val-samples", type=int, default=500)
    parser.add_argument("--pilot-test-samples", type=int, default=500)

    args = parser.parse_args()

    print("=" * 90)
    print("      CUDA AI Acceleration Engine - Phase 9 ResNet-18 & CIFAR-10 Engine")
    print("=" * 90)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print(f"[*] Environment: CUDA={cuda_available} | Device={device_name}")
    print(f"[*] Mode={args.mode} | Epochs={args.epochs} | BatchSize={args.batch_size} | LR={args.lr}\n")

    monitor = HardwareMonitor(device_id=0)
    tracker = ExperimentTracker(results_dir=str(PROJECT_ROOT / "results"), filename_prefix="phase9_")

    # Initial weights snapshot for exact model initialization parity across comparisons
    torch.manual_seed(args.seed)
    seed_model = create_resnet18(num_classes=10, cifar_stem=True)
    initial_weights = copy.deepcopy(seed_model.state_dict())

    # Mode 1: Short Pilot Run
    if args.mode == "pilot":
        print("[*] Starting Stage B Pilot Verification Run (2 epochs, subset samples)...")
        # 1. Pilot on GPU with AMP
        gpu_device = torch.device("cuda:0" if cuda_available else "cpu")
        record = run_single_experiment(
            mode="pilot",
            device=gpu_device,
            use_amp=cuda_available,
            epochs=2,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            max_train_samples=args.pilot_train_samples,
            max_val_samples=args.pilot_val_samples,
            max_test_samples=args.pilot_test_samples,
            initial_weights=initial_weights,
            monitor=monitor,
            early_stopping_patience=0,
        )
        tracker.log_experiment(record)
        print("\n[+] Pilot Run Successful! Recorded to experiments database.")
        print(f"    Train Loss: {record['final_train_loss']:.4f} | Val Acc: {record['best_val_accuracy']:.2f}%")
        print(f"    Throughput: {record['avg_throughput_samples_sec']} samples/sec | Time: {record['total_train_time_ms']:.1f} ms")
        print(f"    Peak VRAM: {record['peak_gpu_memory_mb']} MB (Allocated), {record['peak_vram_reserved_mb']} MB (Reserved)")
        return

    # Mode 2: Full Benchmark Suite (Stage C: CPU vs GPU vs AMP across batch sizes)
    if args.mode == "benchmark":
        print("[*] Starting Stage C GPU Performance Engineering Benchmark Battery...")
        results_list = []

        # Standard benchmark subset to evaluate CPU reference without excessive duration:
        bench_train_samples = 4000
        bench_val_samples = 1000
        bench_test_samples = 1000
        bench_epochs = 2

        # 1. CPU Reference Run
        print("\n>>> [1/4] Running CPU Reference Workload...")
        cpu_record = run_single_experiment(
            mode="benchmark_cpu",
            device=torch.device("cpu"),
            use_amp=False,
            epochs=bench_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            seed=args.seed,
            checkpoint_dir=f"{args.checkpoint_dir}/cpu",
            max_train_samples=bench_train_samples,
            max_val_samples=bench_val_samples,
            max_test_samples=bench_test_samples,
            initial_weights=initial_weights,
            monitor=monitor,
        )
        cpu_time = cpu_record["total_train_time_ms"]
        cpu_record["speedup_vs_cpu"] = 1.0
        results_list.append(cpu_record)
        tracker.log_experiment(cpu_record)

        if not cuda_available:
            print("CUDA unavailable; skipping GPU benchmark arms.")
            return

        # 2. GPU FP32 Run
        print("\n>>> [2/4] Running GPU FP32 Workload...")
        gpu_fp32_record = run_single_experiment(
            mode="benchmark_gpu_fp32",
            device=torch.device("cuda:0"),
            use_amp=False,
            epochs=bench_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            seed=args.seed,
            checkpoint_dir=f"{args.checkpoint_dir}/gpu_fp32",
            max_train_samples=bench_train_samples,
            max_val_samples=bench_val_samples,
            max_test_samples=bench_test_samples,
            initial_weights=initial_weights,
            monitor=monitor,
        )
        gpu_fp32_record["speedup_vs_cpu"] = round(cpu_time / gpu_fp32_record["total_train_time_ms"], 2)
        results_list.append(gpu_fp32_record)
        tracker.log_experiment(gpu_fp32_record)

        # 3. GPU AMP Run
        print("\n>>> [3/4] Running GPU AMP Workload...")
        gpu_amp_record = run_single_experiment(
            mode="benchmark_gpu_amp",
            device=torch.device("cuda:0"),
            use_amp=True,
            epochs=bench_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            seed=args.seed,
            checkpoint_dir=f"{args.checkpoint_dir}/gpu_amp",
            max_train_samples=bench_train_samples,
            max_val_samples=bench_val_samples,
            max_test_samples=bench_test_samples,
            initial_weights=initial_weights,
            monitor=monitor,
        )
        gpu_amp_record["speedup_vs_cpu"] = round(cpu_time / gpu_amp_record["total_train_time_ms"], 2)
        results_list.append(gpu_amp_record)
        tracker.log_experiment(gpu_amp_record)

        # 4. Batch Size Sensitivity Experiments (Batch size = 32, 64, 128)
        print("\n>>> [4/4] Running Batch Size Scaling Experiments (32, 64, 128)...")
        for b_sz in [32, 64]:
            print(f"\n--- Batch Size: {b_sz} ---")
            rec = run_single_experiment(
                mode=f"batch_scaling_{b_sz}",
                device=torch.device("cuda:0"),
                use_amp=True,
                epochs=bench_epochs,
                batch_size=b_sz,
                lr=args.lr,
                optimizer_name=args.optimizer,
                scheduler_name=args.scheduler,
                seed=args.seed,
                checkpoint_dir=f"{args.checkpoint_dir}/b{b_sz}",
                max_train_samples=bench_train_samples,
                max_val_samples=bench_val_samples,
                max_test_samples=bench_test_samples,
                initial_weights=initial_weights,
                monitor=monitor,
            )
            rec["speedup_vs_cpu"] = round(cpu_time / rec["total_train_time_ms"], 2)
            results_list.append(rec)
            tracker.log_experiment(rec)

        print("\n" + "=" * 105)
        print("                        STAGE C BENCHMARK COMPARISON TABLE")
        print("=" * 105)
        print(f"{'Config':<20} | {'Throughput (samp/s)':<20} | {'Total Time (ms)':<16} | {'VRAM Alloc (MB)':<16} | {'Speedup':<8}")
        print("-" * 105)
        for r in results_list:
            tag = f"{r['device']} (AMP={r['use_amp']}) b{r['batch_size']}"
            print(f"{tag:<20} | {r['avg_throughput_samples_sec']:<20.1f} | {r['total_train_time_ms']:<16.1f} | {r['peak_gpu_memory_mb']:<16.1f} | {r['speedup_vs_cpu']}x")
        print("=" * 105)
        return

    # Mode 3: Extended Full Training Run
    if args.mode in ["train", "gpu", "amp"]:
        gpu_device = torch.device("cuda:0" if cuda_available else "cpu")
        use_amp = cuda_available and (args.mode in ["train", "amp"])
        print(f"[*] Launching Extended ResNet-18 Training ({args.epochs} epochs) on {gpu_device} (AMP={use_amp})...")

        record = run_single_experiment(
            mode="full_training",
            device=gpu_device,
            use_amp=use_amp,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_path,
            max_train_samples=None,  # Full 45,000 samples
            max_val_samples=None,    # Full 5,000 samples
            max_test_samples=None,   # Full 10,000 samples
            initial_weights=initial_weights,
            monitor=monitor,
            early_stopping_patience=args.early_stopping,
        )
        tracker.log_experiment(record)
        print(f"\n[+] Full Training Complete: Final Val Acc = {record['best_val_accuracy']:.2f}% | Test Acc = {record['test_accuracy']:.2f}%")
        return

    # Mode 4: Resume Interrupted Training
    if args.mode == "resume":
        resume_file = args.resume_path or f"{args.checkpoint_dir}/latest_checkpoint.pt"
        gpu_device = torch.device("cuda:0" if cuda_available else "cpu")
        print(f"[*] Resuming ResNet-18 training from checkpoint: {resume_file}")
        record = run_single_experiment(
            mode="resume",
            device=gpu_device,
            use_amp=cuda_available,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=resume_file,
            max_train_samples=args.pilot_train_samples,
            max_val_samples=args.pilot_val_samples,
            max_test_samples=args.pilot_test_samples,
            monitor=monitor,
            early_stopping_patience=args.early_stopping,
        )
        tracker.log_experiment(record)
        print(f"\n[+] Resumed Training Complete: Best Val Acc = {record['best_val_accuracy']:.2f}%")
        return


if __name__ == "__main__":
    main()
