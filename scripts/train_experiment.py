import sys
import os
import argparse
import copy
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from engine.device.device_manager import DeviceManager
from engine.models.baseline_mlp import BaselineMLP
from engine.data.synthetic import create_synthetic_dataloaders
from engine.training.trainer import ModelTrainer
from engine.training.experiment_tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainExperiment")


def build_optimizer(name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr)


def main():
    parser = argparse.ArgumentParser(description="CUDA AI Acceleration Engine - Training Experiment CLI")
    parser.add_argument("--mode", type=str, default="compare", choices=["compare", "cpu", "gpu", "resume"],
                        help="Execution mode: compare (CPU vs GPU), cpu, gpu, or resume")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed-precision on CUDA")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of synthetic dataset samples")
    parser.add_argument("--input-dim", type=int, default=128, help="Input feature dimension")
    parser.add_argument("--num-classes", type=int, default=20, help="Number of classification categories")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume-path", type=str, default="checkpoints/latest_checkpoint.pt", help="Checkpoint to resume")

    args = parser.parse_args()

    print("=" * 90)
    print("      CUDA AI Acceleration Engine - Phase 3 Training Pipeline")
    print("=" * 90)

    # Hardware Info
    cuda_available = DeviceManager.is_cuda_available()
    gpu_name = DeviceManager.get_device_name(0) if cuda_available else "N/A"
    print(f"[*] Hardware Environment: CUDA Available = {cuda_available} | Device = {gpu_name}")
    print(f"[*] Experiment Setup: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Seed={args.seed}")
    print(f"[*] Dataset: {args.num_samples} samples, Input Dim={args.input_dim}, Classes={args.num_classes}\n")

    tracker = ExperimentTracker(results_dir="results")

    # 1. Dataset Generation
    train_loader, val_loader = create_synthetic_dataloaders(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        val_split=0.2,
        seed=args.seed,
    )

    criterion = nn.CrossEntropyLoss()

    # 2. Model Initialization (Snapshot initial weights for exact parity)
    torch.manual_seed(args.seed)
    base_model = BaselineMLP(
        input_dim=args.input_dim,
        hidden_dims=[512, 512, 256],
        output_dim=args.num_classes,
        activation="relu",
        dropout=0.05,
    )
    initial_weights = copy.deepcopy(base_model.state_dict())
    num_params = base_model.count_parameters()
    print(f"[*] Model Architecture: BaselineMLP [128 -> 512 -> 512 -> 256 -> 20] ({num_params:,} parameters)\n")

    summary_rows = []

    if args.mode in ["compare", "cpu"]:
        print("-" * 90)
        print(">>> STARTING CPU TRAINING EXPERIMENT")
        print("-" * 90)
        cpu_device = torch.device("cpu")
        cpu_model = BaselineMLP(
            input_dim=args.input_dim,
            hidden_dims=[512, 512, 256],
            output_dim=args.num_classes,
            activation="relu",
            dropout=0.05,
        )
        cpu_model.load_state_dict(initial_weights)
        cpu_opt = build_optimizer(args.optimizer, cpu_model, args.lr)

        cpu_trainer = ModelTrainer(
            model=cpu_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=cpu_opt,
            device=cpu_device,
            use_amp=False,
            checkpoint_dir=os.path.join(args.checkpoint_dir, "cpu"),
        )
        cpu_history = cpu_trainer.fit(epochs=args.epochs, save_checkpoints=True)
        cpu_total_s = cpu_history["total_time_ms"] / 1000.0
        cpu_avg_ep_ms = cpu_history["total_time_ms"] / args.epochs

        print(f"\n[+] CPU Training Finished:")
        print(f"    Total Time:           {cpu_total_s:.3f} s ({cpu_history['total_time_ms']:.1f} ms)")
        print(f"    Average Epoch Time:   {cpu_avg_ep_ms:.1f} ms")
        print(f"    Final Train Loss:     {cpu_history['train_loss'][-1]:.4f}")
        print(f"    Final Val Loss:       {cpu_history['val_loss'][-1]:.4f}")
        print(f"    Best Val Accuracy:    {cpu_history['best_val_accuracy'] * 100:.2f}% (Epoch {cpu_history['best_epoch']})")

        cpu_exp = {
            "experiment_id": f"cpu_mlp_{args.epochs}ep_{args.seed}",
            "model_name": "BaselineMLP",
            "parameters": num_params,
            "device": "cpu",
            "device_name": "Intel Core i5-13450HX",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "use_amp": False,
            "total_train_time_ms": round(cpu_history["total_time_ms"], 2),
            "avg_epoch_time_ms": round(cpu_avg_ep_ms, 2),
            "final_train_loss": round(cpu_history["train_loss"][-1], 4),
            "final_val_loss": round(cpu_history["val_loss"][-1], 4),
            "best_val_accuracy": round(cpu_history["best_val_accuracy"] * 100, 2),
            "peak_gpu_memory_mb": 0.0,
            "speedup_vs_cpu": 1.0,
        }
        tracker.log_experiment(cpu_exp)
        summary_rows.append(cpu_exp)

    if args.mode in ["compare", "gpu"]:
        if not cuda_available:
            print("[-] Cannot run GPU experiment: CUDA is not available!")
            return 1

        print("\n" + "-" * 90)
        print(">>> STARTING NVIDIA RTX 3050 GPU TRAINING EXPERIMENT (FP32)")
        print("-" * 90)
        gpu_device = torch.device("cuda:0")
        gpu_model = BaselineMLP(
            input_dim=args.input_dim,
            hidden_dims=[512, 512, 256],
            output_dim=args.num_classes,
            activation="relu",
            dropout=0.05,
        )
        gpu_model.load_state_dict(initial_weights)
        gpu_opt = build_optimizer(args.optimizer, gpu_model, args.lr)

        gpu_trainer = ModelTrainer(
            model=gpu_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=gpu_opt,
            device=gpu_device,
            use_amp=False,
            checkpoint_dir=os.path.join(args.checkpoint_dir, "gpu_fp32"),
        )
        gpu_history = gpu_trainer.fit(epochs=args.epochs, save_checkpoints=True)
        gpu_total_s = gpu_history["total_time_ms"] / 1000.0
        gpu_avg_ep_ms = gpu_history["total_time_ms"] / args.epochs
        speedup_fp32 = (cpu_history["total_time_ms"] / gpu_history["total_time_ms"]) if "cpu_history" in locals() else 1.0

        print(f"\n[+] GPU (FP32) Training Finished:")
        print(f"    Total Time:           {gpu_total_s:.3f} s ({gpu_history['total_time_ms']:.1f} ms)")
        print(f"    Average Epoch Time:   {gpu_avg_ep_ms:.1f} ms")
        print(f"    Final Train Loss:     {gpu_history['train_loss'][-1]:.4f}")
        print(f"    Final Val Loss:       {gpu_history['val_loss'][-1]:.4f}")
        print(f"    Best Val Accuracy:    {gpu_history['best_val_accuracy'] * 100:.2f}% (Epoch {gpu_history['best_epoch']})")
        print(f"    Peak VRAM Allocation: {gpu_history['peak_gpu_memory_mb']} MB")
        if "cpu_history" in locals():
            print(f"    Speedup vs CPU:       {speedup_fp32:.2f}x")

        gpu_exp = {
            "experiment_id": f"gpu_fp32_mlp_{args.epochs}ep_{args.seed}",
            "model_name": "BaselineMLP",
            "parameters": num_params,
            "device": "cuda:0",
            "device_name": gpu_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "use_amp": False,
            "total_train_time_ms": round(gpu_history["total_time_ms"], 2),
            "avg_epoch_time_ms": round(gpu_avg_ep_ms, 2),
            "final_train_loss": round(gpu_history["train_loss"][-1], 4),
            "final_val_loss": round(gpu_history["val_loss"][-1], 4),
            "best_val_accuracy": round(gpu_history["best_val_accuracy"] * 100, 2),
            "peak_gpu_memory_mb": gpu_history["peak_gpu_memory_mb"],
            "speedup_vs_cpu": round(speedup_fp32, 2),
        }
        tracker.log_experiment(gpu_exp)
        summary_rows.append(gpu_exp)

        # Optional: Mixed Precision GPU Run
        if args.amp or args.mode == "compare":
            print("\n" + "-" * 90)
            print(">>> STARTING NVIDIA RTX 3050 GPU TRAINING EXPERIMENT (AMP Mixed Precision)")
            print("-" * 90)
            amp_model = BaselineMLP(
                input_dim=args.input_dim,
                hidden_dims=[512, 512, 256],
                output_dim=args.num_classes,
                activation="relu",
                dropout=0.05,
            )
            amp_model.load_state_dict(initial_weights)
            amp_opt = build_optimizer(args.optimizer, amp_model, args.lr)

            amp_trainer = ModelTrainer(
                model=amp_model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=amp_opt,
                device=gpu_device,
                use_amp=True,
                checkpoint_dir=os.path.join(args.checkpoint_dir, "gpu_amp"),
            )
            amp_history = amp_trainer.fit(epochs=args.epochs, save_checkpoints=True)
            amp_total_s = amp_history["total_time_ms"] / 1000.0
            amp_avg_ep_ms = amp_history["total_time_ms"] / args.epochs
            speedup_amp = (cpu_history["total_time_ms"] / amp_history["total_time_ms"]) if "cpu_history" in locals() else 1.0

            print(f"\n[+] GPU (AMP) Training Finished:")
            print(f"    Total Time:           {amp_total_s:.3f} s ({amp_history['total_time_ms']:.1f} ms)")
            print(f"    Average Epoch Time:   {amp_avg_ep_ms:.1f} ms")
            print(f"    Final Train Loss:     {amp_history['train_loss'][-1]:.4f}")
            print(f"    Final Val Loss:       {amp_history['val_loss'][-1]:.4f}")
            print(f"    Best Val Accuracy:    {amp_history['best_val_accuracy'] * 100:.2f}% (Epoch {amp_history['best_epoch']})")
            print(f"    Peak VRAM Allocation: {amp_history['peak_gpu_memory_mb']} MB")
            if "cpu_history" in locals():
                print(f"    Speedup vs CPU:       {speedup_amp:.2f}x")

            amp_exp = {
                "experiment_id": f"gpu_amp_mlp_{args.epochs}ep_{args.seed}",
                "model_name": "BaselineMLP",
                "parameters": num_params,
                "device": "cuda:0",
                "device_name": gpu_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "seed": args.seed,
                "use_amp": True,
                "total_train_time_ms": round(amp_history["total_time_ms"], 2),
                "avg_epoch_time_ms": round(amp_avg_ep_ms, 2),
                "final_train_loss": round(amp_history["train_loss"][-1], 4),
                "final_val_loss": round(amp_history["val_loss"][-1], 4),
                "best_val_accuracy": round(amp_history["best_val_accuracy"] * 100, 2),
                "peak_gpu_memory_mb": amp_history["peak_gpu_memory_mb"],
                "speedup_vs_cpu": round(speedup_amp, 2),
            }
            tracker.log_experiment(amp_exp)
            summary_rows.append(amp_exp)

    elif args.mode == "resume":
        print("-" * 90)
        print(f">>> RESUMING TRAINING EXPERIMENT FROM: {args.resume_path}")
        print("-" * 90)
        res_device = torch.device("cuda:0" if cuda_available else "cpu")
        res_model = BaselineMLP(
            input_dim=args.input_dim,
            hidden_dims=[512, 512, 256],
            output_dim=args.num_classes,
            activation="relu",
            dropout=0.05,
        )
        res_opt = build_optimizer(args.optimizer, res_model, args.lr)
        res_trainer = ModelTrainer(
            model=res_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=res_opt,
            device=res_device,
            use_amp=args.amp,
            checkpoint_dir=args.checkpoint_dir,
        )
        res_history = res_trainer.fit(epochs=args.epochs, resume_from=args.resume_path, save_checkpoints=True)
        print(f"\n[+] Resumed Training Completed successfully ({args.epochs} additional epochs).")

    # Final Comparison Summary Table
    if summary_rows:
        print("\n" + "=" * 90)
        print("                        PHASE 3 TRAINING PERFORMANCE BENCHMARK")
        print("=" * 90)
        print(f"{'Experiment':<22} | {'Device':<10} | {'Total Time':<12} | {'Epoch Time':<12} | {'Best Acc':<10} | {'VRAM':<9} | {'Speedup':<8}")
        print("-" * 90)
        for r in summary_rows:
            amp_tag = " (AMP)" if r["use_amp"] else ""
            dev_str = f"{r['device']}{amp_tag}"
            tot_str = f"{r['total_train_time_ms'] / 1000.0:.2f} s"
            ep_str = f"{r['avg_epoch_time_ms']:.1f} ms"
            acc_str = f"{r['best_val_accuracy']:.2f}%"
            vram_str = f"{r['peak_gpu_memory_mb']} MB" if r["peak_gpu_memory_mb"] > 0 else "N/A"
            spd_str = f"{r['speedup_vs_cpu']:.2f}x"
            print(f"{r['experiment_id']:<22} | {dev_str:<10} | {tot_str:<12} | {ep_str:<12} | {acc_str:<10} | {vram_str:<9} | {spd_str:<8}")
        print("=" * 90)
        print(f"[+] Metrics exported to results/training_experiments.json and results/training_experiments.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
