"""
Configurable AI Training Engine for ResNet-18 & CIFAR-10 on CUDA / CPU (Phase 9 & Phase 10).
Provides automatic mixed precision (AMP), learning rate scheduling, gradient accumulation,
cooperative cancellation, checkpoint resumption, throughput tracking, and peak VRAM profiling.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.device.hardware_monitor import HardwareMonitor
from engine.training.cifar10_three_backend_trainer import (
    TrainingCancelledException,
    compute_timing_stats,
)
from .trainer_base import BaseTrainer

logger = logging.getLogger(__name__)


class ResNetTrainer(BaseTrainer):
    """
    Advanced PyTorch trainer designed for ResNet-18 vision pipelines.
    Supports FP32, AMP, CPU reference, gradient accumulation, cooperative cancellation,
    thermal protection, and standardized progress callback streaming.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        backend: str = "GPU_NATIVE",
        use_amp: bool = False,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        checkpoint_path: Optional[Union[str, Path]] = None,
        hardware_monitor: Optional[HardwareMonitor] = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.001,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        cancel_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        progress_interval_batches: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_criterion = criterion or nn.CrossEntropyLoss()
        resolved_optimizer = optimizer or torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=resolved_criterion,
            optimizer=resolved_optimizer,
            device=resolved_device,
            config=config,
        )

        self.backend = backend
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.checkpoint_dir = str(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.hardware_monitor = hardware_monitor or HardwareMonitor(
            device_id=self.device.index or 0 if self.device.type == "cuda" else 0
        )
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.seed = seed
        self.cancel_event = cancel_event
        self.progress_callback = progress_callback
        self.progress_interval_batches = max(1, int(progress_interval_batches))

        # GradScaler for CUDA Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.best_val_acc = -1.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.current_epoch = 0
        self.epochs_without_improvement = 0
        self.best_checkpoint_saved: Optional[Path] = None

    def train_epoch(
        self,
        epoch: int = 1,
        total_epochs: int = 1,
        global_start_time: float = 0.0,
    ) -> Tuple[float, float, float, float, List[float]]:
        """
        Execute one complete training epoch across all training batches with gradient accumulation.
        Returns: (avg_loss, accuracy, epoch_time_ms, throughput, step_latencies)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        step_latencies: List[float] = []
        total_batches = len(self.train_loader)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader, start=1):
            if self.cancel_event and self.cancel_event.is_set():
                raise TrainingCancelledException("ResNet-18 training job cancelled by client request.")

            batch_x = batch_x.to(self.device, non_blocking=(self.device.type == "cuda"))
            batch_y = batch_y.to(self.device, non_blocking=(self.device.type == "cuda"))
            batch_size = batch_x.size(0)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                scaled_loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(scaled_loss).backward()

                if (batch_idx % self.gradient_accumulation_steps == 0) or (batch_idx == total_batches):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()

                if (batch_idx % self.gradient_accumulation_steps == 0) or (batch_idx == total_batches):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            step_lat_ms = (time.perf_counter() - t0) * 1000.0
            step_latencies.append(step_lat_ms)

            with torch.no_grad():
                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_size

            # Periodic batch progress emission
            if self.progress_callback and (
                batch_idx % self.progress_interval_batches == 0 or batch_idx == total_batches
            ):
                current_time = time.perf_counter()
                elapsed = current_time - (global_start_time if global_start_time > 0 else start_time)
                running_loss = total_loss / total_samples
                running_acc = (total_correct / total_samples) * 100.0
                vram_alloc = (
                    torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                    if self.device.type == "cuda"
                    else 0.0
                )
                vram_res = (
                    torch.cuda.memory_reserved(self.device) / (1024 * 1024)
                    if self.device.type == "cuda"
                    else 0.0
                )
                sps = (total_samples / (current_time - start_time)) if (current_time - start_time) > 0 else 0.0

                self.progress_callback({
                    "event": "batch_progress",
                    "epoch": epoch,
                    "current_epoch": epoch,
                    "total_epochs": total_epochs,
                    "batch": batch_idx,
                    "current_step": batch_idx,
                    "total_batches": total_batches,
                    "total_steps": total_batches,
                    "train_loss": round(running_loss, 4),
                    "train_accuracy": round(running_acc, 2),
                    "samples_processed": total_samples,
                    "elapsed_time_sec": round(elapsed, 2),
                    "samples_per_sec": round(sps, 1),
                    "step_latency_ms": round(step_lat_ms, 2),
                    "peak_allocated_vram_mb": round(vram_alloc, 2),
                    "peak_reserved_vram_mb": round(vram_res, 2),
                })

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_sec = time.perf_counter() - start_time
        epoch_time_ms = elapsed_sec * 1000.0
        throughput = total_samples / elapsed_sec if elapsed_sec > 0 else 0.0

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy, epoch_time_ms, throughput, step_latencies

    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate validation or test dataset metrics (loss and top-1 accuracy).
        """
        data_loader = loader if loader is not None else self.val_loader
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                if self.cancel_event and self.cancel_event.is_set():
                    raise TrainingCancelledException("Job cancelled during validation.")
                batch_x = batch_x.to(self.device, non_blocking=(self.device.type == "cuda"))
                batch_y = batch_y.to(self.device, non_blocking=(self.device.type == "cuda"))
                batch_size = batch_x.size(0)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_size

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_sec = time.perf_counter() - start_time
        duration_ms = elapsed_sec * 1000.0
        throughput = total_samples / elapsed_sec if elapsed_sec > 0 else 0.0

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "duration_ms": float(duration_ms),
            "throughput_samples_per_sec": float(throughput),
            "total_samples": int(total_samples),
        }

    def evaluate_test_set(self) -> Dict[str, float]:
        """
        Run final evaluation strictly on the isolated test dataset.
        """
        if self.test_loader is None:
            logger.warning("No test loader supplied; skipping isolated test evaluation.")
            return {"loss": 0.0, "accuracy": 0.0, "duration_ms": 0.0, "throughput_samples_per_sec": 0.0}

        logger.info("Executing final evaluation on isolated CIFAR-10 test set...")
        results = self.validate(loader=self.test_loader)
        logger.info(
            f"Final Test Evaluation: Loss = {results['loss']:.4f} | "
            f"Accuracy = {results['accuracy'] * 100:.2f}% | "
            f"Throughput = {results['throughput_samples_per_sec']:.1f} samples/sec"
        )
        return results

    def fit(
        self,
        epochs: int,
        resume_from: Optional[str] = None,
        save_checkpoints: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training lifecycle execution with metrics tracking, scheduling,
        early stopping, hardware telemetry, gradient accumulation, and checkpointing.
        """
        start_epoch = 1
        history: Dict[str, Any] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_times_ms": [],
            "train_throughput_samples_sec": [],
            "learning_rates": [],
            "vram_allocated_mb": [],
            "vram_reserved_mb": [],
            "gpu_temperature_c": [],
            "gpu_utilization_pct": [],
            "total_time_ms": 0.0,
            "peak_gpu_memory_allocated_mb": 0.0,
            "peak_gpu_memory_reserved_mb": 0.0,
            "device": str(self.device),
            "use_amp": self.use_amp,
            "best_epoch": 0,
            "best_val_accuracy": 0.0,
            "best_val_loss": float("inf"),
            "test_metrics": None,
            "interrupted": False,
            "early_stopped": False,
        }
        all_step_latencies: List[float] = []
        epoch_records: List[Dict[str, Any]] = []

        # Checkpoint Resume
        if resume_from is not None:
            if os.path.exists(resume_from):
                logger.info(f"Resuming training session from: {resume_from}")
                meta = CheckpointManager.load_checkpoint(
                    filepath=resume_from,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    map_location=self.device,
                    restore_rng=True,
                )
                start_epoch = meta.get("epoch", 0) + 1
                prev_metrics = meta.get("metrics", {})
                raw_best_acc = prev_metrics.get("val_accuracy", prev_metrics.get("best_val_acc", -1.0))
                # Normalize fraction if stored as percentage
                self.best_val_acc = (raw_best_acc / 100.0) if raw_best_acc > 1.0 else raw_best_acc
                self.best_val_loss = prev_metrics.get("val_loss", float("inf"))
                logger.info(f"Resumed at epoch {start_epoch}. Prior best val acc: {self.best_val_acc * 100:.2f}%")
            else:
                raise FileNotFoundError(f"Resume checkpoint file not found: {resume_from}")

        # CUDA Sync & Reset Peak Stats
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        total_start = time.perf_counter()
        total_target_epochs = (start_epoch - 1) + epochs

        try:
            for epoch in range(start_epoch, start_epoch + epochs):
                self.current_epoch = epoch
                if self.cancel_event and self.cancel_event.is_set():
                    raise TrainingCancelledException("ResNet-18 training job cancelled by client request.")

                # Thermal Safety Pre-Check
                safe, warning = self.hardware_monitor.check_thermal_safety()
                if not safe:
                    logger.error(f"Halting training at epoch {epoch} due to thermal violation: {warning}")
                    break

                current_lr = self.optimizer.param_groups[0]["lr"]
                history["learning_rates"].append(current_lr)

                # Train epoch
                train_loss, train_acc, ep_time_ms, throughput, step_latencies = self.train_epoch(
                    epoch=epoch,
                    total_epochs=total_target_epochs,
                    global_start_time=total_start,
                )
                all_step_latencies.extend(step_latencies)

                # Validate
                val_metrics = self.validate()
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]

                # Step Learning Rate Scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Telemetry Snapshot
                telemetry = self.hardware_monitor.record_snapshot(
                    epoch=epoch,
                    phase="epoch_end",
                    loss=val_loss,
                    accuracy=val_acc,
                )

                history["train_loss"].append(train_loss)
                history["train_accuracy"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                history["epoch_times_ms"].append(ep_time_ms)
                history["train_throughput_samples_sec"].append(throughput)
                history["vram_allocated_mb"].append(telemetry.get("vram_allocated_mb", 0.0))
                history["vram_reserved_mb"].append(telemetry.get("vram_reserved_mb", 0.0))
                history["gpu_temperature_c"].append(telemetry.get("temperature_c"))
                history["gpu_utilization_pct"].append(telemetry.get("utilization_gpu_pct"))

                epoch_records.append({
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc * 100.0, 2),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc * 100.0, 2),
                    "duration_sec": round(ep_time_ms / 1000.0, 3),
                    "samples_per_sec": round(throughput, 1),
                    "learning_rate": current_lr,
                })

                logger.info(
                    f"[{self.device.type.upper()}{' (AMP)' if self.use_amp else ''}] "
                    f"Epoch {epoch}/{total_target_epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% | "
                    f"Time: {ep_time_ms:.1f}ms ({throughput:.1f} samp/s) | LR: {current_lr:.6f}"
                )

                # Checkpoint Management
                if save_checkpoints:
                    current_metrics = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc * 100.0,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc * 100.0,
                        "throughput": throughput,
                    }

                    if self.checkpoint_path is not None:
                        best_path = str(self.checkpoint_path)
                    else:
                        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")

                    # Best model tracking
                    if val_acc > self.best_val_acc + self.early_stopping_min_delta:
                        self.best_val_acc = val_acc
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch
                        self.epochs_without_improvement = 0

                        CheckpointManager.save_checkpoint(
                            model=self.model,
                            filepath=best_path,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            metrics=current_metrics,
                            metadata={"device": str(self.device), "use_amp": self.use_amp, "is_best": True},
                        )
                        self.best_checkpoint_saved = Path(best_path)
                    else:
                        self.epochs_without_improvement += 1

                # Emit epoch completion progress event
                if self.progress_callback:
                    elapsed = time.perf_counter() - total_start
                    total_train_samples = len(self.train_loader.dataset)
                    sps = (total_train_samples / (ep_time_ms / 1000.0)) if ep_time_ms > 0 else 0.0
                    vram_alloc = (
                        torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                        if self.device.type == "cuda"
                        else 0.0
                    )
                    vram_res = (
                        torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)
                        if self.device.type == "cuda"
                        else 0.0
                    )
                    total_batches = len(self.train_loader)

                    self.progress_callback({
                        "event": "epoch_completed",
                        "epoch": epoch,
                        "current_epoch": epoch,
                        "total_epochs": total_target_epochs,
                        "batch": total_batches,
                        "current_step": total_batches,
                        "total_batches": total_batches,
                        "total_steps": total_batches,
                        "train_loss": round(train_loss, 4),
                        "train_accuracy": round(train_acc * 100.0, 2),
                        "val_loss": round(val_loss, 4),
                        "val_accuracy": round(val_acc * 100.0, 2),
                        "best_val_accuracy": round(self.best_val_acc * 100.0, 2),
                        "epoch_time_sec": round(ep_time_ms / 1000.0, 2),
                        "elapsed_time_sec": round(elapsed, 2),
                        "samples_processed": total_train_samples,
                        "samples_per_sec": round(sps, 1),
                        "peak_allocated_vram_mb": round(vram_alloc, 2),
                        "peak_reserved_vram_mb": round(vram_res, 2),
                    })

                # Early Stopping
                if self.early_stopping_patience > 0 and self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    history["early_stopped"] = True
                    break

        except KeyboardInterrupt:
            logger.warning("Training manually interrupted! Saving emergency checkpoint...")
            history["interrupted"] = True

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_end = time.perf_counter()
        total_wall_time_sec = total_end - total_start

        history["total_time_ms"] = total_wall_time_sec * 1000.0
        history["best_epoch"] = self.best_epoch
        history["best_val_accuracy"] = self.best_val_acc
        history["best_val_loss"] = self.best_val_loss

        # Peak GPU Memory
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
        if self.device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            peak_res = torch.cuda.max_memory_reserved(self.device)
            peak_allocated_mb = round(peak_alloc / (1024.0 * 1024.0), 2)
            peak_reserved_mb = round(peak_res / (1024.0 * 1024.0), 2)
            history["peak_gpu_memory_allocated_mb"] = peak_allocated_mb
            history["peak_gpu_memory_reserved_mb"] = peak_reserved_mb

        # Isolated Final Test Evaluation
        test_metrics = {"loss": 0.0, "accuracy": 0.0}
        if self.test_loader is not None and not history.get("interrupted", False):
            test_metrics = self.evaluate_test_set()
            history["test_metrics"] = test_metrics

        test_acc = test_metrics.get("accuracy", 0.0) * 100.0
        test_loss = test_metrics.get("loss", 0.0)
        final_train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
        overall_samples = sum(len(self.train_loader.dataset) for _ in epoch_records)
        samples_per_sec = (overall_samples / total_wall_time_sec) if total_wall_time_sec > 0 else 0.0

        # Standardized schema fields for orchestrator & UI
        history.update({
            "backend": self.backend,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "Host CPU",
            "epochs": len(epoch_records),
            "total_wall_time_sec": round(total_wall_time_sec, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "final_train_loss": round(final_train_loss, 4),
            "best_val_acc": round(self.best_val_acc * 100.0, 2),
            "test_accuracy": round(test_acc, 2),
            "test_loss": round(test_loss, 4),
            "peak_allocated_vram_mb": peak_allocated_mb,
            "peak_reserved_vram_mb": peak_reserved_mb,
            "overall_step_stats": compute_timing_stats(all_step_latencies),
            "epoch_records": epoch_records,
            "dispatch_summary": {
                "total_custom_dispatches": 0,
                "custom_forward_dispatches": {},
                "custom_backward_dispatches": {},
                "is_custom_dispatched": False,
                "note": "ResNet-18 uses PyTorch native cuDNN/cuBLAS kernels. Custom CUDA autograd kernels are restricted to MLP.",
            },
            "checkpoint_path": str(self.best_checkpoint_saved) if self.best_checkpoint_saved else None,
        })

        return history
