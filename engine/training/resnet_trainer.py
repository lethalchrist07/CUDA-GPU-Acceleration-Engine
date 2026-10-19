"""
Configurable AI Training Engine for ResNet-18 & CIFAR-10 on CUDA / CPU (Phase 9).
Provides automatic mixed precision (AMP), learning rate scheduling, early stopping,
throughput tracking (samples/sec), peak VRAM profiling, isolated test evaluation,
and graceful interruption recovery.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.device.hardware_monitor import HardwareMonitor
from .trainer_base import BaseTrainer

logger = logging.getLogger(__name__)


class ResNetTrainer(BaseTrainer):
    """
    Advanced PyTorch trainer designed for ResNet-18 vision pipelines.
    Supports FP32, AMP, CPU reference, thermal protection, and metrics tracking.
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
        use_amp: bool = False,
        checkpoint_dir: str = "checkpoints",
        hardware_monitor: Optional[HardwareMonitor] = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.001,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_criterion = criterion or nn.CrossEntropyLoss()
        resolved_optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=resolved_criterion,
            optimizer=resolved_optimizer,
            device=resolved_device,
            config=config,
        )

        self.test_loader = test_loader
        self.scheduler = scheduler
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.hardware_monitor = hardware_monitor or HardwareMonitor(
            device_id=self.device.index or 0 if self.device.type == "cuda" else 0
        )
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # GradScaler for CUDA Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.best_val_acc = -1.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.current_epoch = 0
        self.epochs_without_improvement = 0

    def train_epoch(self) -> Tuple[float, float, float, float]:
        """
        Execute one complete training epoch across all training batches.
        Returns: (avg_loss, accuracy, epoch_time_ms, samples_per_second)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            batch_size = batch_x.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_size

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_sec = time.perf_counter() - start_time
        epoch_time_ms = elapsed_sec * 1000.0
        throughput = total_samples / elapsed_sec if elapsed_sec > 0 else 0.0

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy, epoch_time_ms, throughput

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
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
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
        early stopping, hardware telemetry, and checkpointing.
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

        # Checkpoint Resume
        if resume_from is not None:
            if os.path.exists(resume_from):
                logger.info(f"Resuming training session from: {resume_from}")
                meta = CheckpointManager.load_checkpoint(
                    filepath=resume_from,
                    model=self.model,
                    optimizer=self.optimizer,
                    map_location=self.device,
                )
                start_epoch = meta.get("epoch", 0) + 1
                prev_metrics = meta.get("metrics", {})
                self.best_val_acc = prev_metrics.get("val_accuracy", -1.0)
                self.best_val_loss = prev_metrics.get("val_loss", float("inf"))
                logger.info(f"Resumed at epoch {start_epoch}. Prior best val acc: {self.best_val_acc * 100:.2f}%")
            else:
                raise FileNotFoundError(f"Resume checkpoint file not found: {resume_from}")

        # Save initial pre-training weights checkpoint
        if save_checkpoints and start_epoch == 1:
            init_path = os.path.join(self.checkpoint_dir, "initial_model.pt")
            CheckpointManager.save_checkpoint(
                model=self.model,
                filepath=init_path,
                optimizer=self.optimizer,
                epoch=0,
                metrics={"initialized": True},
                metadata={"device": str(self.device), "use_amp": self.use_amp},
            )

        # CUDA Sync & Reset Peak Stats
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        total_start = time.perf_counter()

        try:
            for epoch in range(start_epoch, start_epoch + epochs):
                self.current_epoch = epoch

                # Thermal Safety Pre-Check
                safe, warning = self.hardware_monitor.check_thermal_safety()
                if not safe:
                    logger.error(f"Halting training at epoch {epoch} due to thermal violation: {warning}")
                    break

                current_lr = self.optimizer.param_groups[0]["lr"]
                history["learning_rates"].append(current_lr)

                # Train epoch
                train_loss, train_acc, ep_time_ms, throughput = self.train_epoch()

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

                logger.info(
                    f"[{self.device.type.upper()}{' (AMP)' if self.use_amp else ''}] "
                    f"Epoch {epoch}/{start_epoch + epochs - 1} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% | "
                    f"Time: {ep_time_ms:.1f}ms ({throughput:.1f} samp/s) | LR: {current_lr:.6f}"
                )

                # Checkpoint Management
                if save_checkpoints:
                    current_metrics = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "throughput": throughput,
                    }

                    latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
                    CheckpointManager.save_checkpoint(
                        model=self.model,
                        filepath=latest_path,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=current_metrics,
                        metadata={"device": str(self.device), "use_amp": self.use_amp},
                    )

                    # Best model tracking
                    if val_acc > self.best_val_acc + self.early_stopping_min_delta:
                        self.best_val_acc = val_acc
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch
                        self.epochs_without_improvement = 0

                        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                        CheckpointManager.save_checkpoint(
                            model=self.model,
                            filepath=best_path,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            metrics=current_metrics,
                            metadata={"device": str(self.device), "is_best": True},
                        )
                    else:
                        self.epochs_without_improvement += 1

                # Early Stopping
                if self.early_stopping_patience > 0 and self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered: Validation accuracy has not improved "
                        f"for {self.early_stopping_patience} consecutive epochs. Stopping at epoch {epoch}."
                    )
                    history["early_stopped"] = True
                    break

        except KeyboardInterrupt:
            logger.warning("Training manually interrupted! Saving emergency checkpoint...")
            history["interrupted"] = True
            if save_checkpoints:
                interrupted_path = os.path.join(self.checkpoint_dir, "interrupted_checkpoint.pt")
                CheckpointManager.save_checkpoint(
                    model=self.model,
                    filepath=interrupted_path,
                    optimizer=self.optimizer,
                    epoch=self.current_epoch,
                    metrics={"interrupted": True},
                    metadata={"device": str(self.device)},
                )
                logger.info(f"Interrupted checkpoint saved: {interrupted_path}")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_end = time.perf_counter()

        history["total_time_ms"] = (total_end - total_start) * 1000.0
        history["best_epoch"] = self.best_epoch
        history["best_val_accuracy"] = self.best_val_acc
        history["best_val_loss"] = self.best_val_loss

        # Peak GPU Memory
        if self.device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            peak_res = torch.cuda.max_memory_reserved(self.device)
            history["peak_gpu_memory_allocated_mb"] = round(peak_alloc / (1024.0 * 1024.0), 2)
            history["peak_gpu_memory_reserved_mb"] = round(peak_res / (1024.0 * 1024.0), 2)

        # Isolated Final Test Evaluation
        if self.test_loader is not None and not history.get("interrupted", False):
            history["test_metrics"] = self.evaluate_test_set()

        return history
