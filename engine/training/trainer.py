import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer_base import BaseTrainer
from engine.checkpointing.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class ModelTrainer(BaseTrainer):
    """
    Comprehensive PyTorch training engine supporting CPU and CUDA GPU modes,
    automatic mixed-precision (AMP), memory profiling, loss/accuracy tracking,
    and atomic checkpoint persistence with resume capability.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_amp: bool = False,
        checkpoint_dir: str = "checkpoints",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
        )
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize GradScaler for mixed-precision if active
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.best_val_acc = -1.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.current_epoch = 0

    def train_epoch(self) -> float:
        """Run a single training epoch across all training batches."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

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

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples if total_samples > 0 else 0.0

    def validate(self) -> Dict[str, float]:
        """Evaluate validation metrics (loss and accuracy) across validation set."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)

                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
        }

    def fit(
        self,
        epochs: int,
        resume_from: Optional[str] = None,
        save_checkpoints: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full training and validation pipeline for the requested epochs.
        Tracks accurate timing with CUDA synchronization and GPU peak memory stats.
        """
        start_epoch = 1
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_times_ms": [],
            "total_time_ms": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "device": str(self.device),
            "use_amp": self.use_amp,
            "best_epoch": 0,
            "best_val_accuracy": 0.0,
            "best_val_loss": float("inf"),
        }

        # Handle resume if specified
        if resume_from is not None:
            if os.path.exists(resume_from):
                logger.info(f"Resuming training session from checkpoint: {resume_from}")
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
                logger.info(f"Resumed at epoch {start_epoch}. Best previous accuracy: {self.best_val_acc * 100:.2f}%")
            else:
                raise FileNotFoundError(f"Resume checkpoint file not found: {resume_from}")

        # CUDA Synchronization and Peak Memory Reset
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        total_start = time.perf_counter()

        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            ep_start = time.perf_counter()

            train_loss = self.train_epoch()
            val_metrics = self.validate()

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            ep_end = time.perf_counter()
            ep_time_ms = (ep_end - ep_start) * 1000.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["epoch_times_ms"].append(ep_time_ms)

            logger.info(
                f"[{self.device.type.upper()}] Epoch {epoch}/{start_epoch + epochs - 1} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | Time: {ep_time_ms:.2f} ms"
            )

            # Checkpoint Management
            if save_checkpoints:
                current_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }

                # Save Latest Checkpoint
                latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
                CheckpointManager.save_checkpoint(
                    model=self.model,
                    filepath=latest_path,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=current_metrics,
                    metadata={"device": str(self.device), "use_amp": self.use_amp},
                )

                # Save Best Checkpoint (based on validation accuracy)
                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.best_val_loss = val_metrics["loss"]
                    self.best_epoch = epoch

                    best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                    CheckpointManager.save_checkpoint(
                        model=self.model,
                        filepath=best_path,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=current_metrics,
                        metadata={"device": str(self.device), "is_best": True},
                    )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_end = time.perf_counter()

        history["total_time_ms"] = (total_end - total_start) * 1000.0
        history["best_epoch"] = self.best_epoch
        history["best_val_accuracy"] = self.best_val_acc
        history["best_val_loss"] = self.best_val_loss

        # Query Peak GPU Memory Consumption
        if self.device.type == "cuda":
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            history["peak_gpu_memory_mb"] = round(peak_bytes / (1024.0 * 1024.0), 2)
        else:
            history["peak_gpu_memory_mb"] = 0.0

        return history
