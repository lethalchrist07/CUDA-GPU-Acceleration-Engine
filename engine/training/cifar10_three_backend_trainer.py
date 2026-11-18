"""
Phase 10 Stage 2 Milestone 5: Three-Backend CIFAR-10 Training Engine
Implements unified training lifecycle for:
1. CPU_NATIVE
2. GPU_NATIVE
3. GPU_CUSTOM

Enhanced with:
- Cooperative cancellation tokens (TrainingCancelledException)
- Fine-grained progress callbacks for FastAPI background workers
- Deterministic per-epoch seeding for 100% comparative fairness
- Safe CUDA memory cleanup and atomic checkpoint management
"""
import hashlib
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dispatch_tracker import KernelDispatchTracker
from .reproducibility import reset_dataloader_seed
from engine.checkpointing.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class TrainingCancelledException(Exception):
    """Raised when training is safely interrupted via a cancellation token."""
    pass


def compute_timing_stats(timings_ms: List[float]) -> Dict[str, float]:
    if not timings_ms:
        return {"median_ms": 0.0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "std_ms": 0.0}
    sorted_t = sorted(timings_ms)
    n = len(sorted_t)
    mean_val = sum(sorted_t) / n
    median_val = sorted_t[n // 2] if n % 2 != 0 else (sorted_t[n // 2 - 1] + sorted_t[n // 2]) / 2.0
    var = sum((x - mean_val) ** 2 for x in sorted_t) / (n - 1) if n > 1 else 0.0
    return {
        "median_ms": round(median_val, 4),
        "mean_ms": round(mean_val, 4),
        "min_ms": round(min(sorted_t), 4),
        "max_ms": round(max(sorted_t), 4),
        "std_ms": round(math.sqrt(var), 4),
    }


class ThreeBackendTrainer:
    """
    Modular trainer executing identical training routines across CPU_NATIVE,
    GPU_NATIVE, and GPU_CUSTOM backends with dispatch tracking, memory profiling,
    cancellation checking, and progress streaming.
    """

    def __init__(
        self,
        backend: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        tracker: Optional[KernelDispatchTracker] = None,
        seed: int = 42,
        cancel_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        progress_interval_batches: int = 20,
    ):
        self.backend = backend.upper()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints/phase10_stage2_milestone4")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.tracker = tracker or KernelDispatchTracker()
        self.seed = seed
        self.cancel_event = cancel_event
        self.progress_callback = progress_callback
        self.progress_interval_batches = max(1, progress_interval_batches)

        # Determine target device
        if self.backend == "CPU_NATIVE":
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device required for {self.backend}")
            self.device = torch.device("cuda")

        self.model.to(self.device)
        self.is_cuda = (self.device.type == "cuda")

    def _check_cancellation(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TrainingCancelledException("Training job cancellation requested by client.")

    def train_epoch(self, epoch: int, total_epochs: int = 3, global_start_time: float = 0.0) -> Tuple[float, float, List[float], int]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        step_latencies: List[float] = []
        total_batches = len(self.train_loader)

        # 100% deterministic per-epoch batch order & stochastic augmentation
        reset_dataloader_seed(self.train_loader, self.seed, epoch)

        if self.is_cuda:
            torch.cuda.synchronize()

        for batch_idx, (data, targets) in enumerate(self.train_loader, start=1):
            self._check_cancellation()

            data = data.to(self.device, non_blocking=self.is_cuda)
            targets = targets.to(self.device, non_blocking=self.is_cuda)
            batch_size = data.size(0)

            if self.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            try:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(data)

                # Check output validity
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    raise ValueError(f"NaN or Inf encountered in model outputs at epoch {epoch}, batch {batch_idx}")

                loss = self.criterion(outputs, targets)
                self.tracker.record_fallback("cross_entropy_loss", 1)

                # Backward pass
                loss.backward()

                # Check gradient validity
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            raise ValueError(f"NaN or Inf encountered in gradient of {name} at epoch {epoch}")

                # Optimizer step
                self.optimizer.step()
                self.tracker.record_fallback("optimizer_step", 1)

            except torch.cuda.OutOfMemoryError as oom_err:
                logger.warning(f"CUDA OutOfMemory in ThreeBackendTrainer at epoch {epoch}, batch {batch_idx}: {oom_err}")
                if self.is_cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                import gc
                gc.collect()
                if self.progress_callback:
                    self.progress_callback({
                        "event": "oom_alert",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "message": "CUDA OOM intercepted safely; memory cache flushed and checkpoints preserved intact.",
                    })
                raise

            if self.is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            step_latency_ms = (t1 - t0) * 1000.0
            step_latencies.append(step_latency_ms)

            # Accounting
            loss_val = loss.item()
            total_loss += loss_val * batch_size
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total_samples += batch_size

            # Periodic batch progress emission
            if self.progress_callback and (batch_idx % self.progress_interval_batches == 0 or batch_idx == total_batches):
                current_time = time.perf_counter()
                elapsed = current_time - (global_start_time if global_start_time > 0 else t0)
                running_loss = total_loss / total_samples
                running_acc = (correct / total_samples) * 100.0
                vram_alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024) if self.is_cuda else 0.0
                vram_res = torch.cuda.memory_reserved(self.device) / (1024 * 1024) if self.is_cuda else 0.0

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
                    "step_latency_ms": round(step_latency_ms, 2),
                    "peak_allocated_vram_mb": round(vram_alloc, 2),
                    "peak_reserved_vram_mb": round(vram_res, 2),
                })

        avg_loss = total_loss / total_samples
        accuracy = (correct / total_samples) * 100.0
        return avg_loss, accuracy, step_latencies, total_samples

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in loader:
                self._check_cancellation()

                data = data.to(self.device, non_blocking=self.is_cuda)
                targets = targets.to(self.device, non_blocking=self.is_cuda)
                batch_size = data.size(0)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * batch_size
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = (correct / total_samples) * 100.0 if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def fit(self, epochs: int) -> Dict[str, Any]:
        logger.info(f"Starting training: Backend={self.backend}, Device={self.device}, Epochs={epochs}")
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        self.tracker.reset()
        if self.backend == "GPU_CUSTOM":
            self.tracker.start()

        epoch_records: List[Dict[str, Any]] = []
        all_step_latencies: List[float] = []
        best_val_acc = 0.0
        total_train_samples = 0
        best_checkpoint_saved: Optional[Path] = None

        start_wall_time = time.perf_counter()

        try:
            for epoch in range(1, epochs + 1):
                self._check_cancellation()
                t_ep_start = time.perf_counter()

                train_loss, train_acc, step_times, samples_count = self.train_epoch(
                    epoch=epoch,
                    total_epochs=epochs,
                    global_start_time=start_wall_time,
                )
                all_step_latencies.extend(step_times)
                total_train_samples += samples_count

                self._check_cancellation()
                val_loss, val_acc = self.evaluate(self.val_loader)
                t_ep_end = time.perf_counter()
                epoch_sec = t_ep_end - t_ep_start

                record = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 2),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 2),
                    "epoch_time_sec": round(epoch_sec, 3),
                    "step_stats": compute_timing_stats(step_times),
                }
                epoch_records.append(record)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_checkpoint_saved = self.save_checkpoint(f"{self.backend.lower()}_best.pt", epoch, best_val_acc)

                # Emit epoch completion progress event
                if self.progress_callback:
                    total_batches = len(self.train_loader)
                    elapsed = time.perf_counter() - start_wall_time
                    sps = (total_train_samples / elapsed) if elapsed > 0 else 0.0
                    vram_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if self.is_cuda else 0.0
                    vram_res = torch.cuda.max_memory_reserved(self.device) / (1024 * 1024) if self.is_cuda else 0.0

                    self.progress_callback({
                        "event": "epoch_completed",
                        "epoch": epoch,
                        "current_epoch": epoch,
                        "total_epochs": epochs,
                        "batch": total_batches,
                        "current_step": total_batches,
                        "total_batches": total_batches,
                        "total_steps": total_batches,
                        "train_loss": round(train_loss, 4),
                        "train_accuracy": round(train_acc, 2),
                        "val_loss": round(val_loss, 4),
                        "val_accuracy": round(val_acc, 2),
                        "best_val_accuracy": round(best_val_acc, 2),
                        "epoch_time_sec": round(epoch_sec, 2),
                        "elapsed_time_sec": round(elapsed, 2),
                        "samples_processed": total_train_samples,
                        "samples_per_sec": round(sps, 1),
                        "peak_allocated_vram_mb": round(vram_alloc, 2),
                        "peak_reserved_vram_mb": round(vram_res, 2),
                    })

                logger.info(
                    f"[{self.backend}] Epoch {epoch}/{epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                    f"Time={epoch_sec:.2f}s"
                )

        finally:
            if self.backend == "GPU_CUSTOM":
                self.tracker.stop()
            if self.is_cuda:
                torch.cuda.empty_cache()

        total_wall_time_sec = time.perf_counter() - start_wall_time
        test_loss, test_acc = self.evaluate(self.test_loader)

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if self.is_cuda else 0.0
        )
        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(self.device) / (1024 * 1024) if self.is_cuda else 0.0
        )

        samples_per_sec = (total_train_samples / total_wall_time_sec) if total_wall_time_sec > 0 else 0.0

        dispatch_summary = self.tracker.get_summary() if self.backend == "GPU_CUSTOM" else {
            "total_custom_dispatches": 0,
            "custom_forward_dispatches": {},
            "custom_backward_dispatches": {},
            "is_custom_dispatched": False,
        }

        return {
            "backend": self.backend,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if self.is_cuda else "Host CPU",
            "epochs": epochs,
            "total_wall_time_sec": round(total_wall_time_sec, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "final_train_loss": epoch_records[-1]["train_loss"] if epoch_records else 0.0,
            "best_val_acc": round(best_val_acc, 2),
            "test_accuracy": round(test_acc, 2),
            "test_loss": round(test_loss, 4),
            "peak_allocated_vram_mb": round(peak_allocated_mb, 2),
            "peak_reserved_vram_mb": round(peak_reserved_mb, 2),
            "overall_step_stats": compute_timing_stats(all_step_latencies),
            "epoch_records": epoch_records,
            "dispatch_summary": dispatch_summary,
            "checkpoint_path": str(best_checkpoint_saved) if best_checkpoint_saved else None,
        }

    def save_checkpoint(self, filename: str, epoch: int, accuracy: float) -> Path:
        if self.checkpoint_path is not None:
            save_path = self.checkpoint_path
        else:
            save_path = self.checkpoint_dir / filename

        CheckpointManager.save_checkpoint(
            model=self.model,
            filepath=str(save_path),
            optimizer=self.optimizer,
            epoch=epoch,
            metrics={"val_accuracy": accuracy, "best_val_acc": accuracy, "accuracy": accuracy},
            metadata={"backend": self.backend, "device": str(self.device)},
        )
        return save_path

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        load_path = Path(filename) if Path(filename).is_file() else (self.checkpoint_dir / filename)
        res = CheckpointManager.load_checkpoint(
            filepath=str(load_path),
            model=self.model,
            optimizer=self.optimizer,
            map_location=self.device,
            restore_rng=True,
        )
        if "metrics" in res and isinstance(res["metrics"], dict):
            for k, v in res["metrics"].items():
                if k not in res:
                    res[k] = v
        return res
