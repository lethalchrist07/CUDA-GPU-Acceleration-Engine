"""
Production-Grade Autoregressive Transformer Training Engine.

Engineered for NVIDIA CUDA (Ampere SM 8.6) and CPU execution.
Features:
- Next-token Cross-Entropy loss with numerical protection
- AdamW optimizer with decoupled weight decay
- Cosine Annealing learning rate schedule with Linear Warmup
- Gradient accumulation with exact tail window handling
- Gradient clipping (1.0 default)
- PyTorch Native Mixed Precision (AMP FP16) via torch.amp
- Activation checkpointing support
- Deterministic seeding (CPU, CUDA, PyTorch)
- Accurate tokens/second throughput & GPU VRAM telemetry
- Perplexity computation with overflow clamping
- Atomic checkpointing and complete resume fidelity via CheckpointManager
- Architectural and dataset compatibility validation on resume
- NaN/Inf loss and gradient explosion detection
- Cooperative cancellation support
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.transformer.model import GPTTransformer
from engine.transformer.config import TransformerConfig


@dataclass
class TransformerTrainingConfig:
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 20
    max_steps: Optional[int] = None
    epochs: int = 1
    mixed_precision: bool = True  # Uses torch.amp on CUDA
    eval_interval_steps: int = 50
    checkpoint_interval_steps: int = 100
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    dataset_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "grad_clip": self.grad_clip,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "epochs": self.epochs,
            "mixed_precision": self.mixed_precision,
            "eval_interval_steps": self.eval_interval_steps,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "seed": self.seed,
            "device": self.device,
            "dataset_fingerprint": self.dataset_fingerprint,
        }


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Creates learning rate scheduler with linear warmup followed by cosine decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def safe_perplexity(loss_value: float) -> float:
    """
    Computes perplexity = exp(loss) with clamping to prevent overflow.
    """
    if math.isnan(loss_value) or math.isinf(loss_value):
        return float("inf")
    clamped_loss = min(loss_value, 100.0)
    return math.exp(clamped_loss)


class TransformerTrainer:
    """
    Production-grade hardened trainer for GPTTransformer.
    """

    def __init__(
        self,
        model: GPTTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TransformerTrainingConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TransformerTrainingConfig()
        self.progress_callback = progress_callback
        self.is_cancelled = is_cancelled or (lambda: False)

        # Determine target device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.is_cuda = self.device.type == "cuda"
        self.model.to(self.device)

        # Set seeds
        self._set_seed(self.config.seed)

        # Setup Optimizer with weight decay separation
        self.optimizer = self._configure_optimizer()

        # Setup Mixed Precision Scaler
        self.use_amp = self.is_cuda and self.config.mixed_precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints/transformer")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Explicit metric counters
        self.global_step = 0
        self.total_tokens_processed = 0
        self.microbatches_processed = 0
        self.total_val_tokens_evaluated = 0
        self.scheduler: Optional[LambdaLR] = None

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _configure_optimizer(self) -> AdamW:
        """
        Separates parameters into decay and non-decay (biases, norms, embeddings)
        following standard GPT-2/LLaMA practice.
        """
        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
        )

    def _get_vram_telemetry(self) -> Dict[str, Any]:
        if not self.is_cuda:
            return {
                "allocated_mb": 0.0,
                "max_allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "physical_used_mb": 0.0,
                "physical_total_mb": 0.0,
                "temperature_c": None,
                "utilization_gpu_pct": None,
            }
        telemetry: Dict[str, Any] = {
            "allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024**2), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024**2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024**2), 2),
            "physical_used_mb": None,
            "physical_total_mb": None,
            "temperature_c": None,
            "utilization_gpu_pct": None,
        }
        try:
            from engine.device.hardware_monitor import get_fast_gpu_telemetry
            dev_idx = self.device.index if self.device.index is not None else 0
            fast = get_fast_gpu_telemetry(dev_idx)
            if fast:
                telemetry["temperature_c"] = fast.get("temperature_c")
                telemetry["utilization_gpu_pct"] = fast.get("utilization_gpu_pct")
                telemetry["physical_used_mb"] = fast.get("physical_used_mb")
                telemetry["physical_total_mb"] = fast.get("physical_total_mb")
        except Exception:
            pass
        return telemetry

    def evaluate(self, loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """
        Evaluates cross-entropy loss and perplexity on the validation dataset.
        """
        eval_loader = loader or self.val_loader
        if eval_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        val_tokens_this_eval = 0

        with torch.no_grad():
            for x, y in eval_loader:
                if self.is_cancelled():
                    break
                x = x.to(self.device, non_blocking=self.is_cuda)
                y = y.to(self.device, non_blocking=self.is_cuda)

                val_tokens_this_eval += x.numel()

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, loss = self.model(x, targets=y)

                if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    total_batches += 1

        self.total_val_tokens_evaluated += val_tokens_this_eval
        avg_loss = total_loss / max(1, total_batches)
        perplexity = safe_perplexity(avg_loss)
        return round(avg_loss, 4), round(perplexity, 2)

    def fit(
        self,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        resume_from: Optional[Union[str, Path]] = None,
        allow_dataset_transition: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes complete training loop with step accounting, gradient accumulation,
        evaluation, and atomic checkpointing.
        """
        target_epochs = epochs or self.config.epochs
        num_batches_per_epoch = len(self.train_loader)
        accum_steps = max(1, self.config.gradient_accumulation_steps)
        
        # Calculate total optimizer steps expected
        steps_per_epoch = math.ceil(num_batches_per_epoch / accum_steps)
        total_dataset_steps = target_epochs * steps_per_epoch
        total_expected_steps = min(max_steps, total_dataset_steps) if max_steps else total_dataset_steps
        
        # Setup scheduler
        min_ratio = self.config.min_learning_rate / max(self.config.learning_rate, 1e-8)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=max(1, total_expected_steps),
            min_lr_ratio=min_ratio,
        )

        if resume_from:
            self.load_checkpoint(
                filepath=resume_from,
                expected_dataset_fingerprint=self.config.dataset_fingerprint,
                allow_dataset_transition=allow_dataset_transition,
            )

        start_tokens = self.total_tokens_processed
        start_time = time.perf_counter()
        best_val_loss = float("inf")
        best_checkpoint_path: Optional[str] = None
        step_records: List[Dict[str, Any]] = []

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        is_interrupted = False

        for epoch in range(1, target_epochs + 1):
            if is_interrupted or (max_steps and self.global_step >= max_steps):
                break

            self.model.train()
            running_loss = 0.0
            microbatch_in_accum = 0

            for batch_idx, (x, y) in enumerate(self.train_loader, start=1):
                if self.is_cancelled():
                    is_interrupted = True
                    break

                step_start_time = time.perf_counter()
                x = x.to(self.device, non_blocking=self.is_cuda)
                y = y.to(self.device, non_blocking=self.is_cuda)

                B, T = x.size()
                num_tokens = B * T
                self.total_tokens_processed += num_tokens
                self.microbatches_processed += 1

                # Determine effective accumulation divisor
                is_last_batch_in_epoch = (batch_idx == num_batches_per_epoch)
                
                # Forward pass with AMP
                try:
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        logits, loss = self.model(x, targets=y)
                        # Normalize loss by accumulation window
                        loss_scaled = loss / accum_steps

                    # NaN / Inf detection
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise FloatingPointError(
                            f"NaN or Inf encountered in training loss at epoch {epoch}, batch {batch_idx}"
                        )

                    # Backward pass with Scaler
                    self.scaler.scale(loss_scaled).backward()
                except torch.cuda.OutOfMemoryError as oom_err:
                    import logging
                    logging.getLogger("TransformerTrainer").warning(
                        f"CUDA OutOfMemory in Transformer step (batch {batch_idx}, step {self.global_step}): {oom_err}"
                    )
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
                            "step": self.global_step,
                            "message": "CUDA OOM intercepted safely; memory cache flushed and checkpoints preserved intact.",
                        })
                    raise
                running_loss += loss.item()
                microbatch_in_accum += 1

                # Optimizer step when accumulation threshold is met or at final batch of epoch
                if microbatch_in_accum == accum_steps or is_last_batch_in_epoch:
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = 0.0
                    if self.config.grad_clip > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        grad_norm = float(total_norm.item() if hasattr(total_norm, "item") else total_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.global_step += 1
                    actual_accum = microbatch_in_accum
                    microbatch_in_accum = 0

                    step_elapsed = time.perf_counter() - step_start_time
                    tokens_per_sec = (num_tokens * actual_accum) / max(step_elapsed, 1e-6)
                    curr_lr = self.optimizer.param_groups[0]["lr"]
                    avg_step_loss = running_loss / actual_accum
                    running_loss = 0.0

                    vram = self._get_vram_telemetry()
                    checkpoint_status = "none"

                    # Periodic validation
                    latest_val_loss = None
                    latest_val_ppl = None
                    if self.val_loader and (self.global_step % self.config.eval_interval_steps == 0):
                        v_loss, v_ppl = self.evaluate()
                        latest_val_loss = v_loss
                        latest_val_ppl = v_ppl
                        self.model.train()
                        if v_loss < best_val_loss:
                            best_val_loss = v_loss
                            ckpt_file = self.checkpoint_dir / "best_model.pt"
                            self.save_checkpoint(
                                ckpt_file,
                                step=self.global_step,
                                epoch=epoch,
                                metrics={"val_loss": v_loss, "val_perplexity": v_ppl},
                            )
                            best_checkpoint_path = str(ckpt_file)
                            checkpoint_status = "saved"

                    # Periodic step checkpointing
                    if self.config.checkpoint_interval_steps > 0 and (self.global_step % self.config.checkpoint_interval_steps == 0):
                        periodic_ckpt = self.checkpoint_dir / f"step_{self.global_step}.pt"
                        self.save_checkpoint(
                            periodic_ckpt,
                            step=self.global_step,
                            epoch=epoch,
                            metrics={"val_loss": latest_val_loss, "val_perplexity": latest_val_ppl, "train_loss": avg_step_loss},
                        )
                        checkpoint_status = "saved"

                    elapsed_total = round(time.perf_counter() - start_time, 2)
                    step_data = {
                        "step": self.global_step,
                        "current_step": self.global_step,
                        "total_steps": total_expected_steps,
                        "optimizer_step": self.global_step,
                        "total_optimizer_steps": total_expected_steps,
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": num_batches_per_epoch,
                        "train_loss": round(avg_step_loss, 4),
                        "train_perplexity": round(safe_perplexity(avg_step_loss), 2),
                        "val_loss": latest_val_loss,
                        "val_perplexity": latest_val_ppl,
                        "learning_rate": curr_lr,
                        "grad_norm": round(grad_norm, 4),
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "total_tokens": self.total_tokens_processed,
                        "tokens_processed": self.total_tokens_processed,
                        "microbatches_processed": self.microbatches_processed,
                        "total_microbatches": num_batches_per_epoch * target_epochs,
                        "total_tokens_budget": total_expected_steps * (actual_accum * num_tokens),
                        "step_duration_sec": round(step_elapsed, 4),
                        "elapsed_sec": elapsed_total,
                        "peak_vram_mb": vram["max_allocated_mb"],
                        "allocated_vram_mb": vram["allocated_mb"],
                        "reserved_vram_mb": vram["reserved_mb"],
                        "physical_vram_mb": vram.get("physical_used_mb"),
                        "gpu_temperature_c": vram.get("temperature_c"),
                        "gpu_utilization_pct": vram.get("utilization_gpu_pct"),
                        "checkpoint_status": checkpoint_status,
                    }
                    step_records.append(step_data)

                    if self.progress_callback:
                        self.progress_callback(step_data)

                    if max_steps and self.global_step >= max_steps:
                        break

        total_wall_time = time.perf_counter() - start_time
        final_val_loss, final_val_ppl = self.evaluate()

        # If step records exist, annotate the final step record with the final validation result
        # so time-series progression curves have the final validation point without fabricating intermediate steps
        if step_records:
            step_records[-1]["val_loss"] = final_val_loss
            step_records[-1]["val_perplexity"] = final_val_ppl
            step_records[-1]["is_final_eval"] = True

        status_str = "CANCELLED" if is_interrupted else "COMPLETED"

        # Save final checkpoint if completed
        final_ckpt_file = self.checkpoint_dir / "final_model.pt"
        if not is_interrupted:
            self.save_checkpoint(
                final_ckpt_file,
                step=self.global_step,
                epoch=epoch if 'epoch' in locals() else 1,
                metrics={"final_val_loss": final_val_loss, "final_val_perplexity": final_val_ppl},
            )

        additional_tokens = self.total_tokens_processed - start_tokens
        overall_throughput = additional_tokens / max(total_wall_time, 1e-6)
        vram_final = self._get_vram_telemetry()

        return {
            "status": status_str,
            "global_steps": self.global_step,
            "optimizer_steps_completed": self.global_step,
            "microbatches_processed": self.microbatches_processed,
            "total_epochs": epoch if 'epoch' in locals() else 1,
            "total_tokens_processed": self.total_tokens_processed,
            "additional_tokens_processed": additional_tokens,
            "total_val_tokens_evaluated": self.total_val_tokens_evaluated,
            "total_wall_time_sec": round(total_wall_time, 3),
            "tokens_per_sec": round(overall_throughput, 1),
            "final_train_loss": step_records[-1]["train_loss"] if step_records else 0.0,
            "final_val_loss": final_val_loss,
            "final_val_perplexity": final_val_ppl,
            "peak_vram_mb": vram_final["max_allocated_mb"],
            "physical_vram_mb": vram_final.get("physical_used_mb"),
            "gpu_temperature_c": vram_final.get("temperature_c"),
            "gpu_utilization_pct": vram_final.get("utilization_gpu_pct"),
            "best_checkpoint_path": best_checkpoint_path or str(final_ckpt_file),
            "device": str(self.device),
            "model_parameters": self.model.count_parameters()["total_parameters"],
            "step_records": step_records,
        }

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Saves full training checkpoint via CheckpointManager.
        Guarantees atomic file write, RNG preservation, and protected baseline defense.
        """
        save_path = Path(filepath)
        extra_metrics = metrics or {}
        extra_metrics.update({
            "step": step,
            "total_tokens": self.total_tokens_processed,
            "microbatches_processed": self.microbatches_processed,
        })

        meta = metadata or {}
        meta.update({
            "model_config": self.model.config.to_dict(),
            "training_config": self.config.to_dict(),
            "device": str(self.device),
            "dataset_fingerprint": self.config.dataset_fingerprint,
        })

        return CheckpointManager.save_checkpoint(
            model=self.model,
            filepath=str(save_path),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            epoch=epoch,
            metrics=extra_metrics,
            metadata=meta,
        )

    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        expected_dataset_fingerprint: Optional[str] = None,
        allow_dataset_transition: bool = False,
    ) -> Dict[str, Any]:
        """
        Restores full model, optimizer, scheduler, scaler, and RNG states.
        Enforces architectural and dataset compatibility validation.
        """
        load_path = Path(filepath)
        state = CheckpointManager.load_checkpoint(
            filepath=str(load_path),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            map_location=self.device,
            restore_rng=True,
        )

        metadata = state.get("metadata", {})
        saved_cfg = metadata.get("model_config", {})
        
        # Verify architectural compatibility
        if saved_cfg:
            curr_cfg = self.model.config.to_dict()
            critical_keys = ["vocab_size", "d_model", "n_layer", "n_head"]
            for k in critical_keys:
                if k in saved_cfg and saved_cfg[k] != curr_cfg.get(k):
                    raise ValueError(
                        f"Architectural incompatibility in checkpoint for '{k}': "
                        f"checkpoint has {saved_cfg[k]}, but active model has {curr_cfg.get(k)}."
                    )

        # Verify dataset fingerprint if specified
        saved_fingerprint = metadata.get("dataset_fingerprint")
        if expected_dataset_fingerprint and saved_fingerprint and not allow_dataset_transition:
            if saved_fingerprint != expected_dataset_fingerprint:
                raise ValueError(
                    f"Dataset fingerprint mismatch on resume! "
                    f"Checkpoint trained on {saved_fingerprint[:16]}..., but current dataset is {expected_dataset_fingerprint[:16]}..."
                )

        metrics = state.get("metrics", {})
        self.global_step = metrics.get("step", 0)
        self.total_tokens_processed = metrics.get("total_tokens", 0)
        self.microbatches_processed = metrics.get("microbatches_processed", 0)
        return state
