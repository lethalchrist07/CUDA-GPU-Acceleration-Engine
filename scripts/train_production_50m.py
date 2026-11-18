"""
scripts/train_production_50m.py
Phase 13 Milestone 6: Production-Scale Transformer Pretraining Campaign (50M Cumulative Tokens).
Advances the 86.83M parameter Transformer from Step 2,492 (10,205,184 tokens)
to Step 12,208 (50,000,000 cumulative tokens) on the NVIDIA GeForce RTX 3050 6GB Laptop GPU.

Features:
- Native SDPA Attention (Flash/Memory-Efficient Attention)
- Automatic Mixed Precision (AMP FP16) with GradScaler
- Custom CUDA Fused Bias+GELU autograd kernel
- Scalable memory-mapped zero-copy dataset ingestion
- Continuous GPU temperature, utilization, and VRAM monitoring
- Conservative thermal throttling / safety pause (82°C threshold)
- Atomic double-buffered checkpointing (latest, best, milestone snapshots, final)
- Held-out validation (The Time Machine) and test (Newton's Opticks) evaluation
- Diagnostic mode (--mode diagnostic) for pre-launch verification
- Full production mode (--mode full)
"""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.transformer.config import TransformerConfig
from engine.transformer.model import GPTTransformer
from engine.transformer.scalable_dataset import (
    ScalableMemmapDataset,
    create_scalable_dataloaders,
)
from engine.transformer.trainer import safe_perplexity

def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates learning rate scheduler with linear warmup followed by cosine decay."""
    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", param_group["lr"])

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


import custom_kernels.autograd as cka

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Pretraining50MCampaign")


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def query_gpu_telemetry() -> Dict[str, float]:
    """Queries GPU temperature, utilization, and memory via nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "temperature_c": float(parts[0]),
            "gpu_utilization_pct": float(parts[1]),
            "vram_used_mb": float(parts[2]),
            "vram_total_mb": float(parts[3]),
        }
    except Exception:
        return {
            "temperature_c": 50.0,
            "gpu_utilization_pct": 0.0,
            "vram_used_mb": 0.0,
            "vram_total_mb": 6144.0,
        }


class CustomFusedTransformerMLP(nn.Module):
    """Drop-in MLP replacement using Custom CUDA Fused Bias+GELU."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.d_ff = config.d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        gemm_out = torch.matmul(x, self.c_fc.weight.T)
        gemm_2d = gemm_out.view(-1, self.d_ff)
        h_2d = cka.fused_bias_activation(gemm_2d, self.c_fc.bias, "gelu")
        h = h_2d.view(B, T, self.d_ff)
        h = self.c_proj(h)
        return self.dropout(h)


def inject_custom_fused_mlp(model: GPTTransformer) -> GPTTransformer:
    """Replaces standard MLPs with Custom CUDA Fused Bias+GELU MLPs."""
    for block in model.blocks:
        fused_mlp = CustomFusedTransformerMLP(model.config).to(block.mlp.c_fc.weight.device)
        with torch.no_grad():
            fused_mlp.c_fc.weight.copy_(block.mlp.c_fc.weight)
            fused_mlp.c_fc.bias.copy_(block.mlp.c_fc.bias)
            fused_mlp.c_proj.weight.copy_(block.mlp.c_proj.weight)
            fused_mlp.c_proj.bias.copy_(block.mlp.c_proj.bias)
        block.mlp = fused_mlp
    return model


def evaluate_dataset(
    model: GPTTransformer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluates cross-entropy loss and perplexity on a dataset loader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, targets=y)
            total_loss += loss.item()
            total_batches += 1
    avg_loss = total_loss / max(1, total_batches)
    ppl = safe_perplexity(avg_loss)
    return avg_loss, ppl


def run_campaign(mode: str = "full"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA target required for Phase 13 Milestone 6 training.")

    gpu_name = torch.cuda.get_device_name(0)
    data_dir = PROJECT_ROOT / "data" / "text" / "production_10m"
    resume_ckpt_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_diagnostic" / "final_model.pt"
    campaign_ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_campaign"
    results_dir = PROJECT_ROOT / "results"

    campaign_ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"PHASE 13 MILESTONE 6: 50M-TOKEN EXTENDED PRETRAINING (MODE: {mode.upper()})")
    logger.info(f"Device: {gpu_name} (Ampere SM 8.6, 6GB VRAM)")
    logger.info(f"Resume Checkpoint: {resume_ckpt_path}")
    logger.info(f"Checkpoint Output Directory: {campaign_ckpt_dir}")
    logger.info("=" * 80)

    # 1. Load Data
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    test_bin = data_dir / "test_tokens.bin"
    manifest_path = data_dir / "dataset_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    dataset_fingerprint = manifest["aggregate_content_hash_sha256"]

    train_ds = ScalableMemmapDataset(train_bin, seq_len=256, stride=256, vocab_size=2048)
    val_ds = ScalableMemmapDataset(val_bin, seq_len=256, stride=256, vocab_size=2048)
    test_ds = ScalableMemmapDataset(test_bin, seq_len=256, stride=256, vocab_size=2048)

    loaders = create_scalable_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=16,
        seed=42,
        num_workers=0,
        pin_memory=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    logger.info(f"Loaded Datasets: Train={train_ds.total_tokens:,} tok ({len(train_loader)} batches), Val={val_ds.total_tokens:,} tok, Test={test_ds.total_tokens:,} tok")

    # 2. Load Checkpoint
    logger.info("Loading baseline checkpoint...")
    ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
    metrics = ckpt.get("metrics", {})
    start_step = metrics.get("step", 2492)
    start_tokens = metrics.get("total_tokens", 10_205_184)
    model_cfg_dict = ckpt.get("metadata", {}).get("model_config", {})

    cfg = TransformerConfig(**model_cfg_dict)
    model = GPTTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Inject Custom CUDA Fused Bias+GELU
    logger.info("Integrating Custom CUDA Fused Bias+GELU kernels into TransformerMLP blocks...")
    model = inject_custom_fused_mlp(model)

    # 3. Target Step Calculations
    tokens_per_step = 16 * 256  # 4,096 tokens
    target_lifetime_tokens = 50_000_000

    if mode == "diagnostic":
        target_max_steps = start_step + 5
        eval_interval = 2
        ckpt_interval = 2
        log_interval = 1
        num_epochs = 1
    else:
        # Full campaign: exactly 50M tokens
        remaining_tokens = target_lifetime_tokens - start_tokens
        steps_to_train = math.ceil(remaining_tokens / tokens_per_step)
        target_max_steps = start_step + steps_to_train
        eval_interval = 500
        ckpt_interval = 1000
        log_interval = 50
        steps_per_epoch = len(train_loader)
        num_epochs = math.ceil(steps_to_train / steps_per_epoch) + 1

    logger.info(f"Training Plan: Start Step={start_step}, Target Max Steps={target_max_steps} ({target_max_steps - start_step} steps to train)")
    logger.info(f"Lifetime Token Target: {target_lifetime_tokens:,} tokens (Starting from {start_tokens:,} tokens)")
    logger.info(f"Epochs Planned: {num_epochs} epochs ({len(train_loader)} steps/epoch)")

    # 4. Setup Optimizer, Scaler, Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Successfully restored AdamW optimizer state from checkpoint.")
        except Exception as e:
            logger.warning(f"Could not restore exact optimizer state ({e}); initializing fresh AdamW.")

    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            logger.info("Successfully restored AMP GradScaler state from checkpoint.")
        except Exception as e:
            logger.warning(f"Could not restore scaler state ({e}).")

    scheduler = get_cosine_schedule(
        optimizer=optimizer,
        warmup_steps=200 if mode == "full" else 1,
        max_steps=target_max_steps,
        min_lr_ratio=0.1,  # min_lr = 3e-5
        last_epoch=start_step - 1,
    )

    del ckpt
    torch.cuda.empty_cache()

    # 5. Baseline Evaluation
    logger.info("Evaluating baseline zero-step metrics on held-out validation and test sets...")
    initial_val_loss, initial_val_ppl = evaluate_dataset(model, val_loader, device)
    initial_test_loss, initial_test_ppl = evaluate_dataset(model, test_loader, device)
    logger.info(f"Baseline Validation (The Time Machine): Loss={initial_val_loss:.4f}, PPL={initial_val_ppl:.2f}")
    logger.info(f"Baseline Held-Out Test (Newton Opticks): Loss={initial_test_loss:.4f}, PPL={initial_test_ppl:.2f}")

    # 6. Training Loop Setup
    global_step = start_step
    cumulative_tokens = start_tokens
    best_val_loss = initial_val_loss
    best_val_step = start_step

    thermal_threshold = 82.0
    cooldown_pause_sec = 30.0

    step_telemetry: List[Dict[str, Any]] = []
    checkpoint_history: List[Dict[str, Any]] = []

    def save_atomic_checkpoint(filepath: Path, current_val_loss: Optional[float] = None, current_val_ppl: Optional[float] = None) -> str:
        temp_path = filepath.with_suffix(".tmp")
        save_dict = {
            "epoch": current_epoch,
            "step": global_step,
            "total_tokens": cumulative_tokens,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
            "metrics": {
                "step": global_step,
                "total_tokens": cumulative_tokens,
                "val_loss": current_val_loss,
                "val_perplexity": current_val_ppl,
            },
            "metadata": {
                "model_config": cfg.to_dict(),
                "dataset_fingerprint": dataset_fingerprint,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        torch.save(save_dict, temp_path)
        os.replace(temp_path, filepath)
        h = compute_sha256(filepath)
        checkpoint_history.append({
            "filepath": str(filepath),
            "step": global_step,
            "tokens": cumulative_tokens,
            "sha256": h,
            "size_bytes": filepath.stat().st_size,
        })
        return h

    logger.info("Beginning training loop execution...")
    t_start = time.perf_counter()
    current_epoch = 1

    for epoch in range(1, num_epochs + 1):
        current_epoch = epoch
        if global_step >= target_max_steps:
            break

        model.train()
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if global_step >= target_max_steps:
                break

            t0_step = time.perf_counter()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Thermal Check every 100 steps
            if global_step % 100 == 0:
                gpu_stat = query_gpu_telemetry()
                if gpu_stat["temperature_c"] >= thermal_threshold:
                    logger.warning(f"[THERMAL INTERVENTION] GPU Temperature reached {gpu_stat['temperature_c']}°C (threshold: {thermal_threshold}°C). Pausing {cooldown_pause_sec}s for cooldown...")
                    time.sleep(cooldown_pause_sec)
                    torch.cuda.empty_cache()

            # Forward pass under AMP FP16
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss = model(x, targets=y)

            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(f"NaN/Inf loss detected at step {global_step}!")

            # Backward & Optimizer step with Scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm = float(grad_norm_tensor.item() if hasattr(grad_norm_tensor, "item") else grad_norm_tensor)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            torch.cuda.synchronize()
            t_step_elapsed = time.perf_counter() - t0_step

            global_step += 1
            cumulative_tokens += tokens_per_step

            step_loss = float(loss.item())
            curr_lr = optimizer.param_groups[0]["lr"]
            step_tput = tokens_per_step / max(t_step_elapsed, 1e-6)

            # Telemetry Logging
            if global_step % log_interval == 0 or global_step == target_max_steps:
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                mem_res = torch.cuda.memory_reserved() / (1024 ** 2)
                gpu_stat = query_gpu_telemetry()

                record = {
                    "step": global_step,
                    "tokens": cumulative_tokens,
                    "train_loss": round(step_loss, 4),
                    "train_perplexity": round(safe_perplexity(step_loss), 2),
                    "learning_rate": curr_lr,
                    "grad_norm": round(grad_norm, 4),
                    "tokens_per_sec": round(step_tput, 1),
                    "step_latency_ms": round(t_step_elapsed * 1000, 2),
                    "peak_vram_alloc_mb": round(mem_alloc, 2),
                    "peak_vram_res_mb": round(mem_res, 2),
                    "temperature_c": gpu_stat["temperature_c"],
                    "gpu_utilization_pct": gpu_stat["gpu_utilization_pct"],
                    "elapsed_sec": round(time.perf_counter() - t_start, 2),
                }
                step_telemetry.append(record)
                logger.info(
                    f"Step {global_step:5d} / {target_max_steps:5d} | "
                    f"Loss: {step_loss:.4f} | LR: {curr_lr:.2e} | "
                    f"Grad: {grad_norm:.3f} | {step_tput:6.1f} tok/s | "
                    f"VRAM: {mem_alloc:.0f}/{mem_res:.0f} MB | "
                    f"Temp: {gpu_stat['temperature_c']}°C | "
                    f"Tok: {cumulative_tokens:,}"
                )

            # Periodic Validation & Best Checkpointing
            if global_step % eval_interval == 0 or global_step == target_max_steps:
                val_loss, val_ppl = evaluate_dataset(model, val_loader, device)
                logger.info(f"--- Periodic Validation at Step {global_step}: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f} (Best: {best_val_loss:.4f}) ---")
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = global_step
                    best_ckpt_file = campaign_ckpt_dir / "best_model.pt"
                    save_atomic_checkpoint(best_ckpt_file, val_loss, val_ppl)
                    logger.info(f"[+] Saved NEW BEST model to {best_ckpt_file} (Val Loss: {best_val_loss:.4f})")

            # Periodic Step Checkpointing & Latest Checkpoint
            if global_step % ckpt_interval == 0:
                milestone_ckpt = campaign_ckpt_dir / f"step_{global_step}.pt"
                latest_ckpt = campaign_ckpt_dir / "latest_model.pt"
                save_atomic_checkpoint(milestone_ckpt)
                save_atomic_checkpoint(latest_ckpt)
                logger.info(f"[+] Periodic checkpoint saved: {milestone_ckpt}")

    total_training_time = time.perf_counter() - t_start
    overall_throughput = (cumulative_tokens - start_tokens) / max(total_training_time, 1e-6)

    # 7. Final Checkpoint and Full Evaluation
    latest_ckpt = campaign_ckpt_dir / "latest_model.pt"
    final_ckpt = campaign_ckpt_dir / "final_model.pt"
    save_atomic_checkpoint(latest_ckpt)
    save_atomic_checkpoint(final_ckpt)
    logger.info(f"[+] Final checkpoint saved to: {final_ckpt}")

    logger.info("Evaluating final validation and held-out test metrics on final trained model...")
    final_val_loss, final_val_ppl = evaluate_dataset(model, val_loader, device)
    final_test_loss, final_test_ppl = evaluate_dataset(model, test_loader, device)

    # Evaluate best model as well
    best_model_path = campaign_ckpt_dir / "best_model.pt"
    if best_model_path.exists():
        logger.info(f"Evaluating best model from {best_model_path}...")
        best_ckpt_data = torch.load(best_model_path, map_location=device, weights_only=False)
        best_model = GPTTransformer(cfg).to(device)
        best_model = inject_custom_fused_mlp(best_model)
        best_model.load_state_dict(best_ckpt_data["model_state_dict"])
        best_model_val_loss, best_model_val_ppl = evaluate_dataset(best_model, val_loader, device)
        best_model_test_loss, best_model_test_ppl = evaluate_dataset(best_model, test_loader, device)
    else:
        best_model_val_loss, best_model_val_ppl = final_val_loss, final_val_ppl
        best_model_test_loss, best_model_test_ppl = final_test_loss, final_test_ppl

    logger.info("=" * 80)
    logger.info("TRAINING CAMPAIGN SUMMARY:")
    logger.info(f"  Mode:                       {mode.upper()}")
    logger.info(f"  Start Step -> Final Step:   {start_step} -> {global_step}")
    logger.info(f"  Start Tokens:               {start_tokens:,} tokens")
    logger.info(f"  Additional Tokens:          {cumulative_tokens - start_tokens:,} tokens")
    logger.info(f"  Final Cumulative Tokens:    {cumulative_tokens:,} tokens")
    logger.info(f"  Total Wall-Clock Time:      {total_training_time:.2f} s ({total_training_time / 60:.2f} min)")
    logger.info(f"  Overall Throughput:         {overall_throughput:,.1f} tokens/s")
    logger.info(f"  Validation Loss:            {initial_val_loss:.4f} -> {final_val_loss:.4f} (Best: {best_model_val_loss:.4f} at Step {best_val_step})")
    logger.info(f"  Validation Perplexity:      {initial_val_ppl:.2f} -> {final_val_ppl:.2f} (Best: {best_model_val_ppl:.2f})")
    logger.info(f"  Held-Out Test Loss:         {initial_test_loss:.4f} -> {final_test_loss:.4f} (Best: {best_model_test_loss:.4f})")
    logger.info(f"  Held-Out Test Perplexity:   {initial_test_ppl:.2f} -> {final_test_ppl:.2f} (Best: {best_model_test_ppl:.2f})")
    logger.info("=" * 80)

    # 8. Checkpoint Recovery Parity Test
    logger.info("Validating checkpoint recovery fidelity on final_model.pt...")
    restored_ckpt = torch.load(final_ckpt, map_location=device, weights_only=False)
    recovery_model = GPTTransformer(cfg).to(device)
    recovery_model = inject_custom_fused_mlp(recovery_model)
    recovery_model.load_state_dict(restored_ckpt["model_state_dict"])
    
    test_batch_x, _ = next(iter(val_loader))
    test_batch_x = test_batch_x.to(device)
    model.eval()
    recovery_model.eval()
    with torch.no_grad():
        out_orig, _ = model(test_batch_x)
        out_restored, _ = recovery_model(test_batch_x)
        max_diff = (out_orig - out_restored).abs().max().item()
    logger.info(f"Recovery Validation Max Logit Diff: {max_diff:.2e} (Passed: {max_diff < 1e-4})")

    # Serialize Results
    summary_data = {
        "milestone": "Phase 13 Milestone 6",
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": gpu_name,
            "vram_total_mb": 6143.5,
            "compute_capability": "8.6",
        },
        "token_accounting": {
            "start_step": start_step,
            "final_step": global_step,
            "steps_executed": global_step - start_step,
            "start_tokens": start_tokens,
            "additional_tokens": cumulative_tokens - start_tokens,
            "cumulative_tokens": cumulative_tokens,
            "target_cumulative_tokens": target_lifetime_tokens,
            "tokens_per_step": tokens_per_step,
        },
        "performance": {
            "total_wall_time_sec": round(total_training_time, 2),
            "total_wall_time_min": round(total_training_time / 60, 2),
            "tokens_per_sec": round(overall_throughput, 1),
            "peak_vram_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "peak_vram_res_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
        },
        "scientific_evaluation": {
            "initial_validation_loss": round(initial_val_loss, 4),
            "initial_validation_perplexity": round(initial_val_ppl, 2),
            "final_validation_loss": round(final_val_loss, 4),
            "final_validation_perplexity": round(final_val_ppl, 2),
            "best_validation_loss": round(best_model_val_loss, 4),
            "best_validation_perplexity": round(best_model_val_ppl, 2),
            "best_validation_step": best_val_step,
            "initial_test_loss": round(initial_test_loss, 4),
            "initial_test_perplexity": round(initial_test_ppl, 2),
            "final_test_loss": round(final_test_loss, 4),
            "final_test_perplexity": round(final_test_ppl, 2),
            "best_test_loss": round(best_model_test_loss, 4),
            "best_test_perplexity": round(best_model_test_ppl, 2),
        },
        "recovery_fidelity": {
            "max_logit_diff": max_diff,
            "status": "PASSED" if max_diff < 1e-4 else "FAILED",
        },
        "checkpoints": checkpoint_history,
        "telemetry_sample": step_telemetry[-50:] if len(step_telemetry) > 50 else step_telemetry,
    }

    prefix = "phase13_milestone6_diagnostic" if mode == "diagnostic" else "phase13_milestone6_50m_campaign"
    out_summary_file = results_dir / f"{prefix}_summary.json"
    with open(out_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"[+] Saved execution summary to: {out_summary_file}")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 13 Milestone 6 Pretraining Campaign")
    parser.add_argument("--mode", choices=["diagnostic", "full"], default="diagnostic", help="Execution mode")
    args = parser.parse_args()
    run_campaign(mode=args.mode)
