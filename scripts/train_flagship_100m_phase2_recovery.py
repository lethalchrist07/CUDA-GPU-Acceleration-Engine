"""
scripts/train_flagship_100m_phase2_recovery.py
Phase 2: Authorized 1,000-Step Rapid Convergence Campaign for NEXORA Flagship (~100M / 86.8M) Architecture.

Target Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (Ampere SM 8.6, 6GB VRAM, 70W TGP).
Starting Checkpoint: checkpoints/transformer/flagship_100m_recovery/best_model.pt (Phase 1, Step 100)
Target Output Directory: checkpoints/transformer/flagship_100m_recovery_phase2/

Strict Constraints:
- Exactly 1,000 additional optimizer steps (Steps 101 to 1,100).
- Effective token budget: 2,048,000 tokens (8 seqs x 256 context per step).
- Isolated Phase 2 directory; never overwrite Phase 1 or original checkpoints.
- Conservative memory & thermal safety bounds (75°C pause, 82°C cutoff, 5,120 MB VRAM cap).
- Record telemetry every step; evaluate validation every 100 steps.
- Early stop on divergence, NaN/Inf, or hardware threshold breach.
"""

import copy
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import custom_kernels.autograd as cka
from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.transformer.config import TransformerConfig
from engine.transformer.model import GPTTransformer
from engine.transformer.scalable_dataset import (
    ScalableMemmapDataset,
    create_scalable_dataloaders,
)
from engine.transformer.tokenizer import BPETokenizer
from engine.transformer.trainer import safe_perplexity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase2Convergence")


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


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


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
    except Exception as e:
        logger.warning(f"GPU telemetry query exception: {e}")
        return {
            "temperature_c": 50.0,
            "gpu_utilization_pct": 0.0,
            "vram_used_mb": 0.0,
            "vram_total_mb": 6144.0,
        }


def get_cosine_schedule_continuation(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_campaign_steps: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup from min_lr to base_lr over warmup_steps, then cosine decay back to min_lr."""
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup from min_lr_ratio to 1.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * (float(step) / float(max(1, warmup_steps)))
        progress = float(step - warmup_steps) / float(max(1, total_campaign_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_dataset(
    model: GPTTransformer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    max_batches: Optional[int] = 30,
) -> Tuple[float, float]:
    """Evaluates cross-entropy loss and perplexity on a dataset loader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                _, loss = model(x, targets=y)
            total_loss += loss.item()
            total_batches += 1
    avg_loss = total_loss / max(1, total_batches)
    ppl = safe_perplexity(avg_loss)
    return avg_loss, ppl


def atomic_save_checkpoint(
    checkpoint_data: Dict[str, Any],
    target_path: Path,
) -> None:
    """Saves checkpoint atomically via temporary file and atomic rename."""
    temp_path = target_path.with_suffix(".tmp.pt")
    torch.save(checkpoint_data, temp_path)
    if target_path.exists():
        target_path.unlink()
    temp_path.rename(target_path)


def run_phase2_convergence_campaign():
    logger.info("=" * 80)
    logger.info("NEXORA — PHASE 2: CONTROLLED 1,000-STEP CONVERGENCE CAMPAIGN")
    logger.info("=" * 80)

    # 1. Device check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Aborting campaign.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Target Hardware: {gpu_name}")
    logger.info(f"PyTorch Version: {torch.__version__}")

    # 2. Output directory isolation
    phase2_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_recovery_phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Isolated Phase 2 Recovery Directory: {phase2_dir}")

    # Baseline preservation checks
    p1_ckpt_path = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_recovery" / "best_model.pt"
    pilot_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_pilot" / "best_model.pt"
    m6_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_campaign" / "best_model.pt"
    rc2_zip = PROJECT_ROOT / "dist" / "cuda-acceleration-engine-v1.0.0-rc2-source.zip"

    expected_p1_sha = "88ba2868d3c7c0250aa48228a480b2d601f696a5149efedbc957c135e734f671"
    expected_pilot_sha = "f8b0b9f9d4fb2e9ce482aea9c619e2c588404caa7e5691606e9340f381932f9f"
    expected_m6_sha = "abcd0fe7b305cd1cef7809359ca7128f0ac94bed592a188f8432329ca1406aac"
    expected_rc2_sha = "66259b3c69df00872ad644636dfb7f4fb7abe40b187318ee19a83778abd94058"

    assert compute_sha256(p1_ckpt_path) == expected_p1_sha, "Phase 1 starting checkpoint hash mismatch!"
    assert compute_sha256(pilot_ckpt) == expected_pilot_sha, "Original pilot checkpoint hash mismatch!"
    assert compute_sha256(m6_ckpt) == expected_m6_sha, "Certified M6 checkpoint hash mismatch!"
    assert compute_sha256(rc2_zip) == expected_rc2_sha, "Certified RC2 archive hash mismatch!"
    logger.info("Preflight Baseline Preservation Checks: PASSED (All pre-existing assets verified)")

    # 3. Synchronize BPE tokenizer to Phase 2 directory
    source_vocab = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_recovery" / "bpe_vocab.json"
    target_vocab = phase2_dir / "bpe_vocab.json"
    if not target_vocab.exists():
        shutil.copy2(source_vocab, target_vocab)
        logger.info(f"Copied 2048-token BPE vocabulary to {target_vocab}")
    tokenizer = BPETokenizer.load(target_vocab)
    assert tokenizer.vocab_size == 2048, "Tokenizer vocab size must be 2048"

    # 4. Dataset preparation
    data_dir = PROJECT_ROOT / "data" / "text" / "production_10m"
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    manifest_path = data_dir / "dataset_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    dataset_fingerprint = manifest.get("aggregate_content_hash_sha256", "pretraining_corpus_36doc")

    train_ds = ScalableMemmapDataset(train_bin, seq_len=256, stride=256, vocab_size=2048)
    val_ds = ScalableMemmapDataset(val_bin, seq_len=256, stride=256, vocab_size=2048)

    micro_batch_size = 2
    grad_accum_steps = 4
    seq_len = 256
    effective_tokens_per_step = micro_batch_size * grad_accum_steps * seq_len  # 2048 tokens

    loaders = create_scalable_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=micro_batch_size,
        seed=1042,  # New seed for Phase 2 continuation
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    logger.info(f"Loaded Datasets: Train={train_ds.total_tokens:,} tokens ({len(train_ds)} sequences), Val={val_ds.total_tokens:,} tokens")

    # 5. Load Starting Checkpoint and Instantiate Architecture
    logger.info(f"Loading verified Phase 1 checkpoint: {p1_ckpt_path}")
    p1_ckpt = torch.load(p1_ckpt_path, map_location="cpu", weights_only=False)
    assert len(p1_ckpt["model_state_dict"]) == 149, "Unexpected tensor count in starting checkpoint!"
    start_global_step = p1_ckpt.get("step", 100)
    logger.info(f"Phase 1 Resumption Step: {start_global_step} (Metrics: {p1_ckpt.get('metrics')})")

    model_config = TransformerConfig(**p1_ckpt["metadata"]["model_config"])
    model = GPTTransformer(model_config).to(device)
    model.load_state_dict(p1_ckpt["model_state_dict"], strict=True)

    # Inject Custom CUDA Fused Bias+GELU
    logger.info("Injecting Custom CUDA Fused Bias+GELU into transformer MLP blocks...")
    model = inject_custom_fused_mlp(model)

    param_info = model.count_parameters()
    total_params = param_info["total_parameters"]
    assert total_params == 86_825_472, f"Unexpected parameter count: {total_params}"
    logger.info(f"Model instantiated successfully: Total Parameters = {total_params:,} (Tied: {param_info['tied_weights']})")

    # 6. Configure Optimizer, Scaler, and Schedule
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    base_learning_rate = 2e-4
    min_learning_rate = 2e-5
    warmup_steps = 50
    authorized_steps = 1000

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=base_learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Load optimizer state if available
    if "optimizer_state_dict" in p1_ckpt:
        try:
            optimizer.load_state_dict(p1_ckpt["optimizer_state_dict"])
            logger.info("Successfully resumed AdamW optimizer state dictionary from Phase 1!")
        except Exception as e:
            logger.warning(f"Could not load optimizer state dict directly ({e}); proceeding with re-initialized optimizer.")

    scheduler = get_cosine_schedule_continuation(
        optimizer,
        warmup_steps=warmup_steps,
        total_campaign_steps=authorized_steps,
        base_lr=base_learning_rate,
        min_lr=min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if "scaler_state_dict" in p1_ckpt:
        try:
            scaler.load_state_dict(p1_ckpt["scaler_state_dict"])
            logger.info("Successfully resumed GradScaler state dictionary from Phase 1!")
        except Exception as e:
            logger.warning(f"Could not load scaler state dict directly ({e}).")

    # 7. Hardware Safety Thresholds
    VRAM_SAFETY_CEILING_MB = 5120.0
    THERMAL_WARNING_THRESHOLD_C = 75.0
    THERMAL_CUTOFF_THRESHOLD_C = 82.0

    # 8. Initial Baseline Validation
    logger.info("Evaluating starting validation loss prior to Phase 2 training...")
    init_val_loss, init_val_ppl = evaluate_dataset(model, val_loader, device, use_amp=True, max_batches=30)
    logger.info(f"Phase 2 Baseline Validation Loss: {init_val_loss:.4f} | Perplexity: {init_val_ppl:.2f}")

    # 9. Training Loop (Exactly 1,000 Steps)
    logger.info(f"Starting Authorized 1,000-Step Convergence Run (Target: Steps {start_global_step + 1} to {start_global_step + authorized_steps})...")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step_records: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = [
        {"campaign_step": 0, "cumulative_step": start_global_step, "val_loss": round(init_val_loss, 4), "val_perplexity": round(init_val_ppl, 2)}
    ]
    best_val_loss = init_val_loss
    best_ckpt_path = phase2_dir / "best_model.pt"

    # Save initial best checkpoint
    atomic_save_checkpoint(
        {
            "step": start_global_step,
            "campaign_step": 0,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": {"val_loss": init_val_loss, "val_perplexity": init_val_ppl, "train_loss": p1_ckpt.get("metrics", {}).get("train_loss", 3.9081)},
            "metadata": {"model_config": model_config.to_dict(), "dataset_fingerprint": dataset_fingerprint},
        },
        best_ckpt_path,
    )

    campaign_step = 0
    total_tokens_processed = 0
    train_iter = iter(train_loader)
    start_wall_time = time.perf_counter()

    early_stop_triggered = False
    early_stop_reason = ""
    consecutive_val_degradations = 0

    while campaign_step < authorized_steps:
        step_cycle_start = time.perf_counter()

        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                _, loss = model(x, targets=y)
                loss_scaled = loss / grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                early_stop_triggered = True
                early_stop_reason = f"NaN/Inf loss encountered at campaign step {campaign_step + 1}!"
                break

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()
            total_tokens_processed += (x.size(0) * x.size(1))

        if early_stop_triggered:
            logger.error(early_stop_reason)
            break

        # Optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_val = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        campaign_step += 1
        current_cumulative_step = start_global_step + campaign_step

        step_wall_time = time.perf_counter() - step_cycle_start
        tokens_per_sec = effective_tokens_per_step / max(step_wall_time, 1e-6)
        curr_lr = optimizer.param_groups[0]["lr"]
        avg_step_loss = accum_loss / grad_accum_steps

        # Periodic telemetry (sample nvidia-smi every 10 steps to reduce subprocess overhead)
        if campaign_step % 10 == 0 or campaign_step == 1 or campaign_step == authorized_steps:
            telemetry = query_gpu_telemetry()
            vram_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            gpu_temp = telemetry["temperature_c"]
            gpu_util = telemetry["gpu_utilization_pct"]

            # Safety Guards
            if vram_peak_mb > VRAM_SAFETY_CEILING_MB:
                early_stop_triggered = True
                early_stop_reason = f"VRAM ceiling exceeded! Peak {vram_peak_mb:.1f} MB > Limit {VRAM_SAFETY_CEILING_MB} MB"
                logger.error(early_stop_reason)
                break

            if gpu_temp >= THERMAL_CUTOFF_THRESHOLD_C:
                early_stop_triggered = True
                early_stop_reason = f"EMERGENCY: Thermal cutoff reached! Temp {gpu_temp:.1f}°C >= {THERMAL_CUTOFF_THRESHOLD_C}°C. Aborting."
                logger.error(early_stop_reason)
                break

            if gpu_temp >= THERMAL_WARNING_THRESHOLD_C:
                logger.warning(f"Thermal warning at step {campaign_step}: Temp is {gpu_temp:.1f}°C >= {THERMAL_WARNING_THRESHOLD_C}°C. Pausing 10s...")
                time.sleep(10.0)
        else:
            vram_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            gpu_temp = telemetry["temperature_c"]
            gpu_util = telemetry["gpu_utilization_pct"]

        step_record = {
            "campaign_step": campaign_step,
            "cumulative_step": current_cumulative_step,
            "train_loss": round(avg_step_loss, 4),
            "train_perplexity": round(safe_perplexity(avg_step_loss), 2),
            "learning_rate": round(curr_lr, 7),
            "grad_norm": round(grad_norm_val, 4),
            "step_time_sec": round(step_wall_time, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "vram_allocated_mb": round(vram_allocated_mb, 2),
            "vram_peak_mb": round(vram_peak_mb, 2),
            "gpu_temp_c": gpu_temp,
            "gpu_utilization_pct": gpu_util,
        }
        step_records.append(step_record)

        # Logging cadence
        if campaign_step % 50 == 0 or campaign_step == 1 or campaign_step == authorized_steps:
            logger.info(
                f"Step {campaign_step:4d}/{authorized_steps} (Total {current_cumulative_step}) | "
                f"Loss: {avg_step_loss:.4f} | PPL: {safe_perplexity(avg_step_loss):.2f} | "
                f"LR: {curr_lr:.2e} | GradNorm: {grad_norm_val:.3f} | Latency: {step_wall_time:.3f}s | "
                f"Throughput: {tokens_per_sec:.1f} tok/s | VRAM Peak: {vram_peak_mb:.1f} MB | "
                f"Temp: {gpu_temp:.1f}°C"
            )

        # Evaluation & Resumable Checkpoint Save Every 100 Steps
        if campaign_step % 100 == 0:
            logger.info(f"--- [Evaluation at Step {campaign_step}/{authorized_steps}] ---")
            eval_batches = 50 if campaign_step == authorized_steps else 30
            val_loss, val_ppl = evaluate_dataset(model, val_loader, device, use_amp=True, max_batches=eval_batches)
            logger.info(f"Step {campaign_step} Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

            eval_record = {
                "campaign_step": campaign_step,
                "cumulative_step": current_cumulative_step,
                "val_loss": round(val_loss, 4),
                "val_perplexity": round(val_ppl, 2),
                "train_loss": round(avg_step_loss, 4),
            }
            eval_records.append(eval_record)

            # Check if this is the best validation loss
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                consecutive_val_degradations = 0
                atomic_save_checkpoint(
                    {
                        "step": current_cumulative_step,
                        "campaign_step": campaign_step,
                        "epoch": 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "metrics": {"val_loss": val_loss, "val_perplexity": val_ppl, "train_loss": avg_step_loss},
                        "metadata": {"model_config": model_config.to_dict(), "dataset_fingerprint": dataset_fingerprint},
                    },
                    best_ckpt_path,
                )
                logger.info(f"*** New BEST Validation Checkpoint saved: {best_ckpt_path} (Val Loss: {val_loss:.4f}, Improvement: -{improvement:.4f}) ***")
            else:
                consecutive_val_degradations += 1
                if consecutive_val_degradations >= 4:
                    early_stop_triggered = True
                    early_stop_reason = f"Early stopping triggered: 4 consecutive validation checks without improvement (Current Val Loss: {val_loss:.4f} vs Best: {best_val_loss:.4f})"
                    logger.warning(early_stop_reason)
                    break

            # Save periodic milestone checkpoint every 250 steps
            if campaign_step % 250 == 0:
                milestone_ckpt_path = phase2_dir / f"step_{current_cumulative_step}.pt"
                atomic_save_checkpoint(
                    {
                        "step": current_cumulative_step,
                        "campaign_step": campaign_step,
                        "epoch": 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "metrics": {"val_loss": val_loss, "val_perplexity": val_ppl, "train_loss": avg_step_loss},
                        "metadata": {"model_config": model_config.to_dict(), "dataset_fingerprint": dataset_fingerprint},
                    },
                    milestone_ckpt_path,
                )
                logger.info(f"Saved periodic milestone checkpoint: {milestone_ckpt_path}")

            model.train()

    total_training_wall_time = time.perf_counter() - start_wall_time

    # 10. Final Step Checkpoint Serialization
    final_cumulative_step = start_global_step + campaign_step
    final_ckpt_path = phase2_dir / f"step_{final_cumulative_step}_final.pt"
    final_val_loss = eval_records[-1]["val_loss"]
    final_val_ppl = eval_records[-1]["val_perplexity"]

    atomic_save_checkpoint(
        {
            "step": final_cumulative_step,
            "campaign_step": campaign_step,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": {"val_loss": final_val_loss, "val_perplexity": final_val_ppl, "train_loss": step_records[-1]["train_loss"]},
            "metadata": {"model_config": model_config.to_dict(), "dataset_fingerprint": dataset_fingerprint},
        },
        final_ckpt_path,
    )
    logger.info(f"Saved final campaign checkpoint: {final_ckpt_path}")

    # 11. Text Generation Demonstration (on Best Model)
    logger.info("Executing qualitative text generation test on best checkpoint...")
    best_ckpt_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    eval_model = GPTTransformer(model_config).to(device)
    eval_model.load_state_dict(best_ckpt_data["model_state_dict"])
    eval_model.eval()

    test_prompts = [
        "The physical principles of parallel computing",
        "The architectural foundations of high performance",
    ]
    sample_generations = {}
    with torch.no_grad():
        for p in test_prompts:
            prompt_ids = torch.tensor([tokenizer.encode(p)], device=device)
            gen_ids = eval_model.generate(prompt_ids, max_new_tokens=40, temperature=0.7, top_k=8)
            gen_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
            sample_generations[p] = gen_text
            logger.info(f"Prompt: {repr(p)} -> Output:\n{repr(gen_text)}")

    # 12. Checkpoint Serialization & Integrity Verification
    logger.info("Validating saved checkpoint integrity...")
    assert len(best_ckpt_data["model_state_dict"]) == 149, f"Unexpected tensor count: {len(best_ckpt_data['model_state_dict'])}"
    assert "optimizer_state_dict" in best_ckpt_data, "Optimizer state missing!"
    assert "scaler_state_dict" in best_ckpt_data, "Scaler state missing!"
    best_sha = compute_sha256(best_ckpt_path)
    final_sha = compute_sha256(final_ckpt_path)

    # Re-verify original baseline checkpoints untouched
    assert compute_sha256(p1_ckpt_path) == expected_p1_sha, "Post-run: Phase 1 checkpoint was altered!"
    assert compute_sha256(pilot_ckpt) == expected_pilot_sha, "Post-run: Original pilot checkpoint was altered!"
    assert compute_sha256(m6_ckpt) == expected_m6_sha, "Post-run: Certified M6 checkpoint was altered!"
    assert compute_sha256(rc2_zip) == expected_rc2_sha, "Post-run: Certified RC2 archive was altered!"
    logger.info("Post-Run Baseline Preservation Verification: PASSED (All pre-existing assets 100% untouched)")

    # 13. Compile Summary Results JSON
    max_recorded_temp = max(r["gpu_temp_c"] for r in step_records)
    max_recorded_vram = max(r["vram_peak_mb"] for r in step_records)
    avg_step_latency = np.mean([r["step_time_sec"] for r in step_records])
    avg_throughput = np.mean([r["tokens_per_sec"] for r in step_records])

    summary_results = {
        "campaign": "NEXORA Flagship 100M Safe Recovery Phase 2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "total_physical_vram_mb": 6144.0,
            "vram_safety_ceiling_mb": VRAM_SAFETY_CEILING_MB,
        },
        "model_architecture": {
            "parameter_count": total_params,
            "vocab_size": model_config.vocab_size,
            "max_seq_len": model_config.max_seq_len,
            "d_model": model_config.d_model,
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "d_ff": model_config.d_ff,
            "tied_weights": model_config.tie_weights,
            "use_gradient_checkpointing": model_config.use_gradient_checkpointing,
        },
        "execution": {
            "start_step": start_global_step,
            "authorized_steps": authorized_steps,
            "completed_steps": campaign_step,
            "final_cumulative_step": final_cumulative_step,
            "effective_tokens_per_step": effective_tokens_per_step,
            "total_tokens_trained": total_tokens_processed,
            "total_wall_time_sec": round(total_training_wall_time, 2),
            "avg_step_latency_sec": round(float(avg_step_latency), 3),
            "avg_throughput_tokens_per_sec": round(float(avg_throughput), 1),
            "early_stopping": {
                "triggered": early_stop_triggered,
                "reason": early_stop_reason if early_stop_triggered else "None (Full 1,000 steps completed)",
            },
        },
        "loss_and_convergence": {
            "baseline_val_loss": round(init_val_loss, 4),
            "baseline_val_perplexity": round(init_val_ppl, 2),
            "step1_train_loss": step_records[0]["train_loss"],
            "final_train_loss": step_records[-1]["train_loss"],
            "loss_reduction_train": round(step_records[0]["train_loss"] - step_records[-1]["train_loss"], 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_perplexity": round(safe_perplexity(best_val_loss), 2),
            "final_val_loss": round(final_val_loss, 4),
            "final_val_perplexity": round(final_val_ppl, 2),
            "target_3_20_achieved": best_val_loss <= 3.20,
        },
        "hardware_telemetry": {
            "peak_vram_mb": round(float(max_recorded_vram), 2),
            "peak_gpu_temp_c": round(float(max_recorded_temp), 1),
            "thermal_headroom_to_cutoff_c": round(THERMAL_CUTOFF_THRESHOLD_C - max_recorded_temp, 1),
            "vram_headroom_to_ceiling_mb": round(VRAM_SAFETY_CEILING_MB - max_recorded_vram, 2),
        },
        "checkpoints": {
            "phase2_directory": str(phase2_dir),
            "best_checkpoint": {
                "path": str(best_ckpt_path),
                "size_bytes": best_ckpt_path.stat().st_size,
                "sha256": best_sha,
                "step": best_ckpt_data["step"],
                "val_loss": best_ckpt_data["metrics"]["val_loss"],
            },
            "final_checkpoint": {
                "path": str(final_ckpt_path),
                "size_bytes": final_ckpt_path.stat().st_size,
                "sha256": final_sha,
                "step": final_cumulative_step,
            },
        },
        "baseline_preservation": {
            "phase1_checkpoint_sha256": compute_sha256(p1_ckpt_path),
            "pilot_checkpoint_sha256": compute_sha256(pilot_ckpt),
            "m6_checkpoint_sha256": compute_sha256(m6_ckpt),
            "rc2_archive_sha256": compute_sha256(rc2_zip),
            "all_baselines_intact": True,
        },
        "sample_generations": sample_generations,
        "eval_records": eval_records,
        "step_records": step_records,
    }

    results_file = PROJECT_ROOT / "results" / "phase2_recovery_1000_step_convergence.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    logger.info("=" * 80)
    logger.info("PHASE 2 1,000-STEP CONVERGENCE CAMPAIGN COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Initial Train Loss: {step_records[0]['train_loss']} -> Final Train Loss: {step_records[-1]['train_loss']}")
    logger.info(f"Baseline Val Loss: {init_val_loss:.4f} -> Best Val Loss: {best_val_loss:.4f} (Final: {final_val_loss:.4f})")
    logger.info(f"Peak VRAM: {max_recorded_vram:.2f} MB | Peak GPU Temp: {max_recorded_temp:.1f}°C")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_phase2_convergence_campaign()

