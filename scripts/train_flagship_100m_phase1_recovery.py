"""
scripts/train_flagship_100m_phase1_recovery.py
Phase 1: Authorized 100-Step Safe Recovery Smoke Test for NEXORA Flagship (~100M / 86.8M) Architecture.

Target Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (Ampere SM 8.6, 6GB VRAM, 70W TGP).
Strict Constraints:
- Exactly 100 optimizer steps.
- Isolated output directory: checkpoints/transformer/flagship_100m_recovery/
- Never overwrite existing checkpoints or certified release archives.
- Conservative memory & thermal safety bounds (75°C pause, 82°C cutoff, 5,120 MB VRAM cap).
- Record comprehensive telemetry (loss, step latency, throughput, peak VRAM, GPU temp).
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
from engine.transformer.trainer import safe_perplexity

# Setup clean stdout logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase1SmokeTest")


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
    except Exception as e:
        logger.warning(f"GPU telemetry query exception: {e}")
        return {
            "temperature_c": 50.0,
            "gpu_utilization_pct": 0.0,
            "vram_used_mb": 0.0,
            "vram_total_mb": 6144.0,
        }


def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_dataset(
    model: GPTTransformer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    max_batches: Optional[int] = 50,
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


def run_phase1_smoke_test():
    logger.info("=" * 80)
    logger.info("NEXORA — PHASE 1: AUTHORIZED 100-STEP SAFE RECOVERY SMOKE TEST")
    logger.info("=" * 80)

    # 1. Device check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Aborting smoke test.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Target Hardware: {gpu_name}")
    logger.info(f"PyTorch Version: {torch.__version__}")

    # 2. Output directory isolation
    recovery_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Isolated Recovery Directory: {recovery_dir}")

    # Baseline preservation checks
    pilot_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_pilot" / "best_model.pt"
    m6_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_campaign" / "best_model.pt"
    rc2_zip = PROJECT_ROOT / "dist" / "cuda-acceleration-engine-v1.0.0-rc2-source.zip"

    expected_pilot_sha = "f8b0b9f9d4fb2e9ce482aea9c619e2c588404caa7e5691606e9340f381932f9f"
    expected_m6_sha = "abcd0fe7b305cd1cef7809359ca7128f0ac94bed592a188f8432329ca1406aac"
    expected_rc2_sha = "66259b3c69df00872ad644636dfb7f4fb7abe40b187318ee19a83778abd94058"

    assert compute_sha256(pilot_ckpt).lower() == expected_pilot_sha, "Original pilot checkpoint hash mismatch!"
    assert compute_sha256(m6_ckpt).lower() == expected_m6_sha, "Certified M6 checkpoint hash mismatch!"
    assert compute_sha256(rc2_zip).lower() == expected_rc2_sha, "Certified RC2 archive hash mismatch!"
    logger.info("Preflight Baseline Preservation Checks: PASSED (All original assets 100% verified)")

    # 3. Copy BPE vocab to recovery directory for self-containment
    source_vocab = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_campaign" / "bpe_vocab.json"
    target_vocab = recovery_dir / "bpe_vocab.json"
    if not target_vocab.exists():
        shutil.copy2(source_vocab, target_vocab)
        logger.info(f"Copied 2048-token BPE vocabulary to {target_vocab}")

    # 4. Dataset preparation
    data_dir = PROJECT_ROOT / "data" / "text" / "production_10m"
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    manifest_path = data_dir / "dataset_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    dataset_fingerprint = manifest.get("aggregate_content_hash_sha256", "pretraining_corpus_36doc")

    logger.info("Initializing ScalableMemmapDataset...")
    train_ds = ScalableMemmapDataset(train_bin, seq_len=256, stride=256, vocab_size=2048)
    val_ds = ScalableMemmapDataset(val_bin, seq_len=256, stride=256, vocab_size=2048)

    # Conservative micro-batch = 2, gradient accumulation = 4 -> Effective batch size = 8 sequences (2048 tokens/step)
    micro_batch_size = 2
    grad_accum_steps = 4
    seq_len = 256
    effective_tokens_per_step = micro_batch_size * grad_accum_steps * seq_len  # 2048 tokens

    loaders = create_scalable_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=micro_batch_size,
        seed=42,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    logger.info(f"Loaded Datasets: Train={train_ds.total_tokens:,} tokens ({len(train_ds)} sequences), Val={val_ds.total_tokens:,} tokens")

    # 5. Instantiate Architecture (86,825,472 parameters)
    logger.info("Instantiating clean 86.8M Transformer architecture...")
    model_config = TransformerConfig(
        vocab_size=2048,
        max_seq_len=256,
        d_model=768,
        n_layer=12,
        n_head=12,
        d_ff=3072,
        dropout=0.1,
        attn_dropout=0.1,
        bias=True,
        tie_weights=True,
        use_gradient_checkpointing=True,
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = GPTTransformer(model_config).to(device)

    # Inject Custom CUDA Fused Bias+GELU
    logger.info("Injecting Custom CUDA Fused Bias+GELU into transformer MLP blocks...")
    model = inject_custom_fused_mlp(model)

    param_info = model.count_parameters()
    total_params = param_info["total_parameters"]
    logger.info(f"Model instantiated successfully: Total Parameters = {total_params:,} (Tied: {param_info['tied_weights']})")
    assert total_params == 86_825_472, f"Unexpected parameter count: {total_params}"

    # 6. Configure Optimizer, Scaler, and Schedule
    # Weight decay separation: decay for 2D+ weights, zero decay for biases and norm weights
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    learning_rate = 2e-4
    min_learning_rate = 2e-5
    warmup_steps = 20
    max_steps = 100

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = get_cosine_schedule(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_learning_rate / learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # 7. Hardware Safety Thresholds
    VRAM_SAFETY_CEILING_MB = 5120.0
    THERMAL_WARNING_THRESHOLD_C = 75.0
    THERMAL_CUTOFF_THRESHOLD_C = 82.0

    # 8. Initial Baseline Validation
    logger.info("Performing pre-training baseline evaluation on validation set...")
    init_val_loss, init_val_ppl = evaluate_dataset(model, val_loader, device, use_amp=True, max_batches=30)
    logger.info(f"Baseline (Step 0) Validation Loss: {init_val_loss:.4f} | Perplexity: {init_val_ppl:.2f}")

    # 9. Training Loop (Exactly 100 Steps)
    logger.info(f"Starting Authorized 100-Step Recovery Run (Effective Tokens/Step: {effective_tokens_per_step:,})...")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step_records: List[Dict[str, Any]] = []
    global_step = 0
    microbatch_in_accum = 0
    running_loss = 0.0
    best_val_loss = init_val_loss
    total_tokens_processed = 0

    train_iter = iter(train_loader)
    start_wall_time = time.perf_counter()

    while global_step < max_steps:
        step_cycle_start = time.perf_counter()

        # Accumulate gradients across grad_accum_steps microbatches
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
                raise FloatingPointError(f"NaN or Inf loss encountered at step {global_step + 1}!")

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()
            total_tokens_processed += (x.size(0) * x.size(1))

        # Optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_val = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step += 1
        step_wall_time = time.perf_counter() - step_cycle_start
        tokens_per_sec = effective_tokens_per_step / max(step_wall_time, 1e-6)
        curr_lr = optimizer.param_groups[0]["lr"]
        avg_step_loss = accum_loss / grad_accum_steps

        # GPU Hardware Telemetry Check
        telemetry = query_gpu_telemetry()
        vram_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        gpu_temp = telemetry["temperature_c"]
        gpu_util = telemetry["gpu_utilization_pct"]

        # Hardware Safety Guards
        if vram_peak_mb > VRAM_SAFETY_CEILING_MB:
            raise RuntimeError(f"VRAM ceiling exceeded! Peak {vram_peak_mb:.1f} MB > Limit {VRAM_SAFETY_CEILING_MB} MB")

        if gpu_temp >= THERMAL_CUTOFF_THRESHOLD_C:
            raise RuntimeError(f"EMERGENCY: Thermal cutoff reached! Temp {gpu_temp:.1f}°C >= {THERMAL_CUTOFF_THRESHOLD_C}°C. Aborting.")

        if gpu_temp >= THERMAL_WARNING_THRESHOLD_C:
            logger.warning(f"Thermal warning at step {global_step}: Temp is {gpu_temp:.1f}°C >= {THERMAL_WARNING_THRESHOLD_C}°C. Pausing 10s...")
            time.sleep(10.0)

        step_record = {
            "step": global_step,
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

        if global_step % 10 == 0 or global_step == 1 or global_step == max_steps:
            logger.info(
                f"Step {global_step:3d}/{max_steps} | Loss: {avg_step_loss:.4f} | "
                f"PPL: {safe_perplexity(avg_step_loss):.2f} | LR: {curr_lr:.2e} | "
                f"GradNorm: {grad_norm_val:.3f} | Latency: {step_wall_time:.3f}s | "
                f"Throughput: {tokens_per_sec:.1f} tok/s | VRAM: {vram_allocated_mb:.1f} MB (Peak: {vram_peak_mb:.1f} MB) | "
                f"Temp: {gpu_temp:.1f}°C"
            )

        # Mid-point evaluation at step 50
        if global_step == 50:
            logger.info("Executing Step 50 mid-point validation...")
            val_loss_50, val_ppl_50 = evaluate_dataset(model, val_loader, device, use_amp=True, max_batches=30)
            logger.info(f"Step 50 Validation Loss: {val_loss_50:.4f} | Perplexity: {val_ppl_50:.2f}")
            if val_loss_50 < best_val_loss:
                best_val_loss = val_loss_50
                best_ckpt_path = recovery_dir / "best_model.pt"
                atomic_save_checkpoint(
                    {
                        "step": global_step,
                        "epoch": 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "metrics": {"val_loss": val_loss_50, "val_perplexity": val_ppl_50, "train_loss": avg_step_loss},
                        "metadata": {"model_config": model_config.to_dict(), "dataset_fingerprint": dataset_fingerprint},
                    },
                    best_ckpt_path,
                )
                logger.info(f"Saved new best checkpoint at Step 50: {best_ckpt_path}")
            model.train()

    total_training_wall_time = time.perf_counter() - start_wall_time

    # 10. Step 100 Final Evaluation
    logger.info("Executing Step 100 final validation...")
    final_val_loss, final_val_ppl = evaluate_dataset(model, val_loader, device, use_amp=True, max_batches=50)
    logger.info(f"Final Validation Loss (Step 100): {final_val_loss:.4f} | Perplexity: {final_val_ppl:.2f}")

    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_ckpt_path = recovery_dir / "best_model.pt"
        atomic_save_checkpoint(
            {
                "step": global_step,
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "metrics": {"val_loss": final_val_loss, "val_perplexity": final_val_ppl, "train_loss": avg_step_loss},
                "metadata": {"model_config": model_config.to_dict(), "dataset_fingerprint": dataset_fingerprint},
            },
            best_ckpt_path,
        )
        logger.info(f"Saved final best checkpoint: {best_ckpt_path}")

    # Always save final step checkpoint
    final_ckpt_path = recovery_dir / "step_100_final.pt"
    atomic_save_checkpoint(
        {
            "step": global_step,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": {"val_loss": final_val_loss, "val_perplexity": final_val_ppl, "train_loss": avg_step_loss},
            "metadata": {"model_config": model_config.to_dict(), "dataset_fingerprint": dataset_fingerprint},
        },
        final_ckpt_path,
    )
    logger.info(f"Saved final checkpoint: {final_ckpt_path}")

    # 11. Text Generation Demonstration
    logger.info("Executing text generation test with BPE Tokenizer...")
    from engine.transformer.tokenizer import BPETokenizer
    tokenizer = BPETokenizer.load(target_vocab)
    test_prompt = "The physical principles of parallel computing"
    prompt_ids = torch.tensor([tokenizer.encode(test_prompt)], device=device)
    gen_ids = model.generate(prompt_ids, max_new_tokens=35, temperature=0.7, top_k=8)
    generated_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    logger.info(f"Generation Output:\n{repr(generated_text)}")

    # 12. Checkpoint Serialization & Integrity Verification
    logger.info("Validating saved checkpoint integrity...")
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    assert len(best_ckpt["model_state_dict"]) == 149, f"Unexpected tensor count: {len(best_ckpt['model_state_dict'])}"
    assert "optimizer_state_dict" in best_ckpt, "Optimizer state missing!"
    assert "scaler_state_dict" in best_ckpt, "Scaler state missing!"
    best_sha = compute_sha256(best_ckpt_path)
    final_sha = compute_sha256(final_ckpt_path)

    # Re-verify original baseline checkpoints untouched
    assert compute_sha256(pilot_ckpt).lower() == expected_pilot_sha, "Post-run: Original pilot checkpoint was altered!"
    assert compute_sha256(m6_ckpt).lower() == expected_m6_sha, "Post-run: Certified M6 checkpoint was altered!"
    assert compute_sha256(rc2_zip).lower() == expected_rc2_sha, "Post-run: Certified RC2 archive was altered!"
    logger.info("Post-Run Baseline Preservation Verification: PASSED (All baseline assets 100% untouched)")

    # 13. Compile Summary Results JSON
    final_telemetry = query_gpu_telemetry()
    max_recorded_temp = max(r["gpu_temp_c"] for r in step_records)
    max_recorded_vram = max(r["vram_peak_mb"] for r in step_records)
    avg_step_latency = np.mean([r["step_time_sec"] for r in step_records])
    avg_throughput = np.mean([r["tokens_per_sec"] for r in step_records])

    summary_results = {
        "campaign": "NEXORA GPT-100M Safe Recovery Phase 1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "total_physical_vram_mb": telemetry["vram_total_mb"],
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
            "authorized_steps": max_steps,
            "completed_steps": global_step,
            "effective_tokens_per_step": effective_tokens_per_step,
            "total_tokens_trained": total_tokens_processed,
            "total_wall_time_sec": round(total_training_wall_time, 2),
            "avg_step_latency_sec": round(float(avg_step_latency), 3),
            "avg_throughput_tokens_per_sec": round(float(avg_throughput), 1),
        },
        "loss_and_convergence": {
            "baseline_val_loss": round(init_val_loss, 4),
            "baseline_val_perplexity": round(init_val_ppl, 2),
            "step1_train_loss": step_records[0]["train_loss"],
            "step100_train_loss": step_records[-1]["train_loss"],
            "loss_reduction_train": round(step_records[0]["train_loss"] - step_records[-1]["train_loss"], 4),
            "step50_val_loss": round(val_loss_50, 4),
            "step100_val_loss": round(final_val_loss, 4),
            "step100_val_perplexity": round(final_val_ppl, 2),
            "best_val_loss": round(best_val_loss, 4),
        },
        "hardware_telemetry": {
            "peak_vram_mb": round(float(max_recorded_vram), 2),
            "peak_gpu_temp_c": round(float(max_recorded_temp), 1),
            "thermal_headroom_to_cutoff_c": round(THERMAL_CUTOFF_THRESHOLD_C - max_recorded_temp, 1),
            "vram_headroom_to_ceiling_mb": round(VRAM_SAFETY_CEILING_MB - max_recorded_vram, 2),
        },
        "checkpoints": {
            "recovery_directory": str(recovery_dir),
            "best_checkpoint": {
                "path": str(best_ckpt_path),
                "size_bytes": best_ckpt_path.stat().st_size,
                "sha256": best_sha,
                "tensors_verified": 149,
            },
            "final_checkpoint": {
                "path": str(final_ckpt_path),
                "size_bytes": final_ckpt_path.stat().st_size,
                "sha256": final_sha,
            },
        },
        "baseline_preservation": {
            "pilot_checkpoint_sha256": compute_sha256(pilot_ckpt),
            "m6_checkpoint_sha256": compute_sha256(m6_ckpt),
            "rc2_archive_sha256": compute_sha256(rc2_zip),
            "all_baselines_intact": True,
        },
        "sample_generation": generated_text,
        "step_records": step_records,
    }

    results_file = PROJECT_ROOT / "results" / "phase1_recovery_100_step_smoke_test.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    logger.info("=" * 80)
    logger.info("PHASE 1 100-STEP RECOVERY SMOKE TEST COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Initial Train Loss: {step_records[0]['train_loss']} -> Final Train Loss: {step_records[-1]['train_loss']}")
    logger.info(f"Initial Val Loss: {init_val_loss:.4f} -> Final Val Loss: {final_val_loss:.4f}")
    logger.info(f"Peak VRAM: {max_recorded_vram:.2f} MB | Peak GPU Temp: {max_recorded_temp:.1f}°C")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_phase1_smoke_test()

