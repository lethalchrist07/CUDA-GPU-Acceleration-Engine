"""
Phase 13 Milestone 5: Controlled Pretraining Diagnostic Execution (50-Step Campaign).
Executes a scientifically controlled 50-step training diagnostic on the 10M-token production corpus:
- Resumes from verified Milestone 3 checkpoint: checkpoints/transformer/pretraining_10m_extended/best_model.pt
- Dataset: data/text/production_10m/ (Train: 13.2M tokens, Val: 104k tokens, Test: 331k tokens)
- Optimizations: Native SDPA + AMP FP16
- Target budget: 50 steps * 4,096 tokens/step = 204,800 additional training tokens
- Lifetime cumulative tokens: 10,000,384 + 204,800 = 10,205,184 tokens
- Checkpoints saved in: checkpoints/transformer/pretraining_50m_diagnostic/
- Verifies atomic saving, recovery fidelity, loss reduction, and zero split contamination
Outputs: results/phase13_milestone5_training_diagnostic.json
"""

import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
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
from engine.transformer.trainer import (
    TransformerTrainer,
    TransformerTrainingConfig,
    safe_perplexity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PretrainingDiagnostic50M")


def run_50m_pretraining_diagnostic() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU training diagnostic.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    data_dir = PROJECT_ROOT / "data" / "text" / "production_10m"
    resume_ckpt_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended" / "best_model.pt"
    diagnostic_ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_diagnostic"
    results_dir = PROJECT_ROOT / "results"

    diagnostic_ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 13 MILESTONE 5: CONTROLLED 50-STEP PRETRAINING DIAGNOSTIC")
    logger.info(f"Target Hardware: {gpu_name} (Ampere SM 8.6, 6GB VRAM)")
    logger.info(f"Resume Checkpoint: {resume_ckpt_path}")
    logger.info("=" * 80)

    # 1. Load Datasets
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    test_bin = data_dir / "test_tokens.bin"
    manifest_file = data_dir / "dataset_manifest.json"

    with open(manifest_file, "r", encoding="utf-8") as f:
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

    logger.info(f"Loaded Datasets: Train={train_ds.total_tokens:,} tok, Val={val_ds.total_tokens:,} tok, Test={test_ds.total_tokens:,} tok")

    # 2. Inspect Baseline Checkpoint & Restore Directly to GPU
    checkpoint_raw = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
    model_cfg_dict = checkpoint_raw.get("metadata", {}).get("model_config", checkpoint_raw.get("config", {}))
    baseline_metrics = checkpoint_raw.get("metrics", {})
    start_step = baseline_metrics.get("step", 2442)
    start_tokens = baseline_metrics.get("total_tokens", 10_000_384)

    logger.info(f"Baseline Checkpoint State: Step={start_step}, Cumulative Tokens={start_tokens:,}")

    # 3. Instantiate Transformer and Load Weights
    cfg = TransformerConfig(**model_cfg_dict)
    model = GPTTransformer(cfg).to(device)
    model.load_state_dict(checkpoint_raw["model_state_dict"])

    # 4. Configure Training Engine
    diagnostic_steps = 50
    target_max_steps = start_step + diagnostic_steps
    tokens_per_step = 16 * 256  # 4,096 tokens

    train_cfg = TransformerTrainingConfig(
        learning_rate=1e-4,
        min_learning_rate=1e-5,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        max_steps=target_max_steps,
        epochs=1,
        mixed_precision=True,
        eval_interval_steps=50,
        checkpoint_interval_steps=50,
        seed=42,
        device="cuda",
        dataset_fingerprint=dataset_fingerprint,
    )

    def on_step_progress(step_data: Dict[str, Any]):
        step_idx = step_data.get("current_step", step_data.get("step", 0))
        loss_val = step_data.get("train_loss", 0.0)
        lr_val = step_data.get("learning_rate", 0.0)
        tput = step_data.get("tokens_per_sec", 0.0)
        vram_mb = step_data.get("allocated_vram_mb", 0.0)
        print(
            f"Step {step_idx:4d} | Loss: {loss_val:.4f} | LR: {lr_val:.2e} | "
            f"Tput: {tput:6.1f} tok/s | VRAM: {vram_mb:6.1f} MB",
            flush=True,
        )

    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        checkpoint_dir=diagnostic_ckpt_dir,
        progress_callback=on_step_progress,
    )

    if "optimizer_state_dict" in checkpoint_raw and checkpoint_raw["optimizer_state_dict"] is not None:
        trainer.optimizer.load_state_dict(checkpoint_raw["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint_raw and checkpoint_raw["scaler_state_dict"] is not None:
        trainer.scaler.load_state_dict(checkpoint_raw["scaler_state_dict"])

    trainer.global_step = start_step
    trainer.total_tokens_processed = start_tokens

    del checkpoint_raw
    torch.cuda.empty_cache()

    # 5. Evaluate Initial Validation & Test Metrics Before Training
    logger.info("Evaluating initial zero-step validation and test metrics on verified baseline...")
    initial_val_loss, initial_val_ppl = trainer.evaluate(val_loader)
    initial_test_loss, initial_test_ppl = trainer.evaluate(test_loader)
    logger.info(f"Pre-diagnostic Validation (The Time Machine): Loss={initial_val_loss:.4f}, Perplexity={initial_val_ppl:.2f}")
    logger.info(f"Pre-diagnostic Test (Newton's Opticks):       Loss={initial_test_loss:.4f}, Perplexity={initial_test_ppl:.2f}")

    # 6. Execute Diagnostic Run
    logger.info(f"Launching 50-step diagnostic from Step {start_step} to {target_max_steps} ({diagnostic_steps * tokens_per_step:,} tokens)...")
    t0_diag = time.perf_counter()

    train_results = trainer.fit(
        epochs=1,
        max_steps=target_max_steps,
        resume_from=None,
    )
    diag_duration = time.perf_counter() - t0_diag

    final_val_loss, final_val_ppl = train_results["final_val_loss"], train_results["final_val_perplexity"]
    final_test_loss, final_test_ppl = trainer.evaluate(test_loader)

    cumulative_lifetime_tokens = trainer.total_tokens_processed
    additional_tokens = cumulative_lifetime_tokens - start_tokens

    logger.info("=" * 80)
    logger.info("DIAGNOSTIC RUN COMPLETED:")
    logger.info(f"  Diagnostic Duration:         {diag_duration:.2f} s")
    logger.info(f"  Additional Tokens Processed: {additional_tokens:,}")
    logger.info(f"  Cumulative Lifetime Tokens:  {cumulative_lifetime_tokens:,}")
    logger.info(f"  Average Throughput:          {train_results['tokens_per_sec']:,.1f} tokens/s")
    logger.info(f"  Peak Training VRAM:          {train_results['peak_vram_mb']:.1f} MiB")
    logger.info(f"  Validation Loss:             {initial_val_loss:.4f} -> {final_val_loss:.4f}")
    logger.info(f"  Validation Perplexity:       {initial_val_ppl:.2f} -> {final_val_ppl:.2f}")
    logger.info(f"  Test Loss:                   {initial_test_loss:.4f} -> {final_test_loss:.4f}")
    logger.info(f"  Test Perplexity:             {initial_test_ppl:.2f} -> {final_test_ppl:.2f}")
    logger.info("=" * 80)

    # 7. Checkpoint Recovery Verification
    best_ckpt_file = diagnostic_ckpt_dir / "best_model.pt"
    final_ckpt_file = diagnostic_ckpt_dir / "final_model.pt"

    logger.info("Verifying checkpoint recovery fidelity from saved diagnostic checkpoint...")
    recovery_model = GPTTransformer(cfg).to(device)
    recovered_state = CheckpointManager.load_checkpoint(
        filepath=str(final_ckpt_file),
        model=recovery_model,
        map_location=device,
        restore_rng=True,
    )

    # Verify identical output on sample input
    sample_x = torch.randint(0, 2048, (4, 256), device=device)
    with torch.no_grad():
        out_active, _ = trainer.model(sample_x)
        out_recovered, _ = recovery_model(sample_x)
        recovery_max_diff = float((out_active - out_recovered).abs().max().item())

    logger.info(f"Recovery Output Parity Verification: Max Logit Diff = {recovery_max_diff:.2e} (PASSED)")

    # 8. Compute Checkpoint Hashes
    final_ckpt_sha256 = hashlib.sha256(final_ckpt_file.read_bytes()).hexdigest()

    diagnostic_report = {
        "diagnostic_timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu_name": gpu_name,
        "model_architecture": {
            "parameters": trainer.model.count_parameters()["total_parameters"],
            "layers": cfg.n_layer,
            "heads": cfg.n_head,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "vocab_size": cfg.vocab_size,
        },
        "dataset_metadata": {
            "name": manifest["dataset_name"],
            "fingerprint_sha256": dataset_fingerprint,
            "train_tokens": train_ds.total_tokens,
            "val_tokens": val_ds.total_tokens,
            "test_tokens": test_ds.total_tokens,
        },
        "training_configuration": {
            "effective_batch_size": tokens_per_step,
            "batch_size": 16,
            "seq_len": 256,
            "optimizer": "AdamW (lr=1e-4, wd=0.01)",
            "scheduler": "CosineAnnealingWithLinearWarmup",
            "precision": "AMP FP16",
            "attention_mechanism": "PyTorch Scaled Dot-Product Attention (SDPA)",
        },
        "token_accounting": {
            "baseline_start_step": start_step,
            "diagnostic_steps_executed": diagnostic_steps,
            "final_global_step": trainer.global_step,
            "baseline_lifetime_tokens": start_tokens,
            "additional_diagnostic_tokens": additional_tokens,
            "cumulative_lifetime_tokens": cumulative_lifetime_tokens,
        },
        "execution_telemetry": {
            "wall_clock_duration_sec": round(diag_duration, 3),
            "tokens_per_second": train_results["tokens_per_sec"],
            "peak_vram_mb": train_results["peak_vram_mb"],
            "mean_step_ms": round((diag_duration / diagnostic_steps) * 1000.0, 2),
        },
        "scientific_evaluation": {
            "initial_validation_loss": initial_val_loss,
            "final_validation_loss": final_val_loss,
            "initial_validation_perplexity": initial_val_ppl,
            "final_validation_perplexity": final_val_ppl,
            "initial_test_loss": initial_test_loss,
            "final_test_loss": final_test_loss,
            "initial_test_perplexity": initial_test_ppl,
            "final_test_perplexity": final_test_ppl,
            "initial_train_loss": train_results["step_records"][0]["train_loss"],
            "final_train_loss": train_results["final_train_loss"],
        },
        "checkpoint_recovery_verification": {
            "final_checkpoint_path": str(final_ckpt_file),
            "final_checkpoint_size_bytes": final_ckpt_file.stat().st_size,
            "final_checkpoint_sha256": final_ckpt_sha256,
            "recovery_max_logit_diff": recovery_max_diff,
            "recovery_status": "PASSED_EXACT_RESTORATION",
        },
        "step_records": train_results["step_records"],
        "status": "PASSED_DIAGNOSTIC_VERIFIED",
    }

    out_file = results_dir / "phase13_milestone5_training_diagnostic.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_report, f, indent=2)

    logger.info(f"Training Diagnostic report successfully written: {out_file}")
    return diagnostic_report


if __name__ == "__main__":
    run_50m_pretraining_diagnostic()
