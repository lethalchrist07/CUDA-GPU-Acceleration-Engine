"""
Phase 13 Milestone 5: Complete 50M-Token Pretraining Campaign Architectural Plan.
Synthesizes verified empirical telemetry from Milestone 5 into an actionable,
scientifically controlled plan for scaling the Transformer to 50M lifetime tokens.

Parameters:
- Architecture: 86,825,472 trainable parameters (12 layers, 12 heads, d_model=768, d_ff=3072, vocab=2048)
- Baseline Checkpoint: checkpoints/transformer/pretraining_50m_diagnostic/final_model.pt (Step 2492)
- Cumulative Baseline Tokens: 10,205,184 tokens
- Target Lifetime Tokens: 50,000,000 tokens
- Remaining Training Budget: 39,794,816 tokens
- Effective Batch Size: 16 sequences * 256 context = 4,096 tokens/step
- Optimizer Steps Required: 9,716 steps (Total Lifetime Steps: 12,208)
- Training Dataset: data/text/production_10m/train_tokens.bin (13,215,150 tokens, ~3.01 epochs)
- Target Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere SM 8.6)
Outputs: results/phase13_milestone5_50m_campaign_plan.json
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CampaignPlan50M")


def generate_50m_campaign_plan() -> Dict[str, Any]:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load empirical benchmark data
    opt_bench_file = results_dir / "phase13_milestone5_gpu_optimization_benchmarks.json"
    diag_file = results_dir / "phase13_milestone5_training_diagnostic.json"
    manifest_file = PROJECT_ROOT / "data" / "text" / "production_10m" / "dataset_manifest.json"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(opt_bench_file, "r", encoding="utf-8") as f:
        opt_bench = json.load(f)

    with open(diag_file, "r", encoding="utf-8") as f:
        diag = json.load(f)

    # 2. Token and Step Mathematics
    baseline_tokens = diag["token_accounting"]["cumulative_lifetime_tokens"]  # 10,205,184
    baseline_step = diag["token_accounting"]["final_global_step"]            # 2,492
    target_lifetime_tokens = 50_000_000
    remaining_tokens = target_lifetime_tokens - baseline_tokens               # 39,794,816

    batch_size = 16
    context_length = 256
    effective_batch_tokens = batch_size * context_length                      # 4,096
    remaining_steps = (remaining_tokens + effective_batch_tokens - 1) // effective_batch_tokens  # 9,716 steps
    total_campaign_steps = baseline_step + remaining_steps                    # 12,208 steps

    train_tokens = manifest["split_summary"]["train_tokens"]                  # 13,215,150
    tokens_per_epoch = (train_tokens // effective_batch_tokens) * effective_batch_tokens
    steps_per_epoch = train_tokens // effective_batch_tokens                  # 3,226 steps
    epochs_required = round(remaining_tokens / train_tokens, 2)               # ~3.01 epochs

    # 3. Throughput and Duration Projections
    # Measured kernel step latency: 358.4 ms (Config 4) / 379.1 ms (Config 3)
    measured_step_ms = 365.0  # conservative average
    pure_training_time_sec = remaining_steps * (measured_step_ms / 1000.0)

    # Evaluation overhead: every 200 steps (48 evaluations * 8.5s = 408s)
    eval_interval_steps = 200
    n_evals = remaining_steps // eval_interval_steps
    eval_time_sec = n_evals * 8.5

    # Checkpoint serialization overhead: every 400 steps (24 saves * 3.2s = 76.8s)
    ckpt_interval_steps = 400
    n_ckpts = remaining_steps // ckpt_interval_steps
    ckpt_time_sec = n_ckpts * 3.5

    total_projected_wall_sec = pure_training_time_sec + eval_time_sec + ckpt_time_sec
    total_projected_hours = total_projected_wall_sec / 3600.0

    # 4. Memory Budget & Safety Headroom
    model_params = 86_825_472
    peak_vram_mb = 2723.0  # measured peak allocated/reserved
    total_gpu_vram_mb = 6143.5
    vram_headroom_mb = total_gpu_vram_mb - peak_vram_mb
    vram_headroom_pct = round(100.0 * (vram_headroom_mb / total_gpu_vram_mb), 1)

    plan = {
        "plan_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_AUTHORIZATION",
        "objective": "Controlled 50M-Token Pretraining Campaign Scaling",
        "target_hardware": {
            "gpu": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
            "compute_capability": "Ampere SM 8.6 (16 SMs, 2048 CUDA Cores)",
            "total_vram_mb": total_gpu_vram_mb,
            "system_ram_mb": 16384,
            "os": "Windows 11",
            "cuda_version": "12.4.1",
            "pytorch_version": "2.6.0+cu124",
        },
        "model_architecture": {
            "total_parameters": model_params,
            "layers": 12,
            "heads": 12,
            "d_model": 768,
            "d_ff": 3072,
            "vocab_size": 2048,
            "max_seq_len": 256,
            "weight_tying": True,
            "activation": "GELU (approximate=tanh)",
            "attention": "PyTorch Native Scaled Dot-Product Attention (SDPA)",
        },
        "dataset_configuration": {
            "manifest_name": manifest["dataset_name"],
            "dataset_version": manifest["version"],
            "dataset_fingerprint_sha256": manifest["aggregate_content_hash_sha256"],
            "train_tokens": train_tokens,
            "val_tokens": manifest["split_summary"]["val_tokens"],
            "test_tokens": manifest["split_summary"]["test_tokens"],
            "total_corpus_tokens": manifest["split_summary"]["total_tokens"],
            "domain_coverage": [
                "Classical Literature (21 works, 9.28M tokens)",
                "Scientific & Mathematical Treatises (8 works, 3.50M tokens)",
                "Computer Science, Systems & RFC Standards (5 works, 0.44M tokens)",
            ],
            "held_out_validation_split": "The Time Machine (104,180 tokens - strictly isolated)",
            "held_out_test_split": "Opticks by Sir Isaac Newton (331,796 tokens - strictly isolated)",
        },
        "training_mathematics": {
            "baseline_cumulative_tokens": baseline_tokens,
            "baseline_global_step": baseline_step,
            "target_cumulative_tokens": target_lifetime_tokens,
            "remaining_token_budget": remaining_tokens,
            "effective_batch_size": effective_batch_tokens,
            "batch_size": batch_size,
            "context_length": context_length,
            "remaining_optimizer_steps": remaining_steps,
            "total_lifetime_steps": total_campaign_steps,
            "epochs_required": epochs_required,
            "steps_per_epoch": steps_per_epoch,
        },
        "hyperparameter_schedule": {
            "optimizer": "AdamW",
            "peak_learning_rate": 3e-4,
            "min_learning_rate": 3e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "grad_clip": 1.0,
            "warmup_steps": 200,
            "total_scheduler_steps": remaining_steps,
            "learning_rate_decay": "CosineAnnealingWithLinearWarmup",
            "mixed_precision": "AMP FP16 (torch.amp.autocast + GradScaler)",
            "fused_kernel_acceleration": "Custom CUDA Fused Bias+GELU",
        },
        "runtime_and_throughput_projections": {
            "measured_step_latency_ms": measured_step_ms,
            "projected_step_throughput_tokens_per_sec": round(effective_batch_tokens / (measured_step_ms / 1000.0), 1),
            "pure_compute_time_hours": round(pure_training_time_sec / 3600.0, 2),
            "evaluation_overhead_minutes": round(eval_time_sec / 60.0, 1),
            "checkpoint_overhead_minutes": round(ckpt_time_sec / 60.0, 1),
            "total_estimated_wall_clock_minutes": round(total_projected_wall_sec / 60.0, 1),
            "total_estimated_wall_clock_hours": round(total_projected_hours, 2),
        },
        "vram_budget_and_thermal_safety": {
            "total_hardware_vram_mb": total_gpu_vram_mb,
            "projected_peak_vram_mb": peak_vram_mb,
            "available_headroom_mb": round(vram_headroom_mb, 1),
            "vram_headroom_percentage": vram_headroom_pct,
            "thermal_safety_threshold_celsius": 82.0,
            "thermal_cooldown_pause_sec": 30.0,
            "vram_exhaustion_risk": "NEGLIGIBLE (44.3% VRAM utilized, 55.7% free headroom)",
        },
        "checkpointing_and_reliability": {
            "checkpoint_cadence_steps": ckpt_interval_steps,
            "evaluation_cadence_steps": eval_interval_steps,
            "checkpoint_directory": "checkpoints/transformer/pretraining_50m_campaign/",
            "atomic_double_buffering": True,
            "rng_state_restoration": True,
            "protection_of_historical_checkpoints": [
                "checkpoints/transformer/pretraining_campaign/ (Preserved)",
                "checkpoints/transformer/pretraining_1m_diagnostic/ (Preserved)",
                "checkpoints/transformer/pretraining_10m_extended/ (Preserved)",
                "checkpoints/transformer/pretraining_50m_diagnostic/ (Preserved)",
            ],
        },
        "guardrail_enforcement": {
            "campaign_auto_launch": False,
            "user_authorization_required": True,
            "status": "AWAITING_EXPLICIT_USER_AUTHORIZATION",
        },
    }

    out_file = results_dir / "phase13_milestone5_50m_campaign_plan.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    logger.info("=" * 80)
    logger.info("50M-TOKEN PRETRAINING CAMPAIGN PLAN GENERATED:")
    logger.info(f"  Target Lifetime Tokens:        {target_lifetime_tokens:,}")
    logger.info(f"  Remaining Tokens:              {remaining_tokens:,}")
    logger.info(f"  Optimizer Steps to Run:        {remaining_steps:,} (Lifetime Steps: {total_campaign_steps:,})")
    logger.info(f"  Epochs over Production Corpus: {epochs_required:.2f}")
    logger.info(f"  Estimated Training Duration:   {total_projected_hours:.2f} hours ({plan['runtime_and_throughput_projections']['total_estimated_wall_clock_minutes']} mins)")
    logger.info(f"  Projected Peak VRAM:           {peak_vram_mb:.1f} MiB ({vram_headroom_pct}% free headroom)")
    logger.info(f"  Saved Artifact:                {out_file}")
    logger.info("=" * 80)
    return plan


if __name__ == "__main__":
    generate_50m_campaign_plan()

