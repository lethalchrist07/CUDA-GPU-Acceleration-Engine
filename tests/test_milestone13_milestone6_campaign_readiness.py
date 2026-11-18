"""
Phase 13 Milestone 6: Pretraining Campaign, CUDA Optimization & Reliability Test Suite.
Tests:
1. Stage 1 Preflight Audit Verification.
2. Architecture Invariants (86,825,472 parameters, 12 layers, 12 heads, d_model=768, d_ff=3072).
3. Tokenizer SHA-256 Hash and Vocabulary Integrity.
4. Production Dataset Integrity and Split Isolation.
5. Stage 7 Custom CUDA Performance Benchmark Verification.
6. Stage 4 Diagnostic Launch Execution Summary Verification.
7. Bitwise Checkpoint Recovery Parity (Logit Max Diff == 0.0).
8. Historical Checkpoints Immutable Preservation.
9. Campaign Budget & Hyperparameter Alignment.
"""

import hashlib
import json
import math
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from engine.transformer.config import TransformerConfig
from engine.transformer.model import GPTTransformer


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# TEST 1: Stage 1 Preflight Audit Verification
# ==============================================================================
def test_1_stage1_preflight_audit_report():
    audit_file = PROJECT_ROOT / "results" / "phase13_milestone6_preflight_audit.json"
    assert audit_file.is_file(), f"Preflight audit file missing at {audit_file}"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("all_preflight_checks_passed") is True
    checks = data.get("checks", {})
    assert checks.get("gpu_accessibility", {}).get("status") == "PASS"
    assert checks.get("custom_cuda_kernels", {}).get("status") == "PASS"
    assert checks.get("model_architecture", {}).get("status") == "PASS"
    assert checks.get("tokenizer_integrity", {}).get("status") == "PASS"
    assert checks.get("production_dataset", {}).get("status") == "PASS"
    assert checks.get("checkpoint_integrity", {}).get("status") == "PASS"
    assert checks.get("disk_space", {}).get("status") == "PASS"
    assert checks.get("cuda_stability_probe", {}).get("status") == "PASS"


# ==============================================================================
# TEST 2: Architecture Invariants
# ==============================================================================
def test_2_model_architecture_invariants():
    cfg = TransformerConfig(
        vocab_size=2048,
        max_seq_len=256,
        d_model=768,
        n_layer=12,
        n_head=12,
        d_ff=3072,
        tie_weights=True
    )
    model = GPTTransformer(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params == 86_825_472, f"Param count mismatch: {total_params} vs 86,825,472"
    assert trainable_params == 86_825_472
    assert len(model.blocks) == 12
    for b in model.blocks:
        assert b.attn.n_head == 12
        assert b.attn.d_model == 768
        assert b.mlp.c_fc.out_features == 3072


# ==============================================================================
# TEST 3: Tokenizer SHA-256 Hash and Vocabulary Integrity
# ==============================================================================
def test_3_tokenizer_hash_and_vocab_integrity():
    vocab_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "bpe_vocab.json"
    assert vocab_path.is_file(), f"Missing tokenizer vocab at {vocab_path}"

    vocab_hash = compute_sha256(vocab_path)
    expected_hash = "ffd4aac665204b09488cd84e5428a72aa93223e06708947520768c001c163841"
    assert vocab_hash == expected_hash, f"Tokenizer hash mismatch: {vocab_hash} vs {expected_hash}"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    assert vocab.get("vocab_size") == 2048, f"Vocab size mismatch: {vocab.get('vocab_size')} vs 2048"
    assert len(vocab.get("token_to_id", {})) == 2048


# ==============================================================================
# TEST 4: Production Dataset Integrity and Split Isolation
# ==============================================================================
def test_4_production_dataset_scale_and_isolation():
    data_dir = PROJECT_ROOT / "data" / "text" / "production_10m"
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    test_bin = data_dir / "test_tokens.bin"

    assert train_bin.is_file()
    assert val_bin.is_file()
    assert test_bin.is_file()

    train_tokens = train_bin.stat().st_size // 2
    val_tokens = val_bin.stat().st_size // 2
    test_tokens = test_bin.stat().st_size // 2

    assert train_tokens == 13_215_150, f"Train token mismatch: {train_tokens}"
    assert val_tokens == 104_180, f"Val token mismatch: {val_tokens}"
    assert test_tokens == 331_796, f"Test token mismatch: {test_tokens}"


# ==============================================================================
# TEST 5: Stage 7 Custom CUDA Performance Benchmark Verification
# ==============================================================================
def test_5_stage7_custom_cuda_performance_benchmark():
    bench_file = PROJECT_ROOT / "results" / "phase13_milestone6_cuda_performance_comparison.json"
    assert bench_file.is_file(), f"Missing CUDA benchmark report at {bench_file}"

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    configs = {c["configuration_name"]: c for c in data.get("configurations", [])}
    assert len(configs) == 3

    # Verify speedup of Config B and Config C over Config A
    speedups = data.get("speedup_analysis", {})
    assert speedups.get("config_b_vs_config_a_speedup") >= 2.0
    assert speedups.get("config_c_vs_config_a_total_speedup") >= 2.0
    assert speedups.get("vram_reduction_c_vs_a_percent") >= 40.0


# ==============================================================================
# TEST 6: Stage 4 Diagnostic Launch Execution Summary
# ==============================================================================
def test_6_stage4_diagnostic_execution_summary():
    diag_file = PROJECT_ROOT / "results" / "phase13_milestone6_diagnostic_summary.json"
    assert diag_file.is_file(), f"Missing diagnostic summary at {diag_file}"

    with open(diag_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("mode") == "diagnostic"
    token_acc = data.get("token_accounting", {})
    assert token_acc.get("start_step") == 2492
    assert token_acc.get("final_step") == 2497
    assert token_acc.get("steps_executed") == 5
    assert token_acc.get("start_tokens") == 10_205_184
    assert token_acc.get("cumulative_tokens") == 10_225_664

    sci = data.get("scientific_evaluation", {})
    assert sci.get("final_validation_loss") < sci.get("initial_validation_loss")
    assert sci.get("final_test_loss") < sci.get("initial_test_loss")


# ==============================================================================
# TEST 7: Bitwise Checkpoint Recovery Parity
# ==============================================================================
def test_7_checkpoint_recovery_fidelity():
    diag_file = PROJECT_ROOT / "results" / "phase13_milestone6_diagnostic_summary.json"
    assert diag_file.is_file()

    with open(diag_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    recovery = data.get("recovery_fidelity", {})
    assert recovery.get("status") == "PASSED"
    assert recovery.get("max_logit_diff") == 0.0


# ==============================================================================
# TEST 8: Historical Checkpoints Immutable Preservation
# ==============================================================================
def test_8_historical_checkpoints_preserved():
    protected_checkpoints = [
        PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended" / "best_model.pt",
        PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended" / "final_model.pt",
        PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_diagnostic" / "best_model.pt",
        PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_diagnostic" / "final_model.pt",
    ]
    for ckpt_path in protected_checkpoints:
        assert ckpt_path.is_file(), f"Protected checkpoint missing: {ckpt_path}"
        assert ckpt_path.stat().st_size > 1_000_000_000, f"Checkpoint size corrupted: {ckpt_path}"


# ==============================================================================
# TEST 9: Campaign Budget & Hyperparameter Alignment
# ==============================================================================
def test_9_campaign_budget_and_hyperparameters():
    plan_file = PROJECT_ROOT / "results" / "phase13_milestone5_50m_campaign_plan.json"
    assert plan_file.is_file()

    with open(plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)

    math_plan = plan.get("training_mathematics", {})
    assert math_plan.get("target_cumulative_tokens") == 50_000_000
    assert math_plan.get("effective_batch_size") == 4096
    assert math_plan.get("remaining_optimizer_steps") == 9716
    assert math_plan.get("total_lifetime_steps") == 12208

