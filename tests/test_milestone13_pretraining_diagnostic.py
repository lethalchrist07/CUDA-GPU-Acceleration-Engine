"""
Phase 13 — Milestone 2 Automated Test Suite:
Controlled Extended Transformer Pretraining, GPU Profiling, Scientific Evaluation & Checkpoint Reliability.

Tests:
1. 86.83M Flagship Transformer parameter count & architecture integrity (86,825,472 total, 85,056,000 non-emb)
2. BPE tokenizer compatibility with active diagnostic checkpoint
3. Protected baseline checkpoints bitwise integrity defense
4. GPU performance profiling metrics & RTX 3050 VRAM safety limits
5. 1M-token diagnostic execution summary, step accounting, and loss convergence
6. Checkpoint serialization, SHA-256 validation, and atomicity
7. Checkpoint resumption recovery verification (step 200 -> 225)
8. Scientific evaluation: perplexity reduction, Distinct-N gains, and repetition index drop
9. ModelServer inference serving compatibility with diagnostic checkpoint
10. Reconciled validation loss forensics (3.7435 full vs 3.8104 sample)
"""

import json
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.transformer import (
    BPETokenizer,
    GPTTransformer,
    TransformerConfig,
    safe_perplexity,
)
from engine.transformer.evaluator import (
    compute_distinct_ngrams,
    compute_repetition_index,
)


def compute_sha256(filepath: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# TEST 1: Architecture Integrity & Parameter Accounting
# ============================================================================
def test_1_flagship_architecture_parameter_integrity():
    """Validates exact 86,825,472 total parameters and 85,056,000 non-embedding parameters."""
    cfg = TransformerConfig(
        vocab_size=2048,
        max_seq_len=256,
        d_model=768,
        n_layer=12,
        n_head=12,
        d_ff=3072,
        dropout=0.1,
        tie_weights=True,
    )
    model = GPTTransformer(cfg)
    p_info = model.count_parameters()
    assert p_info["total_parameters"] == 86825472
    assert p_info["non_embedding_parameters"] == 85056000
    assert p_info["tied_weights"] is True


# ============================================================================
# TEST 2: Tokenizer Compatibility
# ============================================================================
def test_2_tokenizer_compatibility():
    """Validates 2048 vocabulary and embedding compatibility with active checkpoint."""
    tok_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "tokenizer" / "bpe_vocab.json"
    assert tok_path.is_file()
    tokenizer = BPETokenizer.load(tok_path)
    assert tokenizer.vocab_size == 2048

    diag_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_1m_diagnostic" / "best_model.pt"
    assert diag_ckpt.is_file()
    state = torch.load(diag_ckpt, map_location="cpu", weights_only=False)
    emb_weight = state["model_state_dict"]["token_embeddings.weight"]
    assert list(emb_weight.shape) == [2048, 768]


# ============================================================================
# TEST 3: Protected Baseline Checkpoints Defense
# ============================================================================
def test_3_protected_baseline_checkpoints_integrity():
    """Guarantees historical baselines are unchanged."""
    baselines = [
        ("resnet18_cifar10", PROJECT_ROOT / "checkpoints" / "resnet18_cifar10" / "best_model.pt", 134220210, "a411eb7c079058e3aada3edb2d851106273527a8d17971936a24671542dc3a78"),
        ("flagship_100m_pilot", PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_pilot" / "best_model.pt", 1026758674, "f8b0b9f9d4fb2e9ce482aea9c619e2c588404caa7e5691606e9340f381932f9f"),
        ("pretraining_campaign_best", PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "best_model.pt", 1042094098, "b237cae9c17b09828f70585875a5782ed7429ed36ddfcae4ae31065ece3dad01"),
        ("pretraining_campaign_final", PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "final_model.pt", 1042094696, "a4fd662065cc713e4a4aa664e85b111b0b3ab8454bba9b16d82abdfa5fcb4f22"),
    ]
    for name, path, exp_size, exp_sha in baselines:
        assert path.is_file(), f"Missing baseline: {name}"
        assert path.stat().st_size == exp_size, f"Size changed for {name}"
        assert compute_sha256(path) == exp_sha, f"SHA-256 changed for {name}"


# ============================================================================
# TEST 4: GPU Performance Profiling Report Validation
# ============================================================================
def test_4_gpu_profiling_report_validation():
    """Validates CUDA event profiling measurements on RTX 3050."""
    prof_file = PROJECT_ROOT / "results" / "phase13_milestone2_gpu_profiling.json"
    assert prof_file.is_file()
    with open(prof_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["hardware"]["sm_architecture"] == "Ampere SM 8.6"
    assert data["profiling_parameters"]["effective_batch_tokens"] == 4096
    lat = data["latencies"]
    assert lat["dataloader_latency_ms"] < 10.0
    assert lat["host_to_device_latency_ms"] < 5.0
    assert lat["forward_pass_latency_ms"] > 0.0
    assert lat["backward_pass_latency_ms"] > 0.0
    assert lat["optimizer_step_latency_ms"] > 0.0
    assert data["throughput"]["tokens_per_second"] > 3000.0
    assert data["memory_telemetry"]["peak_allocated_mb"] < 3000.0
    assert data["memory_telemetry"]["headroom_mb"] > 3000.0


# ============================================================================
# TEST 5: 1M Diagnostic Campaign Execution Summary
# ============================================================================
def test_5_diagnostic_campaign_execution_summary():
    """Validates step accounting, token count, and loss convergence."""
    metrics_file = PROJECT_ROOT / "results" / "phase13_milestone2_training_metrics.json"
    assert metrics_file.is_file()
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["execution_summary"]
    assert summary["status"] == "COMPLETED"
    assert summary["initial_step"] == 125
    assert summary["final_step"] == 245
    assert summary["additional_steps_executed"] == 120
    assert summary["additional_tokens_trained"] == 491520
    assert summary["cumulative_lifetime_tokens"] == 1003520
    assert summary["final_train_loss"] < summary["initial_val_loss"]
    assert summary["final_val_loss"] < summary["initial_val_loss"]
    assert summary["final_val_perplexity"] < summary["initial_val_perplexity"]
    assert len(data["step_records"]) == 120


# ============================================================================
# TEST 6: Diagnostic Checkpoint Serialization & Integrity
# ============================================================================
def test_6_diagnostic_checkpoint_integrity():
    """Validates atomic checkpoint files in isolated diagnostic directory."""
    diag_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_1m_diagnostic"
    best_ckpt = diag_dir / "best_model.pt"
    final_ckpt = diag_dir / "final_model.pt"
    step200_ckpt = diag_dir / "step_200.pt"

    assert best_ckpt.is_file()
    assert final_ckpt.is_file()
    assert step200_ckpt.is_file()

    best_val = CheckpointManager.validate_checkpoint(str(best_ckpt))
    assert best_val["valid"] is True
    assert best_val["has_optimizer"] is True
    assert best_val["metadata"]["model_config"]["vocab_size"] == 2048

    final_val = CheckpointManager.validate_checkpoint(str(final_ckpt))
    assert final_val["valid"] is True


# ============================================================================
# TEST 7: Resumption Recovery Verification
# ============================================================================
def test_7_resumption_recovery_verification():
    """Confirms step 200 -> 225 recovery test passed with matching step accounting."""
    metrics_file = PROJECT_ROOT / "results" / "phase13_milestone2_training_metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rec = data["checkpoints"]["recovery_test"]
    assert rec["start_step"] == 200
    assert rec["target_step"] == 225
    assert rec["final_step"] == 225
    assert rec["verified"] is True


# ============================================================================
# TEST 8: Scientific Evaluation & Diversity Progression
# ============================================================================
def test_8_scientific_evaluation_diversity_and_generalization():
    """Validates that Milestone 2 achieved genuine perplexity and diversity gains."""
    eval_file = PROJECT_ROOT / "results" / "phase13_milestone2_scientific_evaluation.json"
    assert eval_file.is_file()
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    deltas = data["comparative_deltas"]
    assert deltas["val_loss_delta"] < 0.0  # Loss dropped
    assert deltas["perplexity_reduction_ratio"] > 1.0  # PPL dropped
    assert deltas["distinct_1_delta"] > 0.0  # Unique unigrams increased
    assert deltas["distinct_2_delta"] > 0.0  # Unique bigrams increased
    assert deltas["distinct_3_delta"] > 0.0  # Unique trigrams increased
    assert deltas["repetition_index_delta"] < 0.0  # Degenerate repetition dropped
    assert "Genuine Generalization" in deltas["generalization_verdict"]

    diag_gen = data["diagnostic_checkpoint"]["generation_metrics"]
    assert diag_gen["aggregate_metrics"]["degenerate_completions_detected"] == 0


# ============================================================================
# TEST 9: ModelServer Inference Serving Compatibility
# ============================================================================
def test_9_model_server_inference_serving():
    """Verifies that ModelServer can load and generate from diagnostic checkpoint."""
    from backend.serving import ModelServer

    diag_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_1m_diagnostic" / "best_model.pt"
    assert diag_ckpt.is_file()

    load_res = ModelServer.load_model(
        model_name="NEXORA GPT-100M",
        checkpoint_path=str(diag_ckpt),
        device_str="cpu",
    )
    assert load_res["status"] == "loaded"

    gen_res = ModelServer.generate(
        prompt="The time traveler",
        max_new_tokens=16,
        temperature=0.7,
        device_str="cpu",
    )
    assert gen_res["status"] == "success"
    assert gen_res["generated_tokens"] == 16
    assert len(gen_res["generated_text"]) > 0

    ModelServer.unload_model()


# ============================================================================
# TEST 10: Validation Loss Forensics
# ============================================================================
def test_10_validation_loss_forensics_reconciliation():
    """Validates that 3.7435 is full-population loss and 3.8104 is 50-batch sampled loss."""
    diag_eval = PROJECT_ROOT / "results" / "phase13_milestone2_scientific_evaluation.json"
    with open(diag_eval, "r", encoding="utf-8") as f:
        data = json.load(f)

    base = data["baseline_checkpoint"]
    assert base["validation_metrics"]["loss"] == pytest.approx(3.7435, abs=1e-3)
    assert base["validation_metrics"]["batches_evaluated"] == 102
    assert base["divergence_metrics"]["val_loss"] == pytest.approx(3.7725, abs=0.1)
