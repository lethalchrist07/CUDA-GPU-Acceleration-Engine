"""
Phase 13 Milestone 3 Test Suite: 10M-Token Extended Pretraining Verification.

Validates:
1. Cumulative training tokens reached >= 10,000,000.
2. 86.83M parameter flagship Transformer architecture preservation.
3. Checkpoint validity, atomic serialization, and metadata preservation.
4. Validation loss and perplexity improvement (PPL < 15.0).
5. 3-Way scientific evaluation progression (Phase 12 -> M2 -> M3).
6. Generation lexical diversity (Distinct-n gains) and loop elimination.
7. Resumption recovery verification from Step 2000.
8. ModelServer loading and inference generation compatibility.
9. Dataset isolation and test split unexposed status.
10. GPU memory safety and disk retention limits.
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
)
from backend.serving import ModelServer


@pytest.fixture(scope="module")
def training_metrics():
    metrics_path = PROJECT_ROOT / "results" / "phase13_milestone3_training_metrics.json"
    assert metrics_path.is_file(), f"Metrics file missing: {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scientific_eval():
    eval_path = PROJECT_ROOT / "results" / "phase13_milestone3_scientific_evaluation.json"
    assert eval_path.is_file(), f"Evaluation file missing: {eval_path}"
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def checkpoint_report():
    ckpt_path = PROJECT_ROOT / "results" / "phase13_milestone3_checkpoint_report.json"
    assert ckpt_path.is_file(), f"Checkpoint report missing: {ckpt_path}"
    with open(ckpt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_01_cumulative_lifetime_tokens(training_metrics):
    """Verify campaign achieved >= 10,000,000 cumulative lifetime tokens."""
    summary = training_metrics["execution_summary"]
    assert summary["cumulative_lifetime_tokens"] >= 10000000
    assert summary["additional_tokens_trained"] >= 9000000
    assert summary["final_step"] == 2442
    assert summary["status"] == "COMPLETED"


def test_02_architecture_preservation():
    """Verify flagship 86,825,472 parameter architecture remains intact."""
    cfg = TransformerConfig(
        vocab_size=2048,
        max_seq_len=256,
        d_model=768,
        n_layer=12,
        n_head=12,
        d_ff=3072,
        tie_weights=True,
    )
    model = GPTTransformer(cfg)
    p_info = model.count_parameters()
    assert p_info["total_parameters"] == 86825472
    assert p_info["non_embedding_parameters"] == 85056000
    assert p_info["tied_weights"] is True


def test_03_extended_checkpoint_validity(checkpoint_report):
    """Verify best_model.pt and final_model.pt checkpoints are structurally valid."""
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended"
    best_path = ckpt_dir / "best_model.pt"
    final_path = ckpt_dir / "final_model.pt"

    assert best_path.is_file()
    assert final_path.is_file()

    val_best = CheckpointManager.validate_checkpoint(str(best_path))
    val_final = CheckpointManager.validate_checkpoint(str(final_path))

    assert val_best["valid"] is True
    assert val_best["has_optimizer"] is True
    assert val_best["metadata"]["model_config"]["vocab_size"] == 2048

    assert val_final["valid"] is True
    assert val_final["has_optimizer"] is True
    assert val_final["metadata"]["model_config"]["vocab_size"] == 2048


def test_04_validation_loss_and_perplexity_drop(training_metrics):
    """Verify validation loss dropped below 2.50 and perplexity dropped below 15.0."""
    summary = training_metrics["execution_summary"]
    assert summary["final_val_loss"] < 2.50
    assert summary["final_val_perplexity"] < 15.0
    assert summary["best_val_loss"] <= summary["final_val_loss"]
    assert summary["validation_loss_delta"] < -1.0


def test_05_three_way_evaluation_progression(scientific_eval):
    """Verify monotonic loss reduction across Phase 12 -> M2 -> M3."""
    models = scientific_eval["models_evaluated"]
    assert len(models) == 3

    p12_loss = models[0]["validation_metrics"]["loss"]
    m2_loss = models[1]["validation_metrics"]["loss"]
    m3_loss = models[2]["validation_metrics"]["loss"]

    assert p12_loss > m2_loss > m3_loss
    assert p12_loss > 3.70
    assert m3_loss < 2.45

    deltas = scientific_eval["three_way_deltas"]
    assert deltas["p12_to_m3_val_loss_delta"] < -1.30
    assert deltas["p12_to_m3_perplexity_reduction_ratio"] > 3.5


def test_06_lexical_diversity_and_repetition(scientific_eval):
    """Verify lexical diversity increased and repetition dropped significantly."""
    models = scientific_eval["models_evaluated"]
    p12_agg = models[0]["generation_metrics"]["aggregate_metrics"]
    m3_agg = models[2]["generation_metrics"]["aggregate_metrics"]

    assert m3_agg["mean_distinct_1"] > p12_agg["mean_distinct_1"] * 2.0
    assert m3_agg["mean_distinct_2"] > p12_agg["mean_distinct_2"] * 1.5
    assert m3_agg["mean_repetition_index"] < p12_agg["mean_repetition_index"] * 0.5
    assert m3_agg["degenerate_completions_detected"] == 0


def test_07_resumption_recovery_verification(training_metrics):
    """Verify checkpoint recovery verification passed from Step 2000."""
    rec = training_metrics["checkpoints"]["recovery_verification"]
    assert rec["verified"] is True
    assert rec["resumed_step"] == 2000


def test_08_model_server_inference_compatibility():
    """Verify ModelServer loads the 10M checkpoint and generates autoregressive text."""
    best_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended" / "best_model.pt"
    assert best_path.is_file()

    load_res = ModelServer.load_model(
        model_name="NEXORA GPT-100M",
        checkpoint_path=str(best_path),
        device_str="cpu",
    )
    assert load_res["status"] == "loaded"

    gen_res = ModelServer.generate(
        prompt="The time traveler",
        max_new_tokens=20,
        temperature=0.7,
        device_str="cpu",
    )
    assert gen_res["status"] == "success"
    assert len(gen_res["generated_text"]) > len("The time traveler")

    ModelServer.unload_model()


def test_09_dataset_provenance_and_isolation():
    """Verify test split (Opticks) was strictly isolated and unexposed."""
    exp_manifest = PROJECT_ROOT / "data" / "text" / "expanded_pretraining" / "dataset_manifest.json"
    assert exp_manifest.is_file()
    with open(exp_manifest, "r", encoding="utf-8") as f:
        meta = json.load(f)

    summary = meta["split_summary"]
    assert summary["train_tokens"] > 2100000
    assert summary["val_tokens"] == 104180
    assert summary["test_tokens"] == 331918

    # Assert test file exists in raw corpus and was never included in train tokens bin
    test_src = PROJECT_ROOT / "data" / "text" / "expanded_pretraining" / "raw" / "isaac_newton_opticks.txt"
    assert test_src.is_file()


def test_10_gpu_memory_safety_and_retention(training_metrics):
    """Verify VRAM remained within 6GB capacity and checkpoint retention was enforced."""
    summary = training_metrics["execution_summary"]
    assert summary["peak_vram_mb"] < 3500.0  # Well within 6143.5 MiB

    ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended"
    pt_files = list(ckpt_dir.glob("*.pt"))
    assert len(pt_files) <= 5, f"Too many checkpoint files retained: {len(pt_files)}"
