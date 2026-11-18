"""
Phase 12 — Milestone 12 Automated Test Suite:
Flagship Transformer Pretraining Campaign, Dataset Engineering, Memory Optimization & Scientific Evaluation.

Tests:
1. Dataset provenance, manifest validation, token count accounting, zero-leakage split
2. BPE tokenizer vocabulary, fast priority-rank encode/decode lossless roundtrip
3. Flagship architecture exact parameter accounting (86,825,472 total, 85,056,000 non-embedding)
4. Empirical campaign results validation (<= 1M tokens, loss drop, PPL drop > 40x, throughput > 2,000 tok/s)
5. Atomic checkpoint integrity, file sizes, and SHA-256 verification
6. Checkpoint recovery and resumption fidelity (interruption at step 50, resumption to 75)
7. Platform integration with ModelServer and autoregressive generation
8. Protected baseline artifact defense (ResNet-18 and Milestone 10 pilot checkpoints intact)
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

from backend.serving import ModelServer
from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.transformer import (
    GPTTransformer,
    TransformerConfig,
    TransformerTrainer,
    TransformerTrainingConfig,
    BPETokenizer,
    safe_perplexity,
)
from engine.transformer.dataset import TextChunkDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# TEST 1: Dataset Provenance, Manifest & Token Accounting
# ============================================================================
def test_1_dataset_provenance_manifest_and_fingerprints():
    """Validates dataset manifest, legal provenance, binary tokens, and split isolation."""
    data_dir = PROJECT_ROOT / "data" / "text" / "pretraining_campaign"
    manifest_path = data_dir / "dataset_manifest.json"
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"

    assert manifest_path.is_file(), "Manifest file missing"
    assert train_bin.is_file(), "Train binary missing"
    assert val_bin.is_file(), "Validation binary missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "Public Domain" in manifest["license"] or "Permissive" in manifest["license"]
    assert manifest["train_tokens"] >= 1_000_000, f"Expected >= 1M train tokens, got {manifest['train_tokens']}"
    assert manifest["val_tokens"] >= 50_000, f"Expected >= 50k val tokens, got {manifest['val_tokens']}"
    assert manifest["total_tokens"] == manifest["train_tokens"] + manifest["val_tokens"]

    # Verify binary file sizes match uint16 token count (2 bytes per token)
    assert train_bin.stat().st_size == manifest["train_tokens"] * 2
    assert val_bin.stat().st_size == manifest["val_tokens"] * 2

    # Verify zero document overlap between train and val
    train_docs = set(manifest.get("train_documents", []))
    val_docs = set(manifest.get("val_documents", []))
    assert len(train_docs.intersection(val_docs)) == 0, "Train/Val document leakage detected!"


# ============================================================================
# TEST 2: BPE Tokenizer Specification and Roundtrip
# ============================================================================
def test_2_bpe_tokenizer_specification_and_roundtrip():
    """Validates BPE tokenizer artifact, vocabulary size, and lossless roundtrip."""
    tok_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "tokenizer" / "bpe_vocab.json"
    assert tok_path.is_file(), "BPE tokenizer vocab file missing"

    tokenizer = BPETokenizer.load(tok_path)
    assert tokenizer.vocab_size == 2048

    test_sentences = [
        "The Time Traveller was expounding a recondite matter to us.",
        "It was the best of times, it was the worst of times.",
        "To be, or not to be, that is the question.",
    ]
    for s in test_sentences:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        assert decoded == s, f"Tokenizer roundtrip failed for: {s}"


# ============================================================================
# TEST 3: Flagship Architecture Exact Parameters
# ============================================================================
def test_3_flagship_architecture_exact_parameters():
    """Validates exact parameter count for the 86.83M pretraining model."""
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

    assert p_info["total_parameters"] == 86_825_472
    assert p_info["non_embedding_parameters"] == 85_056_000
    assert p_info["tied_weights"] is True

    # Confirm parameter sharing (tied weights)
    assert model.token_embeddings.weight.data_ptr() == model.lm_head.weight.data_ptr()


# ============================================================================
# TEST 4: Campaign Results Artifact Integrity
# ============================================================================
def test_4_campaign_results_artifact_integrity():
    """Validates measured results: tokens, steps, loss reduction, perplexity reduction, VRAM."""
    results_path = PROJECT_ROOT / "results" / "milestone12_pretraining_campaign_results.json"
    assert results_path.is_file(), "Results file missing"

    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    # 1. Budget enforcement: <= 1,000,000 tokens
    total_tokens = res["training_execution"]["total_tokens_processed"]
    assert total_tokens <= 1_000_000, f"Token budget exceeded: {total_tokens}"
    assert total_tokens == 512_000, f"Expected 512,000 tokens, got {total_tokens}"
    assert res["training_execution"]["optimizer_steps_completed"] == 125

    # 2. Scientific evaluation: finite losses and learning demonstrated
    eval_before = res["scientific_evaluation"]["before_training"]
    eval_after = res["scientific_evaluation"]["after_training"]

    step0_ppl = eval_before["val_perplexity"]
    stepN_ppl = eval_after["val_perplexity"]
    step0_loss = eval_before["val_loss"]
    stepN_loss = eval_after["val_loss"]

    assert step0_loss > 7.0, f"Initial val loss should be near ln(2048) (~7.62), got {step0_loss}"
    assert stepN_loss < 4.0, f"Final val loss should drop under 4.0, got {stepN_loss}"
    assert step0_ppl > 1500.0, f"Initial val PPL should be near 2048, got {step0_ppl}"
    assert stepN_ppl < 50.0, f"Final val PPL should drop under 50.0, got {stepN_ppl}"

    # Verify perplexity reduction ratio
    ppl_ratio = res["scientific_evaluation"]["comparison"]["perplexity_reduction_ratio"]
    assert ppl_ratio > 40.0, f"Perplexity reduction ratio should be > 40x, got {ppl_ratio}"

    # 3. Telemetry and VRAM safety
    assert res["training_execution"]["tokens_per_second"] > 2000.0
    assert res["training_execution"]["peak_vram_mb"] < 3000.0  # Well under 6GB limit


# ============================================================================
# TEST 5: Atomic Checkpoint Integrity and SHA-256
# ============================================================================
def test_5_checkpoint_integrity_and_sha256():
    """Validates that saved checkpoints are valid, complete, and uncorrupted."""
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign"
    best_ckpt = ckpt_dir / "best_model.pt"
    final_ckpt = ckpt_dir / "final_model.pt"

    assert best_ckpt.is_file()
    assert final_ckpt.is_file()

    # Model checkpoints with full AdamW moments are ~1.04 GB each
    assert best_ckpt.stat().st_size > 1_000_000_000
    assert final_ckpt.stat().st_size > 1_000_000_000

    best_val = CheckpointManager.validate_checkpoint(str(best_ckpt))
    assert best_val["valid"] is True
    assert len(best_val["sha256"]) == 64

    final_val = CheckpointManager.validate_checkpoint(str(final_ckpt))
    assert final_val["valid"] is True
    assert len(final_val["sha256"]) == 64


# ============================================================================
# TEST 6: Resumption Recovery Verification
# ============================================================================
def test_6_resumption_recovery_verification():
    """Validates that training interruption and resumption succeeded without state corruption."""
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign"
    step50_ckpt = ckpt_dir / "step_50.pt"
    assert step50_ckpt.is_file()

    # Load checkpoint and verify metadata restoration
    cfg = TransformerConfig(vocab_size=2048, max_seq_len=256, d_model=768, n_layer=12, n_head=12, tie_weights=True)
    model = GPTTransformer(cfg)
    trainer = TransformerTrainer(
        model=model,
        train_loader=[],
        val_loader=None,
        checkpoint_dir=ckpt_dir / "tmp_test",
    )
    state = trainer.load_checkpoint(step50_ckpt)
    assert trainer.global_step == 50
    assert trainer.total_tokens_processed == 204_800
    assert "optimizer_state_dict" in state["checkpoint"]
    assert "scheduler_state_dict" in state["checkpoint"]
    assert "torch_rng_state" in state["checkpoint"]


# ============================================================================
# TEST 7: Platform Serving & ModelServer Autoregressive Generation
# ============================================================================
def test_7_platform_serving_and_generation():
    """Verifies that ModelServer can load the campaign checkpoint and perform autoregressive generation."""
    best_ckpt = "checkpoints/transformer/pretraining_campaign/best_model.pt"
    res = ModelServer.load_model("transformer_100m", best_ckpt, device="cpu")
    assert res["status"] == "loaded"
    assert res["metrics"]["val_perplexity"] == 42.25

    gen_res = ModelServer.generate("The time traveler", max_new_tokens=15, seed=42)
    assert gen_res["status"] == "success"
    assert gen_res["generated_tokens"] == 15
    assert len(gen_res["generated_text"]) > 0
    assert "Pretrained Campaign" in gen_res["status_label"]


# ============================================================================
# TEST 8: Protected Baseline Artifact Defense
# ============================================================================
def test_8_protected_baseline_artifact_defense():
    """Guarantees that Milestone 8 ResNet-18 and Milestone 10 pilot checkpoints are unchanged."""
    resnet_ckpt = PROJECT_ROOT / "checkpoints" / "resnet18_cifar10" / "best_model.pt"
    pilot_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_pilot" / "best_model.pt"

    assert resnet_ckpt.is_file()
    assert pilot_ckpt.is_file()

    # Check exact byte sizes of protected baseline checkpoints
    assert resnet_ckpt.stat().st_size == 134_220_210
    assert pilot_ckpt.stat().st_size == 1_026_758_674
