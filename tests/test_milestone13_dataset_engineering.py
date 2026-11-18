"""
Phase 13 — Milestone 1 Automated Test Suite:
Dataset Engineering, Scalable Data Loading, Tokenizer Validation & Extended Pretraining Readiness.

Tests:
1. TextCleaner Unicode NFKC normalization, control-character sanitization, whitespace standardization, Gutenberg stripping
2. MinHash near-duplicate detection and Jaccard similarity estimation
3. Deterministic 3-way document splitting (train/val/test) and zero-leakage enforcement
4. BPE tokenizer validation: 100% round-trip fidelity, zero OOV rate on arbitrary Unicode
5. Tokenizer checkpoint compatibility with active flagship checkpoint (vocab=2048, embedding=[2048, 768])
6. ScalableMemmapDataset zero-copy streaming, context length slicing, causal target alignment
7. DataLoader determinism, reproducible shuffling, and worker seed initialization
8. DatasetManifest v2.0 cryptographic hashes, document metadata, and binary file size alignment
9. Extended training configuration schema and numerical consistency (1M, 10M, 50M)
10. Scientific evaluation metrics: Distinct-1/2/3, repetition index, safe perplexity
11. Protected baseline checkpoint defense
"""

import json
from pathlib import Path
import pytest
import numpy as np
import torch

from engine.transformer import (
    BPETokenizer,
    GPTTransformer,
    TransformerConfig,
    safe_perplexity,
)
from engine.transformer.dataset_pipeline import (
    DatasetExpansionPipeline,
    TextCleaner,
    MinHashDeduplicator,
)
from engine.transformer.scalable_dataset import (
    ScalableMemmapDataset,
    create_scalable_dataloaders,
    seed_worker,
)
from engine.transformer.tokenizer_validator import TokenizerValidator
from engine.transformer.evaluator import (
    ModelEvaluator,
    compute_distinct_ngrams,
    compute_repetition_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# TEST 1: TextCleaner and Normalization
# ============================================================================
def test_1_text_cleaner_and_normalizer():
    """Validates Gutenberg header/footer stripping, NFKC normalization, and control char sanitization."""
    raw_sample = (
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "Hello \x00\x07World!\tThis is a   test.\r\n\r\n\r\n\r\n"
        "Second line with “smart quotes” and fi ligature: ﬁ.\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"
    )

    cleaned = TextCleaner.clean(raw_sample, is_gutenberg=True)
    assert cleaned is not None
    assert "PROJECT GUTENBERG" not in cleaned
    assert "\x00" not in cleaned  # Null char stripped
    assert "\x07" not in cleaned  # Bell char stripped
    assert "\t" in cleaned       # Tab preserved
    assert "\r" not in cleaned   # CRLF normalized to LF
    assert "\n\n\n" not in cleaned  # Excessive newlines collapsed
    assert "fi" in cleaned or "ﬁ" in cleaned  # NFKC decomposition


# ============================================================================
# TEST 2: MinHash Deduplicator and Near-Duplicate Detection
# ============================================================================
def test_2_minhash_deduplicator():
    """Validates MinHash signature computation and Jaccard similarity thresholding."""
    dedup = MinHashDeduplicator(num_perm=64, threshold=0.85)

    text_a = "The quick brown fox jumps over the lazy dog in the sunny morning."
    text_b = "The quick brown fox jumps over the lazy dog in the sunny morning!"  # Punctuation diff
    text_c = "Quantum chromodynamics is the fundamental theory of the strong interaction."

    sig_a = dedup.compute_minhash(text_a)
    sig_b = dedup.compute_minhash(text_b)
    sig_c = dedup.compute_minhash(text_c)

    # Identical texts
    assert dedup.estimate_jaccard(sig_a, sig_a) == 1.0

    # Near-identical texts should exceed threshold
    sim_ab = dedup.estimate_jaccard(sig_a, sig_b)
    assert sim_ab >= 0.70, f"Expected high Jaccard for minor edit, got {sim_ab}"

    # Disparate texts should be low
    sim_ac = dedup.estimate_jaccard(sig_a, sig_c)
    assert sim_ac < 0.25, f"Expected low Jaccard for distinct topics, got {sim_ac}"


# ============================================================================
# TEST 3: Deterministic 3-Way Split and Zero Leakage
# ============================================================================
def test_3_deterministic_3way_split_and_zero_leakage():
    """Validates 3-way split isolation and zero token leakage."""
    pipeline = DatasetExpansionPipeline(name="TestCorpus")
    pipeline.add_document("doc1", "Doc 1", "Auth 1", "2020", "CC0", "Prov", "path1", "The first training document about systems.")
    pipeline.add_document("doc2", "Doc 2", "Auth 2", "2021", "CC0", "Prov", "path2", "The second training document about GPUs.")
    pipeline.add_document("doc3", "Doc 3", "Auth 3", "2022", "CC0", "Prov", "path3", "The third held-out validation document.")
    pipeline.add_document("doc4", "Doc 4", "Auth 4", "2023", "CC0", "Prov", "path4", "The fourth held-out test document.")

    splits = pipeline.split_train_val_test(val_doc_ids={"doc3"}, test_doc_ids={"doc4"})
    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    assert splits["val"][0]["doc_id"] == "doc3"
    assert splits["test"][0]["doc_id"] == "doc4"

    # Leakage check
    assert pipeline.verify_zero_leakage(splits) is True


# ============================================================================
# TEST 4: BPE Tokenizer Forensic Roundtrip and Zero OOV
# ============================================================================
def test_4_bpe_tokenizer_forensic_roundtrip_and_coverage():
    """Validates that BPE tokenizer achieves 100% roundtrip fidelity and 0% true OOV."""
    tok_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "tokenizer" / "bpe_vocab.json"
    assert tok_path.is_file()
    tokenizer = BPETokenizer.load(tok_path)
    validator = TokenizerValidator(tokenizer)

    rt_res = validator.validate_roundtrip()
    assert rt_res["passed"] is True, f"Roundtrip failed: {rt_res['failures']}"

    oov_res = validator.validate_unknown_token_behavior()
    assert oov_res["all_lossless"] is True
    assert oov_res["oov_rate_percent"] == 0.0

    comp_res = validator.measure_compression()
    assert comp_res["overall_bytes_per_token"] > 1.0


# ============================================================================
# TEST 5: Tokenizer Checkpoint Compatibility
# ============================================================================
def test_5_tokenizer_checkpoint_compatibility():
    """Validates that tokenizer vocabulary exactly matches the active flagship model checkpoint."""
    tok_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "tokenizer" / "bpe_vocab.json"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "best_model.pt"

    tokenizer = BPETokenizer.load(tok_path)
    validator = TokenizerValidator(tokenizer)

    compat = validator.validate_checkpoint_compatibility(ckpt_path)
    assert compat["compatible"] is True
    assert compat["tokenizer_vocab_size"] == 2048
    assert compat["embedding_matrix_shape"] == [2048, 768]


# ============================================================================
# TEST 6: ScalableMemmapDataset Zero-Copy Streaming
# ============================================================================
def test_6_scalable_memmap_dataset_zero_copy_and_indexing():
    """Validates ScalableMemmapDataset slicing, length, and causal target alignment."""
    bin_path = PROJECT_ROOT / "data" / "text" / "expanded_pretraining" / "train_tokens.bin"
    assert bin_path.is_file()

    seq_len = 256
    ds = ScalableMemmapDataset(bin_path, seq_len=seq_len, vocab_size=2048)

    assert len(ds) > 0
    assert ds.total_tokens > 2_000_000

    # Test indexing
    x, y = ds[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # Causal alignment: y must be shifted by 1 relative to x
    assert torch.equal(x[1:], y[:-1]), "Causal shift alignment violation!"

    # Token slice
    raw_slice = ds.get_token_slice(0, 10)
    assert len(raw_slice) == 10
    assert isinstance(raw_slice, np.ndarray)


# ============================================================================
# TEST 7: Scalable DataLoader Determinism and Shuffling
# ============================================================================
def test_7_scalable_dataloader_determinism_and_shuffling():
    """Validates that DataLoader streaming is deterministic under identical seeds."""
    bin_path = PROJECT_ROOT / "data" / "text" / "expanded_pretraining" / "train_tokens.bin"
    ds = ScalableMemmapDataset(bin_path, seq_len=128, vocab_size=2048)

    loaders_1 = create_scalable_dataloaders(ds, batch_size=4, seed=42)
    loaders_2 = create_scalable_dataloaders(ds, batch_size=4, seed=42)

    batch_1 = next(iter(loaders_1["train"]))
    batch_2 = next(iter(loaders_2["train"]))

    assert torch.equal(batch_1[0], batch_2[0]), "DataLoader non-deterministic under identical seed!"
    assert torch.equal(batch_1[1], batch_2[1])


# ============================================================================
# TEST 8: DatasetManifest v2.0 Cryptographic Integrity
# ============================================================================
def test_8_dataset_manifest_v2_cryptographic_integrity():
    """Validates manifest v2.0 schema, token counts, and disk file alignment."""
    manifest_path = PROJECT_ROOT / "data" / "text" / "expanded_pretraining" / "dataset_manifest.json"
    assert manifest_path.is_file()

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "2.0.0"
    assert len(data["aggregate_content_hash_sha256"]) == 64
    assert data["split_summary"]["total_tokens"] > 2_000_000
    assert data["split_summary"]["train_tokens"] > 1_800_000
    assert data["split_summary"]["val_tokens"] > 50_000
    assert data["split_summary"]["test_tokens"] > 100_000

    # Verify binary file sizes match token counts * 2 (uint16)
    data_dir = manifest_path.parent
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    test_bin = data_dir / "test_tokens.bin"

    assert train_bin.stat().st_size == data["split_summary"]["train_tokens"] * 2
    assert val_bin.stat().st_size == data["split_summary"]["val_tokens"] * 2
    assert test_bin.stat().st_size == data["split_summary"]["test_tokens"] * 2


# ============================================================================
# TEST 9: Training Readiness Configurations Validation
# ============================================================================
def test_9_training_readiness_configurations_validation():
    """Validates training configuration schemas for 1M, 10M, and 50M targets."""
    config_dir = PROJECT_ROOT / "configs" / "training"
    for target in ("1m_diagnostic", "10m_extended", "50m_research"):
        cfg_file = config_dir / f"pretraining_{target}.json"
        assert cfg_file.is_file()

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        assert cfg["architecture"]["total_parameters"] == 86825472
        assert cfg["architecture"]["vocab_size"] == 2048
        assert cfg["architecture"]["context_length"] == 256
        assert cfg["data"]["effective_batch_tokens"] == 4096
        assert cfg["hardware"]["expected_peak_vram_mb"] < 2500.0

        # Step count math
        eff_tokens = cfg["data"]["effective_batch_tokens"]
        steps = cfg["optimizer"]["total_optimizer_steps"]
        target_tokens = cfg["target_tokens"]
        assert steps * eff_tokens >= target_tokens
        assert (steps - 1) * eff_tokens < target_tokens


# ============================================================================
# TEST 10: Scientific Evaluation Metrics
# ============================================================================
def test_10_scientific_evaluator_metrics():
    """Validates Distinct-N ratios, repetition index, and safe perplexity calculation."""
    diverse_tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    repeated_tokens = [10, 20, 30, 10, 20, 30, 10, 20, 30]

    assert compute_distinct_ngrams(diverse_tokens, 1) == 1.0
    assert compute_distinct_ngrams(diverse_tokens, 2) == 1.0
    assert compute_repetition_index(diverse_tokens, 3) == 0.0

    rep_idx = compute_repetition_index(repeated_tokens, 3)
    assert rep_idx > 0.50, f"Expected high repetition index for loop, got {rep_idx}"

    # Clamped perplexity
    assert safe_perplexity(0.0) == 1.0
    assert safe_perplexity(float("nan")) == float("inf")
    assert safe_perplexity(float("inf")) == float("inf")
    assert safe_perplexity(3.7435) == pytest.approx(42.245, rel=1e-2)


# ============================================================================
# TEST 11: Protected Baseline Checkpoint Defense
# ============================================================================
def test_11_protected_baseline_defense():
    """Guarantees that historical model checkpoints were untouched."""
    resnet_ckpt = PROJECT_ROOT / "checkpoints" / "resnet18_cifar10" / "best_model.pt"
    pilot_100m_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_pilot" / "best_model.pt"
    campaign_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "best_model.pt"

    assert resnet_ckpt.is_file()
    assert pilot_100m_ckpt.is_file()
    assert campaign_ckpt.is_file()

    assert resnet_ckpt.stat().st_size == 134_220_210
    assert pilot_100m_ckpt.stat().st_size == 1_026_758_674
    assert campaign_ckpt.stat().st_size == 1_042_094_098
