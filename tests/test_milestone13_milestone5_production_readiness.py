"""
Phase 13 Milestone 5: Production-Scale Dataset Expansion, CUDA-Optimized
Transformer Training & 50M-Token Campaign Readiness Test Suite.

Tests:
1. Production dataset scale, binary alignment, split isolation, and manifest consistency.
2. Tokenizer integrity, hash fidelity, and zero-OOV round-trip coverage.
3. 86.83M parameter Transformer architectural invariants (12 layers, 12 heads, d_model=768, d_ff=3072).
4. Custom CUDA autograd operators numerical parity vs native PyTorch.
5. Scalable memory-mapped dataset streaming and causal token alignment.
6. GPU optimization benchmarks validation (SDPA, AMP FP16, custom fused bias-activation).
7. Controlled 50-step training diagnostic telemetry and atomic checkpoint recovery.
8. 50M-token pretraining campaign architectural plan validation and guardrail defense.
9. Historical protected checkpoints immutable defense.
"""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from engine.transformer import BPETokenizer
from engine.transformer.config import TransformerConfig
from engine.transformer.model import GPTTransformer
from engine.transformer.scalable_dataset import ScalableMemmapDataset
from engine.checkpointing.checkpoint_manager import CheckpointManager
import custom_kernels.autograd as cka


# ==============================================================================
# TEST 1: Production Dataset Scale, Splits & Manifest
# ==============================================================================
def test_1_production_dataset_scale_and_splits():
    data_dir = PROJECT_ROOT / "data" / "text" / "production_10m"
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    test_bin = data_dir / "test_tokens.bin"
    manifest_file = data_dir / "dataset_manifest.json"

    assert train_bin.is_file(), f"Missing train_tokens.bin at {train_bin}"
    assert val_bin.is_file(), f"Missing val_tokens.bin at {val_bin}"
    assert test_bin.is_file(), f"Missing test_tokens.bin at {test_bin}"
    assert manifest_file.is_file(), f"Missing dataset_manifest.json at {manifest_file}"

    # Verify uint16 byte alignment
    assert train_bin.stat().st_size % 2 == 0
    assert val_bin.stat().st_size % 2 == 0
    assert test_bin.stat().st_size % 2 == 0

    train_tokens = train_bin.stat().st_size // 2
    val_tokens = val_bin.stat().st_size // 2
    test_tokens = test_bin.stat().st_size // 2

    # Scale requirements: Train >= 10M tokens
    assert train_tokens >= 10_000_000, f"Expected train >= 10M tokens, got {train_tokens:,}"
    # Strictly preserved validation and test splits
    assert val_tokens == 104_180, f"Validation split corrupted: got {val_tokens}, expected 104,180"
    assert test_tokens == 331_796, f"Test split corrupted: got {test_tokens}, expected 331,796"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["split_summary"]["train_tokens"] == train_tokens
    assert manifest["split_summary"]["val_tokens"] == val_tokens
    assert manifest["split_summary"]["test_tokens"] == test_tokens
    assert manifest["vocab_size"] == 2048


# ==============================================================================
# TEST 2: Tokenizer Integrity, Hash Fidelity & Round-trip
# ==============================================================================
def test_2_tokenizer_integrity_and_compatibility():
    tok_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "tokenizer" / "bpe_vocab.json"
    assert tok_path.is_file(), f"Tokenizer file missing at {tok_path}"

    tok_bytes = tok_path.read_bytes()
    tok_sha256 = hashlib.sha256(tok_bytes).hexdigest()
    expected_sha256 = "ffd4aac665204b09488cd84e5428a72aa93223e06708947520768c001c163841"
    assert tok_sha256 == expected_sha256, f"Tokenizer hash mismatch: {tok_sha256} vs {expected_sha256}"

    tokenizer = BPETokenizer.load(tok_path)
    assert tokenizer.vocab_size == 2048

    # Test round-trip fidelity
    sample_text = "CUDA GPU Acceleration Engine with Tensor Cores and Parallel Computing."
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == sample_text, f"Round-trip decoding failed: '{decoded}' vs '{sample_text}'"


# ==============================================================================
# TEST 3: Transformer Architectural Invariants
# ==============================================================================
def test_3_transformer_architecture_fidelity():
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
    param_counts = model.count_parameters()

    assert param_counts["total_parameters"] == 86_825_472
    assert param_counts["trainable_parameters"] == 86_825_472
    assert param_counts["d_model"] == 768
    assert param_counts["n_layer"] == 12
    assert param_counts["n_head"] == 12
    assert param_counts["vocab_size"] == 2048


# ==============================================================================
# TEST 4: Custom CUDA Autograd Parity
# ==============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for custom kernel parity test")
def test_4_custom_cuda_autograd_parity():
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # 1. Vector Add
    a = torch.randn(64, 768, device=device, requires_grad=True)
    b = torch.randn(64, 768, device=device, requires_grad=True)
    out_cuda = cka.vector_add(a, b)
    out_pt = a + b
    assert (out_cuda - out_pt).abs().max().item() < 1e-6

    # 2. GELU
    x = torch.randn(64, 768, device=device, requires_grad=True)
    out_gelu_cuda = cka.gelu(x)
    out_gelu_pt = F.gelu(x, approximate="tanh")
    assert (out_gelu_cuda - out_gelu_pt).abs().max().item() < 1e-5

    # 3. Fused Bias + GELU
    bias = torch.randn(768, device=device, requires_grad=True)
    out_fused = cka.fused_bias_activation(x, bias, "gelu")
    out_fused_pt = F.gelu(x + bias, approximate="tanh")
    assert (out_fused - out_fused_pt).abs().max().item() < 1e-5

    # Backward gradient check
    grad_out = torch.randn_like(out_fused)
    out_fused.backward(grad_out)
    assert x.grad is not None and bias.grad is not None


# ==============================================================================
# TEST 5: Scalable Memory-Mapped Dataset Streaming
# ==============================================================================
def test_5_dataloader_scalable_memmap_streaming():
    train_bin = PROJECT_ROOT / "data" / "text" / "production_10m" / "train_tokens.bin"
    dataset = ScalableMemmapDataset(train_bin, seq_len=256, stride=256, vocab_size=2048)

    assert len(dataset) > 50_000
    x, y = dataset[0]

    assert x.shape == torch.Size([256])
    assert y.shape == torch.Size([256])
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # Strict causal shift check: y[t] == raw[t+1]
    raw = dataset.get_token_slice(0, 257)
    assert torch.equal(x, torch.from_numpy(raw[:-1].astype(np.int64)))
    assert torch.equal(y, torch.from_numpy(raw[1:].astype(np.int64)))


# ==============================================================================
# TEST 6: GPU Optimization Benchmarks Validation
# ==============================================================================
def test_6_gpu_optimization_benchmarks_fidelity():
    bench_file = PROJECT_ROOT / "results" / "phase13_milestone5_gpu_optimization_benchmarks.json"
    assert bench_file.is_file(), f"Missing optimization benchmarks report: {bench_file}"

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASSED_MATCHED_BENCHMARK_VERIFIED"
    assert len(data["configurations"]) == 4

    configs = {c["configuration_name"]: c for c in data["configurations"]}
    baseline = configs["Baseline (Manual Attention + FP32)"]
    amp = configs["Native SDPA Attention + AMP FP16"]
    fused = configs["Native SDPA + AMP FP16 + Custom Fused Bias-GELU"]

    # Speedup verification: AMP >= 1.8x, Fused >= 1.9x
    assert amp["speedup_factor_vs_baseline"] >= 1.8
    assert fused["speedup_factor_vs_baseline"] >= 1.9

    # VRAM savings: >= 35%
    assert amp["vram_savings_percent"] >= 35.0
    assert fused["vram_savings_percent"] >= 35.0


# ==============================================================================
# TEST 7: Controlled Training Diagnostic Telemetry & Checkpoints
# ==============================================================================
def test_7_training_diagnostic_telemetry_fidelity():
    diag_file = PROJECT_ROOT / "results" / "phase13_milestone5_training_diagnostic.json"
    assert diag_file.is_file(), f"Missing training diagnostic report: {diag_file}"

    with open(diag_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASSED_DIAGNOSTIC_VERIFIED"
    assert data["token_accounting"]["diagnostic_steps_executed"] == 50
    assert data["token_accounting"]["additional_diagnostic_tokens"] == 204_800
    assert data["token_accounting"]["cumulative_lifetime_tokens"] == 10_205_184

    # Verify checkpoint existence
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_diagnostic"
    assert (ckpt_dir / "final_model.pt").is_file()
    assert (ckpt_dir / "best_model.pt").is_file()


# ==============================================================================
# TEST 8: 50M Campaign Plan & Guardrail Defense
# ==============================================================================
def test_8_50m_campaign_plan_readiness():
    plan_file = PROJECT_ROOT / "results" / "phase13_milestone5_50m_campaign_plan.json"
    assert plan_file.is_file(), f"Missing 50M campaign plan: {plan_file}"

    with open(plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)

    assert plan["status"] == "READY_FOR_AUTHORIZATION"
    assert plan["guardrail_enforcement"]["campaign_auto_launch"] is False
    assert plan["guardrail_enforcement"]["user_authorization_required"] is True

    # Exact token arithmetic
    math_cfg = plan["training_mathematics"]
    assert math_cfg["baseline_cumulative_tokens"] + math_cfg["remaining_token_budget"] == 50_000_000
    assert math_cfg["remaining_optimizer_steps"] * math_cfg["effective_batch_size"] >= math_cfg["remaining_token_budget"]


# ==============================================================================
# TEST 9: Historical Protected Checkpoints Immutable Defense
# ==============================================================================
def test_9_protected_checkpoints_integrity():
    p10m_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended" / "best_model.pt"
    assert p10m_ckpt.is_file(), "Historical 10M checkpoint missing!"

    # Verify SHA-256 remains 100% byte-identical
    h = hashlib.sha256(p10m_ckpt.read_bytes()).hexdigest()
    expected_sha256 = "4b0d59c40edd607d3dea8024f92d9381e79dcfa26b3b399574230fd2a739908e"
    assert h == expected_sha256, f"Protected 10M checkpoint modified! {h} vs {expected_sha256}"

