"""
Phase 13 Milestone 4 Test Suite:
Scientific Generalization Audit, GPU Training Optimization & 50M-Token Pretraining Readiness.

Validates:
1. Checkpoint integrity audit and hash consistency.
2. Held-out test split evaluation on unexposed Newton Opticks corpus (PPL < 25.0, Top-1 Accuracy > 40%).
3. Generation lexical diversity (Distinct-2 > 0.80) and degenerate loop elimination (0 / 10).
4. Memorization analysis confirming zero verbatim text regurgitations.
5. GPU optimization benchmarks: SDPA attention speedup (> 1.20x) and numerical parity.
6. Precision benchmarking: AMP FP16 speedup (> 1.50x) over FP32 baseline.
7. Custom CUDA kernel acceleration: Fused Bias+Activation speedup (> 1.50x) over PyTorch separate.
8. Comparative Training Laboratory autograd parity (GELU & Linear forward/backward gradients).
9. 50M-token dataset readiness assessment and 3 documented training configurations.
10. Backend ModelServer serving compatibility with audited 10M checkpoint.
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
def audit_report():
    p = PROJECT_ROOT / "results" / "phase13_milestone4_checkpoint_audit.json"
    assert p.is_file(), f"Audit file missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_eval_report():
    p = PROJECT_ROOT / "results" / "phase13_milestone4_test_evaluation.json"
    assert p.is_file(), f"Test eval file missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def gen_memorization_report():
    p = PROJECT_ROOT / "results" / "phase13_milestone4_generation_memorization.json"
    assert p.is_file(), f"Generation file missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def gpu_optimization_report():
    p = PROJECT_ROOT / "results" / "phase13_milestone4_gpu_optimization.json"
    assert p.is_file(), f"Optimization file missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dataset_readiness_report():
    p = PROJECT_ROOT / "results" / "phase13_milestone4_dataset_readiness.json"
    assert p.is_file(), f"Dataset readiness file missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def comparative_lab_report():
    p = PROJECT_ROOT / "results" / "phase13_milestone4_comparative_lab_audit.json"
    assert p.is_file(), f"Comparative lab file missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def test_01_checkpoint_integrity_audit(audit_report):
    """Verify Stage 1 checkpoint audit certified all 10 integrity items."""
    assert audit_report["audit_status"] == "ALL_10_CHECKS_VERIFIED"
    arch = audit_report["architecture"]
    assert arch["parameters"] == 86825472
    assert arch["non_embedding_parameters"] == 85056000
    assert arch["tied_weights"] is True

    best_ckpt = audit_report["checkpoints"]["best_model"]
    assert best_ckpt["cumulative_tokens"] >= 10000000
    assert best_ckpt["sha256"] == "4b0d59c40edd607d3dea8024f92d9381e79dcfa26b3b399574230fd2a739908e"
    assert audit_report["cpu_gpu_numerical_parity_max_diff"] < 1e-4


def test_02_held_out_test_evaluation(test_eval_report):
    """Verify Stage 2 held-out test evaluation on untouched Newton Opticks corpus."""
    models = test_eval_report["models"]
    assert len(models) == 3

    m3 = models[2]
    assert m3["label"] == "Phase 13 M3 Extended (10.0M Tokens)"
    assert m3["test_loss"] < 3.20
    assert m3["test_perplexity"] < 25.0
    assert m3["top1_token_accuracy_pct"] > 40.0
    assert m3["test_throughput_tok_per_sec"] > 20000.0

    comparison = test_eval_report["held_out_comparison"]
    assert comparison["test_loss_reduction"] < -1.0
    assert comparison["test_perplexity_ratio"] > 2.5


def test_03_generation_diversity_and_loop_elimination(gen_memorization_report):
    """Verify Stage 3 generation quality: high diversity and zero degenerate loops."""
    models = gen_memorization_report["models_evaluated"]
    assert len(models) == 3

    m3 = models[2]
    agg = m3["aggregate_metrics"]
    assert agg["mean_distinct_1"] > 0.50
    assert agg["mean_distinct_2"] > 0.80
    assert agg["mean_distinct_3"] > 0.90
    assert agg["mean_repetition_index"] < 0.15
    assert agg["degenerate_loops_count"] == 0


def test_04_memorization_absence_verification(gen_memorization_report):
    """Verify Stage 3 memorization analysis: zero verbatim regurgitation cases."""
    models = gen_memorization_report["models_evaluated"]
    m3 = models[2]
    agg = m3["aggregate_metrics"]
    assert agg["verbatim_regurgitations_count"] == 0
    assert agg["mean_longest_overlap_tokens"] < 15.0


def test_05_sdpa_attention_optimization(gpu_optimization_report):
    """Verify Stage 4 PyTorch SDPA attention delivers speedup over manual attention."""
    bench = gpu_optimization_report["benchmarks"]["attention_mechanism"]
    assert bench["speedup_ratio"] > 1.20
    assert bench["numerical_parity"] is True
    assert bench["max_absolute_diff"] < 1e-3


def test_06_precision_policy_benchmarks(gpu_optimization_report):
    """Verify Stage 4 AMP FP16 and BF16 speedups over native FP32 baseline."""
    bench = gpu_optimization_report["benchmarks"]["precision_policies"]
    assert bench["amp_fp16_speedup"] > 1.50
    assert bench["amp_bf16_speedup"] > 1.50
    assert bench["amp_fp16_peak_vram_mb"] < bench["fp32_peak_vram_mb"]


def test_07_custom_cuda_fused_bias_activation(gpu_optimization_report):
    """Verify Stage 4 Custom CUDA Fused Bias+Activation achieves speedup over PyTorch separate."""
    bench = gpu_optimization_report["benchmarks"]["custom_fused_bias_activation"]
    assert bench["speedup_ratio"] > 1.20
    assert bench["numerical_parity"] is True
    assert bench["max_absolute_diff"] < 1e-5


def test_08_dataset_readiness_and_configurations(dataset_readiness_report):
    """Verify Stage 5 dataset readiness assessment and 3 defensible configurations."""
    verdict = dataset_readiness_report["readiness_verdict"]["verdict"]
    assert verdict == "CONDITIONAL_APPROVAL_UPON_EXPANSION"

    configs = dataset_readiness_report["configurations"]
    assert len(configs) >= 3
    assert configs[0]["config_id"] == "config_a_balanced_multidomain"
    assert configs[0]["target_training_tokens"] == 50000000
    assert configs[0]["effective_corpus_epochs"] == 5.0


def test_09_comparative_laboratory_autograd_parity(comparative_lab_report):
    """Verify Stage 6 Comparative Training Laboratory autograd numerical parity."""
    parity = comparative_lab_report["parity_verification"]
    assert parity["numerical_parity_status"] == "VERIFIED_BITWISE_ACCURATE"
    assert parity["gelu_fwd_max_diff"] < 1e-5
    assert parity["gelu_bwd_max_diff"] < 1e-5
    assert parity["linear_weight_grad_max_diff"] < 1e-4

    inventory = comparative_lab_report["kernel_inventory"]
    assert len(inventory) >= 6
    ready_kernels = [k["name"] for k in inventory if k["training_status"] == "FULLY_READY"]
    assert "GELU Activation" in ready_kernels
    assert "Fused Bias + Activation" in ready_kernels
    assert "Vector Add (Residual Add)" in ready_kernels


def test_10_model_server_compatibility():
    """Verify ModelServer loads the audited 10M checkpoint and generates completions."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended" / "best_model.pt"
    assert ckpt_path.is_file()

    load_res = ModelServer.load_model(
        model_name="NEXORA GPT-100M",
        checkpoint_path=str(ckpt_path),
        device_str="cpu",
    )
    assert load_res["status"] == "loaded"

    gen_res = ModelServer.generate(
        prompt="In the beginning",
        max_new_tokens=15,
        temperature=0.7,
        device_str="cpu",
    )
    assert gen_res["status"] == "success"
    assert len(gen_res["generated_text"]) > len("In the beginning")

    ModelServer.unload_model()
