"""
tests/test_milestone13_milestone7_comparative_lab.py
Phase 13 Milestone 7 Regression Test Suite:
Validates artifact audit, text generation capabilities, comparative training laboratory,
custom CUDA performance investigation, and FastAPI laboratory backend endpoints.
"""

import json
from pathlib import Path
import pytest
import torch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from backend.app import create_app
from backend.serving import ModelServer
from engine.training.orchestrator import TrainingOrchestrator


@pytest.fixture(scope="module")
def api_client():
    app = create_app(PROJECT_ROOT)
    return TestClient(app)


def test_milestone6_artifacts_and_checkpoint_hashes():
    """Verify Milestone 6 checkpoints exist, hashes match audit records, and params equal 86,825,472."""
    audit_file = PROJECT_ROOT / "results" / "phase13_milestone7_artifact_audit.json"
    assert audit_file.is_file(), "phase13_milestone7_artifact_audit.json must exist"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASSED", f"Artifact audit failed: {data}"
    checks = data["checks"]
    assert checks["final_model_exists"] is True
    assert checks["best_model_exists"] is True
    assert checks["param_count_exactly_86825472"] is True
    assert checks["final_step_is_12208"] is True
    assert checks["cumulative_tokens_is_50001920"] is True
    assert checks["best_step_is_12000"] is True
    assert checks["optimizer_state_recoverable"] is True
    assert checks["scheduler_state_recoverable"] is True
    assert checks["scaler_state_recoverable"] is True
    assert checks["final_hash_matches_manifest"] is True
    assert checks["best_hash_matches_manifest"] is True


def test_historical_checkpoints_preserved():
    """Verify historical checkpoints remain intact and were not overwritten."""
    audit_file = PROJECT_ROOT / "results" / "phase13_milestone7_artifact_audit.json"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    historical = data.get("historical_integrity", {})
    assert len(historical) >= 4, "Must track at least 4 historical checkpoints"
    for name, info in historical.items():
        assert info["exists"] is True, f"Historical checkpoint {name} missing"
        assert info["size_bytes"] > 0, f"Historical checkpoint {name} is empty"


def test_text_generation_capabilities_and_diversity():
    """Verify comprehensive model capability evaluation across 4 sampling regimes."""
    gen_file = PROJECT_ROOT / "results" / "phase13_milestone7_generation_capabilities.json"
    assert gen_file.is_file(), "phase13_milestone7_generation_capabilities.json must exist"

    with open(gen_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["stage"] == "Stage 2 - Model Capability Evaluation"
    overall = data["overall_metrics"]
    assert overall["total_evaluations"] == 44
    assert overall["mean_throughput_tokens_per_sec"] > 100.0
    assert overall["mean_latency_per_token_ms"] < 15.0

    regimes = data["regime_summaries"]
    assert "Greedy (T=0.0)" in regimes
    assert "Balanced (T=0.7)" in regimes
    assert "Exploratory (T=1.0)" in regimes

    # Balanced regime should achieve high diversity and low repetition
    balanced = regimes["Balanced (T=0.7)"]
    assert balanced["mean_distinct_2"] > 0.70
    assert balanced["mean_repetition_index"] < 0.25

    # Exploratory regime reaches maximum unigram/bigram diversity
    exploratory = regimes["Exploratory (T=1.0)"]
    assert exploratory["mean_distinct_2"] > 0.80
    assert exploratory["mean_repetition_index"] < 0.15

    # Text completion vs instruction alignment documented
    insights = data["scientific_insights"]
    assert "text_completion_vs_instruction_alignment" in insights
    assert len(insights["text_completion_vs_instruction_alignment"]) > 50


def test_comparative_training_laboratory_results():
    """Verify comparative training lab results for Config A, B, and C."""
    lab_file = PROJECT_ROOT / "results" / "phase13_milestone7_comparative_lab.json"
    assert lab_file.is_file(), "phase13_milestone7_comparative_lab.json must exist"

    with open(lab_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    configs = {c["configuration_name"]: c for c in data["configurations"]}
    assert len(configs) == 3

    # Config A: Standard PyTorch FP32
    cfg_a = next(c for name, c in configs.items() if "Configuration A" in name)
    assert cfg_a["use_sdpa"] is False
    assert cfg_a["use_amp"] is False
    assert cfg_a["use_custom_fused"] is False

    # Config B: Optimized PyTorch SDPA + AMP FP16
    cfg_b = next(c for name, c in configs.items() if "Configuration B" in name)
    assert cfg_b["use_sdpa"] is True
    assert cfg_b["use_amp"] is True
    assert cfg_b["use_custom_fused"] is False
    assert cfg_b["speedup_factor"] >= 2.0
    assert cfg_b["vram_savings_pct"] >= 40.0

    # Config C: Optimized PyTorch + Custom CUDA
    cfg_c = next(c for name, c in configs.items() if "Configuration C" in name)
    assert cfg_c["use_sdpa"] is True
    assert cfg_c["use_amp"] is True
    assert cfg_c["use_custom_fused"] is True
    assert cfg_c["speedup_factor"] >= 2.0

    # Verify ~0.5% differential between C and B
    delta_pct = data["comparative_metrics"]["config_c_vs_config_b_delta_pct"]
    assert 0.1 <= delta_pct <= 1.0, f"Expected delta in ~0.5% range, got {delta_pct}%"


def test_numerical_parity_and_erf_tanh_analysis():
    """Verify numerical parity between native MLP and custom CUDA fused bias+GELU."""
    lab_file = PROJECT_ROOT / "results" / "phase13_milestone7_comparative_lab.json"
    with open(lab_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    parity = data.get("numerical_parity", {})
    assert "mlp_diff_vs_pytorch_tanh" in parity
    assert parity["mlp_diff_vs_pytorch_tanh"] < 1.0e-5, (
        f"Custom kernel should match PyTorch tanh GELU to within single-precision machine epsilon, got {parity['mlp_diff_vs_pytorch_tanh']}"
    )


def test_custom_cuda_kernel_bottleneck_profiling():
    """Verify deep-dive profiling metrics for CUDA kernel bottlenecks."""
    prof_file = PROJECT_ROOT / "results" / "phase13_milestone7_kernel_bottleneck_analysis.json"
    assert prof_file.is_file(), "phase13_milestone7_kernel_bottleneck_analysis.json must exist"

    with open(prof_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    bw = data["memory_bandwidth_analysis"]
    assert bw["forward_bandwidth_gb_s"] > 90.0, f"Forward bandwidth should exceed 90 GB/s, got {bw['forward_bandwidth_gb_s']}"
    assert bw["backward_bandwidth_gb_s"] > 80.0, f"Backward bandwidth should exceed 80 GB/s, got {bw['backward_bandwidth_gb_s']}"

    recon = data["layer_step_reconciliation"]
    assert recon["num_layers"] == 12
    assert "py_autograd_fwd_overhead_per_layer_ms" in recon
    assert "unfused_gemm_penalty_ms" in recon


def test_laboratory_backend_configurations_api(api_client):
    """Verify GET /api/laboratory/configurations endpoint."""
    res = api_client.get("/api/laboratory/configurations")
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "available"
    assert payload["total_configurations"] >= 3
    ids = [c["id"] for c in payload["configurations"]]
    assert "config_a" in ids
    assert "config_b" in ids
    assert "config_c" in ids


def test_laboratory_backend_gpu_monitoring_and_locking(api_client):
    """Verify GET /api/laboratory/gpu-monitoring and POST /api/laboratory/experiments locking."""
    # 1. Monitoring state
    res = api_client.get("/api/laboratory/gpu-monitoring")
    assert res.status_code == 200
    mon = res.json()
    assert "device_name" in mon
    assert "vram_total_mb" in mon
    assert "can_launch_experiment" in mon

    # 2. Concurrency rejection when lock active
    orch = TrainingOrchestrator.get_instance()
    orch.active_gpu_job_id = "test_contention_lock_01"
    try:
        lock_res = api_client.post(
            "/api/laboratory/experiments",
            json={"name": "Colliding Job", "steps": 10},
        )
        assert lock_res.status_code == 409
        assert "actively reserved" in lock_res.json()["detail"]
    finally:
        orch.active_gpu_job_id = None

    # 3. Successful experiment creation when free
    ok_res = api_client.post(
        "/api/laboratory/experiments",
        json={"name": "Verified Run", "steps": 25},
    )
    assert ok_res.status_code == 201
    assert ok_res.json()["status"] == "created"


def test_laboratory_backend_evaluations_and_checkpoints_api(api_client):
    """Verify GET /api/laboratory/evaluations and GET /api/laboratory/checkpoints."""
    # Checkpoints
    res_ck = api_client.get("/api/laboratory/checkpoints")
    assert res_ck.status_code == 200
    ck_data = res_ck.json()
    assert ck_data["total_checkpoints"] >= 2
    filenames = [c["filename"] for c in ck_data["checkpoints"]]
    assert "best_model.pt" in filenames
    assert "final_model.pt" in filenames

    # Evaluations
    res_ev = api_client.get("/api/laboratory/evaluations")
    assert res_ev.status_code == 200
    ev_data = res_ev.json()
    assert ev_data["sample_count"] == 44
    assert "regime_summaries" in ev_data
    assert "scientific_insights" in ev_data


def test_model_server_inference_serving():
    """Verify ModelServer loads best_model.pt and generates autoregressive text."""
    ckpt_path = "checkpoints/transformer/pretraining_50m_campaign/best_model.pt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    load_res = ModelServer.load_model(
        model_name="NEXORA GPT-100M",
        checkpoint_path=ckpt_path,
        device_str=device,
    )
    assert load_res["status"] == "loaded"
    assert load_res["metrics"]["step"] == 12000

    gen_res = ModelServer.generate(
        prompt="The Time Machine",
        max_new_tokens=15,
        temperature=0.7,
        device_str=device,
    )
    assert "generated_text" in gen_res
    assert len(gen_res["generated_text"]) > 0

    unload_res = ModelServer.unload_model()
    assert unload_res["status"] == "unloaded"

