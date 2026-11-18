"""
Unit and Integration Tests for Phase 13 Milestone 8: Adaptive Kernel Selection Engine
Tests deterministic fallback, hardware-informed selection, caching, invalidation, and layer swapping.
"""

import threading
import pytest
import torch
import torch.nn as nn

from engine.transformer.config import TransformerConfig
from engine.transformer.model import GPTTransformer, TransformerMLP
from engine.transformer.adaptive_engine import (
    AdaptiveExecutionEngine,
    ExecutionPath,
    EngineWorkloadType,
    HardwareProfile,
    SelectionDecision,
    CustomFusedTransformerMLP,
)


@pytest.fixture
def small_config():
    return TransformerConfig(
        vocab_size=256,
        d_model=64,
        n_layer=2,
        n_head=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def small_model(small_config):
    return GPTTransformer(small_config)


def test_adaptive_engine_singleton():
    engine1 = AdaptiveExecutionEngine()
    engine2 = AdaptiveExecutionEngine()
    assert engine1 is engine2


def test_hardware_profile_detection():
    engine = AdaptiveExecutionEngine()
    hw = engine.get_hardware_profile()
    assert isinstance(hw, HardwareProfile)
    if torch.cuda.is_available():
        assert hw.sm_count > 0
        assert hw.total_vram_mb > 0
        assert hw.major >= 5
    else:
        assert hw.device_name == "CPU"


def test_default_training_selection_amp():
    engine = AdaptiveExecutionEngine()
    engine.invalidate_cache()
    decision = engine.select_execution_path(
        workload=EngineWorkloadType.TRAINING,
        precision="amp_fp16",
        batch_size=16,
        sequence_length=256,
    )
    assert isinstance(decision, SelectionDecision)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        assert decision.selected_path == ExecutionPath.OPTIMIZED_PYTORCH
        assert decision.use_sdpa is True
        assert decision.use_amp is True
        assert decision.use_custom_fused is False
        assert "cuBLASLt" in decision.rationale or "Optimized PyTorch" in decision.rationale


def test_fp32_selection_standard():
    engine = AdaptiveExecutionEngine()
    decision = engine.select_execution_path(
        workload=EngineWorkloadType.TRAINING,
        precision="fp32",
        batch_size=8,
        sequence_length=128,
    )
    assert decision.selected_path == ExecutionPath.STANDARD_PYTORCH
    assert decision.use_sdpa is False
    assert decision.use_amp is False
    assert decision.use_custom_fused is False


def test_custom_cuda_force_selection():
    engine = AdaptiveExecutionEngine()
    decision = engine.select_execution_path(
        workload=EngineWorkloadType.TRAINING,
        precision="amp_fp16",
        force_path=ExecutionPath.CUSTOM_CUDA,
    )
    if torch.cuda.is_available():
        assert decision.selected_path == ExecutionPath.CUSTOM_CUDA
        assert decision.use_custom_fused is True
        assert decision.use_sdpa is True


def test_caching_and_invalidation():
    engine = AdaptiveExecutionEngine()
    engine.invalidate_cache()
    stats_init = engine.get_cache_stats()
    assert stats_init["cached_decisions_count"] == 0

    d1 = engine.select_execution_path(workload=EngineWorkloadType.TRAINING, precision="amp_fp16", batch_size=4, sequence_length=64)
    assert d1.cache_hit is False

    d2 = engine.select_execution_path(workload=EngineWorkloadType.TRAINING, precision="amp_fp16", batch_size=4, sequence_length=64)
    assert d2.cache_hit is True
    assert d2.selected_path == d1.selected_path

    invalidated = engine.invalidate_cache()
    assert invalidated >= 1
    stats_after = engine.get_cache_stats()
    assert stats_after["cached_decisions_count"] == 0


def test_apply_execution_path_swap_and_restore(small_model):
    engine = AdaptiveExecutionEngine()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    small_model = small_model.to(device)

    # Initially standard MLP
    for block in small_model.blocks:
        assert isinstance(block.mlp, TransformerMLP)

    # Apply CUSTOM_CUDA
    decision_custom = SelectionDecision(
        selected_path=ExecutionPath.CUSTOM_CUDA,
        workload_type=EngineWorkloadType.TRAINING,
        precision="amp_fp16",
        use_sdpa=True,
        use_amp=True,
        use_custom_fused=True,
        compiler_backend=None,
        rationale="Unit test custom swap",
        fallback_reason=None,
        benchmark_expected_latency_ms=317.46,
        confidence_score=1.0,
        cache_hit=False,
        cache_key="test_custom",
    )
    small_model = engine.apply_execution_path(small_model, decision_custom)
    for block in small_model.blocks:
        assert isinstance(block.mlp, CustomFusedTransformerMLP)

    # Revert to OPTIMIZED_PYTORCH
    decision_opt = SelectionDecision(
        selected_path=ExecutionPath.OPTIMIZED_PYTORCH,
        workload_type=EngineWorkloadType.TRAINING,
        precision="amp_fp16",
        use_sdpa=True,
        use_amp=True,
        use_custom_fused=False,
        compiler_backend=None,
        rationale="Unit test revert",
        fallback_reason=None,
        benchmark_expected_latency_ms=316.08,
        confidence_score=1.0,
        cache_hit=False,
        cache_key="test_opt",
    )
    small_model = engine.apply_execution_path(small_model, decision_opt)
    for block in small_model.blocks:
        assert isinstance(block.mlp, TransformerMLP)


def test_status_summary():
    engine = AdaptiveExecutionEngine()
    summary = engine.get_status_summary()
    assert summary["engine_status"] == "ONLINE"
    assert "hardware" in summary
    assert "available_paths" in summary
    assert "empirical_benchmarks_ms" in summary
    assert "cache_stats" in summary


def test_multithreaded_concurrency():
    engine = AdaptiveExecutionEngine()
    results = []

    def worker(idx):
        for _ in range(20):
            d = engine.select_execution_path(
                workload=EngineWorkloadType.TRAINING,
                precision="amp_fp16" if idx % 2 == 0 else "fp32",
                batch_size=8 + (idx % 4),
                sequence_length=128,
            )
            results.append(d.selected_path)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 160


def test_api_laboratory_configurations_includes_adaptive():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from backend.app import create_app
    app = create_app(Path(__file__).resolve().parents[1])
    client = TestClient(app)

    res = client.get("/api/laboratory/configurations")
    assert res.status_code == 200
    data = res.json()
    assert data["total_configurations"] >= 4
    config_ids = [c["id"] for c in data["configurations"]]
    assert "config_d" in config_ids


def test_api_laboratory_adaptive_status_endpoint():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from backend.app import create_app
    app = create_app(Path(__file__).resolve().parents[1])
    client = TestClient(app)

    res = client.get("/api/laboratory/adaptive-status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert "hardware" in data
    assert "available_paths" in data
    assert len(data["available_paths"]) == 4
    assert data["numerical_certification_status"] in ["CERTIFIED", "PENDING_VERIFICATION"]


def test_api_laboratory_adaptive_select_endpoint():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from backend.app import create_app
    app = create_app(Path(__file__).resolve().parents[1])
    client = TestClient(app)

    payload = {
        "workload": "training",
        "precision": "amp_fp16",
        "batch_size": 16,
        "sequence_length": 256,
    }
    res = client.post("/api/laboratory/adaptive-select", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "decision" in data
    dec = data["decision"]
    assert dec["selected_path"] in ["optimized_pytorch", "standard_pytorch", "custom_cuda"]
    assert "path_name" in dec and isinstance(dec["path_name"], str)
    assert "execution_mode" in dec and dec["execution_mode"] in ["GPU_NATIVE", "GPU_CUSTOM", "COMPILER_ASSISTED", "CPU_NATIVE"]
    assert "kernel_dispatch" in dec and isinstance(dec["kernel_dispatch"], str)
    assert "empirical_throughput_tokens_per_sec" in dec and isinstance(dec["empirical_throughput_tokens_per_sec"], (int, float))
    assert dec["throughput_is_estimated"] is True
    assert "empirical_vram_mb" in dec and isinstance(dec["empirical_vram_mb"], (int, float))
    assert "numerical_parity_verified" in dec and isinstance(dec["numerical_parity_verified"], bool)
    assert dec["numerical_parity_verified"] is True
    assert "rationale" in dec and len(dec["rationale"]) > 0


def test_api_laboratory_adaptive_select_inference_contract():
    from pathlib import Path
    from fastapi.testclient import TestClient
    from backend.app import create_app
    app = create_app(Path(__file__).resolve().parents[1])
    client = TestClient(app)

    payload = {
        "workload": "inference",
        "precision": "amp_fp16",
        "batch_size": 4,
        "sequence_length": 128,
    }
    res = client.post("/api/laboratory/adaptive-select", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    dec = data["decision"]
    assert dec["workload_type"] == "inference"
    assert dec["empirical_throughput_tokens_per_sec"] > 0
    assert dec["empirical_vram_mb"] > 0

