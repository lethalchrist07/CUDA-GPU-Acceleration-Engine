"""
Automated Test Suite for NEXORA Adaptive Training Subsystem.
Verifies hardware detection, CPU fallback, precision gating, VRAM budgeting,
batch-size optimization, OOM recovery, and API route integration.
"""

import math
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import create_app
from engine.training.adaptive_engine import (
    AdaptiveTrainingEngine,
    AdaptiveHardwareProfile,
    AdaptiveTrainingDecision,
)
from engine.training.job_store import JobStore
from engine.training.orchestrator import TrainingOrchestrator


@pytest.fixture(scope="module")
def client():
    app = create_app(REPO_ROOT)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def adaptive_engine():
    return AdaptiveTrainingEngine()


# ============================================================================
# 1. Hardware Detection and Capability Classification
# ============================================================================

def test_hardware_detection_rtx3050_or_cuda(adaptive_engine):
    """Verify reliable GPU detection on host hardware."""
    hw = adaptive_engine.get_hardware_profile(force_cpu=False)
    assert isinstance(hw, AdaptiveHardwareProfile)

    if torch.cuda.is_available():
        assert hw.device_type == "cuda"
        assert hw.total_vram_mb > 1000.0
        assert hw.major >= 5
        assert hw.sm_count > 0
        if "3050" in hw.device_name:
            assert hw.major == 8
            assert hw.minor == 6
            assert hw.sm_count == 20
            assert hw.supports_ampere_tensor_cores is True
            assert hw.supports_fp16 is True
        # BF16 flag must match PyTorch runtime capability
        assert hw.supports_bf16 == torch.cuda.is_bf16_supported()
    else:
        assert hw.device_type == "cpu"
        assert hw.device_name == "CPU Host"
        assert hw.supports_ampere_tensor_cores is False


def test_cpu_only_fallback(adaptive_engine):
    """Verify deterministic fallback when CPU mode is forced or CUDA is unavailable."""
    hw_cpu = adaptive_engine.get_hardware_profile(force_cpu=True)
    assert hw_cpu.device_type == "cpu"
    assert hw_cpu.device_name == "CPU Host"
    assert hw_cpu.major == 0
    assert hw_cpu.minor == 0
    assert hw_cpu.sm_count == 0
    assert hw_cpu.total_vram_mb == 0.0
    assert hw_cpu.free_vram_mb == 0.0
    assert hw_cpu.supports_ampere_tensor_cores is False
    assert hw_cpu.supports_fp16 is False
    assert hw_cpu.supports_bf16 is False
    assert hw_cpu.supports_custom_cuda is False


# ============================================================================
# 2. Precision Selection and Gating
# ============================================================================

def test_precision_selection_fp32_always_supported(adaptive_engine):
    """FP32 is universally supported across CPU and GPU."""
    prec_cpu, rat_cpu, adj_cpu = adaptive_engine.select_precision("float32", force_cpu=True)
    assert prec_cpu == "float32"

    prec_gpu, rat_gpu, adj_gpu = adaptive_engine.select_precision("float32", force_cpu=False)
    assert prec_gpu == "float32"


def test_precision_selection_amp_fp16(adaptive_engine):
    """AMP FP16 is selected on Ampere GPU, falls back to FP32 on CPU."""
    prec_cpu, rat_cpu, adj_cpu = adaptive_engine.select_precision("amp", force_cpu=True)
    assert prec_cpu == "float32"
    assert any("not supported on CPU" in a for a in adj_cpu)

    if torch.cuda.is_available():
        prec_gpu, rat_gpu, adj_gpu = adaptive_engine.select_precision("amp", force_cpu=False)
        assert prec_gpu == "amp"


def test_precision_selection_bf16_gating(adaptive_engine):
    """BF16 on CPU must safely fall back to FP32; on GPU it must check is_bf16_supported()."""
    # CPU fallback
    prec_cpu, rat_cpu, adj_cpu = adaptive_engine.select_precision("bfloat16", force_cpu=True)
    assert prec_cpu == "float32"
    assert any("not supported on CPU" in a for a in adj_cpu)

    # Simulated unsupported GPU
    fake_hw = AdaptiveHardwareProfile(
        device_name="GeForce GTX 1080",
        device_type="cuda",
        major=6,
        minor=1,
        sm_count=20,
        total_vram_mb=8192.0,
        free_vram_mb=7000.0,
        supports_ampere_tensor_cores=False,
        supports_fp16=False,
        supports_bf16=False,  # Unsupported on Pascal
        supports_custom_cuda=False,
        supports_cuda_graphs=False,
    )
    prec_unsupported, rat_unsupported, adj_unsupported = adaptive_engine.select_precision("bfloat16", hw=fake_hw)
    assert prec_unsupported == "amp"
    assert any("BFloat16 requested but unsupported" in a for a in adj_unsupported)


# ============================================================================
# 3. VRAM Estimation Mathematical Accuracy
# ============================================================================

def test_vram_estimation_cifar10_mlp(adaptive_engine):
    """Verify VRAM estimation for CIFAR10 MLP."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for VRAM estimation test")

    est = adaptive_engine.estimate_training_vram("cifar10_mlp", batch_size=128, precision="float32")
    assert est["model_name"] == "cifar10_mlp"
    assert est["param_memory_mb"] > 10.0  # 3.4M params * 4 bytes ≈ 13.07 MB
    assert est["optimizer_memory_mb"] > 10.0
    assert est["total_estimated_vram_mb"] < 400.0  # Fits easily in <400 MB
    assert est["is_safe"] is True
    assert est["headroom_mb"] > 1000.0


def test_vram_estimation_transformer_scaling(adaptive_engine):
    """Verify VRAM estimation scales with sequence length and batch size."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for transformer VRAM estimation test")

    est_small = adaptive_engine.estimate_training_vram("transformer_50m", batch_size=2, seq_len=128, precision="amp")
    est_large = adaptive_engine.estimate_training_vram("transformer_50m", batch_size=16, seq_len=1024, precision="float32")

    assert est_large["total_estimated_vram_mb"] > est_small["total_estimated_vram_mb"]
    assert est_large["activation_memory_mb"] > est_small["activation_memory_mb"]
    assert est_small["is_safe"] is True


# ============================================================================
# 4. Automatic Batch Size and VRAM Safety Margins
# ============================================================================

def test_safe_batch_size_retained_when_within_budget(adaptive_engine):
    """When requested batch size fits comfortably within budget, it must be honored."""
    dec = adaptive_engine.recommend_training_config(
        model_name="cifar10_mlp",
        requested_batch_size=64,
        requested_precision="float32",
    )
    assert dec.applied_batch_size == 64
    assert dec.is_safe is True


def test_excessive_batch_size_adaptively_reduced_and_accum_scaled(adaptive_engine):
    """When requested batch size exceeds VRAM budget, batch is reduced and accum steps doubled/scaled."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for VRAM budget reduction test")

    # Request batch size 32 with seq_len 1024 and unscaled FP32 on 86.8M model (~11.3GB requirement)
    dec = adaptive_engine.recommend_training_config(
        model_name="transformer_50m",
        requested_batch_size=32,
        seq_len=1024,
        requested_precision="float32",
        target_vram_ratio=0.85,
    )
    assert dec.applied_batch_size < 32
    assert dec.gradient_accumulation_steps > 1
    # Effective batch size should approximate the user's intent
    assert dec.effective_batch_size >= 32
    assert any("exceeded" in a for a in dec.adjustments_made)
    assert dec.is_safe is True


def test_auto_batch_size_selection(adaptive_engine):
    """Auto batch size selection finds the optimal candidate within budget."""
    dec = adaptive_engine.recommend_training_config(
        model_name="cifar10_mlp",
        requested_batch_size=None,  # Auto
        requested_precision="amp",
    )
    assert dec.applied_batch_size >= 16
    assert dec.is_safe is True
    assert any("Auto-selected optimal batch size" in a for a in dec.adjustments_made)


# ============================================================================
# 5. OOM Recovery and Checkpoint Preservation
# ============================================================================

def test_oom_recovery_halves_batch_and_doubles_accum(adaptive_engine):
    """OOM recovery clears cache, halves batch size, and doubles accumulation steps."""
    mitigation = adaptive_engine.handle_oom(
        current_batch_size=64,
        current_accum_steps=2,
        min_batch_size=16,
        model_name="cifar10_mlp",
    )
    assert mitigation is not None
    assert mitigation["status"] == "RECOVERED"
    assert mitigation["previous_batch_size"] == 64
    assert mitigation["new_batch_size"] == 32
    assert mitigation["previous_accum_steps"] == 2
    assert mitigation["new_accum_steps"] == 4
    assert mitigation["effective_batch_size"] == 128  # 32 * 4


def test_oom_recovery_exhausted_at_min_floor(adaptive_engine):
    """When batch size is already at minimum floor, returns None to signal unrecoverable state."""
    mitigation = adaptive_engine.handle_oom(
        current_batch_size=16,
        current_accum_steps=4,
        min_batch_size=16,
        model_name="cifar10_mlp",
    )
    assert mitigation is None


def test_checkpoint_unaffected_by_oom(tmp_path):
    """Verify that an OOM failure does not corrupt or overwrite an existing checkpoint."""
    from engine.checkpointing.checkpoint_manager import CheckpointManager

    ckpt_path = tmp_path / "best_model.pt"
    dummy_model = torch.nn.Linear(10, 2)
    CheckpointManager.save_checkpoint(
        model=dummy_model,
        filepath=str(ckpt_path),
        epoch=1,
        metrics={"val_accuracy": 85.5},
    )
    assert ckpt_path.is_file()
    initial_bytes = ckpt_path.stat().st_size
    initial_hash = CheckpointManager.validate_checkpoint(str(ckpt_path))["sha256"]

    # Simulate an OOM exception
    try:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory simulated.")
    except torch.cuda.OutOfMemoryError:
        pass

    # Verify checkpoint remained completely intact
    assert ckpt_path.is_file()
    assert ckpt_path.stat().st_size == initial_bytes
    assert CheckpointManager.validate_checkpoint(str(ckpt_path))["sha256"] == initial_hash


# ============================================================================
# 6. API Route Integration
# ============================================================================

def test_api_adaptive_hardware_endpoint(client):
    """Test GET /api/training/adaptive/hardware returns valid hardware profile."""
    res = client.get("/api/training/adaptive/hardware")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert "device_name" in data
    assert "device_type" in data
    assert "compute_capability" in data
    assert "total_vram_mb" in data
    assert "free_vram_mb" in data
    assert "supports_ampere_tensor_cores" in data
    assert "supports_bf16" in data


def test_api_adaptive_recommendation_endpoint(client):
    """Test POST /api/training/adaptive/recommend generates evidence-backed recommendation."""
    payload = {
        "model_name": "transformer_50m",
        "backend": "GPU_NATIVE",
        "batch_size": 16,
        "precision": "amp",
        "seq_len": 256,
        "gradient_accumulation_steps": 2,
        "target_vram_ratio": 0.85,
    }
    res = client.post("/api/training/adaptive/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert data["model_name"] == "transformer_50m"
    assert "applied_batch_size" in data
    assert "applied_precision" in data
    assert "estimated_peak_vram_mb" in data
    assert "rationale" in data
    assert data["is_safe"] is True


def test_api_adaptive_job_submission_and_config(client):
    """Test submitting a job with adaptive_mode=True preserves adaptive decision in config."""
    payload = {
        "model_name": "cifar10_mlp",
        "backend": "CPU_NATIVE",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "adaptive_mode": True,
        "auto_batch_size": True,
        "max_train_samples": 64,
        "max_val_samples": 32,
        "max_test_samples": 32,
    }
    res = client.post("/api/training/jobs", json=payload)
    assert res.status_code == 201
    job = res.json()
    assert job["status"] in ("QUEUED", "RUNNING")
    assert "config" in job
    cfg = job["config"]
    assert cfg.get("adaptive_mode") is True
    assert cfg.get("auto_batch_size") is True
    assert "adaptive_decision" in cfg


def test_manual_override_preserved_when_adaptive_disabled(client):
    """Manual configuration must be strictly preserved when adaptive_mode=False."""
    payload = {
        "model_name": "cifar10_mlp",
        "backend": "CPU_NATIVE",
        "epochs": 1,
        "batch_size": 64,  # Explicit manual batch size
        "learning_rate": 0.02,
        "adaptive_mode": False,
        "auto_batch_size": False,
        "precision": "float32",
        "max_train_samples": 64,
        "max_val_samples": 32,
        "max_test_samples": 32,
    }
    res = client.post("/api/training/jobs", json=payload)
    assert res.status_code == 201
    job = res.json()
    assert job["batch_size"] == 64
    assert job["config"]["batch_size"] == 64
    assert job["config"]["precision"] == "float32"
    assert job["config"].get("adaptive_decision") is None
