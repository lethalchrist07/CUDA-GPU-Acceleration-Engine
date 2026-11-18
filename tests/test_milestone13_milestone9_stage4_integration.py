"""
tests/test_milestone13_milestone9_stage4_integration.py
Phase 13 Milestone 9 Stage 4: Backend-to-Engine Integration & Live System Verification.

Comprehensive Automated Verification Suite:
1. Dual-slot serving isolation on physical GPU (ResNet-18 + NEXORA GPT-86.8M).
2. End-to-end FastAPI serving API routes (load, list, predict, generate, unload).
3. Training Orchestrator lifecycle for production 86.83M preset (QUEUED -> RUNNING -> COMPLETED).
4. Training Orchestrator cooperative cancellation and GPU lock release (CANCEL_REQUESTED -> CANCELLED).
5. Hardware resource concurrency guard (HTTP 409 Conflict prevention during GPU occupancy).
6. Comparative Laboratory 4-configuration contracts, adaptive selection, and evaluation catalog.
7. Live GPU hardware telemetry and thermal safety monitoring.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.schemas.training import TrainingJobCreateRequest
from backend.serving import ModelServer
import backend.serving as serving
from engine.training.orchestrator import SUPPORTED_MODELS, TrainingOrchestrator
from engine.transformer.config import get_preset_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    app = create_app(PROJECT_ROOT)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_serving_and_orchestrator():
    """Ensure clean serving and GPU lock state before and after each test."""
    ModelServer.unload_model("all")
    orch = TrainingOrchestrator.get_instance()
    orch.active_gpu_job_id = None
    try:
        orch.gpu_lock.release()
    except RuntimeError:
        pass
    yield
    ModelServer.unload_model("all")
    orch.active_gpu_job_id = None
    try:
        orch.gpu_lock.release()
    except RuntimeError:
        pass


# -----------------------------------------------------------------------------
# 1. Real Engine Dual-Slot Serving & Cross-Modal GPU Inference
# -----------------------------------------------------------------------------
def test_stage4_serving_dual_slot_gpu_isolation_and_inference():
    """
    Verify independent vision and transformer slots execute on GPU without cross-modal
    state collision, preserve 86,825,472 parameter counts, and release VRAM on unload.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1. Load ResNet-18 into Vision Slot
    vis_res = ModelServer.load_model(
        model_name="resnet18",
        checkpoint_path="checkpoints/resnet18_cifar10/best_model.pt",
        device_str=device,
    )
    assert vis_res["status"] == "loaded"
    assert serving._VISION_MODEL is not None
    assert serving._VISION_MODEL_NAME == "ResNet18"

    # 2. Load NEXORA GPT-86.8M into Transformer Slot
    trans_res = ModelServer.load_model(
        model_name="transformer",
        checkpoint_path="checkpoints/transformer/pretraining_50m_campaign/best_model.pt",
        device_str=device,
    )
    assert trans_res["status"] == "loaded"
    assert trans_res["parameters"] == 86825472
    assert serving._TRANSFORMER_MODEL is not None
    assert "NEXORA GPT-86.8M" in serving._TRANSFORMER_MODEL_NAME

    # Both slots active concurrently
    assert serving._VISION_MODEL is not None
    assert serving._TRANSFORMER_MODEL is not None

    # 3. Vision Inference (predict sample index 0)
    vis_out = ModelServer.predict(sample_index=0)
    assert vis_out["status"] == "success"
    assert "predicted_index" in vis_out
    assert "predicted_class" in vis_out
    assert "confidence" in vis_out
    assert 0 <= vis_out["predicted_index"] < 10

    # 4. Transformer Autoregressive Generation
    prompt = "In the deep space between stars"
    gen_out = ModelServer.generate(prompt=prompt, max_new_tokens=10, temperature=0.8)
    assert gen_out["status"] == "success"
    assert "generated_text" in gen_out
    assert "generated_tokens" in gen_out
    assert gen_out["generated_tokens"] > 0
    assert "tokens_per_second" in gen_out
    assert gen_out["is_partially_trained"] is False

    # 5. Cross-Modal Stability: Vision again immediately after generation
    vis_out_2 = ModelServer.predict(sample_index=0)
    assert vis_out_2["predicted_index"] == vis_out["predicted_index"]

    # 6. Unload Vision: Transformer must remain completely operational
    ModelServer.unload_model("vision")
    assert serving._VISION_MODEL is None
    assert serving._TRANSFORMER_MODEL is not None
    gen_out_2 = ModelServer.generate(prompt=prompt, max_new_tokens=8, temperature=0.8)
    assert gen_out_2["generated_tokens"] > 0

    # 7. Unload Transformer: Full cleanup
    ModelServer.unload_model("transformer")
    assert serving._TRANSFORMER_MODEL is None


# -----------------------------------------------------------------------------
# 2. FastAPI Serving API Routes End-to-End
# -----------------------------------------------------------------------------
def test_stage4_api_serving_routes(client: TestClient):
    """Verify REST contracts across /api/models endpoints with truthful payloads."""
    # 1. Unload all to start clean
    resp = client.post("/api/models/unload")
    assert resp.status_code == 200

    # 2. Query available models list
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"
    assert len(data["models"]) >= 2

    # 3. Load Transformer via API
    load_payload = {
        "model_name": "transformer",
        "checkpoint": "checkpoints/transformer/pretraining_50m_campaign/best_model.pt",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }
    resp = client.post("/api/models/load", json=load_payload)
    assert resp.status_code == 200
    load_data = resp.json()
    assert load_data["status"] == "loaded"
    assert load_data["parameters"] == 86825472

    # 4. Generate Text via API
    gen_payload = {
        "prompt": "The discovery of gravitation",
        "max_new_tokens": 12,
        "temperature": 0.7,
        "model_name": "NEXORA GPT-86.8M",
    }
    resp = client.post("/api/models/generate", json=gen_payload)
    assert resp.status_code == 200
    gen_res = resp.json()
    assert "generated_text" in gen_res
    assert gen_res["generated_tokens"] > 0
    assert "NEXORA GPT-86.8M" in gen_res["model_name"]
    assert gen_res["tokens_per_second"] > 0.0

    # 5. Clean up
    resp = client.post("/api/models/unload")
    assert resp.status_code == 200


# -----------------------------------------------------------------------------
# 3. Training Orchestrator 86.83M Preset & Lifecycle
# -----------------------------------------------------------------------------
def test_stage4_training_orchestrator_86m_preset_lifecycle():
    """
    Verify production_86m geometry preset and controlled training execution
    transitions from QUEUED -> RUNNING -> COMPLETED with checkpoint emission.
    """
    # 1. Preset configuration verification
    preset_cfg = get_preset_config("production_86m", vocab_size=2048)
    assert preset_cfg.n_layer == 12
    assert preset_cfg.n_head == 12
    assert preset_cfg.d_model == 768
    assert preset_cfg.vocab_size == 2048
    assert preset_cfg.tie_weights is True

    # 2. Supported model registration
    assert "transformer_50m" in SUPPORTED_MODELS
    info = SUPPORTED_MODELS["transformer_50m"]
    assert info["parameters"] == 86825472
    assert info["name"] == "NEXORA GPT-86.8M"

    # 3. Normalization through Pydantic Request schema
    req = TrainingJobCreateRequest(
        model_name="nexora_gpt_86.8m",
        backend="GPU_NATIVE" if torch.cuda.is_available() else "CPU_NATIVE",
        epochs=1,
        batch_size=2,
        learning_rate=3e-4,
    )
    assert req.model_name == "transformer_50m"

    # 4. Launch small 2-step training smoke job
    orchestrator = TrainingOrchestrator.get_instance()
    backend_choice = "GPU_NATIVE" if torch.cuda.is_available() else "CPU_NATIVE"

    job_info = orchestrator.submit_job(
        model_name="transformer_50m",
        backend=backend_choice,
        dataset="MultiDocumentCorpus",
        epochs=1,
        batch_size=2,
        learning_rate=3e-4,
        extra_config={
            "max_steps": 2,
            "precision": "amp" if torch.cuda.is_available() else "float32",
            "gradient_accumulation_steps": 1,
            "eval_interval": 2,
            "save_interval": 2,
            "seq_len": 64,
        },
    )
    job_id = job_info["job_id"]
    assert job_info["status"] == "QUEUED"

    # Wait for completion
    t0 = time.time()
    final_job = None
    while time.time() - t0 < 60:
        j = orchestrator.get_job(job_id)
        if j["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            final_job = j
            break
        time.sleep(0.5)

    assert final_job is not None, f"Job {job_id} did not complete within timeout"
    assert final_job["status"] == "COMPLETED", f"Job failed: {final_job.get('error_message')}"
    assert orchestrator.active_gpu_job_id is None, "GPU job lock must be released on completion"

    # Verify physical and registered checkpoints
    ckpt_dir = (PROJECT_ROOT / orchestrator.checkpoints_base_dir / job_id).resolve()
    assert ckpt_dir.is_dir()
    ckpts = list(ckpt_dir.glob("*.pt"))
    assert len(ckpts) > 0, "Expected at least 1 checkpoint file saved"

    registered = orchestrator.get_job_checkpoints(job_id)
    assert len(registered) > 0


# -----------------------------------------------------------------------------
# 4. Cooperative Cancellation & Safe GPU Lock Release
# -----------------------------------------------------------------------------
def test_stage4_training_orchestrator_cooperative_cancellation():
    """Verify cooperative cancellation transitions safely to CANCELLED and frees GPU."""
    orchestrator = TrainingOrchestrator.get_instance()
    backend_choice = "GPU_NATIVE" if torch.cuda.is_available() else "CPU_NATIVE"

    job_info = orchestrator.submit_job(
        model_name="transformer_50m",
        backend=backend_choice,
        dataset="MultiDocumentCorpus",
        epochs=10,
        batch_size=2,
        learning_rate=3e-4,
        extra_config={"max_steps": 200, "seq_len": 64},
    )
    job_id = job_info["job_id"]

    # Wait for job to enter RUNNING state
    t0 = time.time()
    while time.time() - t0 < 15:
        j = orchestrator.get_job(job_id)
        if j["status"] == "RUNNING":
            break
        time.sleep(0.1)

    # Issue cancellation
    cancelled = orchestrator.cancel_job(job_id)
    assert cancelled is True

    # Await CANCELLED terminal state
    t0 = time.time()
    final_job = None
    while time.time() - t0 < 20:
        j = orchestrator.get_job(job_id)
        if j["status"] in ("CANCELLED", "COMPLETED", "FAILED"):
            final_job = j
            break
        time.sleep(0.2)

    assert final_job is not None
    assert final_job["status"] == "CANCELLED"
    assert orchestrator.active_gpu_job_id is None, "GPU lock must be released on cancellation"


# -----------------------------------------------------------------------------
# 5. GPU Resource Mutual Exclusion Guard
# -----------------------------------------------------------------------------
def test_stage4_gpu_concurrency_lock_and_laboratory_conflict(client: TestClient):
    """Verify laboratory rejects experiment initiation with HTTP 409 when GPU is occupied."""
    orchestrator = TrainingOrchestrator.get_instance()

    # Simulate active GPU training reservation
    orchestrator.active_gpu_job_id = "test_active_training_job"

    # Attempt to launch comparative laboratory experiment
    resp = client.post(
        "/api/laboratory/experiments",
        json={"configurations": ["Config A", "Config B"]},
    )
    assert resp.status_code == 409
    assert "actively reserved" in resp.json()["detail"]

    # Clear active GPU reservation
    orchestrator.active_gpu_job_id = None

    # Now experiment creation should succeed
    resp_ok = client.post(
        "/api/laboratory/experiments",
        json={"configurations": ["Config A", "Config B"]},
    )
    assert resp_ok.status_code == 201
    assert resp_ok.json()["status"] == "created"


# -----------------------------------------------------------------------------
# 6. Comparative Laboratory 4-Configuration Contracts & Adaptive Select
# -----------------------------------------------------------------------------
def test_stage4_laboratory_configurations_and_benchmarks(client: TestClient):
    """Verify laboratory routes return verified artifacts and route adaptive requests."""
    # 1. Configurations
    resp = client.get("/api/laboratory/configurations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_configurations"] == 4
    cfg_ids = [c["id"] for c in data["configurations"]]
    assert cfg_ids == ["config_a", "config_b", "config_c", "config_d"]
    for c in data["configurations"]:
        assert c["parameters"] == 86825472

    # 2. Historical Experiments
    resp = client.get("/api/laboratory/experiments")
    assert resp.status_code == 200
    exp_data = resp.json()
    assert exp_data["status"] == "available"
    assert "experiment_data" in exp_data

    # 3. Kernel Microbenchmarks
    resp = client.get("/api/laboratory/cuda-benchmarks")
    assert resp.status_code == 200
    bench_data = resp.json()
    assert bench_data["status"] == "available"
    assert "benchmarks" in bench_data

    # 4. Checkpoints Metadata
    resp = client.get("/api/laboratory/checkpoints")
    assert resp.status_code == 200
    ckpts_data = resp.json()
    assert ckpts_data["total_checkpoints"] > 0
    filenames = [c["filename"] for c in ckpts_data["checkpoints"]]
    assert "best_model.pt" in filenames or "final_model.pt" in filenames

    # 5. Adaptive Status
    resp = client.get("/api/laboratory/adaptive-status")
    assert resp.status_code == 200
    adapt_status = resp.json()
    assert adapt_status["status"] == "available"
    assert len(adapt_status["available_paths"]) >= 2

    # 6. Adaptive Selection Query
    select_payload = {
        "workload": "training",
        "precision": "amp_fp16",
        "batch_size": 16,
        "sequence_length": 256,
    }
    resp = client.post("/api/laboratory/adaptive-select", json=select_payload)
    assert resp.status_code == 200
    sel_res = resp.json()
    assert sel_res["status"] == "success"
    assert "decision" in sel_res
    assert sel_res["decision"]["selected_path"] == "optimized_pytorch"


# -----------------------------------------------------------------------------
# 7. Hardware NVML Telemetry & Safety Guard
# -----------------------------------------------------------------------------
def test_stage4_hardware_nvml_telemetry(client: TestClient):
    """Verify live GPU telemetry reports truthful metrics and respects safety limits."""
    resp = client.get("/api/laboratory/gpu-monitoring")
    assert resp.status_code == 200
    data = resp.json()

    if torch.cuda.is_available():
        assert "NVIDIA" in data["device_name"] or "RTX" in data["device_name"] or "GeForce" in data["device_name"]
        assert data["vram_total_mb"] > 0
        assert data["vram_free_mb"] >= 0
        assert data["thermal_throttling_active"] is False
        assert data["temperature_celsius"] < 85.0
        assert data["can_launch_experiment"] is True
    else:
        assert data["device_name"] == "CPU Emulation Mode"
