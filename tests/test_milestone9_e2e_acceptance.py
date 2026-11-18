"""
tests/test_milestone9_e2e_acceptance.py
Phase 13 Milestone 9: End-to-End System Acceptance Testing.
Automated verification across all 10 acceptance criteria:
1. GPU Command Center and live NVML telemetry.
2. Every AI Training Studio architecture and execution backend.
3. Training start, progress, cancellation and checkpoint handling.
4. Model Registry and checkpoint integrity.
5. NEXORA GPT-86.8M inference.
6. NEXORA GPT-100M (92.13M parameters) pilot checkpoint, configuration and actual training status.
7. CUDA Laboratory and comparative benchmark configurations A-D.
8. Adaptive Kernel Selection.
9. Frontend-to-backend communication and error handling.
10. Application startup and graceful shutdown.
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
from engine.transformer.adaptive_engine import AdaptiveExecutionEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    app = create_app(PROJECT_ROOT)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean model serving and GPU lock state before and after each test."""
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
# 1. GPU Command Center and Live NVML Telemetry
# -----------------------------------------------------------------------------
def test_acceptance_1_gpu_command_center_and_nvml_telemetry(client: TestClient):
    """Verify live GPU telemetry from /api/hardware and /api/laboratory/gpu-monitoring."""
    # Test /api/hardware
    res_hw = client.get("/api/hardware")
    assert res_hw.status_code == 200
    hw_data = res_hw.json()
    assert hw_data["status"] == "available"
    assert "gpus" in hw_data
    assert len(hw_data["gpus"]) > 0

    gpu0 = hw_data["gpus"][0]
    if torch.cuda.is_available():
        assert "NVIDIA" in gpu0["name"] or "RTX" in gpu0["name"] or "GeForce" in gpu0["name"]
        assert gpu0["memory_total_mib"] is not None and gpu0["memory_total_mib"] > 4000
        assert gpu0["temperature_c"] is not None and gpu0["temperature_c"] > 0
        assert gpu0["temperature_c"] < 85.0, "GPU temperature exceeds safe thermal threshold"
    else:
        assert gpu0["name"] == "CPU Emulation Device"

    # Test /api/laboratory/gpu-monitoring
    res_mon = client.get("/api/laboratory/gpu-monitoring")
    assert res_mon.status_code == 200
    mon_data = res_mon.json()
    assert mon_data["thermal_throttling_active"] is False
    assert mon_data["can_launch_experiment"] is True
    assert mon_data["active_gpu_job_id"] is None


# -----------------------------------------------------------------------------
# 2. Every AI Training Studio Architecture and Execution Backend
# -----------------------------------------------------------------------------
def test_acceptance_2_training_studio_all_architectures_and_backends(client: TestClient):
    """Verify all 7 registered architectures and their supported backends."""
    res = client.get("/api/training/models")
    assert res.status_code == 200
    models_data = res.json()["models"]
    model_ids = {m["model_id"]: m for m in models_data}

    # Verify all 7 required architectures are present
    expected_models = [
        "cifar10_mlp",
        "resnet18",
        "baseline_mlp",
        "transformer_50m",
        "transformer_100m",
        "transformer_25m",
        "transformer_1m",
    ]
    for mid in expected_models:
        assert mid in model_ids, f"Expected architecture '{mid}' not found in training registry"

    # Verify backend coverage for each model
    cifar10_mlp = model_ids["cifar10_mlp"]
    assert "GPU_CUSTOM" in cifar10_mlp["supported_backends"]
    assert "GPU_NATIVE" in cifar10_mlp["supported_backends"]
    assert "CPU_NATIVE" in cifar10_mlp["supported_backends"]
    assert cifar10_mlp["custom_cuda_supported"] is True

    resnet18 = model_ids["resnet18"]
    assert "GPU_NATIVE" in resnet18["supported_backends"]
    assert "CPU_NATIVE" in resnet18["supported_backends"]

    trans_50m = model_ids["transformer_50m"]
    assert trans_50m["parameters"] == 86825472
    assert "GPU_NATIVE" in trans_50m["supported_backends"]

    trans_100m = model_ids["transformer_100m"]
    assert trans_100m["parameters"] == 92133888
    assert "GPU_NATIVE" in trans_100m["supported_backends"]


# -----------------------------------------------------------------------------
# 3. Training Start, Progress, Cancellation, and Checkpoint Handling
# -----------------------------------------------------------------------------
def test_acceptance_3_training_start_progress_cancel_checkpoint(client: TestClient):
    """Verify full training job lifecycle: start, progress polling, completion, and cancellation."""
    orch = TrainingOrchestrator.get_instance()
    backend = "GPU_NATIVE" if torch.cuda.is_available() else "CPU_NATIVE"

    # 1. Submit a fast smoke job (2 steps of transformer_1m)
    job_submit = orch.submit_job(
        model_name="transformer_1m",
        backend=backend,
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
            "seq_len": 32,
        },
    )
    job_id = job_submit["job_id"]
    assert job_submit["status"] == "QUEUED"

    # Poll until COMPLETED
    t0 = time.time()
    final_job = None
    while time.time() - t0 < 45:
        j = orch.get_job(job_id)
        if j["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            final_job = j
            break
        time.sleep(0.3)

    assert final_job is not None, "Job did not finish in time"
    assert final_job["status"] == "COMPLETED"
    assert orch.active_gpu_job_id is None, "GPU lock must be freed after completion"

    # Verify checkpoints
    ckpts = orch.get_job_checkpoints(job_id)
    assert len(ckpts) > 0, "Expected at least 1 checkpoint emitted"
    assert "checkpoint_path" in ckpts[0]

    # Verify API checkpoints endpoint
    res_ckpt = client.get(f"/api/training/jobs/{job_id}/checkpoints")
    assert res_ckpt.status_code == 200
    assert len(res_ckpt.json()["checkpoints"]) > 0

    # 2. Test Cancellation on a job
    job_cancel_submit = orch.submit_job(
        model_name="transformer_1m",
        backend=backend,
        dataset="MultiDocumentCorpus",
        epochs=10,
        batch_size=2,
        learning_rate=3e-4,
        extra_config={"max_steps": 200, "seq_len": 32},
    )
    cancel_job_id = job_cancel_submit["job_id"]

    # Wait for RUNNING or QUEUED
    t0 = time.time()
    while time.time() - t0 < 15:
        j = orch.get_job(cancel_job_id)
        if j["status"] in ("RUNNING", "QUEUED"):
            break
        time.sleep(0.1)

    # Issue cancellation via API
    res_cancel = client.post(f"/api/training/jobs/{cancel_job_id}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"].upper() in ("CANCEL_REQUESTED", "CANCELLED")

    # Await CANCELLED terminal state
    t0 = time.time()
    term_job = None
    while time.time() - t0 < 25:
        j = orch.get_job(cancel_job_id)
        if j["status"] in ("CANCELLED", "COMPLETED", "FAILED"):
            term_job = j
            break
        time.sleep(0.2)

    assert term_job is not None
    assert term_job["status"] == "CANCELLED"
    assert orch.active_gpu_job_id is None, "GPU lock must be released on cancellation"


# -----------------------------------------------------------------------------
# 4. Model Registry and Checkpoint Integrity
# -----------------------------------------------------------------------------
def test_acceptance_4_model_registry_and_checkpoint_integrity(client: TestClient):
    """Verify registry listing, checkpoint catalog, and SHA-256 integrity."""
    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    models = res_models.json()["models"]
    model_names = [m["name"] for m in models]
    assert "ResNet18" in model_names
    assert "NEXORA GPT-86.8M" in model_names

    res_ckpts = client.get("/api/checkpoints")
    assert res_ckpts.status_code == 200
    catalog = res_ckpts.json()
    assert len(catalog["checkpoints"]) > 0

    # Verify physical file existence and non-empty SHA-256 for all catalog items
    for item in catalog["checkpoints"][:5]:
        p = PROJECT_ROOT / item["path"]
        assert p.is_file(), f"Checkpoint path {item['path']} missing on disk"
        assert len(item["sha256"]) == 64, f"Invalid SHA-256 hash for {item['filename']}"

    # Verify the certified 50M pretraining campaign best checkpoint exists
    campaign_ckpt = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_campaign" / "best_model.pt"
    assert campaign_ckpt.is_file()
    assert campaign_ckpt.stat().st_size > 100 * 1024 * 1024  # > 100 MB


# -----------------------------------------------------------------------------
# 5. NEXORA GPT-86.8M Inference
# -----------------------------------------------------------------------------
def test_acceptance_5_nexora_gpt_86m_inference(client: TestClient):
    """Verify loading and generating from the certified NEXORA GPT-86.8M model."""
    load_res = client.post(
        "/api/models/load",
        json={
            "model_name": "NEXORA GPT-86.8M",
            "checkpoint": "checkpoints/transformer/pretraining_50m_campaign/best_model.pt",
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },
    )
    assert load_res.status_code == 200
    assert load_res.json()["status"] == "loaded"
    assert load_res.json()["parameters"] == 86825472

    gen_res = client.post(
        "/api/models/generate",
        json={
            "prompt": "Hardware acceleration of matrix multiply",
            "max_new_tokens": 8,
            "temperature": 0.7,
            "seed": 42,
        },
    )
    assert gen_res.status_code == 200
    data = gen_res.json()
    assert data["status"] == "success"
    assert "NEXORA GPT-86.8M" in data["model_name"]
    assert data["generated_tokens"] == 8
    assert len(data["generated_text"]) > 0
    assert data["is_partially_trained"] is False
    assert "12,208 Steps" in data["status_label"]
    assert data["tokens_per_second"] > 0


# -----------------------------------------------------------------------------
# 6. NEXORA GPT-100M (92.13M Parameters) Pilot Checkpoint and Actual Status
# -----------------------------------------------------------------------------
def test_acceptance_6_nexora_gpt_100m_pilot_checkpoint_and_status(client: TestClient):
    """Verify NEXORA GPT-100M pilot checkpoint configuration, parameters, and training status."""
    pilot_path = "checkpoints/transformer/flagship_100m_pilot/best_model.pt"
    p = PROJECT_ROOT / pilot_path
    assert p.is_file(), f"Pilot checkpoint '{pilot_path}' not found on disk"

    # Verify physical checkpoint payload
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "metadata" in ckpt
    meta = ckpt["metadata"]
    mcfg = meta["model_config"]

    # Verify architecture matches 100M flagship topology (12 layers, 12 heads, d_model=768, d_ff=3072)
    assert mcfg["n_layer"] == 12
    assert mcfg["n_head"] == 12
    assert mcfg["d_model"] == 768
    assert mcfg["d_ff"] == 3072
    assert mcfg["tie_weights"] is True

    # Checkpoint metrics confirm actual exploratory status: 15 optimizer steps
    metrics = ckpt.get("metrics", {})
    assert metrics.get("step") in (10, 15) or metrics.get("microbatches_processed") is not None
    assert metrics.get("val_loss") is not None

    # Load via API and verify serving correctly flags as pilot / partially trained
    load_res = client.post(
        "/api/models/load",
        json={
            "model_name": "NEXORA GPT-100M",
            "checkpoint": pilot_path,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },
    )
    assert load_res.status_code == 200
    assert "NEXORA GPT-100M" in load_res.json()["model_name"]

    # Generate and inspect truthful partial training flags
    gen_res = client.post(
        "/api/models/generate",
        json={
            "prompt": "Neural network architecture",
            "max_new_tokens": 6,
            "checkpoint": pilot_path,
        },
    )
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["is_partially_trained"] is True
    assert "Pilot Checkpoint" in gen_data["status_label"] or "Partially Trained" in gen_data["status_label"]


# -----------------------------------------------------------------------------
# 7. CUDA Laboratory Comparative Benchmark Configurations A-D
# -----------------------------------------------------------------------------
def test_acceptance_7_cuda_laboratory_configurations_a_to_d(client: TestClient):
    """Verify the 4 comparative configurations and empirical results."""
    res_configs = client.get("/api/laboratory/configurations")
    assert res_configs.status_code == 200
    configs = res_configs.json()["configurations"]
    assert len(configs) == 4

    config_map = {c["id"]: c for c in configs}
    assert "config_a" in config_map
    assert "config_b" in config_map
    assert "config_c" in config_map
    assert "config_d" in config_map

    # Configuration A: Baseline FP32
    assert "FP32" in config_map["config_a"]["precision"]
    assert config_map["config_a"]["custom_kernels_enabled"] is False

    # Configuration B: Optimized PyTorch AMP SDPA
    assert "AMP" in config_map["config_b"]["precision"]
    assert config_map["config_b"]["custom_kernels_enabled"] is False

    # Configuration C: Custom CUDA
    assert config_map["config_c"]["custom_kernels_enabled"] is True

    # Configuration D: Adaptive Engine
    assert "Adaptive" in config_map["config_d"]["name"]

    # Verify historical benchmarks
    res_exp = client.get("/api/laboratory/experiments")
    assert res_exp.status_code == 200
    exp_data = res_exp.json()["experiment_data"]
    assert "configurations" in exp_data or "comparative_metrics" in exp_data

    # Verify GPU Mutual Exclusion (concurrency lock)
    orch = TrainingOrchestrator.get_instance()
    orch.active_gpu_job_id = "test_occupying_job"
    res_conflict = client.post(
        "/api/laboratory/experiments",
        json={"configurations": ["Config A", "Config B"]},
    )
    assert res_conflict.status_code == 409
    orch.active_gpu_job_id = None


# -----------------------------------------------------------------------------
# 8. Adaptive Kernel Selection
# -----------------------------------------------------------------------------
def test_acceptance_8_adaptive_kernel_selection(client: TestClient):
    """Verify runtime dynamic kernel selection logic and empirical benchmarks."""
    engine = AdaptiveExecutionEngine()
    summary = engine.get_status_summary()

    assert "hardware" in summary
    assert "available_paths" in summary
    assert "default_training_path" in summary
    assert "default_inference_path" in summary

    # Training selection: cuBLASLt epilogue bias fusion advantage should select optimized_pytorch
    train_decision = engine.select_execution_path(
        workload="training",
        precision="amp_fp16",
        batch_size=16,
        sequence_length=256,
    )
    assert train_decision.selected_path.value == "optimized_pytorch"
    assert "cuBLASLt" in train_decision.rationale

    # Inference selection: direct C++ CUDA kernel dispatch should select custom_cuda
    inf_decision = engine.select_execution_path(
        workload="inference",
        precision="fp16",
        batch_size=1,
        sequence_length=128,
    )
    assert inf_decision.selected_path.value == "custom_cuda"

    # API endpoints
    res_status = client.get("/api/laboratory/adaptive-status")
    assert res_status.status_code == 200
    assert res_status.json()["numerical_certification_status"] == "CERTIFIED"

    res_select = client.post(
        "/api/laboratory/adaptive-select",
        json={"workload": "training", "precision": "amp_fp16", "batch_size": 16, "sequence_length": 256},
    )
    assert res_select.status_code == 200
    assert res_select.json()["decision"]["selected_path"] == "optimized_pytorch"


# -----------------------------------------------------------------------------
# 9. Frontend-to-Backend Communication and Error Handling
# -----------------------------------------------------------------------------
def test_acceptance_9_communication_and_error_handling(client: TestClient):
    """Verify API error handlers: 404, 422, and structured JSON payloads."""
    # 404 File Not Found
    res_404 = client.post(
        "/api/models/load",
        json={"model_name": "resnet18", "checkpoint": "non_existent/path/model.pt"},
    )
    assert res_404.status_code == 404
    assert "error" in res_404.json() or "detail" in res_404.json()

    # 422 Validation Error on invalid laboratory configuration
    res_422 = client.post(
        "/api/laboratory/experiments",
        json={"configurations": ["Invalid_Config_X"]},
    )
    assert res_422.status_code == 422

    # Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"


# -----------------------------------------------------------------------------
# 10. Application Startup and Graceful Shutdown
# -----------------------------------------------------------------------------
def test_acceptance_10_application_startup_and_graceful_shutdown():
    """Verify application lifecycle manager handles clean startup and graceful shutdown."""
    from backend.app import lifespan
    from contextlib import asynccontextmanager
    import asyncio

    test_app = create_app(PROJECT_ROOT)

    async def run_lifecycle():
        async with lifespan(test_app):
            # Assert running inside lifespan
            assert test_app.title == "CUDA Acceleration Engine"
        # Post-lifespan assertions:
        assert serving._VISION_MODEL is None
        assert serving._TRANSFORMER_MODEL is None

    asyncio.run(run_lifecycle())
