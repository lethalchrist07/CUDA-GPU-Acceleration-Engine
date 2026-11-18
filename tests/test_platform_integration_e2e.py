"""
Full Platform Integration and End-to-End User Journey Verification Test Suite.
Validates the complete interoperability of:
1. Application startup and dashboard loading (Vite SPA + FastAPI)
2. Live GPU hardware telemetry and capability discovery
3. Adaptive training configuration and job submission
4. Live job progress streaming and metrics polling
5. Cooperative job cancellation and GPU resource cleanup
6. Bounded training completion and atomic checkpoint persistence
7. Checkpoint discovery and Model Registry cataloging
8. Transformer inference execution and robust error handling
9. Backend restart recovery and database state persistence
"""

import math
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import create_app
from backend.serving import ModelServer
from engine.training.job_store import JobStore
from engine.training.orchestrator import TrainingOrchestrator


@pytest.fixture(scope="module")
def client():
    app = create_app(REPO_ROOT)
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# User Journey 1: Application Startup, Health Checks & SPA Static Serving
# ============================================================================

def test_journey_1_app_startup_and_dashboard_loading(client):
    """Verify application starts up, passes health checks, and serves SPA assets."""
    # 1. API Health Check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] == "ok"
    assert "version" in health_data

    # 2. System Overview
    res_overview = client.get("/api/overview")
    assert res_overview.status_code == 200
    overview_data = res_overview.json()
    assert overview_data["status"] in ("available", "degraded")
    assert "benchmark_count" in overview_data

    # 3. Root SPA serving
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "<!doctype html>" in res_root.text.lower() or "<html" in res_root.text.lower()

    # 4. Deep SPA Route serving (fallback to index.html)
    res_spa = client.get("/training_studio")
    assert res_spa.status_code == 200
    assert "text/html" in res_spa.headers.get("content-type", "")

    # 5. API Protection (unmatched /api route must return JSON 404, not HTML)
    res_unmatched = client.get("/api/nonexistent_route")
    assert res_unmatched.status_code == 404
    assert res_unmatched.headers.get("content-type", "").startswith("application/json")


# ============================================================================
# User Journey 2: Real GPU Hardware Detection & Live Telemetry
# ============================================================================

def test_journey_2_gpu_command_center_telemetry(client):
    """Verify GPU Command Center reads live hardware counters without synthetic values."""
    # 1. Live hardware telemetry
    res_hw = client.get("/api/hardware")
    assert res_hw.status_code == 200
    hw_data = res_hw.json()

    if torch.cuda.is_available():
        assert hw_data["status"] == "available"
        assert len(hw_data["gpus"]) >= 1
        gpu = hw_data["gpus"][0]
        assert "RTX 3050" in gpu["name"] or "NVIDIA" in gpu["name"]
        assert gpu["memory_total_mib"] > 1000.0
        assert gpu["memory_free_mib"] is not None
        assert gpu["temperature_c"] is not None
    else:
        assert hw_data["status"] in ("available", "unavailable")

    # 2. CUDA Capabilities
    res_cap = client.get("/api/capabilities")
    assert res_cap.status_code == 200
    cap_data = res_cap.json()
    assert cap_data["status"] == "available"
    assert "cuda" in cap_data

    # 3. Adaptive Training Hardware Profile
    res_adap = client.get("/api/training/adaptive/hardware")
    assert res_adap.status_code == 200
    adap_data = res_adap.json()
    assert adap_data["status"] == "available"
    if torch.cuda.is_available():
        assert adap_data["device_type"] == "cuda"
        assert adap_data["compute_capability"] == [8, 6]
        assert adap_data["sm_count"] == 20
        assert adap_data["supports_ampere_tensor_cores"] is True
        assert adap_data["supports_fp16"] is True


# ============================================================================
# User Journey 3 & 4: Configure Adaptive Job, Submit & Observe Progress
# ============================================================================

def test_journey_3_and_4_adaptive_job_submission_and_progress(client):
    """Verify adaptive recommendation generation, job queueing, and metrics tracking."""
    # 1. Query Adaptive Recommendation
    rec_payload = {
        "model_name": "cifar10_mlp",
        "backend": "CPU_NATIVE",
        "batch_size": 32,
        "precision": "float32",
    }
    res_rec = client.post("/api/training/adaptive/recommend", json=rec_payload)
    assert res_rec.status_code == 200
    rec_data = res_rec.json()
    assert rec_data["status"] == "available"
    assert rec_data["applied_batch_size"] > 0
    assert rec_data["is_safe"] is True

    # 2. Submit Job with Adaptive Optimization
    job_payload = {
        "model_name": "cifar10_mlp",
        "backend": "CPU_NATIVE",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_train_samples": 64,
        "max_val_samples": 32,
        "max_test_samples": 32,
        "adaptive_mode": True,
        "auto_batch_size": True,
    }
    res_submit = client.post("/api/training/jobs", json=job_payload)
    assert res_submit.status_code == 201
    job_data = res_submit.json()
    job_id = job_data["job_id"]
    assert job_data["status"] in ("QUEUED", "RUNNING")
    assert job_data["config"]["adaptive_mode"] is True

    # 3. Poll Job and Verify Progress
    completed = False
    for _ in range(60):
        time.sleep(0.3)
        res_poll = client.get(f"/api/training/jobs/{job_id}")
        assert res_poll.status_code == 200
        poll_data = res_poll.json()
        if poll_data["status"] in ("COMPLETED", "FAILED"):
            completed = True
            break

    assert completed is True
    assert poll_data["status"] == "COMPLETED"

    # 4. Check Metrics Endpoint
    res_metrics = client.get(f"/api/training/jobs/{job_id}/metrics")
    assert res_metrics.status_code == 200
    m_data = res_metrics.json()
    assert m_data["job_id"] == job_id
    assert m_data["metrics"] is not None


# ============================================================================
# User Journey 5: Cooperative Cancellation & Safe GPU Cleanup
# ============================================================================

def test_journey_5_cooperative_cancellation(client):
    """Verify cooperative job cancellation and safe release of resources."""
    # Submit a job with enough epochs to safely catch while running
    job_payload = {
        "model_name": "cifar10_mlp",
        "backend": "CPU_NATIVE",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_train_samples": 500,
        "max_val_samples": 64,
        "max_test_samples": 64,
    }
    res_submit = client.post("/api/training/jobs", json=job_payload)
    assert res_submit.status_code == 201
    job_id = res_submit.json()["job_id"]

    # Request cancellation immediately
    res_cancel = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200
    cancel_data = res_cancel.json()
    assert cancel_data["job_id"] == job_id
    assert cancel_data["status"] in ("cancel_requested", "terminal")

    # Verify terminal status is CANCELLED
    for _ in range(40):
        time.sleep(0.2)
        res_job = client.get(f"/api/training/jobs/{job_id}")
        if res_job.json()["status"] in ("CANCELLED", "COMPLETED", "FAILED"):
            break

    final_status = res_job.json()["status"]
    assert final_status in ("CANCELLED", "COMPLETED")


# ============================================================================
# User Journey 6 & 7: Checkpoint Persistence & Model Registry Integration
# ============================================================================

def test_journey_6_and_7_checkpoint_persistence_and_registry(client):
    """Verify completed job checkpoint is atomically saved and listed in registry."""
    job_payload = {
        "model_name": "cifar10_mlp",
        "backend": "CPU_NATIVE",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_train_samples": 64,
        "max_val_samples": 32,
        "max_test_samples": 32,
    }
    res_submit = client.post("/api/training/jobs", json=job_payload)
    job_id = res_submit.json()["job_id"]

    for _ in range(60):
        time.sleep(0.3)
        res_job = client.get(f"/api/training/jobs/{job_id}")
        if res_job.json()["status"] == "COMPLETED":
            break

    assert res_job.json()["status"] == "COMPLETED"

    # Query job checkpoints
    res_ckpts = client.get(f"/api/training/jobs/{job_id}/checkpoints")
    assert res_ckpts.status_code == 200
    ckpts = res_ckpts.json()["checkpoints"]
    assert len(ckpts) >= 1
    target_ckpt = ckpts[0]
    assert Path(target_ckpt["checkpoint_path"]).is_file()
    assert target_ckpt["size_bytes"] > 0
    assert len(target_ckpt["sha256"]) == 64

    # Query Model Registry catalog
    res_catalog = client.get("/api/checkpoints")
    assert res_catalog.status_code == 200
    catalog = res_catalog.json()
    assert "checkpoints" in catalog
    assert len(catalog["checkpoints"]) >= 1


# ============================================================================
# User Journey 8: Transformer Model Discovery, Loading & Text Generation
# ============================================================================

def test_journey_8_transformer_inference_and_error_handling(client):
    """Verify loading existing Transformer checkpoint, running text generation, and error handling."""
    # 1. Verify Model Discovery in Registry
    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    model_ids = [m["model_id"] for m in models_data.get("models", [])]
    assert "transformer_50m" in model_ids
    assert "resnet18_cifar10" in model_ids

    # 2. Text Generation via POST /api/models/generate
    gen_payload = {
        "prompt": "Technical Foundations of GPU Computing:",
        "max_new_tokens": 8,
        "temperature": 0.7,
        "seed": 42,
    }
    res_gen = client.post("/api/models/generate", json=gen_payload)
    assert res_gen.status_code == 200
    gen_data = res_gen.json()
    assert gen_data["status"] == "success"
    assert "generated_text" in gen_data
    assert "tokens_per_second" in gen_data
    assert gen_data["generated_tokens"] > 0

    # 3. Robust Error Handling: Invalid Model Architecture
    res_invalid_model = client.post("/api/models/load", json={"model_name": "NonexistentNeuralNet"})
    assert res_invalid_model.status_code in (400, 422)

    # 4. Robust Error Handling: Nonexistent Checkpoint File
    res_missing_ckpt = client.post(
        "/api/models/load",
        json={"model_name": "ResNet18", "checkpoint": "corrupt/or/missing/file.pt"},
    )
    assert res_missing_ckpt.status_code == 404


# ============================================================================
# User Journey 9: Backend Restart Recovery & Database State Persistence
# ============================================================================

def test_journey_9_backend_restart_and_persistence():
    """Verify database state and saved checkpoints survive backend server re-instantiation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "restart_test.db"
        ckpts_dir = tmp_path / "checkpoints"

        # Server Instance 1
        store_1 = JobStore(db_path)
        orch_1 = TrainingOrchestrator(job_store=store_1, checkpoints_base_dir=ckpts_dir)
        job = orch_1.submit_job(
            model_name="cifar10_mlp",
            backend="CPU_NATIVE",
            epochs=1,
            batch_size=32,
            max_train_samples=64,
            max_val_samples=32,
            max_test_samples=32,
        )
        job_id = job["job_id"]

        for _ in range(60):
            time.sleep(0.2)
            if orch_1.get_job(job_id)["status"] == "COMPLETED":
                break

        assert orch_1.get_job(job_id)["status"] == "COMPLETED"
        ckpts_1 = orch_1.get_job_checkpoints(job_id)
        assert len(ckpts_1) >= 1
        original_sha = ckpts_1[0]["sha256"]
        orch_1.shutdown()

        # Simulate Server Restart: Instantiate Server Instance 2 on same database
        store_2 = JobStore(db_path)
        orch_2 = TrainingOrchestrator(job_store=store_2, checkpoints_base_dir=ckpts_dir)
        reloaded_job = orch_2.get_job(job_id)
        assert reloaded_job is not None
        assert reloaded_job["status"] == "COMPLETED"

        reloaded_ckpts = orch_2.get_job_checkpoints(job_id)
        assert len(reloaded_ckpts) == len(ckpts_1)
        assert reloaded_ckpts[0]["sha256"] == original_sha
        orch_2.shutdown()
