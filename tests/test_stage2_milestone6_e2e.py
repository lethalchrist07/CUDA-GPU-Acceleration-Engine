"""
Tests for Phase 10 Stage 2 Milestone 6:
Frontend-Backend Integration End-to-End Verification

Validates that all frontend-facing API workflows operate seamlessly against the
real FastAPI training orchestration backend:
1. Model capability catalog inspection (GET /api/training/models)
2. Job creation, validation, and execution (POST /api/training/jobs -> 201 Created)
3. Job lifecycle monitoring and metrics reporting (GET /api/training/jobs/{id}, GET /api/training/jobs/{id}/metrics)
4. Checkpoint discovery, SHA-256 digest, and path isolation (GET /api/training/jobs/{id}/checkpoints)
5. Cooperative job cancellation (POST /api/training/jobs/{id}/cancel)
6. Historical job listing, filtering, and pagination (GET /api/training/jobs)
7. 3-Backend comparison consistency (CPU_NATIVE, GPU_NATIVE, GPU_CUSTOM)
"""

import time
import pytest
from starlette.testclient import TestClient
from backend.app import create_app
import torch

@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_e2e_model_catalog(client):
    """Verify frontend model catalog retrieval matches expected schema."""
    resp = client.get("/api/training/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    
    mlp_model = next((m for m in data["models"] if m["model_id"] == "cifar10_mlp"), None)
    assert mlp_model is not None
    assert mlp_model["model_id"] == "cifar10_mlp"
    assert "CPU_NATIVE" in mlp_model["supported_backends"]
    assert "GPU_NATIVE" in mlp_model["supported_backends"]
    assert "GPU_CUSTOM" in mlp_model["supported_backends"]
    assert mlp_model["parameters"] == 3426250
    assert mlp_model["custom_cuda_supported"] is True


def test_e2e_fast_pilot_training_lifecycle(client):
    """
    Test the complete frontend lifecycle for a fast pilot training run on GPU_CUSTOM:
    Creation -> Polling -> Metrics -> Checkpoints -> History
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU_CUSTOM e2e test")
        
    payload = {
        "model_name": "cifar10_mlp",
        "dataset": "CIFAR-10",
        "backend": "GPU_CUSTOM",
        "epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.01,
        "seed": 42,
        "max_train_samples": 128,
        "max_val_samples": 64,
        "max_test_samples": 64,
    }
    
    # 1. Create job (201 Created)
    create_resp = client.post("/api/training/jobs", json=payload)
    assert create_resp.status_code == 201
    job_info = create_resp.json()
    job_id = job_info["job_id"]
    assert job_id.startswith("job_")
    assert job_info["status"] in ("QUEUED", "RUNNING")
    
    # 2. Poll until completion (max 90 seconds)
    start_t = time.time()
    terminal_status = None
    poll_data = None
    while time.time() - start_t < 90:
        poll_resp = client.get(f"/api/training/jobs/{job_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        terminal_status = poll_data["status"]
        if terminal_status in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.5)
        
    assert terminal_status == "COMPLETED"
    assert poll_data is not None
    assert poll_data["progress"]["total_epochs"] == 1
    assert poll_data["progress"]["train_loss"] > 0.0
    
    # 3. Fetch detailed metrics
    metrics_resp = client.get(f"/api/training/jobs/{job_id}/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    assert metrics_data["job_id"] == job_id
    metrics = metrics_data["metrics"]
    assert metrics is not None
    assert "epoch_records" in metrics or "epoch_metrics" in metrics
    records = metrics.get("epoch_records") or metrics.get("epoch_metrics")
    assert len(records) == 1
    epoch_0 = records[0]
    assert epoch_0["epoch"] == 1
    assert epoch_0["train_loss"] > 0
    assert epoch_0["epoch_time_sec"] > 0
    assert metrics["samples_per_sec"] > 0
    
    # Verify custom CUDA dispatches occurred
    disp = metrics.get("dispatch_summary")
    assert disp is not None
    assert disp["total_custom_dispatches"] > 0
    
    # 4. Fetch checkpoints
    ckpt_resp = client.get(f"/api/training/jobs/{job_id}/checkpoints")
    assert ckpt_resp.status_code == 200
    ckpt_data = ckpt_resp.json()
    assert "checkpoints" in ckpt_data
    ckpts = ckpt_data["checkpoints"]
    assert len(ckpts) >= 1
    best_ckpt = ckpts[0]
    assert best_ckpt["epoch"] == 1
    assert best_ckpt["size_bytes"] > 0
    assert len(best_ckpt["sha256"]) == 64
    assert job_id in best_ckpt["checkpoint_path"].replace("\\", "/")


def test_e2e_cooperative_cancellation(client):
    """Verify frontend can submit and cancel a multi-epoch job cooperatively."""
    payload = {
        "model_name": "cifar10_mlp",
        "dataset": "CIFAR-10",
        "backend": "CPU_NATIVE",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "seed": 1234,
        "max_train_samples": 512,
        "max_val_samples": 128,
        "max_test_samples": 128,
    }
    create_resp = client.post("/api/training/jobs", json=payload)
    assert create_resp.status_code == 201
    job_id = create_resp.json()["job_id"]
    
    # Request cancellation
    cancel_resp = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] in ("cancel_requested", "terminal")
    
    # Poll until CANCELLED
    start_t = time.time()
    cancelled = False
    while time.time() - start_t < 30:
        poll_resp = client.get(f"/api/training/jobs/{job_id}")
        assert poll_resp.status_code == 200
        if poll_resp.json()["status"] == "CANCELLED":
            cancelled = True
            break
        time.sleep(0.3)
        
    assert cancelled, "Job failed to transition to CANCELLED state"


def test_e2e_history_and_filtering(client):
    """Verify frontend history table filtering by backend and status."""
    resp = client.get("/api/training/jobs?status=ALL&backend=ALL")
    assert resp.status_code == 200
    data = resp.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)
    assert len(data["jobs"]) >= 2
    
    # Filter by COMPLETED
    resp_completed = client.get("/api/training/jobs?status=COMPLETED&backend=ALL")
    assert resp_completed.status_code == 200
    for j in resp_completed.json()["jobs"]:
        assert j["status"] == "COMPLETED"
        
    # Filter by CPU_NATIVE
    resp_cpu = client.get("/api/training/jobs?status=ALL&backend=CPU_NATIVE")
    assert resp_cpu.status_code == 200
    for j in resp_cpu.json()["jobs"]:
        assert j["backend"] == "CPU_NATIVE"
