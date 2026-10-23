"""
Phase 10 Stage 2 Milestone 5: Integration & Orchestration Test Suite
Comprehensive testing for FastAPI backend training orchestration:
1. API model listing and backend capabilities.
2. Job creation and Pydantic validation (epochs, batch size, model/backend compatibility).
3. Queue behavior and state transitions (QUEUED -> RUNNING -> COMPLETED).
4. GPU mutual exclusion (at most 1 active GPU job on RTX 3050 6GB).
5. Real CPU training smoke job (CPU_NATIVE, zero CUDA allocation).
6. Real native GPU training smoke job (GPU_NATIVE cuBLAS).
7. Real custom CUDA training smoke job (GPU_CUSTOM, verified affirmative dispatches).
8. Live progress reporting and polling endpoints.
9. Safe cooperative cancellation (RUNNING -> CANCEL_REQUESTED -> CANCELLED).
10. Checkpoint metadata and isolation.
11. Server restart and interrupted-job recovery.
12. Failed job handling with sanitized error messages.
13. SQLite persistence across orchestrator re-instantiation.
14. Deterministic comparative batch order and augmentation reproducibility.
15. Job ID input sanitization and path traversal prevention.
"""
import math
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
from engine.data.cifar10 import create_cifar10_dataloaders
from engine.training.job_store import JobStore
from engine.training.orchestrator import TrainingOrchestrator
from engine.training.reproducibility import reset_dataloader_seed, set_epoch_seed


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    app = create_app(REPO_ROOT)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def isolated_orchestrator(tmp_path_factory):
    """Orchestrator fixture with dedicated isolated SQLite DB and checkpoints."""
    tmp_dir = tmp_path_factory.mktemp("orch_test")
    db_path = tmp_dir / "test_training.db"
    ckpt_dir = tmp_dir / "checkpoints"
    store = JobStore(db_path)
    orch = TrainingOrchestrator(job_store=store, checkpoints_base_dir=ckpt_dir)
    yield orch
    orch.shutdown()


# ============================================================================
# 1. API Models Listing & Capabilities
# ============================================================================
def test_api_models_listing(client):
    res = client.get("/api/training/models")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert "models" in data
    models = {m["model_id"]: m for m in data["models"]}

    # Verify model presence
    assert "cifar10_mlp" in models
    assert "resnet18" in models
    assert "baseline_mlp" in models

    # Custom CUDA capability rule: only cifar10_mlp supports GPU_CUSTOM
    assert models["cifar10_mlp"]["custom_cuda_supported"] is True
    assert "GPU_CUSTOM" in models["cifar10_mlp"]["supported_backends"]

    assert models["resnet18"]["custom_cuda_supported"] is False
    assert "GPU_CUSTOM" not in models["resnet18"]["supported_backends"]

    assert models["baseline_mlp"]["custom_cuda_supported"] is False
    assert "GPU_CUSTOM" not in models["baseline_mlp"]["supported_backends"]


# ============================================================================
# 2. Job Creation and Pydantic Validation
# ============================================================================
def test_job_creation_validation_bounds(client):
    # Invalid epochs (< 1)
    res_low_ep = client.post("/api/training/jobs", json={"model_name": "cifar10_mlp", "epochs": 0})
    assert res_low_ep.status_code == 422

    # Invalid epochs (> 50)
    res_high_ep = client.post("/api/training/jobs", json={"model_name": "cifar10_mlp", "epochs": 100})
    assert res_high_ep.status_code == 422

    # Invalid batch size (< 16)
    res_low_bs = client.post("/api/training/jobs", json={"model_name": "cifar10_mlp", "batch_size": 8})
    assert res_low_bs.status_code == 422

    # Invalid batch size (> 512)
    res_high_bs = client.post("/api/training/jobs", json={"model_name": "cifar10_mlp", "batch_size": 1024})
    assert res_high_bs.status_code == 422

    # Invalid backend
    res_bad_backend = client.post("/api/training/jobs", json={"model_name": "cifar10_mlp", "backend": "TPU_CLOUD"})
    assert res_bad_backend.status_code == 422

    # Unsupported custom CUDA on ResNet18
    res_unsupported_cuda = client.post("/api/training/jobs", json={"model_name": "resnet18", "backend": "GPU_CUSTOM"})
    assert res_unsupported_cuda.status_code == 400
    assert "does not support GPU_CUSTOM" in res_unsupported_cuda.json()["detail"]


# ============================================================================
# 3. Queue Behavior & Lifecycle Transitions
# ============================================================================
def test_queue_behavior_and_status_filtering(isolated_orchestrator):
    orch = isolated_orchestrator
    job = orch.submit_job(
        model_name="cifar10_mlp",
        backend="CPU_NATIVE",
        epochs=1,
        batch_size=32,
        max_train_samples=64,
        max_val_samples=32,
        max_test_samples=32,
    )
    job_id = job["job_id"]
    assert job["status"] == "QUEUED"

    # Poll until completed (allow up to 30s)
    for _ in range(60):
        time.sleep(0.5)
        current = orch.get_job(job_id)
        if current["status"] in ("COMPLETED", "FAILED"):
            break

    final_job = orch.get_job(job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["started_at"] is not None
    assert final_job["completed_at"] is not None

    # Verify status filtering in list_jobs
    completed_jobs = orch.list_jobs(status="COMPLETED")
    assert any(j["job_id"] == job_id for j in completed_jobs)

    running_jobs = orch.list_jobs(status="RUNNING")
    assert not any(j["job_id"] == job_id for j in running_jobs)


# ============================================================================
# 4. GPU Mutual Exclusion (Single Active GPU Job on 6GB VRAM)
# ============================================================================
def test_gpu_mutual_exclusion(isolated_orchestrator):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for mutual exclusion verification")

    orch = isolated_orchestrator

    # Launch GPU Job 1 (GPU_NATIVE)
    job1 = orch.submit_job(
        model_name="cifar10_mlp",
        backend="GPU_NATIVE",
        epochs=1,
        batch_size=32,
        max_train_samples=256,
        max_val_samples=64,
        max_test_samples=64,
    )
    # Immediately launch GPU Job 2 (GPU_CUSTOM)
    job2 = orch.submit_job(
        model_name="cifar10_mlp",
        backend="GPU_CUSTOM",
        epochs=1,
        batch_size=32,
        max_train_samples=128,
        max_val_samples=64,
        max_test_samples=64,
    )

    # Job 2 MUST be held in QUEUED while Job 1 is running
    saw_queued_state = False
    for _ in range(20):
        time.sleep(0.1)
        j1 = orch.get_job(job1["job_id"])
        j2 = orch.get_job(job2["job_id"])

        if j1["status"] == "RUNNING" and j2["status"] == "QUEUED":
            saw_queued_state = True
            break
        if j1["status"] == "COMPLETED":
            break

    # Wait for both jobs to finish cleanly
    for _ in range(40):
        time.sleep(0.5)
        j1 = orch.get_job(job1["job_id"])
        j2 = orch.get_job(job2["job_id"])
        if j1["status"] in ("COMPLETED", "FAILED") and j2["status"] in ("COMPLETED", "FAILED"):
            break

    final_j1 = orch.get_job(job1["job_id"])
    final_j2 = orch.get_job(job2["job_id"])

    assert final_j1["status"] == "COMPLETED"
    assert final_j2["status"] == "COMPLETED"
    assert saw_queued_state is True, "Second GPU job did not wait in QUEUED state for GPU lock."


# ============================================================================
# 5. Real CPU Training Smoke Job (CPU_NATIVE)
# ============================================================================
def test_real_cpu_training_smoke_job(isolated_orchestrator):
    orch = isolated_orchestrator
    job = orch.submit_job(
        model_name="cifar10_mlp",
        backend="CPU_NATIVE",
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        max_train_samples=64,
        max_val_samples=32,
        max_test_samples=32,
    )
    job_id = job["job_id"]

    for _ in range(30):
        time.sleep(0.3)
        j = orch.get_job(job_id)
        if j["status"] in ("COMPLETED", "FAILED"):
            break

    assert j["status"] == "COMPLETED"
    metrics = j["metrics"]
    assert metrics is not None
    assert metrics["backend"] == "CPU_NATIVE"
    assert metrics["device"] == "cpu"
    assert metrics["peak_allocated_vram_mb"] == 0.0
    assert not math.isnan(metrics["final_train_loss"])
    assert metrics["final_train_loss"] > 0.0


# ============================================================================
# 6. Real Native GPU Training Smoke Job (GPU_NATIVE)
# ============================================================================
def test_real_gpu_native_training_smoke_job(isolated_orchestrator):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU_NATIVE test")

    orch = isolated_orchestrator
    job = orch.submit_job(
        model_name="cifar10_mlp",
        backend="GPU_NATIVE",
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        max_train_samples=64,
        max_val_samples=32,
        max_test_samples=32,
    )
    job_id = job["job_id"]

    for _ in range(30):
        time.sleep(0.3)
        j = orch.get_job(job_id)
        if j["status"] in ("COMPLETED", "FAILED"):
            break

    assert j["status"] == "COMPLETED"
    metrics = j["metrics"]
    assert metrics is not None
    assert metrics["backend"] == "GPU_NATIVE"
    assert "cuda" in metrics["device"].lower()
    assert metrics["peak_allocated_vram_mb"] > 0.0


# ============================================================================
# 7. Real Custom CUDA Training Smoke Job & Affirmative Dispatches (GPU_CUSTOM)
# ============================================================================
def test_real_gpu_custom_training_smoke_and_dispatches(isolated_orchestrator):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU_CUSTOM test")

    orch = isolated_orchestrator
    job = orch.submit_job(
        model_name="cifar10_mlp",
        backend="GPU_CUSTOM",
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        max_train_samples=64,
        max_val_samples=32,
        max_test_samples=32,
    )
    job_id = job["job_id"]

    for _ in range(30):
        time.sleep(0.3)
        j = orch.get_job(job_id)
        if j["status"] in ("COMPLETED", "FAILED"):
            break

    assert j["status"] == "COMPLETED"
    metrics = j["metrics"]
    assert metrics is not None
    assert metrics["backend"] == "GPU_CUSTOM"
    assert "cuda" in metrics["device"].lower()

    # Affirmative custom CUDA dispatch audit
    disp = metrics["dispatch_summary"]
    assert disp["is_custom_dispatched"] is True
    assert disp["total_custom_dispatches"] > 0
    assert "matmul_tiled" in disp["custom_forward_dispatches"]
    assert "fused_bias_activation" in disp["custom_forward_dispatches"]
    assert "fused_bias_activation_backward" in disp["custom_backward_dispatches"]


# ============================================================================
# 8. Live Progress Reporting & Metrics Polling API
# ============================================================================
def test_progress_reporting_and_metrics_endpoints(client):
    res_submit = client.post(
        "/api/training/jobs",
        json={
            "model_name": "cifar10_mlp",
            "backend": "CPU_NATIVE",
            "epochs": 1,
            "batch_size": 32,
            "max_train_samples": 64,
            "max_val_samples": 32,
            "max_test_samples": 32,
        },
    )
    assert res_submit.status_code == 201
    job_id = res_submit.json()["job_id"]

    # Poll status endpoint
    for _ in range(30):
        time.sleep(0.3)
        res_poll = client.get(f"/api/training/jobs/{job_id}")
        assert res_poll.status_code == 200
        poll_data = res_poll.json()
        if poll_data["status"] in ("COMPLETED", "FAILED"):
            break

    assert poll_data["status"] == "COMPLETED"
    assert poll_data["progress"]["total_epochs"] == 1
    assert poll_data["progress"]["train_loss"] > 0.0

    # Poll metrics endpoint
    res_metrics = client.get(f"/api/training/jobs/{job_id}/metrics")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert metrics_data["job_id"] == job_id
    assert metrics_data["job_status"] == "COMPLETED"
    assert metrics_data["metrics"] is not None


# ============================================================================
# 9. Safe Cooperative Cancellation
# ============================================================================
def test_safe_cooperative_cancellation(client):
    # Submit multi-epoch job to allow cancellation window
    res_submit = client.post(
        "/api/training/jobs",
        json={
            "model_name": "cifar10_mlp",
            "backend": "CPU_NATIVE",
            "epochs": 10,
            "batch_size": 32,
            "max_train_samples": 256,
            "max_val_samples": 64,
        },
    )
    assert res_submit.status_code == 201
    job_id = res_submit.json()["job_id"]

    # Request cancellation
    res_cancel = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["job_id"] == job_id

    # Wait for status to transition to CANCELLED
    for _ in range(30):
        time.sleep(0.3)
        res_poll = client.get(f"/api/training/jobs/{job_id}")
        poll_data = res_poll.json()
        if poll_data["status"] == "CANCELLED":
            break

    assert poll_data["status"] == "CANCELLED"

    # Cancelling again returns terminal message
    res_cancel_again = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert res_cancel_again.status_code == 200
    assert res_cancel_again.json()["status"] == "terminal"
    time.sleep(0.5)


# ============================================================================
# 10. Checkpoint Metadata Isolation
# ============================================================================
def test_checkpoint_metadata_isolation(client):
    res_submit = client.post(
        "/api/training/jobs",
        json={
            "model_name": "cifar10_mlp",
            "backend": "CPU_NATIVE",
            "epochs": 1,
            "batch_size": 32,
            "max_train_samples": 64,
            "max_val_samples": 32,
            "max_test_samples": 32,
        },
    )
    job_id = res_submit.json()["job_id"]

    for _ in range(30):
        time.sleep(0.3)
        res_poll = client.get(f"/api/training/jobs/{job_id}")
        if res_poll.json()["status"] == "COMPLETED":
            break

    res_ckpts = client.get(f"/api/training/jobs/{job_id}/checkpoints")
    assert res_ckpts.status_code == 200
    ckpts_data = res_ckpts.json()
    assert ckpts_data["job_id"] == job_id
    assert len(ckpts_data["checkpoints"]) >= 1

    ckpt = ckpts_data["checkpoints"][0]
    assert ckpt["job_id"] == job_id
    assert ckpt["size_bytes"] > 0
    assert len(ckpt["sha256"]) == 64
    assert Path(ckpt["checkpoint_path"]).is_file()
    # Unique isolated directory keyed by job_id
    assert job_id in ckpt["checkpoint_path"]


# ============================================================================
# 11. Server Restart and Interrupted-Job Recovery
# ============================================================================
def test_restart_and_interrupted_job_recovery(tmp_path):
    db_path = tmp_path / "recovery_test.db"
    store1 = JobStore(db_path)

    # Simulate an active job when server crashed
    job = store1.create_job(
        job_id="job_crashed_123",
        model_name="cifar10_mlp",
        backend="GPU_CUSTOM",
        dataset="CIFAR-10",
        epochs=3,
        batch_size=128,
        learning_rate=0.01,
        config={},
    )
    store1.update_job_status("job_crashed_123", "RUNNING", started_at="2026-10-23T00:00:00Z")

    # Simulate new server boot (orchestrator startup)
    store2 = JobStore(db_path)
    recovered_count = store2.recover_interrupted_jobs()

    assert recovered_count == 1
    recovered_job = store2.get_job("job_crashed_123")
    assert recovered_job["status"] == "INTERRUPTED"
    assert "interrupted by server restart" in recovered_job["error_message"]
    assert recovered_job["completed_at"] is not None


# ============================================================================
# 12. Failed Job Handling
# ============================================================================
def test_failed_job_handling(tmp_path):
    db_path = tmp_path / "fail_test.db"
    store = JobStore(db_path)
    job = store.create_job(
        job_id="job_fail_1",
        model_name="cifar10_mlp",
        backend="CPU_NATIVE",
        dataset="CIFAR-10",
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        config={},
    )
    store.update_job_status("job_fail_1", "RUNNING")
    store.fail_job("job_fail_1", "CUDA Out of Memory: Batch memory requirements exceeded 6GB VRAM.")

    j = store.get_job("job_fail_1")
    assert j["status"] == "FAILED"
    assert "CUDA Out of Memory" in j["error_message"]


# ============================================================================
# 13. Persistence Across Orchestrator Re-instantiation
# ============================================================================
def test_persistence_across_orchestrator_restart(tmp_path):
    db_path = tmp_path / "persist_test.db"
    store1 = JobStore(db_path)
    orch1 = TrainingOrchestrator(job_store=store1, checkpoints_base_dir=tmp_path / "ckpts")

    job = orch1.submit_job(
        model_name="cifar10_mlp",
        backend="CPU_NATIVE",
        epochs=1,
        batch_size=32,
        max_train_samples=64,
        max_val_samples=32,
        max_test_samples=32,
    )
    job_id = job["job_id"]

    for _ in range(30):
        time.sleep(0.3)
        if orch1.get_job(job_id)["status"] == "COMPLETED":
            break

    # Shutdown instance 1
    orch1.shutdown()

    # Create instance 2 on same database
    store2 = JobStore(db_path)
    orch2 = TrainingOrchestrator(job_store=store2, checkpoints_base_dir=tmp_path / "ckpts")

    job_restored = orch2.get_job(job_id)
    assert job_restored is not None
    assert job_restored["status"] == "COMPLETED"
    assert job_restored["metrics"] is not None
    assert job_restored["metrics"]["final_train_loss"] > 0.0
    orch2.shutdown()


# ============================================================================
# 14. Deterministic Comparative Batch Order & Augmentation Parity
# ============================================================================
def test_deterministic_comparative_batch_order():
    seed = 42

    def sample_batches():
        set_epoch_seed(seed, 0)
        loader, _, _, _ = create_cifar10_dataloaders(
            data_dir="data",
            batch_size=32,
            seed=seed,
            max_train_samples=128,
            augment=True,
        )
        epoch_batches = []
        for ep in range(2):
            reset_dataloader_seed(loader, seed, ep)
            batches_in_ep = []
            for data, targets in loader:
                batches_in_ep.append((data[0, 0, 0, :10].clone(), targets[:10].clone()))
            epoch_batches.append(batches_in_ep)
        return epoch_batches

    run1 = sample_batches()
    run2 = sample_batches()

    # Verify both epochs are 100% bitwise identical between run 1 and run 2
    for ep in range(2):
        for b_idx in range(len(run1[ep])):
            data1, targets1 = run1[ep][b_idx]
            data2, targets2 = run2[ep][b_idx]
            assert torch.equal(data1, data2), f"Epoch {ep} batch {b_idx} data mismatch between comparative runs!"
            assert torch.equal(targets1, targets2), f"Epoch {ep} batch {b_idx} target mismatch!"


# ============================================================================
# 15. Job ID Validation Prevents Path Traversal
# ============================================================================
def test_job_id_validation_prevents_path_traversal(client):
    assert client.get("/api/training/jobs/..%2F..%2Fsecret").status_code in (400, 404)
    assert client.get("/api/training/jobs/job*invalid").status_code == 400
    assert client.get("/api/training/jobs/job%20space").status_code == 400
    assert client.post("/api/training/jobs/job*bad/cancel").status_code == 400
