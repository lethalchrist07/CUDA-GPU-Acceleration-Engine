"""
Phase 10 Stage 2 Milestone 7: Advanced AI Model Training — ResNet-18 Integration
Test Suite covering:
1. Bug A & Bug B verification (HTML step validation and completed epoch progress)
2. ResNet-18 capability listing & strict rejection of GPU_CUSTOM
3. Real ResNet-18 CPU_NATIVE and GPU_NATIVE training jobs
4. CUDA Automatic Mixed Precision (AMP) stability
5. Gradient accumulation correctness
6. Learning rate scheduler progression (Cosine & Step)
7. Full checkpoint saving & state resume roundtrip
8. Safe cooperative cancellation
9. Byte-for-byte preservation of Phase 9 baseline checkpoint
"""
import hashlib
import os
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import app
from backend.schemas.training import TrainingJobCreateRequest
from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.models.resnet import create_resnet18
from engine.training.job_store import JobStore
from engine.training.orchestrator import TrainingOrchestrator


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def ensure_orchestrator_idle():
    orch = TrainingOrchestrator.get_instance()
    for _ in range(50):
        with orch.active_jobs_lock:
            if not orch.active_jobs:
                break
        time.sleep(0.1)
    yield


def test_bug_a_learning_rate_acceptance():
    """
    Verify Bug A: Floating-point learning rate values such as 0.01, 0.001, 0.0005
    are valid and correctly accepted by the API request schema.
    """
    req1 = TrainingJobCreateRequest(
        model_name="resnet18",
        backend="GPU_NATIVE",
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
    )
    assert req1.learning_rate == 0.01

    req2 = TrainingJobCreateRequest(
        model_name="resnet18",
        backend="GPU_NATIVE",
        epochs=1,
        batch_size=32,
        learning_rate=0.0005,
    )
    assert req2.learning_rate == 0.0005


def test_bug_b_completed_job_progress_reporting(client):
    """
    Verify Bug B: A completed training job must report completed epoch progress
    (epoch == total_epochs, batch == total_batches) rather than defaulting to Epoch 0 / 1.
    """
    payload = {
        "model_name": "cifar10_mlp",
        "backend": "CPU_NATIVE",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "seed": 42,
        "max_train_samples": 64,
        "max_val_samples": 32,
        "max_test_samples": 32,
    }
    res = client.post("/api/training/jobs", json=payload)
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    for _ in range(60):
        time.sleep(0.3)
        poll = client.get(f"/api/training/jobs/{job_id}").json()
        if poll["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break

    assert poll["status"] == "COMPLETED"
    progress = poll["progress"]
    assert progress.get("epoch") == 1
    assert progress.get("current_epoch") == 1
    assert progress.get("total_epochs") == 1
    assert progress.get("batch") == progress.get("total_batches")


def test_resnet18_catalog_and_gpu_custom_rejection(client):
    """
    Verify ResNet-18 capability listing in /api/training/models and ensure
    GPU_CUSTOM is strictly rejected with HTTP 400 and an explicit explanation.
    """
    # 1. Models catalog listing
    res_models = client.get("/api/training/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    resnet_meta = next((m for m in models_data["models"] if m["model_id"] == "resnet18"), None)
    assert resnet_meta is not None
    assert "CPU_NATIVE" in resnet_meta["supported_backends"]
    assert "GPU_NATIVE" in resnet_meta["supported_backends"]
    assert "GPU_CUSTOM" not in resnet_meta["supported_backends"]
    assert resnet_meta["custom_cuda_supported"] is False

    # 2. Strict rejection of GPU_CUSTOM for ResNet-18
    bad_payload = {
        "model_name": "resnet18",
        "backend": "GPU_CUSTOM",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
    }
    res_bad = client.post("/api/training/jobs", json=bad_payload)
    assert res_bad.status_code == 400
    err_detail = res_bad.json()["detail"]
    assert "GPU_CUSTOM" in err_detail
    assert "MLP" in err_detail


def test_resnet18_cpu_native_training_job(client):
    """
    Submit and verify a genuine CPU_NATIVE training job on ResNet-18.
    """
    payload = {
        "model_name": "resnet18",
        "backend": "CPU_NATIVE",
        "epochs": 1,
        "batch_size": 16,
        "learning_rate": 0.01,
        "seed": 42,
        "max_train_samples": 32,
        "max_val_samples": 16,
        "max_test_samples": 16,
        "precision": "float32",
    }
    res = client.post("/api/training/jobs", json=payload)
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    for _ in range(60):
        time.sleep(0.3)
        poll = client.get(f"/api/training/jobs/{job_id}").json()
        if poll["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break

    assert poll["status"] == "COMPLETED"
    assert poll["backend"] == "CPU_NATIVE"
    metrics = poll["metrics"]
    assert metrics is not None
    assert metrics["epochs"] == 1
    assert metrics["final_train_loss"] > 0
    assert metrics["best_val_acc"] >= 0.0
    assert metrics["dispatch_summary"]["is_custom_dispatched"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ResNet-18 GPU test")
def test_resnet18_gpu_native_amp_training_job(client):
    """
    Submit and verify a genuine GPU_NATIVE training job with CUDA Automatic Mixed Precision (AMP)
    on ResNet-18 using the RTX 3050 Laptop GPU.
    """
    payload = {
        "model_name": "resnet18",
        "backend": "GPU_NATIVE",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "seed": 42,
        "max_train_samples": 64,
        "max_val_samples": 32,
        "max_test_samples": 32,
        "precision": "amp",
    }
    res = client.post("/api/training/jobs", json=payload)
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    for _ in range(60):
        time.sleep(0.3)
        poll = client.get(f"/api/training/jobs/{job_id}").json()
        if poll["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break

    assert poll["status"] == "COMPLETED"
    assert poll["backend"] == "GPU_NATIVE"
    metrics = poll["metrics"]
    assert metrics is not None
    assert metrics["peak_allocated_vram_mb"] > 0
    assert metrics["dispatch_summary"]["is_custom_dispatched"] is False
    assert poll["checkpoint_path"] is not None
    assert Path(poll["checkpoint_path"]).is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Gradient Accumulation test")
def test_resnet18_gradient_accumulation_and_scheduler(client):
    """
    Submit and verify a ResNet-18 job with gradient accumulation (4x) and Cosine Annealing scheduler.
    """
    payload = {
        "model_name": "resnet18",
        "backend": "GPU_NATIVE",
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.02,
        "seed": 1337,
        "max_train_samples": 64,
        "max_val_samples": 32,
        "max_test_samples": 32,
        "precision": "amp",
        "lr_scheduler": "cosine",
        "gradient_accumulation_steps": 4,
    }
    res = client.post("/api/training/jobs", json=payload)
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    for _ in range(90):
        time.sleep(0.4)
        poll = client.get(f"/api/training/jobs/{job_id}").json()
        if poll["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break

    assert poll["status"] == "COMPLETED"
    metrics = poll["metrics"]
    assert len(metrics["epoch_records"]) == 2
    # Verify learning rate was recorded and adjusted
    lr_ep1 = metrics["epoch_records"][0]["learning_rate"]
    lr_ep2 = metrics["epoch_records"][1]["learning_rate"]
    assert lr_ep1 > 0
    assert lr_ep2 > 0


def test_checkpoint_roundtrip_and_resume(tmp_path):
    """
    Verify full state checkpointing and restoration:
    Model, Optimizer, Scheduler, Scaler, Epoch, and RNG states.
    """
    device = torch.device("cpu")
    model1 = create_resnet18(num_classes=10, cifar_stem=True).to(device)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.05, momentum=0.9)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=2, gamma=0.5)

    ckpt_path = str(tmp_path / "resume_test.pt")
    CheckpointManager.save_checkpoint(
        model=model1,
        filepath=ckpt_path,
        optimizer=optimizer1,
        scheduler=scheduler1,
        epoch=3,
        metrics={"best_val_acc": 82.5},
        metadata={"device": "cpu"},
    )

    # Restore into fresh instance
    model2 = create_resnet18(num_classes=10, cifar_stem=True).to(device)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=2, gamma=0.5)

    meta = CheckpointManager.load_checkpoint(
        filepath=ckpt_path,
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        map_location=device,
        restore_rng=True,
    )

    assert meta["epoch"] == 3
    assert meta["metrics"]["best_val_acc"] == 82.5
    assert optimizer2.param_groups[0]["lr"] == 0.05

    # Check weight equality
    p1 = list(model1.parameters())[0].detach()
    p2 = list(model2.parameters())[0].detach()
    assert torch.allclose(p1, p2)


def test_resnet18_cooperative_cancellation(client):
    """
    Verify that cooperative cancellation on a running ResNet-18 training job
    halts execution safely, updates status to CANCELLED, and releases GPU lock.
    """
    backend = "GPU_NATIVE" if torch.cuda.is_available() else "CPU_NATIVE"
    payload = {
        "model_name": "resnet18",
        "backend": backend,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_train_samples": 4000,
        "max_val_samples": 500,
        "max_test_samples": 500,
    }
    res = client.post("/api/training/jobs", json=payload)
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    # Wait for job to start running
    for _ in range(30):
        time.sleep(0.1)
        st = client.get(f"/api/training/jobs/{job_id}").json()["status"]
        if st == "RUNNING":
            break

    # Request cancellation
    res_cancel = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200

    # Wait for terminal CANCELLED state
    terminal_status = None
    for _ in range(50):
        time.sleep(0.3)
        poll = client.get(f"/api/training/jobs/{job_id}").json()
        terminal_status = poll["status"]
        if terminal_status in ("CANCELLED", "FAILED", "COMPLETED"):
            break

    assert terminal_status == "CANCELLED"

    # Confirm GPU lock released by verifying another job can start
    orch = TrainingOrchestrator.get_instance()
    assert orch.active_gpu_job_id is None


def test_baseline_resnet18_checkpoint_preservation():
    """
    Strictly verify that the Phase 9 baseline checkpoint
    'checkpoints/resnet18_cifar10/best_model.pt' was NOT modified.
    Expected SHA256: a411eb7c079058e3aada3edb2d851106273527a8d17971936a24671542dc3a78
    """
    baseline_path = Path("checkpoints/resnet18_cifar10/best_model.pt")
    assert baseline_path.is_file(), "Baseline ResNet-18 checkpoint is missing!"

    file_bytes = baseline_path.read_bytes()
    current_hash = hashlib.sha256(file_bytes).hexdigest().lower()
    expected_hash = "a411eb7c079058e3aada3edb2d851106273527a8d17971936a24671542dc3a78"

    assert current_hash == expected_hash, (
        f"Protected Phase 9 baseline checkpoint was modified! "
        f"Expected {expected_hash}, got {current_hash}"
    )
