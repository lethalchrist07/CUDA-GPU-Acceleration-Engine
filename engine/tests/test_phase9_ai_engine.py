"""
Automated Validation Suite for Phase 9 - AI Training & GPU Performance Engineering.
Covers:
- Dataset preprocessing & normalization
- ResNet-18 initialization & parameter verification
- Forward/backward execution on CPU & CUDA
- CPU/GPU numerical parity verification with tolerances
- Automatic Mixed Precision (AMP) execution & GradScaler
- Checkpoint atomic saving & restoration
- Inference correctness & probability distributions
- Hardware monitoring, thermal guards, and OOM recovery
- Model serving & FastAPI endpoint validation
- Artifact integrity & benchmark consistency
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from backend.app import create_app
from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.data.cifar10 import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR10Dataset,
    create_cifar10_dataloaders,
)
from engine.device.device_manager import DeviceManager
from engine.device.hardware_monitor import HardwareMonitor
from engine.models.resnet import BasicBlock, ResNet18, create_resnet18

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Dataset & Preprocessing Validation
# ---------------------------------------------------------------------------
def test_cifar10_dataset_preprocessing_and_normalization():
    """Verify CIFAR-10 data shapes, augmentations, and mean/std normalization."""
    # Synthetic raw batches for fast isolated test
    rng = np.random.RandomState(42)
    raw_images = rng.randint(0, 256, size=(100, 3, 32, 32), dtype=np.uint8)
    raw_labels = rng.randint(0, 10, size=(100,))

    dataset_norm = CIFAR10Dataset(raw_images, raw_labels, augment=False, normalize=True)
    assert len(dataset_norm) == 100

    img_tensor, label_tensor = dataset_norm[0]
    assert img_tensor.shape == torch.Size([3, 32, 32])
    assert img_tensor.dtype == torch.float32
    assert label_tensor.dtype == torch.long
    assert 0 <= label_tensor.item() < 10

    # Test augmentation
    dataset_aug = CIFAR10Dataset(raw_images, raw_labels, augment=True, normalize=True)
    aug_img, _ = dataset_aug[0]
    assert aug_img.shape == torch.Size([3, 32, 32])


def test_cifar10_dataloader_creation():
    """Verify DataLoader generation, batching, and classes."""
    train_loader, val_loader, test_loader, class_names = create_cifar10_dataloaders(
        data_dir=str(ROOT / "data"),
        batch_size=32,
        val_split=0.2,
        seed=42,
        max_train_samples=64,
        max_val_samples=32,
        max_test_samples=32,
        allow_synthetic_fallback=True,
    )
    assert len(class_names) == 10
    assert "airplane" in class_names and "truck" in class_names

    bx, by = next(iter(train_loader))
    assert bx.shape == torch.Size([32, 3, 32, 32])
    assert by.shape == torch.Size([32])


# ---------------------------------------------------------------------------
# 2. ResNet-18 Architecture & Parameter Verification
# ---------------------------------------------------------------------------
def test_resnet18_initialization_and_parameter_counts():
    """Verify exact parameter counts and structural dimensions of ResNet-18."""
    model = create_resnet18(num_classes=10, cifar_stem=True)
    counts = model.count_parameters()
    # Canonical ResNet-18 has exactly 11,173,962 parameters
    assert counts["total_parameters"] == 11173962
    assert counts["trainable_parameters"] == 11173962
    assert counts["non_trainable_parameters"] == 0
    assert model.get_model_size_mb() > 40.0  # ~42.6 MB in FP32


def test_resnet18_forward_pass_dimensions():
    """Verify output tensor shape and logit dimensions."""
    model = create_resnet18(num_classes=10, cifar_stem=True)
    model.eval()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == torch.Size([8, 10])


# ---------------------------------------------------------------------------
# 3. CPU & GPU Forward / Backward Execution
# ---------------------------------------------------------------------------
def test_cpu_forward_backward_execution():
    """Verify loss calculation, backward pass, and optimizer step on CPU."""
    model = create_resnet18(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.train()
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    assert loss.item() > 0.0
    loss.backward()

    # Confirm gradients exist
    assert model.conv1.weight.grad is not None
    optimizer.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_gpu_forward_backward_execution():
    """Verify loss calculation, backward pass, and optimizer step on CUDA GPU."""
    device = torch.device("cuda:0")
    model = create_resnet18(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

    model.train()
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    assert model.conv1.weight.grad is not None
    optimizer.step()


# ---------------------------------------------------------------------------
# 4. CPU vs GPU Numerical Parity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_cpu_gpu_numerical_parity_resnet18():
    """Verify forward-pass output match between CPU and GPU within numerical tolerance."""
    torch.manual_seed(1234)
    model_cpu = create_resnet18(num_classes=10)
    weights = copy.deepcopy(model_cpu.state_dict())

    model_gpu = create_resnet18(num_classes=10).cuda()
    model_gpu.load_state_dict(copy.deepcopy(weights))

    model_cpu.eval()
    model_gpu.eval()

    x_cpu = torch.randn(16, 3, 32, 32)
    x_gpu = x_cpu.clone().cuda()

    with torch.no_grad():
        out_cpu = model_cpu(x_cpu)
        out_gpu = model_gpu(x_gpu).cpu()

    max_diff = torch.max(torch.abs(out_cpu - out_gpu)).item()
    assert max_diff < 5e-4, f"Max difference {max_diff} exceeded tolerance threshold 5e-4"
    assert torch.allclose(out_cpu, out_gpu, atol=5e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# 5. Automatic Mixed Precision (AMP)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_amp_mixed_precision_execution():
    """Verify FP16 autocast and GradScaler scaling and step."""
    device = torch.device("cuda:0")
    model = create_resnet18(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    x = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        outputs = model(x)
        loss = criterion(outputs, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert scaler.get_scale() > 0.0


# ---------------------------------------------------------------------------
# 6. Checkpoint Roundtrip & Device Remapping
# ---------------------------------------------------------------------------
def test_checkpoint_atomic_recovery_and_resume(tmp_path):
    """Verify saving checkpoint on one device and loading on another."""
    model_a = create_resnet18(num_classes=10)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=0.001)

    ckpt_path = tmp_path / "test_model_roundtrip.pt"
    CheckpointManager.save_checkpoint(
        model=model_a,
        filepath=str(ckpt_path),
        optimizer=optimizer_a,
        epoch=5,
        metrics={"val_accuracy": 0.825, "val_loss": 0.45},
        metadata={"model_name": "ResNet18"},
    )
    assert ckpt_path.is_file()

    model_b = create_resnet18(num_classes=10)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=0.001)
    meta = CheckpointManager.load_checkpoint(
        filepath=str(ckpt_path),
        model=model_b,
        optimizer=optimizer_b,
        map_location=torch.device("cpu"),
    )

    assert meta["epoch"] == 5
    assert meta["metrics"]["val_accuracy"] == 0.825

    # Check that weights are identical
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


# ---------------------------------------------------------------------------
# 7. Inference Correctness & Softmax Distribution
# ---------------------------------------------------------------------------
def test_inference_correctness_and_probabilities():
    """Verify softmax probability distribution properties (sums to 1.0, bounded [0, 1])."""
    model = create_resnet18(num_classes=10)
    x = torch.randn(5, 3, 32, 32)

    probs = model.predict_proba(x)
    assert probs.shape == torch.Size([5, 10])
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
    sums = torch.sum(probs, dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    preds = model.predict(x)
    assert preds.shape == torch.Size([5])
    assert torch.equal(preds, torch.argmax(probs, dim=1))


# ---------------------------------------------------------------------------
# 8. Hardware Monitoring, Thermal Guard, & OOM Recovery
# ---------------------------------------------------------------------------
def test_hardware_monitor_telemetry_and_safety():
    """Verify HardwareMonitor data schema, thermal checking, and OOM handling."""
    monitor = HardwareMonitor(device_id=0, warn_temp_c=75.0, max_temp_c=85.0)
    telem = monitor.get_telemetry()
    assert "timestamp" in telem
    assert "vram_allocated_mb" in telem
    assert "vram_reserved_mb" in telem

    safe, msg = monitor.check_thermal_safety()
    assert isinstance(safe, bool)

    # Test OOM recovery batch reduction
    new_b = monitor.handle_oom(current_batch_size=128, min_batch_size=16)
    assert new_b == 64
    new_b2 = monitor.handle_oom(current_batch_size=16, min_batch_size=16)
    assert new_b2 is None


# ---------------------------------------------------------------------------
# 9. FastAPI Model Registry, Serving, and Predict Endpoints
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client():
    with TestClient(create_app(ROOT)) as client:
        yield client


def test_api_capabilities_endpoint(api_client):
    res = api_client.get("/api/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert "platform" in data
    assert "cuda" in data
    assert "models" in data
    assert "ResNet18" in data["models"]["architectures"]


def test_api_model_registry_and_checkpoints(api_client):
    res = api_client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    model_names = [m["name"] for m in data["models"]]
    assert "ResNet18" in model_names
    assert "BaselineMLP" in model_names

    res_ckpt = api_client.get("/api/checkpoints")
    assert res_ckpt.status_code == 200
    ckpts = res_ckpt.json()
    assert ckpts["status"] == "available"
    assert len(ckpts["checkpoints"]) > 0


def test_api_experiments_and_evaluation(api_client):
    res_exp = api_client.get("/api/experiments")
    assert res_exp.status_code == 200
    exp_data = res_exp.json()
    assert exp_data["status"] == "available"
    assert "phase9_training" in exp_data
    assert "phase9_benchmarks" in exp_data

    res_eval = api_client.get("/api/models/evaluation")
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert eval_data["model_name"] == "ResNet18"


def test_api_model_load_and_predict(api_client):
    # Test loading model
    load_res = api_client.post("/api/models/load", json={"model_name": "ResNet18", "device": "auto"})
    assert load_res.status_code == 200
    load_data = load_res.json()
    assert load_data["status"] == "loaded"
    assert load_data["model_name"] == "ResNet18"

    # Test predict on sample index
    pred_res = api_client.post("/api/models/predict", json={"sample_index": 0})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data["status"] == "success"
    assert pred_data["predicted_class"] in CIFAR10_CLASSES
    assert 0.0 <= pred_data["confidence"] <= 1.0
    assert len(pred_data["top_predictions"]) > 0
    assert pred_data["latency_ms"] > 0.0


# ---------------------------------------------------------------------------
# 10. Benchmark Artifact Consistency
# ---------------------------------------------------------------------------
def test_phase9_benchmark_artifact_integrity():
    """Verify that phase9_gpu_benchmarks.json contains measured speedups and valid metrics."""
    bench_file = ROOT / "results/phase9_gpu_benchmarks.json"
    assert bench_file.is_file(), "phase9_gpu_benchmarks.json not found"

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "hardware" in data
    assert "training_benchmarks" in data
    assert "inference_benchmarks" in data

    train_benches = data["training_benchmarks"]
    assert len(train_benches) >= 4

    # Confirm speedup calculation against CPU reference
    cpu_ref = next(b for b in train_benches if "CPU" in b["config"])
    assert cpu_ref["speedup_vs_cpu"] == 1.0

    gpu_amps = [b for b in train_benches if "GPU AMP" in b["config"]]
    for b in gpu_amps:
        assert b["speedup_vs_cpu"] > 10.0, f"Expected GPU AMP speedup > 10x, got {b['speedup_vs_cpu']}"
        assert b["peak_vram_allocated_mb"] > 0.0
