import os
import pytest
import torch
import torch.nn as nn
from engine.device.device_manager import DeviceManager, get_device
from engine.models.baseline_mlp import BaselineMLP
from engine.evaluation.metrics import compare_tensors, compute_accuracy
from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.data.synthetic import SyntheticClassificationDataset, create_synthetic_dataloaders


def test_cuda_detection():
    """Verify PyTorch recognizes the physical NVIDIA GPU and CUDA driver."""
    assert torch.cuda.is_available(), "CUDA must be available in the active PyTorch environment"
    assert torch.cuda.device_count() >= 1, "At least 1 CUDA device must be detected"

    dev_name = torch.cuda.get_device_name(0)
    assert "RTX 3050" in dev_name or "GeForce" in dev_name, f"Expected RTX 3050, found: {dev_name}"

    cap = torch.cuda.get_device_capability(0)
    assert cap == (8, 6), f"Expected Ampere Compute Capability (8, 6), found: {cap}"


def test_device_manager():
    """Verify DeviceManager device resolution and memory safety queries."""
    cuda_dev = DeviceManager.get_device("cuda")
    assert cuda_dev.type == "cuda"
    assert cuda_dev.index == 0

    cpu_dev = DeviceManager.get_device("cpu")
    assert cpu_dev.type == "cpu"

    mem_info = DeviceManager.get_memory_info(0)
    assert mem_info["total_mb"] > 5000.0, f"Expected ~6GB VRAM, got: {mem_info['total_mb']} MB"
    assert mem_info["free_mb"] > 1000.0, f"Free VRAM unexpectedly low: {mem_info['free_mb']} MB"

    headroom_ok = DeviceManager.check_memory_headroom(required_mb=100.0)
    assert headroom_ok is True


def test_model_cpu_execution():
    """Verify BaselineMLP model forward pass on CPU."""
    model = BaselineMLP(input_dim=64, hidden_dims=[128, 64], output_dim=10)
    model.to("cpu")
    model.eval()

    x = torch.randn(16, 64, device="cpu")
    with torch.no_grad():
        out = model(x)

    assert out.shape == (16, 10)
    assert torch.isfinite(out).all()
    assert model.device.type == "cpu"


def test_model_gpu_execution():
    """Verify BaselineMLP model forward pass on CUDA GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    model = BaselineMLP(input_dim=64, hidden_dims=[128, 64], output_dim=10)
    model.to(device)
    model.eval()

    x = torch.randn(16, 64, device=device)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (16, 10)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
    assert model.device.type == "cuda"


def test_cpu_gpu_numerical_parity():
    """Verify that forward pass produces identical numerical results on CPU vs GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    model_cpu = BaselineMLP(input_dim=64, hidden_dims=[128, 64], output_dim=10, dropout=0.0)
    model_cpu.eval()

    # Create identical GPU model with shared state_dict
    model_gpu = BaselineMLP(input_dim=64, hidden_dims=[128, 64], output_dim=10, dropout=0.0)
    model_gpu.load_state_dict(model_cpu.state_dict())
    model_gpu.to("cuda:0")
    model_gpu.eval()

    torch.manual_seed(123)
    x_cpu = torch.randn(32, 64)
    x_gpu = x_cpu.clone().to("cuda:0")

    with torch.no_grad():
        out_cpu = model_cpu(x_cpu)
        out_gpu = model_gpu(x_gpu)

    parity = compare_tensors(out_cpu, out_gpu, tolerance=1e-5)
    assert parity["is_close"] is True, f"Parity mismatch! Max error: {parity['max_abs_diff']}"
    assert parity["mismatched_elements"] == 0


def test_checkpoint_roundtrip(tmp_path):
    """Verify atomic checkpoint serialization and deserialization across devices."""
    checkpoint_file = str(tmp_path / "test_model.pt")

    model = BaselineMLP(input_dim=32, hidden_dims=[64], output_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    CheckpointManager.save_checkpoint(
        model=model,
        filepath=checkpoint_file,
        optimizer=optimizer,
        epoch=1,
        metrics={"loss": 0.45},
    )
    assert os.path.exists(checkpoint_file)

    # Restore into fresh model
    fresh_model = BaselineMLP(input_dim=32, hidden_dims=[64], output_dim=4)
    fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)

    meta = CheckpointManager.load_checkpoint(
        filepath=checkpoint_file,
        model=fresh_model,
        optimizer=fresh_opt,
        map_location=torch.device("cpu"),
    )

    assert meta["epoch"] == 1
    assert meta["metrics"]["loss"] == 0.45

    # Confirm weights match exactly
    x = torch.randn(8, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), fresh_model(x), atol=1e-6)
