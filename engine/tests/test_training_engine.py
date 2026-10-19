import os
import copy
import pytest
import torch
import torch.nn as nn
from engine.models.baseline_mlp import BaselineMLP
from engine.data.synthetic import create_synthetic_dataloaders
from engine.training.trainer import ModelTrainer
from engine.training.experiment_tracker import ExperimentTracker


def test_cpu_training(tmp_path):
    """Verify that ModelTrainer runs cleanly on CPU and loss decreases."""
    torch.manual_seed(42)
    train_loader, val_loader = create_synthetic_dataloaders(
        num_samples=200, input_dim=32, num_classes=4, batch_size=32, seed=42
    )
    model = BaselineMLP(input_dim=32, hidden_dims=[64, 32], output_dim=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "checkpoints_cpu"),
    )
    history = trainer.fit(epochs=3, save_checkpoints=True)

    assert len(history["train_loss"]) == 3
    assert history["train_loss"][-1] < history["train_loss"][0], "Loss should decrease during training"
    assert history["total_time_ms"] > 0.0
    assert history["peak_gpu_memory_mb"] == 0.0


def test_gpu_training(tmp_path):
    """Verify that ModelTrainer runs on CUDA GPU, loss decreases, and tracks VRAM."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    train_loader, val_loader = create_synthetic_dataloaders(
        num_samples=300, input_dim=32, num_classes=4, batch_size=32, seed=42
    )
    model = BaselineMLP(input_dim=32, hidden_dims=[64, 32], output_dim=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cuda:0"),
        checkpoint_dir=str(tmp_path / "checkpoints_gpu"),
    )
    history = trainer.fit(epochs=3, save_checkpoints=True)

    assert len(history["train_loss"]) == 3
    assert history["train_loss"][-1] < history["train_loss"][0], "Loss should decrease during training"
    assert history["peak_gpu_memory_mb"] > 0.0, "Should record non-zero peak GPU VRAM"
    assert os.path.exists(tmp_path / "checkpoints_gpu" / "best_model.pt")


def test_amp_training(tmp_path):
    """Verify automatic mixed-precision (AMP) training on CUDA GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    train_loader, val_loader = create_synthetic_dataloaders(
        num_samples=200, input_dim=32, num_classes=4, batch_size=32, seed=42
    )
    model = BaselineMLP(input_dim=32, hidden_dims=[64, 32], output_dim=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cuda:0"),
        use_amp=True,
        checkpoint_dir=str(tmp_path / "checkpoints_amp"),
    )
    history = trainer.fit(epochs=3, save_checkpoints=True)

    assert history["use_amp"] is True
    assert len(history["train_loss"]) == 3
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))


def test_checkpoint_and_resume(tmp_path):
    """Verify that training can be paused, checkpointed, and resumed seamlessly."""
    ckpt_dir = str(tmp_path / "checkpoints_resume")
    torch.manual_seed(42)
    train_loader, val_loader = create_synthetic_dataloaders(
        num_samples=200, input_dim=32, num_classes=4, batch_size=32, seed=42
    )
    model = BaselineMLP(input_dim=32, hidden_dims=[64, 32], output_dim=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    trainer1 = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
    )
    # Train for 2 epochs
    history1 = trainer1.fit(epochs=2, save_checkpoints=True)
    latest_ckpt = os.path.join(ckpt_dir, "latest_checkpoint.pt")
    assert os.path.exists(latest_ckpt)

    # Resume with fresh model and optimizer
    model_resume = BaselineMLP(input_dim=32, hidden_dims=[64, 32], output_dim=4)
    optimizer_resume = torch.optim.Adam(model_resume.parameters(), lr=1e-2)
    trainer2 = ModelTrainer(
        model=model_resume,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_resume,
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
    )
    # Resume for 2 more epochs
    history2 = trainer2.fit(epochs=2, resume_from=latest_ckpt, save_checkpoints=True)

    assert trainer2.current_epoch == 4, "Current epoch should reach 4 after resuming"
    assert len(history2["train_loss"]) == 2


def test_experiment_tracker(tmp_path):
    """Verify JSON and CSV export functionality."""
    results_dir = str(tmp_path / "test_results")
    tracker = ExperimentTracker(results_dir=results_dir)

    exp_data = {
        "experiment_id": "test_exp_01",
        "model_name": "BaselineMLP",
        "parameters": 5000,
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adamw",
        "seed": 42,
        "use_amp": False,
        "total_train_time_ms": 1250.0,
        "avg_epoch_time_ms": 250.0,
        "final_train_loss": 0.25,
        "final_val_loss": 0.30,
        "best_val_accuracy": 92.5,
        "peak_gpu_memory_mb": 150.0,
        "speedup_vs_cpu": 3.2,
    }
    tracker.log_experiment(exp_data)

    assert os.path.exists(tracker.json_file)
    assert os.path.exists(tracker.csv_file)
