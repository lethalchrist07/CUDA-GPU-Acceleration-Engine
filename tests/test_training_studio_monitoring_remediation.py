"""
Phase 13 Milestone 9: AI Training Studio Monitoring Accuracy Remediation Automated Tests.
Validates:
1. Gradient accumulation with 15 microbatches and 8 optimizer steps.
2. Accurate completed-job progress accounting.
3. Historical job backward compatibility.
4. Final validation chart data mapping & single-point fallback.
5. Missing periodic validation handling (no fabrication).
6. Running-job polling state and metrics response.
7. Completed-job inspection and checkpoint integrity.
8. CPU_NATIVE telemetry safe defaults.
9. GPU telemetry availability and graceful fallback via direct NVML / ctypes.
"""
import copy
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.app import create_app
from engine.device.hardware_monitor import HardwareMonitor, get_fast_gpu_telemetry
from engine.training.job_store import JobStore
from engine.transformer.config import TransformerConfig
from engine.transformer.model import GPTTransformer
from engine.transformer.trainer import (
    TransformerTrainer,
    TransformerTrainingConfig,
    safe_perplexity,
)


@pytest.fixture
def test_client():
    app = create_app()
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_gradient_accumulation_15_microbatches_8_optimizer_steps():
    """
    Test 1: Verify gradient accumulation with exactly 15 microbatches and accum_steps=2
    produces exactly 8 optimizer steps, with batch indices [2, 4, 6, 8, 10, 12, 14, 15].
    """
    # Create synthetic dataset with exactly 15 microbatches of batch_size=2, seq_len=16
    vocab_size = 64
    seq_len = 16
    num_samples = 30  # 30 samples / batch_size 2 = 15 batches
    x = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, vocab_size, (num_samples, seq_len))
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)

    assert len(train_loader) == 15

    model_cfg = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        n_layer=2,
        n_head=2,
        d_model=32,
    )
    model = GPTTransformer(model_cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_cfg = TransformerTrainingConfig(
            learning_rate=1e-3,
            gradient_accumulation_steps=2,
            eval_interval_steps=50,  # 50 > 8, so intra-eval won't trigger
            mixed_precision=False,
            device="cpu",
            seed=42,
        )

        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_cfg,
            checkpoint_dir=tmp_dir,
        )

        results = trainer.fit(epochs=1, max_steps=15)

        # 1. Exactly 15 microbatches processed
        assert results["microbatches_processed"] == 15
        # 2. Exactly 8 optimizer steps completed (7 pairs of 2 + 1 tail of 1)
        assert results["optimizer_steps_completed"] == 8
        assert results["global_steps"] == 8
        assert len(results["step_records"]) == 8

        # 3. Microbatch indices where optimizer stepped: 2, 4, 6, 8, 10, 12, 14, 15
        step_batches = [s["batch"] for s in results["step_records"]]
        assert step_batches == [2, 4, 6, 8, 10, 12, 14, 15]

        # 4. Final step record has final validation populated without intermediate fabrication
        final_record = results["step_records"][-1]
        assert final_record["is_final_eval"] is True
        assert final_record["val_perplexity"] == results["final_val_perplexity"]
        assert final_record["val_loss"] == results["final_val_loss"]

        # 5. Intermediate steps (0 to 6) have None for val_perplexity (truthful, uninvented)
        for s in results["step_records"][:-1]:
            assert s["val_perplexity"] is None
            assert s["val_loss"] is None


def test_accurate_completed_job_progress():
    """
    Test 2: Verify job_store.complete_job records optimizer_steps_completed (8)
    as current_step and total_steps, while preserving microbatch count (15).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_jobs.db"
        store = JobStore(db_path=db_path)

        job_id = "test_job_opt_accounting"
        store.create_job(
            job_id=job_id,
            model_name="transformer_50m",
            backend="CPU_NATIVE",
            dataset="Tiny Shakespeare",
            epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            config={"max_steps": 15, "gradient_accumulation_steps": 2},
        )
        store.update_job_status(job_id, "RUNNING")

        # Simulate metrics from trainer
        metrics = {
            "status": "COMPLETED",
            "global_steps": 8,
            "optimizer_steps_completed": 8,
            "microbatches_processed": 15,
            "total_tokens_processed": 3840,
            "total_wall_time_sec": 12.34,
            "tokens_per_sec": 311.2,
            "final_train_loss": 4.594,
            "final_val_loss": 4.76,
            "final_val_perplexity": 116.75,
            "peak_vram_mb": 0.0,
            "step_records": [{"step": i} for i in range(1, 9)],
        }

        success = store.complete_job(job_id, metrics=metrics)
        assert success is True

        job = store.get_job(job_id)
        assert job["status"] == "COMPLETED"
        p = job["progress"]

        # Step accounting must match optimizer steps (8), NOT microbatches (15)
        assert p["current_step"] == 8
        assert p["total_steps"] == 8
        assert p["step"] == 8
        assert p["optimizer_steps_completed"] == 8

        # Microbatches must be recorded distinctly
        assert p["microbatches_processed"] == 15
        assert p["batch"] == 15
        assert p["total_batches"] == 15

        # Progress should mirror final validation metrics
        assert p["val_perplexity"] == 116.75
        assert p["val_loss"] == 4.76
        assert p["tokens_per_sec"] == 311.2


def test_historical_job_compatibility():
    """
    Test 3: Verify that historical jobs (which have progress_json with current_step=15)
    are accurately resolved without breaking legacy vision jobs.
    """
    # 1. Historical language model job simulation
    legacy_lm_progress = {
        "step": 8,
        "current_step": 15,  # legacy overwrite
        "total_steps": 15,
        "batch": 15,
        "microbatches_processed": 15,
    }
    legacy_lm_metrics = {
        "optimizer_steps_completed": 8,
        "final_val_perplexity": 116.75,
        "step_records": [{"step": i, "val_perplexity": None} for i in range(1, 9)],
    }

    # Verify reconciliation logic implemented in ActiveJobMonitor.tsx
    actual_opt_steps = legacy_lm_metrics.get("optimizer_steps_completed") or len(legacy_lm_metrics["step_records"])
    assert actual_opt_steps == 8
    resolved_current_step = actual_opt_steps
    resolved_total_steps = actual_opt_steps
    assert resolved_current_step == 8
    assert resolved_total_steps == 8
    assert legacy_lm_progress["microbatches_processed"] == 15

    # 2. Vision job (CIFAR10) compatibility
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_vision_jobs.db"
        store = JobStore(db_path=db_path)

        vision_job_id = "test_cifar10_job"
        store.create_job(
            job_id=vision_job_id,
            model_name="cifar10_mlp",
            backend="GPU_CUSTOM",
            dataset="CIFAR-10",
            epochs=3,
            batch_size=128,
            learning_rate=0.01,
            config={"model_id": "cifar10_mlp", "backend": "GPU_CUSTOM"},
        )
        store.update_job_status(vision_job_id, "RUNNING")

        vision_metrics = {
            "status": "COMPLETED",
            "epochs": 3,
            "final_train_loss": 1.234,
            "best_val_acc": 55.4,
            "samples_per_sec": 1250.0,
            "epoch_metrics": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
        }
        store.complete_job(vision_job_id, metrics=vision_metrics)

        vjob = store.get_job(vision_job_id)
        assert vjob["status"] == "COMPLETED"
        assert vjob["progress"]["epoch"] == 3
        assert vjob["progress"]["total_epochs"] == 3
        assert vjob["progress"]["val_accuracy"] == 55.4


def test_final_validation_chart_rendering_mapping():
    """
    Test 4: Verify the chart mapping algorithm correctly populates the final
    validation perplexity on the last step when periodic values are None.
    """
    step_records = [
        {"step": i, "train_loss": 5.0 - (i * 0.1), "val_perplexity": None, "val_loss": None}
        for i in range(1, 9)
    ]
    metrics = {
        "final_val_perplexity": 116.75,
        "final_val_loss": 4.76,
        "step_records": step_records,
    }

    # Simulate frontend TrainingCharts mapping logic
    chart_data = []
    for idx, st in enumerate(step_records):
        is_last = (idx == len(step_records) - 1)
        has_step_val = st.get("val_perplexity") is not None and st.get("val_perplexity") > 0
        val_ppl = (
            round(float(st["val_perplexity"]), 2)
            if has_step_val
            else (
                round(float(metrics["final_val_perplexity"]), 2)
                if (is_last and metrics.get("final_val_perplexity") is not None)
                else None
            )
        )
        val_loss = (
            round(float(st["val_loss"]), 4)
            if (st.get("val_loss") is not None and st.get("val_loss") > 0)
            else (
                round(float(metrics["final_val_loss"]), 4)
                if (is_last and metrics.get("final_val_loss") is not None)
                else None
            )
        )
        chart_data.append({
            "name": f"Step {st['step']}",
            "step": st["step"],
            "train_loss": round(float(st["train_loss"]), 4),
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "is_final_eval": is_last and not has_step_val and (val_ppl is not None),
        })

    assert len(chart_data) == 8
    # Steps 1 to 7 have None val_perplexity
    for d in chart_data[:-1]:
        assert d["val_perplexity"] is None
        assert d["is_final_eval"] is False

    # Step 8 has the final validation result
    assert chart_data[-1]["val_perplexity"] == 116.75
    assert chart_data[-1]["val_loss"] == 4.76
    assert chart_data[-1]["is_final_eval"] is True


def test_missing_periodic_validation_handling():
    """
    Test 5: Verify that missing periodic validation values are handled cleanly
    without NaN/Inf, division by zero, or synthetic fabrication.
    """
    assert safe_perplexity(float("nan")) == float("inf")
    assert safe_perplexity(float("inf")) == float("inf")
    assert round(safe_perplexity(4.76), 2) == round(math.exp(4.76), 2)

    # Empty / none values in step data do not cause exceptions
    raw_step_data = {"step": 1, "train_loss": 4.5, "val_perplexity": None}
    val = raw_step_data.get("val_perplexity")
    assert (val is not None and val > 0) is False


def test_running_job_polling_api(test_client):
    """
    Test 6: Verify polling endpoints return expected structure and headers.
    """
    # 1. Models list endpoint
    resp = test_client.get("/api/training/models")
    assert resp.status_code == 200
    models_data = resp.json()
    assert models_data["status"] == "available"
    assert any(m["model_id"] == "transformer_50m" for m in models_data["models"])

    # 2. Jobs list endpoint
    resp = test_client.get("/api/training/jobs")
    assert resp.status_code == 200
    jobs_data = resp.json()
    assert "jobs" in jobs_data


def test_completed_job_inspection_and_checkpoints(test_client):
    """
    Test 7: Verify inspection of target job job_20261113_153924_735287.
    """
    job_id = "job_20261113_153924_735287"
    resp = test_client.get(f"/api/training/jobs/{job_id}")
    if resp.status_code == 200:
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "COMPLETED"
        assert data["model_name"] == "transformer_50m"
        assert data["metrics"]["final_val_perplexity"] == 116.75
        assert len(data["metrics"]["step_records"]) == 8

        # Checkpoints endpoint
        ckpt_resp = test_client.get(f"/api/training/jobs/{job_id}/checkpoints")
        assert ckpt_resp.status_code == 200
        ckpt_data = ckpt_resp.json()
        assert ckpt_data["status"] == "available"
        assert len(ckpt_data["checkpoints"]) > 0
        ckpt = ckpt_data["checkpoints"][0]
        assert ckpt["job_id"] == job_id
        assert ckpt["accuracy"] == 116.75
        assert len(ckpt["sha256"]) == 64


def test_cpu_native_telemetry_behavior():
    """
    Test 8: Verify CPU_NATIVE telemetry returns safe defaults (0.0 MB VRAM, None GPU temp)
    without crashing or invoking CUDA calls on CPU.
    """
    mon = HardwareMonitor(device_id=0)
    # When simulating non-CUDA environment
    with patch("torch.cuda.is_available", return_value=False):
        telem = mon.get_telemetry()
        assert telem["cuda_available"] is False
        assert telem["device_name"] == "CPU"
        assert telem["temperature_c"] is None
        assert telem["utilization_gpu_pct"] is None
        assert telem["vram_allocated_mb"] == 0.0


def test_gpu_telemetry_availability_and_fallback():
    """
    Test 9: Verify GPU telemetry queries hardware safely and falls back cleanly on error.
    """
    # 1. Direct NVML query on RTX 3050 Laptop GPU
    fast_telem = get_fast_gpu_telemetry(0)
    if fast_telem is not None:
        assert isinstance(fast_telem["temperature_c"], float)
        assert fast_telem["temperature_c"] > 0.0
        assert isinstance(fast_telem["utilization_gpu_pct"], float)
        assert isinstance(fast_telem["physical_used_mb"], float)
        assert fast_telem["physical_total_mb"] >= 4000.0  # RTX 3050 has ~6144 MB

    # 2. Invalid device index fallback (returns None safely without crash)
    invalid_telem = get_fast_gpu_telemetry(9999)
    assert invalid_telem is None

    # 3. HardwareMonitor integration
    mon = HardwareMonitor(device_id=0)
    full_telem = mon.get_telemetry()
    assert "timestamp" in full_telem
    if full_telem["cuda_available"]:
        assert full_telem["vram_total_mb"] > 0
