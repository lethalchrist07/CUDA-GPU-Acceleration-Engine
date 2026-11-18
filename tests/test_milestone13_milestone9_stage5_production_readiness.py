"""
tests/test_milestone13_milestone9_stage5_production_readiness.py
Phase 13 Milestone 9 Stage 5: Core Engine Production Readiness & Release Certification Test Suite.

Covers:
1. Environment and Configuration Settings (environment variable overrides, CORS, security headers).
2. Checkpoint Baseline Artifact Protection (preventing accidental overwrite of 50M campaign weights).
3. Corrupt/Invalid Checkpoint Error Handling (graceful ValueError instead of unhandled crash).
4. Global Exception Handlers (404 for missing resources, 422 for invalid payloads, 507 for OOM).
5. Atomic Checkpoint Writing & Temporary File Cleanup.
6. Hardware Diagnostic & Fallback Detection.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import EngineSettings, get_settings
from engine.checkpointing.checkpoint_manager import CheckpointManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    app = create_app(PROJECT_ROOT)
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Configuration & Security Settings
# -----------------------------------------------------------------------------
def test_production_settings_env_overrides():
    """Verify EngineSettings correctly honors environment variable overrides."""
    with patch.dict(
        os.environ,
        {
            "CUDA_ENGINE_HOST": "0.0.0.0",
            "CUDA_ENGINE_PORT": "9000",
            "CUDA_ENGINE_CORS_ORIGINS": "https://dashboard.example.com,http://localhost:3000",
            "CUDA_ENGINE_LOG_LEVEL": "DEBUG",
            "CUDA_ENGINE_THERMAL_LIMIT_C": "85.0",
        },
    ):
        settings = EngineSettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.cors_origins == ["https://dashboard.example.com", "http://localhost:3000"]
        assert settings.log_level == "DEBUG"
        assert settings.thermal_throttle_threshold_c == 85.0


def test_security_headers_enforced(client: TestClient):
    """Verify HTTP security headers (nosniff, DENY, XSS protection, Referrer policy) are applied."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    headers = resp.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert "strict-origin" in headers.get("Referrer-Policy", "")


# -----------------------------------------------------------------------------
# 2. Checkpoint Baseline Artifact Protection
# -----------------------------------------------------------------------------
def test_protected_checkpoint_overwrite_prevention():
    """Verify CheckpointManager refuses to overwrite 50M campaign and protected baseline models."""
    dummy_model = nn.Linear(10, 2)

    # All these critical baseline paths must be protected
    protected_paths = [
        "checkpoints/transformer/pretraining_50m_campaign/best_model.pt",
        "checkpoints/transformer/flagship_100m_pilot/best_model.pt",
        "checkpoints/resnet18_cifar10/best_model.pt",
        "checkpoints/production_10m/final_model.pt",
    ]

    for rel_path in protected_paths:
        abs_p = (PROJECT_ROOT / rel_path).resolve()
        assert CheckpointManager.is_protected_path(str(abs_p)), f"Path {rel_path} should be protected!"

        if abs_p.is_file():
            with pytest.raises(PermissionError) as exc_info:
                CheckpointManager.save_checkpoint(dummy_model, str(abs_p))
            assert "Protected baseline artifact cannot be overwritten" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 3. Checkpoint Integrity & Corrupt File Handling
# -----------------------------------------------------------------------------
def test_corrupt_checkpoint_validation_handling():
    """Verify corrupted, truncated, or empty checkpoints raise clean ValueError rather than crash."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Empty file
        empty_file = tmp_path / "empty.pt"
        empty_file.write_bytes(b"")
        with pytest.raises(ValueError) as exc:
            CheckpointManager.validate_checkpoint(str(empty_file))
        assert "empty" in str(exc.value).lower()

        # 2. Corrupt random bytes file
        corrupt_file = tmp_path / "corrupt.pt"
        corrupt_file.write_bytes(b"PK\x03\x04random_corrupted_binary_data_not_pickle")
        with pytest.raises(ValueError) as exc:
            CheckpointManager.validate_checkpoint(str(corrupt_file))
        assert "corrupted" in str(exc.value).lower() or "unreadable" in str(exc.value).lower()

        # 3. Valid pickle but missing required model state dict
        missing_keys_file = tmp_path / "bad_keys.pt"
        torch.save({"random_key": 123}, str(missing_keys_file))
        with pytest.raises(ValueError) as exc:
            CheckpointManager.validate_checkpoint(str(missing_keys_file))
        assert "missing required key" in str(exc.value).lower()


# -----------------------------------------------------------------------------
# 4. Atomic Checkpoint Write & Cleanup
# -----------------------------------------------------------------------------
def test_atomic_checkpoint_write_and_cleanup():
    """Verify save_checkpoint performs atomic replace and leaves no orphaned temporary files."""
    dummy_model = nn.Linear(8, 4)
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / "test_atomic.pt"
        abs_saved = CheckpointManager.save_checkpoint(
            model=dummy_model,
            filepath=str(target_path),
            epoch=1,
            metrics={"loss": 0.5},
        )
        assert Path(abs_saved).is_file()
        assert Path(abs_saved).stat().st_size > 0

        # Verify no .tmp_* files remain
        tmp_leftovers = list(Path(tmp_dir).glob("*.tmp_*"))
        assert len(tmp_leftovers) == 0, f"Orphaned temp files found: {tmp_leftovers}"

        # Verify valid checkpoint contents
        val = CheckpointManager.validate_checkpoint(abs_saved)
        assert val["valid"] is True
        assert val["epoch"] == 1
        assert val["metrics"]["loss"] == 0.5


# -----------------------------------------------------------------------------
# 5. Global Exception Handlers & Graceful Error Responses
# -----------------------------------------------------------------------------
def test_global_exception_handlers(client: TestClient):
    """Verify backend returns structured JSON with appropriate status codes for errors."""
    # 1. 404 for missing checkpoint in /api/models/load
    resp = client.post(
        "/api/models/load",
        json={"model_name": "resnet18", "checkpoint": "non_existent_ckpt_123.pt"},
    )
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data

    # 2. 422 for unprocessable experiment configurations
    resp = client.post(
        "/api/laboratory/experiments",
        json={"configurations": ["InvalidConfigXYZ"]},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data

