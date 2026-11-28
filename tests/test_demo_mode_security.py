"""
tests/test_demo_mode_security.py
Stage 7B: Automated Verification Suite for Public Demo Security Mode (CUDA_ENGINE_DEMO_MODE=1).

Verifies:
1. Mutating route protection: blocks training, laboratory experiments, adaptive selection, and model unloading with HTTP 403.
2. Public model whitelist: only resnet18_cifar10, transformer_50m, baseline_mlp permitted; arbitrary checkpoint paths and unapproved models rejected with HTTP 403.
3. Pre-route size limiting: payloads > 32 KB rejected with HTTP 413.
4. Atomic non-blocking inference lock: returns HTTP 429 when predict or generate is running concurrently.
5. Inference safety bounds: 48-token limit, 256-character prompt clamping, and 128-float tensor limit (HTTP 400).
6. Hardware safeguards: 1,024 MB VRAM reserve, 75°C thermal limit, NVML failure fail-closed (HTTP 503), CPU bypass.
7. Public GET endpoint privacy: hardware UUID masking, OS platform sanitization, private model exclusion, artifact log filtering.
8. Standard mode invariance: when CUDA_ENGINE_DEMO_MODE=0, normal developer behavior is preserved.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import os
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import get_settings, reload_settings
from backend.serving import _INFERENCE_LOCK, ModelServer
from backend.data import clear_catalog_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def ensure_clean_settings():
    """Ensure settings are cleanly reloaded and caches flushed for each test."""
    clear_catalog_cache()
    yield
    clear_catalog_cache()
    if "CUDA_ENGINE_DEMO_MODE" in os.environ:
        del os.environ["CUDA_ENGINE_DEMO_MODE"]
    reload_settings()


@pytest.fixture
def demo_client():
    """Test client configured with CUDA_ENGINE_DEMO_MODE=1."""
    os.environ["CUDA_ENGINE_DEMO_MODE"] = "1"
    reload_settings()
    clear_catalog_cache()
    app = create_app(PROJECT_ROOT)
    with TestClient(app) as client:
        yield client
    if "CUDA_ENGINE_DEMO_MODE" in os.environ:
        del os.environ["CUDA_ENGINE_DEMO_MODE"]
    reload_settings()


@pytest.fixture
def standard_client():
    """Test client configured in standard developer mode (CUDA_ENGINE_DEMO_MODE=0)."""
    os.environ["CUDA_ENGINE_DEMO_MODE"] = "0"
    reload_settings()
    clear_catalog_cache()
    app = create_app(PROJECT_ROOT)
    with TestClient(app) as client:
        yield client
    reload_settings()


class TestMutatingRouteProtection:
    """Verify mutating routes return HTTP 403 in public demo mode."""

    def test_training_job_submission_blocked(self, demo_client):
        resp = demo_client.post("/api/training/jobs", json={
            "model_name": "ResNet18",
            "backend": "CPU_NATIVE",
            "dataset": "CIFAR-10",
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
        })
        assert resp.status_code == 403
        assert "disabled in public demo mode" in resp.json()["detail"]

    def test_training_job_cancellation_blocked(self, demo_client):
        resp = demo_client.post("/api/training/jobs/job_12345/cancel")
        assert resp.status_code == 403
        assert "disabled in public demo mode" in resp.json()["detail"]

    def test_training_adaptive_recommend_blocked(self, demo_client):
        resp = demo_client.post("/api/training/adaptive/recommend", json={
            "model_name": "ResNet18",
            "backend": "CPU_NATIVE",
        })
        assert resp.status_code == 403
        assert "disabled in public demo mode" in resp.json()["detail"]

    def test_laboratory_experiment_blocked(self, demo_client):
        resp = demo_client.post("/api/laboratory/experiments", json={
            "configurations": ["Config A"],
        })
        assert resp.status_code == 403
        assert "disabled in public demo mode" in resp.json()["detail"]

    def test_laboratory_adaptive_select_blocked(self, demo_client):
        resp = demo_client.post("/api/laboratory/adaptive-select", json={
            "workload": "inference",
            "precision": "FP32",
            "batch_size": 1,
            "sequence_length": 64,
        })
        assert resp.status_code == 403
        assert "disabled in public demo mode" in resp.json()["detail"]

    def test_model_unload_blocked(self, demo_client):
        resp = demo_client.post("/api/models/unload")
        assert resp.status_code == 403
        assert "Model unloading is forbidden in public demo mode" in resp.json()["detail"]


class TestPublicModelWhitelist:
    """Verify only approved public model IDs are permitted and arbitrary paths rejected."""

    def test_arbitrary_checkpoint_path_rejected(self, demo_client):
        resp = demo_client.post("/api/models/load", json={
            "model_name": "ResNet18",
            "checkpoint": "checkpoints/transformer/flagship_100m_pilot/best_model.pt",
        })
        assert resp.status_code == 403
        assert "Arbitrary checkpoint paths are forbidden" in resp.json()["detail"]

    def test_unapproved_model_name_rejected(self, demo_client):
        resp = demo_client.post("/api/models/load", json={
            "model_name": "Flagship100M",
        })
        assert resp.status_code == 403
        assert "not in the approved public demo whitelist" in resp.json()["detail"]

    def test_path_traversal_attempt_rejected(self, demo_client):
        resp = demo_client.post("/api/models/load", json={
            "model_name": "../../Windows/win.ini",
            "checkpoint": "../../Windows/win.ini",
        })
        assert resp.status_code == 403

    def test_approved_models_accepted(self, demo_client):
        for model_id in ("resnet18_cifar10", "transformer_50m", "baseline_mlp"):
            with patch.object(ModelServer, "load_model", return_value={"status": "loaded", "model_name": model_id}):
                resp = demo_client.post("/api/models/load", json={"model_name": model_id})
                assert resp.status_code == 200


class TestRequestSizeLimit:
    """Verify request payloads > 32 KB are rejected with HTTP 413 before body processing."""

    def test_oversized_payload_rejected_with_413(self, demo_client):
        oversized_data = {"prompt": "A" * 35000}
        resp = demo_client.post("/api/models/generate", json=oversized_data)
        assert resp.status_code == 413
        assert "PAYLOAD_TOO_LARGE" in resp.text

    def test_valid_size_payload_accepted(self, demo_client):
        valid_data = {"prompt": "Hello world", "max_new_tokens": 10}
        with patch.object(ModelServer, "generate", return_value={"status": "success", "generated_text": "demo output"}):
            resp = demo_client.post("/api/models/generate", json=valid_data)
            assert resp.status_code == 200


class TestAtomicInferenceLock:
    """Verify atomic non-blocking lock returns HTTP 429 when busy."""

    def test_predict_returns_429_when_lock_held(self, demo_client):
        acquired = _INFERENCE_LOCK.acquire(blocking=False)
        assert acquired, "Failed to acquire lock for test setup"
        try:
            resp = demo_client.post("/api/models/predict", json={"sample_index": 0})
            assert resp.status_code == 429
            assert "Concurrency is limited to 1 active inference job" in resp.json()["detail"]
        finally:
            _INFERENCE_LOCK.release()

    def test_generate_returns_429_when_lock_held(self, demo_client):
        acquired = _INFERENCE_LOCK.acquire(blocking=False)
        assert acquired, "Failed to acquire lock for test setup"
        try:
            resp = demo_client.post("/api/models/generate", json={"prompt": "test"})
            assert resp.status_code == 429
            assert "Concurrency is limited to 1 active inference job" in resp.json()["detail"]
        finally:
            _INFERENCE_LOCK.release()


class TestInferenceParameterLimits:
    """Verify prompt length clamping, token clamping, and tensor size limits."""

    def test_prompt_clamped_to_256_chars(self, demo_client):
        long_prompt = "A" * 500
        with patch.object(ModelServer, "generate", return_value={"status": "success"}) as mock_gen:
            demo_client.post("/api/models/generate", json={"prompt": long_prompt})
            mock_gen.assert_called_once()
            called_prompt = mock_gen.call_args[1]["prompt"]
            assert len(called_prompt) == 256

    def test_tokens_clamped_to_48(self, demo_client):
        with patch.object(ModelServer, "generate", return_value={"status": "success"}) as mock_gen:
            demo_client.post("/api/models/generate", json={"prompt": "hi", "max_new_tokens": 200})
            mock_gen.assert_called_once()
            assert mock_gen.call_args[1]["max_new_tokens"] == 48

    def test_tensor_float_limit_enforced(self, demo_client):
        large_tensor = [0.1] * 200  # Exceeds 128 floats
        resp = demo_client.post("/api/models/predict", json={"tensor": large_tensor})
        assert resp.status_code == 400
        assert "exceeds maximum allowed 128 floats" in resp.json()["detail"]

    def test_valid_tensor_under_128_floats_accepted(self, demo_client):
        valid_tensor = [0.1] * 64
        with patch.object(ModelServer, "predict", return_value={"status": "success"}):
            resp = demo_client.post("/api/models/predict", json={"tensor": valid_tensor})
            assert resp.status_code == 200


class TestHardwareSafeguards:
    """Verify VRAM reserve, 75°C thermal limit, and NVML failure behavior."""

    def test_vram_under_1024mb_rejected(self, demo_client):
        import torch
        low_vram = (500 * 1024 * 1024, 6000 * 1024 * 1024)  # 500 MB free < 1024 MB
        with patch.object(torch.cuda, "mem_get_info", return_value=low_vram), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(ModelServer, "_get_gpu_temperature", return_value=60.0):
            with pytest.raises(Exception) as exc_info:
                ModelServer._enforce_demo_hardware_safeguards(torch.device("cuda:0"))
            assert "Insufficient GPU VRAM" in str(exc_info.value)
            assert exc_info.value.status_code == 503

    def test_thermal_limit_75c_exceeded_rejected(self, demo_client):
        import torch
        ok_vram = (3000 * 1024 * 1024, 6000 * 1024 * 1024)
        with patch.object(torch.cuda, "mem_get_info", return_value=ok_vram), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(ModelServer, "_get_gpu_temperature", return_value=76.5):  # > 75.0 C
            with pytest.raises(Exception) as exc_info:
                ModelServer._enforce_demo_hardware_safeguards(torch.device("cuda:0"))
            assert "thermal safety threshold exceeded" in str(exc_info.value)
            assert exc_info.value.status_code == 503

    def test_nvml_failure_fails_closed_with_503(self, demo_client):
        import torch
        ok_vram = (3000 * 1024 * 1024, 6000 * 1024 * 1024)
        with patch.object(torch.cuda, "mem_get_info", return_value=ok_vram), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(ModelServer, "_get_gpu_temperature", return_value=None):  # NVML failure
            with pytest.raises(Exception) as exc_info:
                ModelServer._enforce_demo_hardware_safeguards(torch.device("cuda:0"))
            assert "GPU thermal telemetry unavailable" in str(exc_info.value)
            assert exc_info.value.status_code == 503

    def test_cpu_mode_safely_bypasses_gpu_checks(self, demo_client):
        import torch
        # CPU path should never call or fail on GPU telemetry
        ModelServer._enforce_demo_hardware_safeguards(torch.device("cpu"))


class TestPublicGETEndpointsPrivacy:
    """Verify hardware UUID masking, OS sanitization, private model exclusion, and artifact filtering."""

    def test_health_reports_public_demo_mode(self, demo_client):
        resp = demo_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "public-demo"

    def test_hardware_redacts_gpu_uuid_and_masks_os(self, demo_client):
        resp = demo_client.get("/api/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert data["host"]["platform"] == "Windows 11 (64-bit)"
        for gpu in data.get("gpus", []):
            assert gpu["uuid"] == "GPU-DEMO-REDACTED"

    def test_capabilities_masks_os_platform(self, demo_client):
        resp = demo_client.get("/api/capabilities")
        assert resp.status_code == 200
        assert resp.json()["platform"]["os"] == "Windows 11 (64-bit)"

    def test_checkpoints_catalog_hides_pilot_and_private_models(self, demo_client):
        resp = demo_client.get("/api/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        for ckpt in data.get("checkpoints", []):
            rel = ckpt["path"].lower()
            assert "flagship_100m" not in rel
            assert "pilot" not in rel
            assert "training_jobs" not in rel

    def test_models_list_hides_pilot_checkpoints(self, demo_client):
        resp = demo_client.get("/api/models")
        assert resp.status_code == 200
        models = resp.json().get("models", [])
        for m in models:
            assert m["model_id"] in ("resnet18_cifar10", "transformer_50m", "baseline_mlp")
            for c in m.get("checkpoints", []):
                assert "pilot" not in c.get("relative_path", "").lower()

    def test_artifact_log_retrieval_forbidden(self, demo_client):
        # In demo mode, raw execution logs must not be retrievable
        for log_id in ("benchmark-log", "python-test-log", "native-test-log"):
            resp = demo_client.get(f"/api/artifacts/{log_id}")
            assert resp.status_code == 404

    def test_training_jobs_list_returns_empty(self, demo_client):
        resp = demo_client.get("/api/training/jobs")
        assert resp.status_code == 200
        assert resp.json()["total_jobs"] == 0
        assert resp.json()["jobs"] == []

    def test_training_job_detail_returns_404(self, demo_client):
        resp = demo_client.get("/api/training/jobs/any_job_id")
        assert resp.status_code == 404


class TestStandardModePreservation:
    """Verify that when CUDA_ENGINE_DEMO_MODE=0, standard developer behavior is 100% preserved."""

    def test_health_reports_read_write(self, standard_client):
        resp = standard_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "read-write"

    def test_standard_mode_allows_model_unload(self, standard_client):
        with patch.object(ModelServer, "unload_model", return_value={"status": "unloaded"}):
            resp = standard_client.post("/api/models/unload")
            assert resp.status_code == 200
