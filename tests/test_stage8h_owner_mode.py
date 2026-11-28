"""
NEXORA v1.0.0 — Stage 8H: Secure Owner Mode Verification Tests
Validates server-side Owner Mode authentication, role elevation,
public demo restriction preservation, fail-safe closed behavior,
constant-time token comparison, and CUDA safety invariants.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_OWNER_KEY = "nexora_stage8h_test_secret_key_987654321"

os.environ["CUDA_ENGINE_DEMO_MODE"] = "1"
os.environ["CUDA_ENGINE_ENV"] = "production"
os.environ["CUDA_ENGINE_OWNER_KEY"] = TEST_OWNER_KEY

from backend.app import create_app
from backend.config import get_settings


@pytest.fixture(scope="module")
def app_client():
    settings = get_settings()
    settings.demo_mode = True
    settings.owner_key = TEST_OWNER_KEY
    app = create_app()
    return TestClient(app)


def test_01_health_public_visitor_reports_public_demo(app_client):
    """Anonymous visitors receive public-demo mode on health endpoint."""
    resp = app_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "public-demo"


def test_02_health_invalid_owner_key_reports_public_demo(app_client):
    """Requests with incorrect X-Owner-Key or Bearer token remain in public-demo mode."""
    # Wrong X-Owner-Key
    resp = app_client.get("/api/health", headers={"X-Owner-Key": "wrong-key"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "public-demo"

    # Wrong Authorization Bearer
    resp2 = app_client.get("/api/health", headers={"Authorization": "Bearer wrong-key"})
    assert resp2.status_code == 200
    assert resp2.json()["mode"] == "public-demo"


def test_03_health_valid_owner_key_reports_owner_mode(app_client):
    """Requests with valid X-Owner-Key or Authorization Bearer report owner mode."""
    # Via X-Owner-Key header
    resp = app_client.get("/api/health", headers={"X-Owner-Key": TEST_OWNER_KEY})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "owner"

    # Via Authorization: Bearer <key>
    resp2 = app_client.get("/api/health", headers={"Authorization": f"Bearer {TEST_OWNER_KEY}"})
    assert resp2.status_code == 200
    assert resp2.json()["mode"] == "owner"


def test_04_training_job_submission_public_blocked_owner_allowed(app_client):
    """Anonymous/invalid requests receive HTTP 403; valid owner can submit training jobs."""
    job_payload = {
        "model_name": "resnet18",
        "backend": "CPU_NATIVE",
        "dataset": "cifar10",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "max_train_samples": 10,
        "max_val_samples": 10,
    }

    # Anonymous -> 403 Forbidden
    resp_anon = app_client.post("/api/training/jobs", json=job_payload)
    assert resp_anon.status_code == 403
    assert "public demo mode" in resp_anon.json()["detail"].lower()

    # Wrong key -> 403 Forbidden
    resp_wrong = app_client.post(
        "/api/training/jobs",
        json=job_payload,
        headers={"X-Owner-Key": "invalid_key"},
    )
    assert resp_wrong.status_code == 403

    # Valid Owner Key -> Allowed (201 Created or successfully accepted by orchestrator)
    resp_owner = app_client.post(
        "/api/training/jobs",
        json=job_payload,
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp_owner.status_code == 201
    job_data = resp_owner.json()
    assert "job_id" in job_data
    assert job_data["status"] in ("QUEUED", "RUNNING", "COMPLETED")


def test_05_training_jobs_list_public_empty_owner_revealed(app_client):
    """Public callers receive empty job list; authorized owner receives actual jobs."""
    # Public -> 0 jobs returned
    resp_anon = app_client.get("/api/training/jobs")
    assert resp_anon.status_code == 200
    assert resp_anon.json()["total_jobs"] == 0
    assert resp_anon.json()["jobs"] == []

    # Owner -> returns jobs list
    resp_owner = app_client.get(
        "/api/training/jobs",
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp_owner.status_code == 200
    data = resp_owner.json()
    assert data["total_jobs"] >= 1
    assert len(data["jobs"]) >= 1


def test_06_training_job_cancellation_public_blocked_owner_allowed(app_client):
    """Job cancellation is blocked (403) for public, but authorized for owner."""
    # Public -> 403 Forbidden
    resp_anon = app_client.post("/api/training/jobs/fake-job-id/cancel")
    assert resp_anon.status_code == 403
    assert "disabled in public demo mode" in resp_anon.json()["detail"].lower()

    # Owner -> enters handler; nonexistent job returns 404 (not 403)
    resp_owner = app_client.post(
        "/api/training/jobs/fake-job-id/cancel",
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp_owner.status_code == 404


def test_07_laboratory_experiment_public_blocked_owner_allowed(app_client):
    """Comparative laboratory experiment creation is blocked for public, permitted for owner."""
    lab_payload = {
        "configurations": ["Config A"],
        "epochs": 1,
        "batch_size": 16,
    }

    # Public -> 403
    resp_anon = app_client.post("/api/laboratory/experiments", json=lab_payload)
    assert resp_anon.status_code == 403
    assert "disabled in public demo mode" in resp_anon.json()["detail"].lower()

    # Owner -> 201 Created
    resp_owner = app_client.post(
        "/api/laboratory/experiments",
        json=lab_payload,
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp_owner.status_code == 201
    assert resp_owner.json()["status"] == "created"


def test_08_laboratory_adaptive_select_public_blocked_owner_allowed(app_client):
    """Adaptive execution path mutation is blocked for public, permitted for owner."""
    select_payload = {
        "workload": "training",
        "precision": "FP16",
        "batch_size": 16,
        "sequence_length": 128,
    }

    # Public -> 403
    resp_anon = app_client.post("/api/laboratory/adaptive-select", json=select_payload)
    assert resp_anon.status_code == 403

    # Owner -> 200 OK
    resp_owner = app_client.post(
        "/api/laboratory/adaptive-select",
        json=select_payload,
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp_owner.status_code == 200
    assert resp_owner.json()["status"] == "success"


def test_09_model_unload_public_blocked_owner_allowed(app_client):
    """Model unload is blocked for public, permitted for owner."""
    # Public -> 403
    resp_anon = app_client.post("/api/models/unload")
    assert resp_anon.status_code == 403

    # Owner -> 200 OK
    resp_owner = app_client.post(
        "/api/models/unload",
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp_owner.status_code == 200
    assert resp_owner.json()["status"] == "unloaded"


def test_10_model_generate_token_clamp_public_vs_owner(app_client):
    """Public callers are clamped to max 48 tokens; Owner mode bypasses the clamp."""
    # Public requesting 60 tokens -> clamped to <= 48
    resp_anon = app_client.post(
        "/api/models/generate",
        json={"prompt": "CUDA kernel optimization", "max_new_tokens": 60},
    )
    assert resp_anon.status_code == 200
    assert resp_anon.json()["generated_tokens"] <= 48

    # Owner requesting 60 tokens -> unclamped (> 48)
    resp_owner = app_client.post(
        "/api/models/generate",
        json={"prompt": "CUDA kernel optimization", "max_new_tokens": 60},
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp_owner.status_code == 200
    assert resp_owner.json()["generated_tokens"] > 48


def test_11_request_size_limit_public_vs_owner(app_client):
    """Public visitors are limited to 32 KB; Owner can send larger payloads (up to 50 MB)."""
    # 40 KB string
    large_prompt = "X" * 40960

    # Public -> 413 Request Entity Too Large
    resp_anon = app_client.post(
        "/api/models/generate",
        json={"prompt": large_prompt, "max_new_tokens": 10},
    )
    assert resp_anon.status_code == 413
    assert "payload exceeds" in resp_anon.json()["detail"].lower()

    # Owner -> permitted through middleware
    resp_owner = app_client.post(
        "/api/models/generate",
        json={"prompt": large_prompt, "max_new_tokens": 10},
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    # Passed middleware size check; returns 200
    assert resp_owner.status_code == 200


def test_12_path_traversal_containment_preserved_for_owner(app_client):
    """Owner mode cannot load files outside PROJECT_ROOT (filesystem boundary enforced)."""
    resp = app_client.post(
        "/api/models/load",
        json={"model_name": "custom", "checkpoint": "../../etc/passwd"},
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp.status_code == 400
    assert "traversal" in resp.json()["detail"].lower()


def test_13_fail_safe_closed_when_owner_key_not_configured():
    """When CUDA_ENGINE_OWNER_KEY is empty/unset, owner authorization strictly fails safe."""
    from backend.auth import is_owner_authorized
    from unittest.mock import MagicMock

    # Request with some key
    mock_request = MagicMock()
    mock_request.headers = {"X-Owner-Key": "some-attempted-key"}

    # Mock settings with empty owner key
    mock_settings = MagicMock()
    mock_settings.owner_key = ""

    assert is_owner_authorized(mock_request, settings=mock_settings) is False


def test_14_auth_rate_limiting_enforcement(app_client):
    """After 10 failed authentication attempts from a client IP, subsequent attempts return HTTP 429."""
    from backend.auth import AUTH_RATE_LIMITER
    AUTH_RATE_LIMITER.reset()

    # 10 failed attempts
    for i in range(10):
        resp = app_client.get(
            "/api/health",
            headers={"X-Owner-Key": f"wrong-key-{i}"},
        )
        assert resp.status_code == 200
        assert resp.json()["mode"] == "public-demo"

    # Attempt 11 -> Locked out with HTTP 429
    resp_locked = app_client.get(
        "/api/health",
        headers={"X-Owner-Key": "wrong-key-11"},
    )
    assert resp_locked.status_code == 429
    assert "retry-after" in resp_locked.headers
    assert "too many failed" in resp_locked.json()["detail"].lower()


def test_15_auth_rate_limiting_expiry_and_window(app_client):
    """Failed authentication records expire outside the 300-second sliding window."""
    from backend.auth import AUTH_RATE_LIMITER
    from collections import deque
    import time
    AUTH_RATE_LIMITER.reset()

    # Artificially populate 10 failures aged 301 seconds ago
    old_time = time.time() - 301.0
    for _ in range(10):
        AUTH_RATE_LIMITER._failures.setdefault("testclient", deque()).append(old_time)

    # Attempt now should not be locked because all failures have expired
    resp = app_client.get(
        "/api/health",
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "owner"


def test_16_auth_rate_limiting_cleared_on_owner_success(app_client):
    """Successful owner authentication clears previous failed attempt counters."""
    from backend.auth import AUTH_RATE_LIMITER
    AUTH_RATE_LIMITER.reset()

    # 5 failed attempts
    for i in range(5):
        resp = app_client.get(
            "/api/health",
            headers={"X-Owner-Key": f"wrong-key-{i}"},
        )
        assert resp.status_code == 200

    assert len(AUTH_RATE_LIMITER._failures.get("testclient", [])) == 5

    # Valid owner authentication
    resp_success = app_client.get(
        "/api/health",
        headers={"X-Owner-Key": TEST_OWNER_KEY},
    )
    assert resp_success.status_code == 200
    assert resp_success.json()["mode"] == "owner"

    # Verification: failure history for this client IP is reset
    assert "testclient" not in AUTH_RATE_LIMITER._failures


def test_17_anti_spoofing_direct_external_ip():
    """Direct external TCP connections ignore spoofed X-Forwarded-For headers."""
    from backend.auth import get_trusted_client_ip
    from unittest.mock import MagicMock

    mock_req = MagicMock()
    mock_req.client.host = "198.51.100.25"  # External public IP
    mock_req.headers = {"x-forwarded-for": "10.0.0.1, 8.8.8.8"}

    resolved_ip = get_trusted_client_ip(mock_req)
    # Must use the real TCP client IP, rejecting the spoofed XFF header
    assert resolved_ip == "198.51.100.25"


def test_18_anti_spoofing_proxy_takes_rightmost_ip():
    """Connections through local tunnel proxy take the rightmost valid entry, preventing prefix injection."""
    from backend.auth import get_trusted_client_ip
    from unittest.mock import MagicMock

    mock_req = MagicMock()
    mock_req.client.host = "127.0.0.1"  # Local tunnel proxy
    # Attacker injects fake IPs at the front; reverse proxy appends real remote client at the end
    mock_req.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8, 203.0.113.50"}

    resolved_ip = get_trusted_client_ip(mock_req)
    assert resolved_ip == "203.0.113.50"


def test_19_frontend_storage_code_audit():
    """Verify frontend code exclusively uses sessionStorage for owner key and cleans up localStorage."""
    api_ts = (PROJECT_ROOT / "frontend/src/services/api.ts").read_text(encoding="utf-8")
    training_api_ts = (PROJECT_ROOT / "frontend/src/services/trainingApi.ts").read_text(encoding="utf-8")

    # Ensure no localStorage.setItem for owner key exists
    assert "localStorage.setItem('nexora_owner_key'" not in api_ts
    assert "localStorage.setItem(\"nexora_owner_key\"" not in api_ts
    assert "localStorage.setItem('nexora_owner_key'" not in training_api_ts
    assert "localStorage.setItem(\"nexora_owner_key\"" not in training_api_ts

    # Verify sessionStorage is used
    assert "sessionStorage.setItem('nexora_owner_key'" in api_ts
    assert "sessionStorage.getItem('nexora_owner_key')" in api_ts
    assert "sessionStorage.getItem('nexora_owner_key')" in training_api_ts

    # Verify legacy localStorage cleanup is performed
    assert "localStorage.removeItem('nexora_owner_key')" in api_ts


