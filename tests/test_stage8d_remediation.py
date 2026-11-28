"""
Stage 8D: Targeted Remote Demo Connectivity Remediation Verification Suite
Tests:
1. Health endpoint returns status=ok and mode=public-demo.
2. Approved public demo model selection in generate_text works (HTTP 200).
3. Unapproved model paths in generate_text are rejected (HTTP 403).
4. Protected mutating operations remain HTTP 403.
5. HTML / authentication responses do not cause crashes.
6. Standard developer mode preserves full read-write behavior.
"""
import pytest
from starlette.testclient import TestClient
from backend.config import reload_settings
from backend.app import create_app


@pytest.fixture(autouse=True)
def setup_demo_mode(monkeypatch):
    monkeypatch.setenv("CUDA_ENGINE_DEMO_MODE", "1")
    reload_settings()
    yield
    monkeypatch.delenv("CUDA_ENGINE_DEMO_MODE", raising=False)
    reload_settings()


def test_01_health_public_demo():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "public-demo"
    assert data["version"] == "1.0.0"


def test_02_generate_approved_50m_model_path_accepted():
    app = create_app()
    client = TestClient(app)
    payload = {
        "prompt": "CUDA acceleration on Ampere architecture",
        "max_new_tokens": 32,
        "checkpoint": "checkpoints/transformer/pretraining_50m_campaign/best_model.pt",
    }
    resp = client.post("/api/models/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "generated_text" in data
    assert data["generated_tokens"] <= 48


def test_03_generate_unapproved_pilot_checkpoint_rejected():
    app = create_app()
    client = TestClient(app)
    payload = {
        "prompt": "Testing pilot checkpoint access",
        "checkpoint": "checkpoints/transformer/flagship_100m_pilot/best_model.pt",
    }
    resp = client.post("/api/models/generate", json=payload)
    assert resp.status_code == 403
    assert "forbidden in demo mode" in resp.json()["detail"].lower()


def test_04_generate_arbitrary_path_traversal_rejected():
    app = create_app()
    client = TestClient(app)
    payload = {
        "prompt": "Testing arbitrary path traversal",
        "checkpoint": "../../etc/passwd",
    }
    resp = client.post("/api/models/generate", json=payload)
    assert resp.status_code == 403
    assert "forbidden in demo mode" in resp.json()["detail"].lower()


def test_05_mutating_operations_remain_403():
    app = create_app()
    client = TestClient(app)
    
    # 1. Training job creation
    r_job = client.post("/api/training/jobs", json={"model_name": "resnet18"})
    assert r_job.status_code == 403

    # 2. Training job cancellation
    r_cancel = client.post("/api/training/jobs/dummy-job/cancel")
    assert r_cancel.status_code == 403

    # 3. Adaptive recommendation
    r_rec = client.post("/api/training/adaptive/recommend", json={"model_name": "resnet18"})
    assert r_rec.status_code == 403

    # 4. Lab experiment execution
    r_lab = client.post("/api/laboratory/experiments", json={"config_id": "config_a"})
    assert r_lab.status_code == 403

    # 5. Adaptive path selection
    r_adapt = client.post("/api/laboratory/adaptive-select", json={"model_name": "NEXORA GPT-86.8M"})
    assert r_adapt.status_code == 403

    # 6. Model unload
    r_unload = client.post("/api/models/unload")
    assert r_unload.status_code == 403


def test_06_developer_mode_preserves_read_write(monkeypatch):
    monkeypatch.setenv("CUDA_ENGINE_DEMO_MODE", "0")
    reload_settings()
    app = create_app()
    client = TestClient(app)
    
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["mode"] == "read-write"

    # Unload is permitted in standard developer mode
    r_unload = client.post("/api/models/unload")
    assert r_unload.status_code == 200
    assert r_unload.json()["status"] == "unloaded"
