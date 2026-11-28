"""
NEXORA v1.0.0 — Stage 8G: ngrok Remote Connectivity Verification Tests
Validates CORS preflight, ngrok origin regex, ngrok-skip-browser-warning header,
and end-to-end API operations under ngrok tunneling.
"""
from __future__ import annotations

import os
import pytest
from starlette.testclient import TestClient

os.environ["CUDA_ENGINE_DEMO_MODE"] = "1"
os.environ["CUDA_ENGINE_ENV"] = "production"

from backend.app import create_app
from backend.config import get_settings


@pytest.fixture(scope="module")
def demo_client():
    settings = get_settings()
    settings.demo_mode = True
    app = create_app()
    return TestClient(app)


def test_01_cors_options_preflight_ngrok_free(demo_client):
    """Verify CORS preflight OPTIONS succeeds for *.ngrok-free.app."""
    origin = "https://a1b2-c3d4.ngrok-free.app"
    resp = demo_client.options(
        "/api/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "accept,authorization,ngrok-skip-browser-warning",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"
    allow_hdrs = resp.headers.get("access-control-allow-headers", "").lower()
    assert "ngrok-skip-browser-warning" in allow_hdrs or "*" in allow_hdrs


def test_02_cors_options_preflight_ngrok_app(demo_client):
    """Verify CORS preflight OPTIONS succeeds for *.ngrok.app."""
    origin = "https://custom-demo.ngrok.app"
    resp = demo_client.options(
        "/api/models/load",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,accept,ngrok-skip-browser-warning",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_03_cors_options_preflight_ngrok_io(demo_client):
    """Verify CORS preflight OPTIONS succeeds for legacy *.ngrok.io."""
    origin = "https://nexora.ngrok.io"
    resp = demo_client.options(
        "/api/overview",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_04_ngrok_skip_browser_warning_header_accepted(demo_client):
    """Verify backend accepts ngrok-skip-browser-warning header without failure."""
    origin = "https://demo.ngrok-free.app"
    resp = demo_client.get(
        "/api/health",
        headers={
            "Origin": origin,
            "ngrok-skip-browser-warning": "true",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "public-demo"


def test_05_remote_hardware_telemetry_ngrok(demo_client):
    """Verify hardware status endpoint works remotely via ngrok origin."""
    origin = "https://demo.ngrok-free.app"
    resp = demo_client.get(
        "/api/hardware",
        headers={
            "Origin": origin,
            "ngrok-skip-browser-warning": "true",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    data = resp.json()
    assert "gpus" in data or "devices" in data or "cuda_available" in data or "status" in data


def test_06_model_load_reaches_backend_ngrok(demo_client):
    """Verify model load request reaches backend for approved demo models under ngrok."""
    origin = "https://demo.ngrok-free.app"
    resp = demo_client.post(
        "/api/models/load",
        headers={
            "Origin": origin,
            "ngrok-skip-browser-warning": "true",
        },
        json={"model_name": "ResNet18", "device": "cpu"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    data = resp.json()
    assert data["status"] == "loaded"


def test_07_model_generate_text_ngrok(demo_client):
    """Verify text generation works under ngrok origin within demo clamps."""
    origin = "https://demo.ngrok-free.app"
    resp = demo_client.post(
        "/api/models/generate",
        headers={
            "Origin": origin,
            "ngrok-skip-browser-warning": "true",
        },
        json={
            "prompt": "CUDA acceleration engine provides",
            "max_new_tokens": 16,
            "temperature": 0.7,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    data = resp.json()
    assert "generated_text" in data
    assert data["generated_tokens"] <= 48


def test_08_training_launch_reaches_backend_blocked_ngrok(demo_client):
    """Verify training launch reaches backend and is safely rejected under ngrok."""
    origin = "https://demo.ngrok-free.app"
    resp = demo_client.post(
        "/api/training/jobs",
        headers={
            "Origin": origin,
            "ngrok-skip-browser-warning": "true",
        },
        json={"model_name": "ResNet18"},
    )
    assert resp.status_code == 403
    assert resp.headers.get("access-control-allow-origin") == origin
    assert "disabled in public demo mode" in resp.json()["detail"]


def test_09_disallowed_origin_rejected_ngrok(demo_client):
    """Verify unauthorized external origins do not receive allow-origin header."""
    origin = "https://attacker.evil.com"
    resp = demo_client.options(
        "/api/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") is None or resp.status_code == 400


def test_10_frontend_index_and_assets_served_ngrok(demo_client):
    """Verify root / serves the current index.html with the latest ngrok-ready asset."""
    resp = demo_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "/assets/index-" in resp.text and ".js" in resp.text
