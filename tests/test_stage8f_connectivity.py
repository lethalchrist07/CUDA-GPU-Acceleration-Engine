"""
NEXORA v1.0.0 — Stage 8F: Cloudflare Remote API Connectivity Verification Tests
Validates CORS preflight, credentials, origin regex, and API endpoint behavior under Cloudflare Quick Tunnel.
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


def test_01_cors_options_preflight_trycloudflare(demo_client):
    """Verify CORS preflight OPTIONS succeeds for *.trycloudflare.com."""
    origin = "https://peace-peaceful-random-words.trycloudflare.com"
    resp = demo_client.options(
        "/api/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "accept,authorization",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_02_cors_options_preflight_post_with_headers(demo_client):
    """Verify CORS preflight OPTIONS for POST with custom headers succeeds."""
    origin = "https://quick-tunnel-demo.trycloudflare.com"
    resp = demo_client.options(
        "/api/models/load",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,accept,cf-access-token",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"
    allow_hdrs = resp.headers.get("access-control-allow-headers", "").lower()
    assert "content-type" in allow_hdrs or "*" in allow_hdrs


def test_03_cors_options_preflight_github_pages(demo_client):
    """Verify CORS preflight OPTIONS succeeds for GitHub Pages origin."""
    origin = "https://lethalchrist07.github.io"
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


def test_04_cors_disallowed_origin_rejected(demo_client):
    """Verify unauthorized external origins do not receive allow-origin header."""
    origin = "https://malicious-attacker-site.com"
    resp = demo_client.options(
        "/api/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette CORSMiddleware returns 400 or omits access-control-allow-origin
    assert resp.headers.get("access-control-allow-origin") is None or resp.status_code == 400


def test_05_remote_health_endpoint(demo_client):
    """Verify /api/health works with remote tunnel origin and returns public-demo mode."""
    origin = "https://my-tunnel.trycloudflare.com"
    resp = demo_client.get("/api/health", headers={"Origin": origin})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    data = resp.json()
    assert data["status"] in ("ok", "available")
    assert data["mode"] == "public-demo"


def test_06_model_load_reaches_backend(demo_client):
    """Verify model load request reaches backend for approved demo models."""
    origin = "https://my-tunnel.trycloudflare.com"
    resp = demo_client.post(
        "/api/models/load",
        headers={"Origin": origin},
        json={"model_name": "ResNet18", "device": "cpu"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    data = resp.json()
    assert data["status"] == "loaded"


def test_07_unapproved_model_rejected_cleanly(demo_client):
    """Verify unapproved model load returns HTTP 403 with descriptive detail."""
    resp = demo_client.post(
        "/api/models/load",
        json={"model_name": "flagship_100m_pilot"},
    )
    assert resp.status_code == 403
    assert "approved public demo whitelist" in resp.json()["detail"]


def test_08_training_launch_reaches_backend_and_blocks_safely(demo_client):
    """Verify training launch reaches backend and is safely rejected in demo mode."""
    origin = "https://my-tunnel.trycloudflare.com"
    resp = demo_client.post(
        "/api/training/jobs",
        headers={"Origin": origin},
        json={
            "model_name": "ResNet18",
            "backend": "GPU_CUSTOM",
            "dataset": "CIFAR-10",
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    )
    assert resp.status_code == 403
    assert resp.headers.get("access-control-allow-origin") == origin
    assert "disabled in public demo mode" in resp.json()["detail"]


def test_09_model_registry_checkpoints_endpoint(demo_client):
    """Verify /api/checkpoints returns approved checkpoints without private leaks."""
    origin = "https://my-tunnel.trycloudflare.com"
    resp = demo_client.get("/api/checkpoints", headers={"Origin": origin})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    data = resp.json()
    assert data["status"] in ("ok", "available")
    assert isinstance(data["checkpoints"], list)


def test_10_frontend_index_and_assets_served(demo_client):
    """Verify root / serves the updated index.html with the new build asset."""
    resp = demo_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "/assets/index-" in resp.text and ".js" in resp.text
