"""
tests/test_single_server_deployment.py
Automated Verification Suite for BLK-03 Single-Server Local Production Deployment.

Validates:
1. Root URL (GET /) serves production-built React SPA index.html.
2. Static assets (CSS/JS in /assets/...) are served with correct MIME types.
3. Client-side routes (GET /overview, /training_studio, etc.) serve index.html for SPA routing.
4. API routes (GET /api/...) return JSON responses, and unmatched API routes return JSON 404s, never HTML.
5. Swagger documentation (GET /docs, /openapi.json) remains functional.
6. Degraded environments (missing frontend/dist) fail gracefully without crashing API endpoints.
7. Single-server origins (http://localhost:8000, http://127.0.0.1:8000) are permitted by CORS.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    with TestClient(create_app(PROJECT_ROOT)) as c:
        yield c


class TestSingleServerDeployment:
    """Verification battery for BLK-03 single-server deployment."""

    def test_root_serves_frontend_index_html(self, client):
        """Verify GET / returns production index.html with 200 OK and text/html MIME."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        html = response.text
        assert '<div id="root"></div>' in html
        assert "/assets/index-" in html
        assert "CUDA GPU Acceleration Engine" in html

    def test_static_assets_served_correctly(self, client):
        """Verify compiled JS and CSS bundles in /assets are served with valid MIME types."""
        dist_assets = PROJECT_ROOT / "frontend" / "dist" / "assets"
        assert dist_assets.is_dir(), "frontend/dist/assets directory does not exist"

        css_files = list(dist_assets.glob("*.css"))
        js_files = list(dist_assets.glob("*.js"))

        assert len(css_files) > 0, "No CSS bundles found in frontend/dist/assets"
        assert len(js_files) > 0, "No JS bundles found in frontend/dist/assets"

        # Verify CSS bundle
        css_url = f"/assets/{css_files[0].name}"
        res_css = client.get(css_url)
        assert res_css.status_code == 200
        assert "text/css" in res_css.headers.get("content-type", "")
        assert len(res_css.content) > 1000

        # Verify JS bundle
        js_url = f"/assets/{js_files[0].name}"
        res_js = client.get(js_url)
        assert res_js.status_code == 200
        assert any(ct in res_js.headers.get("content-type", "") for ct in ("javascript", "application/javascript", "text/javascript"))
        assert len(res_js.content) > 10000

    @pytest.mark.parametrize("route", [
        "/overview",
        "/gpu_monitoring",
        "/training_studio",
        "/model_registry",
        "/cuda_lab",
        "/analytics",
        "/inference",
        "/verification",
    ])
    def test_spa_client_routes_serve_index_html(self, client, route):
        """Verify client-side routes return index.html for React SPA mounting on refresh."""
        response = client.get(route)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert '<div id="root"></div>' in response.text

    def test_valid_api_endpoints_return_json(self, client):
        """Verify API endpoints return structured JSON with 200 OK."""
        api_endpoints = [
            "/api/health",
            "/api/overview",
            "/api/hardware",
            "/api/models",
            "/api/training/models",
        ]
        for ep in api_endpoints:
            res = client.get(ep)
            assert res.status_code == 200, f"Endpoint {ep} failed with status {res.status_code}"
            assert "application/json" in res.headers.get("content-type", "")
            data = res.json()
            assert isinstance(data, dict)

    def test_unmatched_api_routes_return_json_404_not_html(self, client):
        """Verify that non-existent API routes return JSON 404, never index.html."""
        # Non-existent endpoint
        res = client.get("/api/completely_nonexistent_endpoint_12345")
        assert res.status_code == 404
        assert "application/json" in res.headers.get("content-type", "")
        body = res.json()
        assert "detail" in body
        assert "<html" not in res.text.lower()

        # Non-existent sub-path under API
        res_sub = client.get("/api/training/nonexistent_sub_action")
        assert res_sub.status_code == 404
        assert "application/json" in res_sub.headers.get("content-type", "")

    def test_docs_endpoints_accessible(self, client):
        """Verify FastAPI documentation endpoints remain accessible in single-server mode."""
        res_docs = client.get("/docs")
        assert res_docs.status_code == 200
        assert "text/html" in res_docs.headers.get("content-type", "")

        res_openapi = client.get("/openapi.json")
        assert res_openapi.status_code == 200
        assert "application/json" in res_openapi.headers.get("content-type", "")
        assert "paths" in res_openapi.json()

    def test_missing_frontend_build_graceful_handling(self, tmp_path):
        """Verify server initializes and handles requests cleanly when frontend/dist is absent."""
        empty_root = tmp_path / "empty_nexora_root"
        empty_root.mkdir()

        with TestClient(create_app(empty_root)) as sterile_client:
            # API endpoint must still work
            res_health = sterile_client.get("/api/health")
            assert res_health.status_code == 200

            # Root request should return descriptive 404, not unhandled 500
            res_root = sterile_client.get("/")
            assert res_root.status_code == 404
            assert "Frontend static build not found" in res_root.json()["detail"]

    def test_cors_allows_single_server_origins(self, client):
        """Verify CORS headers permit single-server local deployment origins."""
        for origin in ["http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:5173"]:
            res = client.get("/api/health", headers={"Origin": origin})
            assert res.status_code == 200
            assert res.headers.get("access-control-allow-origin") == origin
