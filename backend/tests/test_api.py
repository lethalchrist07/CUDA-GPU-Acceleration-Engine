from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client():
    with TestClient(create_app(ROOT)) as test_client:
        yield test_client


@pytest.mark.parametrize("section", ["overview", "benchmarks", "training", "inference", "hardware", "verification"])
def test_six_sections_return_backend_data(client, section):
    response = client.get(f"/api/{section}")
    assert response.status_code == 200
    data = response.json()
    assert data["fetched_at"]
    assert data["status"] in ("available", "unavailable", "degraded")
    if section != "hardware":
        assert data["status"] == "available"


def test_benchmark_api_matches_file_exactly(client):
    import json
    path = ROOT / "results/phase2_verification_20261016/final_run/benchmark_results.json"
    assert client.get("/api/benchmarks").json()["rows"] == json.loads(path.read_text())["benchmarks"]


def test_download_is_exact_artifact(client):
    path = ROOT / "results/phase2_verification_20261016/final_run/benchmark_results.csv"
    response = client.get("/api/artifacts/benchmark-csv")
    assert response.status_code == 200
    assert response.content == path.read_bytes()


def test_invalid_artifact_and_write_request_are_rejected(client):
    assert client.get("/api/artifacts/unknown").status_code == 404
    assert client.get("/api/artifacts/..%2F..%2Frequirements.txt").status_code == 404
    assert client.get("/api/artifacts/invalid*id").status_code == 404
    assert client.get("/api/artifacts/benchmark%20csv").status_code == 404
    assert client.post("/api/benchmarks", json={}).status_code == 405


def test_missing_data_stays_explicitly_unavailable(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/benchmarks").json()["status"] == "unavailable"
        assert client.get("/api/overview").json()["status"] == "degraded"


def test_cors_allows_only_local_frontend(client):
    assert client.get("/api/health", headers={"Origin": "http://127.0.0.1:5173"}).headers[
        "access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "access-control-allow-origin" not in client.get(
        "/api/health", headers={"Origin": "https://unrelated.example"}).headers


def test_security_headers_present_on_responses(client):
    res = client.get("/api/health")
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["x-xss-protection"] == "1; mode=block"
    assert res.headers["referrer-policy"] == "strict-origin-when-cross-origin"

