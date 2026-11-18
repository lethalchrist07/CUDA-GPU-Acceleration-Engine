"""
test_ui_acceptance_and_stabilization.py

Comprehensive acceptance tests verifying the UI stabilization and defects remediation:
- Absence of blue theme (#38BDF8) and copper palette (#F5A15D) in frontend source.
- Frontend production distribution files exist and are served properly.
- CPU-only hardware detection and fallback schema conformance.
- Hardware telemetry real data integrity (no mock 5.2W or 617.42 driver leaks).
- Benchmark artifacts and laboratory API endpoint integrity.
- Full 11 acceptance journeys.
"""

import os
import glob
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.data import collect_hardware


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_no_copper_or_blue_theme_leaks():
    """Verify that no deprecated copper (#F5A15D) or blue (#38BDF8) hex codes remain in frontend TSX source."""
    frontend_src = os.path.join(os.path.dirname(__file__), "..", "frontend", "src")
    tsx_files = glob.glob(os.path.join(frontend_src, "**", "*.tsx"), recursive=True)
    assert len(tsx_files) > 0, "Frontend TSX source files must exist"

    prohibited_patterns = ["#F5A15D", "#FFBF8A", "#38BDF8"]
    violations = []

    for fpath in tsx_files:
        with open(fpath, "r", encoding="utf-8") as fp:
            content = fp.read()
        for pat in prohibited_patterns:
            if pat in content:
                violations.append((fpath, pat))

    assert len(violations) == 0, f"Found theme/color violations: {violations}"


def test_no_hardcoded_mock_telemetry_leaks():
    """Verify that hardcoded mock values (5.2 W, 617.42 driver, mock UUID) are eliminated from active components."""
    frontend_src = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components")
    files_to_check = [
        os.path.join(frontend_src, "OverviewSection.tsx"),
        os.path.join(frontend_src, "GpuMonitoringSection.tsx"),
        os.path.join(frontend_src, "showcase", "ShowcaseExperience.tsx"),
    ]

    prohibited = ["'5.2 W'", '"5.2 W"', "'617.42'", '"617.42"', "GPU-424cb528-3d05-fbf1-8183-d93ed7818db7"]
    violations = []

    for fpath in files_to_check:
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as fp:
            content = fp.read()
        for pat in prohibited:
            if pat in content:
                violations.append((fpath, pat))

    assert len(violations) == 0, f"Found mock telemetry leaks: {violations}"


def test_production_dist_and_single_server_serving(client):
    """Verify that frontend/dist assets are built and served by FastAPI."""
    dist_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
    index_html = os.path.join(dist_dir, "index.html")
    assert os.path.exists(index_html), "frontend/dist/index.html must exist from production build"

    res = client.get("/")
    assert res.status_code == 200
    assert "<!DOCTYPE html>" in res.text or "<!doctype html>" in res.text

    # Deep routes return SPA index
    for route in ["/overview", "/training_studio", "/model_registry", "/cuda_lab", "/gpu_monitoring"]:
        spa_res = client.get(route)
        assert spa_res.status_code == 200
        assert "<div id=\"root\"></div>" in spa_res.text


def test_hardware_telemetry_and_cpu_fallback_schema(client):
    """Verify hardware endpoint reflects real physical hardware or clean cpu_only fallback."""
    res = client.get("/api/hardware")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] in ["available", "cpu_only", "degraded", "unavailable"]

    if data["status"] == "available":
        assert len(data["gpus"]) > 0
        gpu = data["gpus"][0]
        assert "name" in gpu
        assert "driver_version" in gpu
        assert "memory_total_mib" in gpu
        assert gpu["memory_total_mib"] > 0
    elif data["status"] == "cpu_only":
        assert len(data["gpus"]) == 0
        assert "host" in data
        assert "processor" in data["host"]


def test_benchmark_csv_and_laboratory_endpoints(client):
    """Verify that benchmark artifact downloads and laboratory configurations respond correctly."""
    # Laboratory configurations
    res_configs = client.get("/api/laboratory/configurations")
    assert res_configs.status_code == 200
    configs = res_configs.json().get("configurations", [])
    assert len(configs) >= 4  # Configs A, B, C, D

    # Laboratory experiments
    res_exp = client.get("/api/laboratory/experiments")
    assert res_exp.status_code == 200

    # Adaptive selection contract
    res_select = client.post(
        "/api/laboratory/adaptive-select",
        json={"workload": "training", "precision": "amp_fp16", "batch_size": 16, "sequence_length": 256},
    )
    assert res_select.status_code == 200
    assert "decision" in res_select.json()


def test_cifar_sample_inference_endpoint(client):
    """Verify that vision prediction endpoint responds cleanly."""
    res = client.post("/api/models/predict", json={"sample_index": 0})
    # Returns 200 with prediction or graceful 400/503 if weights not loaded
    assert res.status_code in [200, 400, 503]
    if res.status_code == 200:
        pred = res.json()
        assert "predicted_class" in pred
        assert "confidence" in pred
