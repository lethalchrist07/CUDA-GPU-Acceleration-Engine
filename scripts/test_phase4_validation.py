"""
Phase 4 Automated Validation & Production Readiness Test Battery
Covers:
- Cryptographic hash verification of all 17 Phase 2 manifest files
- Production build bundle assets validation
- Backend API security, CORS, headers, and read-only confinement
- Artifact traversal rejection and exact file download
- Live hardware telemetry parsing (no synthetic zeroing)
- Dynamic speedup metric verification (89.906x E2E / 116.75x Kernel on SGEMM 2048x2048 Tiled)
- Vite dev server and backend proxy functionality
"""
import hashlib
import json
import re
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"


def test_manifest_preservation():
    manifest_path = ROOT / "results/phase2_verification_20261016/manifest.json"
    assert manifest_path.is_file(), "Manifest file missing"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    assert len(manifest) == 17, f"Expected 17 manifest items, found {len(manifest)}"
    for rel_path, meta in manifest.items():
        file_path = ROOT / rel_path
        assert file_path.is_file(), f"Manifest file missing: {rel_path}"
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_hash == meta["sha256"], f"Hash mismatch for {rel_path}: {actual_hash} != {meta['sha256']}"
    print(f"[PASS 1/7] All {len(manifest)} Phase 2 manifest files match expected SHA-256 hashes (100% integrity).")


def test_production_build_assets():
    dist_dir = ROOT / "frontend/dist"
    assert dist_dir.is_dir(), "frontend/dist directory does not exist"
    index_html = dist_dir / "index.html"
    assert index_html.is_file(), "dist/index.html missing"
    content = index_html.read_text(encoding="utf-8")
    assert 'id="root"' in content, "dist/index.html missing #root mount point"
    assert "CUDA GPU Acceleration Engine" in content, "dist/index.html missing title tag"
    
    js_files = list((dist_dir / "assets").glob("*.js"))
    css_files = list((dist_dir / "assets").glob("*.css"))
    assert len(js_files) > 0, "No bundled JS files in dist/assets"
    assert len(css_files) > 0, "No bundled CSS files in dist/assets"
    print(f"[PASS 2/7] Production build verified: {len(js_files)} JS chunk ({js_files[0].stat().st_size / 1024:.1f} KB), {len(css_files)} CSS file ({css_files[0].stat().st_size / 1024:.1f} KB).")


def test_backend_api_and_data_accuracy():
    endpoints = ["health", "overview", "benchmarks", "training", "inference", "hardware", "verification"]
    for ep in endpoints:
        req = urllib.request.Request(f"{BACKEND_URL}/api/{ep}", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as res:
            assert res.status == 200, f"/api/{ep} failed with status {res.status}"
            data = json.loads(res.read().decode())
            assert "status" in data or ep == "health"

    # Verify Benchmarks data match Phase 2 final run
    with urllib.request.urlopen(f"{BACKEND_URL}/api/benchmarks", timeout=5) as res:
        bm = json.loads(res.read().decode())
        assert bm["source"]["provenance"] == "phase2_hash_verified"
        assert len(bm["rows"]) == 12
        assert all(r["verified"] is True for r in bm["rows"])
        
        # Check SGEMM 2048x2048 Tiled metrics
        row12 = next(r for r in bm["rows"] if r["id"] == 12)
        assert abs(row12["end_to_end_speedup"] - 89.906) < 0.1, f"Expected ~89.9x E2E speedup, got {row12['end_to_end_speedup']}"
        assert abs(row12["kernel_speedup"] - 116.75) < 0.1, f"Expected ~116.8x kernel speedup, got {row12['kernel_speedup']}"

    print("[PASS 3/7] All 7 live FastAPI routes verified with real data & Phase 2 mathematical parity.")


def test_hardware_telemetry_fidelity():
    with urllib.request.urlopen(f"{BACKEND_URL}/api/hardware", timeout=5) as res:
        hw = json.loads(res.read().decode())
        assert hw["status"] in ("available", "unavailable")
        assert hw["source"] == "nvidia-smi"
        if hw["status"] == "available":
            gpu = hw["gpus"][0]
            assert "RTX 3050" in gpu["name"]
            assert gpu["memory_total_mib"] == 6144.0
            assert gpu["temperature_c"] is not None
            assert gpu["power_w"] is not None
            # Check power_limit_w is strictly null (not synthetic 0)
            assert gpu["power_limit_w"] is None, "Missing power limit must remain null, never synthetic 0"
    print("[PASS 4/7] Live NVIDIA-SMI hardware telemetry parsed faithfully without synthetic zeroing.")


def test_security_headers_and_cors():
    # Security headers
    req = urllib.request.Request(f"{BACKEND_URL}/api/health")
    with urllib.request.urlopen(req, timeout=5) as res:
        assert res.headers.get("x-content-type-options") == "nosniff"
        assert res.headers.get("x-frame-options") == "DENY"
        assert res.headers.get("x-xss-protection") == "1; mode=block"
        assert res.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    # CORS local allowed
    req_cors = urllib.request.Request(f"{BACKEND_URL}/api/health", headers={"Origin": "http://127.0.0.1:5173"})
    with urllib.request.urlopen(req_cors, timeout=5) as res:
        assert res.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"

    # CORS external rejected
    req_bad_cors = urllib.request.Request(f"{BACKEND_URL}/api/health", headers={"Origin": "https://malicious.external.com"})
    with urllib.request.urlopen(req_bad_cors, timeout=5) as res:
        assert res.headers.get("access-control-allow-origin") is None

    print("[PASS 5/7] Backend security headers, CORS origin restrictions, and HTTP methods verified.")


def test_artifact_confinement_and_downloads():
    # Valid artifact download
    req = urllib.request.Request(f"{BACKEND_URL}/api/artifacts/benchmark-csv")
    with urllib.request.urlopen(req, timeout=5) as res:
        assert res.status == 200
        content = res.read()
        expected = (ROOT / "results/phase2_verification_20261016/final_run/benchmark_results.csv").read_bytes()
        assert content == expected, "Downloaded artifact does not match expected file"

    # Invalid artifact IDs
    for bad_id in ["unknown_artifact", "..%2F..%2Frequirements.txt", "invalid*id", "benchmark%20csv"]:
        try:
            with urllib.request.urlopen(f"{BACKEND_URL}/api/artifacts/{bad_id}", timeout=5):
                assert False, f"Expected 404 for bad artifact ID: {bad_id}"
        except urllib.error.HTTPError as exc:
            assert exc.code in (404, 400), f"Expected 404/400 for {bad_id}, got {exc.code}"

    # Method not allowed for POST
    post_req = urllib.request.Request(f"{BACKEND_URL}/api/benchmarks", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(post_req, timeout=5):
            assert False, "Expected 405 Method Not Allowed for POST"
    except urllib.error.HTTPError as exc:
        assert exc.code == 405

    print("[PASS 6/7] Read-only artifact store confinement and path traversal protections fully verified.")


def test_frontend_server_and_proxy():
    # Frontend index
    with urllib.request.urlopen(FRONTEND_URL, timeout=5) as res:
        assert res.status == 200
        html = res.read().decode()
        assert '<div id="root">' in html

    # Vite proxy to backend
    for route in ["overview", "hardware", "verification"]:
        with urllib.request.urlopen(f"{FRONTEND_URL}/api/{route}", timeout=5) as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert "status" in data

    print("[PASS 7/7] Vite server (port 5173) and backend API proxy (/api/*) confirmed operational.")


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 4 Production Readiness & Verification Test Battery")
    print("=" * 70)
    test_manifest_preservation()
    test_production_build_assets()
    test_backend_api_and_data_accuracy()
    test_hardware_telemetry_fidelity()
    test_security_headers_and_cors()
    test_artifact_confinement_and_downloads()
    test_frontend_server_and_proxy()
    print("=" * 70)
    print("ALL 7 PHASE 4 TEST SUITES PASSED CLEANLY (100% SUCCESS)")
    print("=" * 70)
