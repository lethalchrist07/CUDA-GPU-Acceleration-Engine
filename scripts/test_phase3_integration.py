"""
Comprehensive Integration Test Suite for Phase 3:
- Tests Backend REST API endpoints
- Tests Frontend Dev Server & Proxy
- Tests Production Bundle Build Assets
- Tests Preservation of Phase 2 Hash Manifest
"""
import hashlib
import json
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
    print(f"[PASS] All {len(manifest)} Phase 2 manifest files match expected SHA-256 hashes.")


def test_frontend_dist():
    dist_dir = ROOT / "frontend/dist"
    assert dist_dir.is_dir(), "frontend/dist directory does not exist"
    index_html = dist_dir / "index.html"
    assert index_html.is_file(), "dist/index.html is missing"
    html_content = index_html.read_text(encoding="utf-8")
    assert 'id="root"' in html_content, "dist/index.html missing #root div"
    
    js_files = list((dist_dir / "assets").glob("*.js"))
    css_files = list((dist_dir / "assets").glob("*.css"))
    assert len(js_files) > 0, "No bundled JS files in dist/assets"
    assert len(css_files) > 0, "No bundled CSS files in dist/assets"
    print(f"[PASS] Frontend production build verified: {len(js_files)} JS chunk(s), {len(css_files)} CSS file(s).")


def test_backend_live_endpoints():
    routes = [
        "health",
        "overview",
        "benchmarks",
        "training",
        "inference",
        "hardware",
        "verification",
    ]
    for route in routes:
        url = f"{BACKEND_URL}/api/{route}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as res:
            assert res.status == 200, f"Backend route {route} returned status {res.status}"
            data = json.loads(res.read().decode())
            assert "status" in data or route == "health"
    print(f"[PASS] All {len(routes)} live FastAPI routes returned HTTP 200.")


def test_frontend_server_and_proxy():
    # Test index
    with urllib.request.urlopen(FRONTEND_URL, timeout=5) as res:
        assert res.status == 200, f"Frontend index returned {res.status}"
        body = res.read().decode()
        assert '<div id="root">' in body, "Frontend response missing #root container"

    # Test Vite proxy to backend
    proxied_routes = ["overview", "hardware", "verification"]
    for route in proxied_routes:
        url = f"{FRONTEND_URL}/api/{route}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as res:
            assert res.status == 200, f"Proxied route {route} returned {res.status}"
            data = json.loads(res.read().decode())
            assert data.get("status") in ("available", "degraded", "unavailable")
    print(f"[PASS] Vite server (port 5173) and backend proxy (/api/*) fully functional.")


if __name__ == "__main__":
    print("=== Running Phase 3 Frontend & Backend Integration Tests ===")
    test_manifest_preservation()
    test_frontend_dist()
    test_backend_live_endpoints()
    test_frontend_server_and_proxy()
    print("=== ALL INTEGRATION TESTS PASSED SUCCESSFULLY ===")
