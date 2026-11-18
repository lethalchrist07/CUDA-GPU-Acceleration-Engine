"""
scripts/verify_single_server_runtime.py
End-to-End Live Runtime Verification of BLK-03 Single-Server Deployment.
Launches uvicorn server on localhost:8000 and executes live HTTP requests against:
- Root frontend SPA (index.html)
- Compiled JS & CSS bundles (/assets/...)
- SPA client-side routes (/training_studio, /overview, etc.)
- Backend API endpoints (/api/health, /api/hardware, /api/models, etc.)
- API 404 JSON response integrity
- Swagger documentation (/docs, /openapi.json)
"""

import os
import re
import sys
import time
import socket
import subprocess
import urllib.request
import urllib.error
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"


def wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)
    return False


def http_get(url: str, accept: str = "*/*") -> tuple[int, str, bytes, dict]:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "NEXORA-Verification/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
            headers = dict(resp.headers)
            return status, content_type, data, headers
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read(), dict(e.headers)


def run_live_verification():
    print("=" * 70)
    print("  NEXORA BLK-03 — LIVE SINGLE-SERVER RUNTIME VERIFICATION")
    print("=" * 70)

    # 1. Start uvicorn server process
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    server_cmd = [
        sys.executable, "-m", "uvicorn", "backend.app:app",
        "--host", HOST, "--port", str(PORT),
        "--log-level", "warning"
    ]
    print(f"[*] Launching unified single-server daemon: {' '.join(server_cmd)}")
    proc = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        print(f"[*] Waiting for server to bind on {BASE_URL}...")
        if not wait_for_port(HOST, PORT, timeout=20.0):
            stdout, stderr = proc.communicate(timeout=5)
            print("[-] FATAL: Server failed to bind to port 8000!")
            print(f"STDOUT:\n{stdout.decode('utf-8', errors='replace')}")
            print(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
            sys.exit(1)

        print(f"[+] Server successfully bound to {BASE_URL}\n")
        tests_passed = 0
        total_tests = 0

        # --- TEST 1: Root URL (GET /) ---
        total_tests += 1
        status, ct, data, _ = http_get(f"{BASE_URL}/", accept="text/html")
        html_str = data.decode("utf-8", errors="replace")
        assert status == 200, f"Expected 200, got {status}"
        assert "text/html" in ct, f"Expected text/html, got {ct}"
        assert '<div id="root"></div>' in html_str, "Missing #root in HTML"
        assert "CUDA GPU Acceleration Engine" in html_str, "Missing title in HTML"
        print(f"[PASS 1/8] Root entry point (GET /) served React SPA index.html ({len(data):,} bytes)")
        tests_passed += 1

        # Extract bundle paths from HTML
        js_match = re.search(r'src="(/assets/[^"]+\.js)"', html_str)
        css_match = re.search(r'href="(/assets/[^"]+\.css)"', html_str)
        assert js_match, "Could not find JS bundle link in index.html"
        assert css_match, "Could not find CSS bundle link in index.html"
        js_path = js_match.group(1)
        css_path = css_match.group(1)

        # --- TEST 2: Static Assets (/assets/...) ---
        total_tests += 1
        status_css, ct_css, data_css, _ = http_get(f"{BASE_URL}{css_path}")
        assert status_css == 200, f"CSS asset failed: {status_css}"
        assert "text/css" in ct_css, f"CSS MIME error: {ct_css}"

        status_js, ct_js, data_js, _ = http_get(f"{BASE_URL}{js_path}")
        assert status_js == 200, f"JS asset failed: {status_js}"
        assert any(m in ct_js for m in ("javascript", "application/javascript")), f"JS MIME error: {ct_js}"
        print(f"[PASS 2/8] Static bundles served successfully: CSS ({len(data_css):,} bytes), JS ({len(data_js):,} bytes)")
        tests_passed += 1

        # --- TEST 3: SPA Client Routes (Page Refresh Simulation) ---
        total_tests += 1
        spa_routes = [
            "/overview",
            "/gpu_monitoring",
            "/training_studio",
            "/model_registry",
            "/cuda_lab",
            "/analytics",
            "/inference",
            "/verification",
        ]
        for route in spa_routes:
            s, c, d, _ = http_get(f"{BASE_URL}{route}", accept="text/html")
            assert s == 200, f"Route {route} returned status {s}"
            assert "text/html" in c, f"Route {route} did not return text/html"
            assert '<div id="root"></div>' in d.decode("utf-8", errors="replace")
        print(f"[PASS 3/8] SPA client-side routes ({len(spa_routes)} routes) verified for seamless page refresh")
        tests_passed += 1

        # --- TEST 4: Backend Core Telemetry API ---
        total_tests += 1
        status_health, ct_health, data_health, _ = http_get(f"{BASE_URL}/api/health", accept="application/json")
        assert status_health == 200
        health_json = json.loads(data_health)
        assert health_json.get("status") == "ok"

        status_hw, ct_hw, data_hw, _ = http_get(f"{BASE_URL}/api/hardware", accept="application/json")
        assert status_hw == 200
        hw_json = json.loads(data_hw)
        assert "gpus" in hw_json
        print(f"[PASS 4/8] Health & Hardware Telemetry API verified: {len(hw_json.get('gpus', []))} GPU(s) reported")
        tests_passed += 1

        # --- TEST 5: Model Registry & Capabilities API ---
        total_tests += 1
        s_models, _, d_models, _ = http_get(f"{BASE_URL}/api/models", accept="application/json")
        assert s_models == 200
        models_json = json.loads(d_models)
        assert "models" in models_json

        s_caps, _, d_caps, _ = http_get(f"{BASE_URL}/api/capabilities", accept="application/json")
        assert s_caps == 200
        caps_json = json.loads(d_caps)
        assert "cuda" in caps_json
        print(f"[PASS 5/8] Model Registry & Platform Capabilities API verified ({len(models_json.get('models', []))} models registered)")
        tests_passed += 1

        # --- TEST 6: Training Orchestration API ---
        total_tests += 1
        s_train_models, _, d_train_models, _ = http_get(f"{BASE_URL}/api/training/models", accept="application/json")
        assert s_train_models == 200
        train_models_json = json.loads(d_train_models)
        assert len(train_models_json.get("models", [])) > 0
        print(f"[PASS 6/8] Training Studio API verified ({len(train_models_json.get('models', []))} trainable models discoverable)")
        tests_passed += 1

        # --- TEST 7: API 404 Response Integrity (JSON, NOT HTML) ---
        total_tests += 1
        s_404, ct_404, d_404, _ = http_get(f"{BASE_URL}/api/nonexistent_endpoint_xyz", accept="application/json")
        assert s_404 == 404, f"Expected 404, got {s_404}"
        assert "application/json" in ct_404, f"API 404 returned non-JSON MIME: {ct_404}"
        err_json = json.loads(d_404)
        assert "detail" in err_json
        assert "<html" not in d_404.decode("utf-8", errors="replace").lower()
        print(f"[PASS 7/8] API 404 isolation confirmed: Unmatched /api/* routes return JSON 404, never index.html")
        tests_passed += 1

        # --- TEST 8: Swagger API Documentation ---
        total_tests += 1
        s_docs, ct_docs, _, _ = http_get(f"{BASE_URL}/docs", accept="text/html")
        assert s_docs == 200
        assert "text/html" in ct_docs

        s_openapi, ct_openapi, d_openapi, _ = http_get(f"{BASE_URL}/openapi.json", accept="application/json")
        assert s_openapi == 200
        openapi_json = json.loads(d_openapi)
        assert "paths" in openapi_json
        print(f"[PASS 8/8] OpenAPI Documentation endpoints verified (/docs & /openapi.json accessible)")
        tests_passed += 1

        print("\n" + "=" * 70)
        print(f"  ALL {tests_passed}/{total_tests} LIVE RUNTIME VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print(f"  SINGLE-SERVER LOCAL DEPLOYMENT (http://{HOST}:{PORT}) IS FULLY OPERATIONAL.")
        print("=" * 70)

    finally:
        print("[*] Terminating live test server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[+] Test server cleanly terminated.")


if __name__ == "__main__":
    run_live_verification()
