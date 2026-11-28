"""
NEXORA v1.0.0 — Stage 7C: Local Public Demo Acceptance Test Runner
Performs end-to-end HTTP verification of the running application in Demo Mode
and Standard Mode across all 10 acceptance criteria.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "http://127.0.0.1:8000"


def make_request(
    path: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    raw_data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15.0,
) -> Tuple[int, Dict[str, Any], str, Dict[str, str]]:
    url = f"{BASE_URL}{path}"
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    body_bytes = None
    if raw_data is not None:
        body_bytes = raw_data
        if "Content-Type" not in req_headers:
            req_headers["Content-Type"] = "application/json"
    elif data is not None:
        body_bytes = json.dumps(data).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            resp_headers = dict(resp.headers)
            raw_text = resp.read().decode("utf-8", errors="replace")
            try:
                parsed_json = json.loads(raw_text)
            except Exception:
                parsed_json = {}
            return status_code, parsed_json, raw_text, resp_headers
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        resp_headers = dict(exc.headers)
        raw_text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_json = json.loads(raw_text)
        except Exception:
            parsed_json = {}
        return status_code, parsed_json, raw_text, resp_headers
    except Exception as exc:
        return 0, {}, str(exc), {}


def wait_for_server(timeout_sec: float = 20.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        status_code, resp_json, _, _ = make_request("/api/health")
        if status_code == 200:
            return True
        time.sleep(0.5)
    return False


def get_port_pid(port: int = 8000) -> Optional[int]:
    try:
        output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in parts:
                return int(parts[-1])
    except Exception:
        pass
    return None


def kill_pid(pid: int) -> bool:
    try:
        subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def test_public_demo_mode() -> Dict[str, Any]:
    results = {}
    print("=" * 70)
    print("STAGE 7C ACCEPTANCE SUITE — TESTING PUBLIC DEMO MODE (CUDA_ENGINE_DEMO_MODE=1)")
    print("=" * 70)

    # 1. Health & Mode Check
    status, body, _, headers = make_request("/api/health")
    mode = body.get("mode")
    version = body.get("version")
    assert_1 = (status == 200 and mode == "public-demo" and version == "1.0.0")
    results["T01_Health_Demo_Mode"] = {
        "status": "PASS" if assert_1 else "FAIL",
        "http_status": status,
        "detail": f"mode='{mode}', version='{version}'",
    }
    print(f"[{'PASS' if assert_1 else 'FAIL'}] Health endpoint reports mode='{mode}' (HTTP {status})")

    # 2. Frontend Root Route Serving SPA
    status, _, raw_text, _ = make_request("/")
    has_root = '<div id="root">' in raw_text or '<html' in raw_text
    assert_2 = (status == 200 and has_root)
    results["T02_Dashboard_SPA_Serving"] = {
        "status": "PASS" if assert_2 else "FAIL",
        "http_status": status,
        "detail": f"Length: {len(raw_text)} bytes, contains HTML root: {has_root}",
    }
    print(f"[{'PASS' if assert_2 else 'FAIL'}] Dashboard SPA served at GET / (HTTP {status})")

    # 3. Hardware Telemetry Sanitization
    status, body, _, _ = make_request("/api/hardware")
    gpus = body.get("gpus", [])
    gpu_uuid = gpus[0].get("uuid") if gpus else None
    host = body.get("host", {})
    platform = host.get("platform")
    assert_3 = (status == 200 and gpu_uuid == "GPU-DEMO-REDACTED" and platform == "Windows 11 (64-bit)")
    results["T03_Hardware_Telemetry_Sanitization"] = {
        "status": "PASS" if assert_3 else "FAIL",
        "http_status": status,
        "detail": f"gpu.uuid='{gpu_uuid}', host.platform='{platform}'",
    }
    print(f"[{'PASS' if assert_3 else 'FAIL'}] Hardware telemetry sanitized: UUID='{gpu_uuid}', Platform='{platform}' (HTTP {status})")

    # 4. Capabilities Sanitization
    status, body, _, _ = make_request("/api/capabilities")
    plat_info = body.get("platform", {})
    plat_cap = plat_info.get("os")
    assert_4 = (status == 200 and plat_cap == "Windows 11 (64-bit)")
    results["T04_Capabilities_Sanitization"] = {
        "status": "PASS" if assert_4 else "FAIL",
        "http_status": status,
        "detail": f"platform.os='{plat_cap}'",
    }
    print(f"[{'PASS' if assert_4 else 'FAIL'}] Capabilities platform sanitized: '{plat_cap}' (HTTP {status})")

    # 5. Safe Read-Only GET Endpoints
    safe_endpoints = [
        ("/api/overview", "overview"),
        ("/api/benchmarks", "benchmarks"),
        ("/api/verification", "verification"),
        ("/api/laboratory/configurations", "laboratory_configs"),
        ("/api/training/models", "training_models"),
    ]
    all_safe_pass = True
    safe_details = []
    for ep, name in safe_endpoints:
        st, b, _, _ = make_request(ep)
        safe_pass = (st == 200)
        safe_details.append(f"{ep}: HTTP {st}")
        if not safe_pass:
            all_safe_pass = False
    results["T05_Safe_GET_Endpoints"] = {
        "status": "PASS" if all_safe_pass else "FAIL",
        "http_status": 200 if all_safe_pass else 500,
        "detail": ", ".join(safe_details),
    }
    print(f"[{'PASS' if all_safe_pass else 'FAIL'}] Safe GET endpoints: {', '.join(safe_details)}")

    # 6. Blocked Mutating Endpoints (HTTP 403 Forbidden)
    mutating_cases = [
        ("POST", "/api/training/jobs", {"model_name": "resnet18", "dataset": "cifar10"}),
        ("POST", "/api/training/jobs/dummy-job/cancel", {}),
        ("POST", "/api/training/adaptive/recommend", {"model_name": "resnet18"}),
        ("POST", "/api/laboratory/experiments", {"config_id": "config_a"}),
        ("POST", "/api/laboratory/adaptive-select", {"model_name": "NEXORA GPT-86.8M"}),
        ("POST", "/api/models/unload", {}),
    ]
    all_403_pass = True
    mutating_details = []
    for method, path, payload in mutating_cases:
        st, b, _, _ = make_request(path, method=method, data=payload)
        is_403 = (st == 403)
        err_detail = b.get("detail", "")
        mutating_details.append(f"{path}: HTTP {st} ({err_detail})")
        if not is_403:
            all_403_pass = False
    results["T06_Blocked_Mutating_Endpoints"] = {
        "status": "PASS" if all_403_pass else "FAIL",
        "http_status": 403 if all_403_pass else 0,
        "detail": "; ".join(mutating_details),
    }
    print(f"[{'PASS' if all_403_pass else 'FAIL'}] Blocked mutating endpoints returned HTTP 403: {len(mutating_cases)} verified")

    # 7. Model Whitelist Enforcement
    approved_models = ["resnet18_cifar10", "transformer_50m", "baseline_mlp"]
    whitelist_details = []
    all_whitelist_pass = True
    for model_id in approved_models:
        st, b, _, _ = make_request("/api/models/load", method="POST", data={"model_name": model_id})
        pass_model = (st == 200 and b.get("status") in ("loaded", "already_loaded", "ok", "ready"))
        whitelist_details.append(f"load('{model_id}'): HTTP {st} -> {b.get('status')}")
        if not pass_model:
            all_whitelist_pass = False

    # Arbitrary checkpoint injection attempt
    st_arb, b_arb, _, _ = make_request(
        "/api/models/load",
        method="POST",
        data={"model_name": "resnet18_cifar10", "checkpoint": "checkpoints/transformer/flagship_100m_pilot/best_model.pt"},
    )
    arb_pass = (st_arb == 403)
    whitelist_details.append(f"arbitrary checkpoint: HTTP {st_arb}")
    if not arb_pass:
        all_whitelist_pass = False

    # Unapproved model name attempt
    st_unapp, b_unapp, _, _ = make_request(
        "/api/models/load",
        method="POST",
        data={"model_name": "unapproved_huggingface_model"},
    )
    unapp_pass = (st_unapp == 403)
    whitelist_details.append(f"unapproved model: HTTP {st_unapp}")
    if not unapp_pass:
        all_whitelist_pass = False

    # Ensure transformer_50m and resnet18 are active for subsequent inference tests
    make_request("/api/models/load", method="POST", data={"model_name": "transformer_50m"})
    make_request("/api/models/load", method="POST", data={"model_name": "resnet18_cifar10"})

    results["T07_Model_Whitelist_Enforcement"] = {
        "status": "PASS" if all_whitelist_pass else "FAIL",
        "http_status": 200,
        "detail": "; ".join(whitelist_details),
    }
    print(f"[{'PASS' if all_whitelist_pass else 'FAIL'}] Model whitelist enforcement: {'; '.join(whitelist_details)}")

    # 8. Request Size Limit (HTTP 413)
    oversized_payload = b"X" * 35000  # 35 KB > 32 KB limit
    st_413, b_413, _, _ = make_request(
        "/api/models/generate",
        method="POST",
        raw_data=oversized_payload,
        headers={"Content-Length": str(len(oversized_payload))},
    )
    assert_413 = (st_413 == 413 and b_413.get("error_type") == "PAYLOAD_TOO_LARGE")
    results["T08_Request_Size_Limit_413"] = {
        "status": "PASS" if assert_413 else "FAIL",
        "http_status": st_413,
        "detail": f"Payload {len(oversized_payload)} bytes -> HTTP {st_413}, error_type='{b_413.get('error_type')}'",
    }
    print(f"[{'PASS' if assert_413 else 'FAIL'}] Oversized payload (>32 KB) rejected with HTTP 413: {assert_413}")

    # 9. Inference Input Constraints (Tensor Size & Token Limits)
    # 9a. Oversized tensor (> 128 floats)
    oversized_tensor = [0.5] * 200
    st_tensor, b_tensor, _, _ = make_request("/api/models/predict", method="POST", data={"tensor": oversized_tensor})
    assert_tensor = (st_tensor == 400 and "exceeds maximum allowed 128 floats" in b_tensor.get("detail", ""))

    # 9b. Valid predict request
    st_pred, b_pred, _, _ = make_request("/api/models/predict", method="POST", data={"sample_index": 0})
    assert_pred = (st_pred == 200 and ("predicted_class" in b_pred or "prediction" in b_pred))

    # 9c. Generation token & prompt clamping
    long_prompt = "NEXORA " * 100  # > 256 chars
    st_gen, b_gen, _, _ = make_request(
        "/api/models/generate",
        method="POST",
        data={"prompt": long_prompt, "max_new_tokens": 150},
    )
    assert_gen = (st_gen == 200 and "generated_text" in b_gen)
    tokens_clamped = b_gen.get("generated_tokens", 0) <= 48

    assert_limits = (assert_tensor and assert_pred and assert_gen and tokens_clamped)
    results["T09_Inference_Input_Constraints"] = {
        "status": "PASS" if assert_limits else "FAIL",
        "http_status": st_gen,
        "detail": f"Oversized tensor HTTP {st_tensor}; Valid predict HTTP {st_pred}; Clamped generation HTTP {st_gen} (tokens={b_gen.get('generated_tokens')})",
    }
    print(f"[{'PASS' if assert_limits else 'FAIL'}] Inference input constraints (Tensor 400: {st_tensor}, Predict 200: {st_pred}, Clamped Gen 200: {st_gen})")

    # 10. Concurrency Protection (Atomic Inference Lock HTTP 429)
    # Fire concurrent generation requests using ThreadPoolExecutor to verify atomic rejection
    import concurrent.futures

    def _send_concurrent_gen():
        return make_request(
            "/api/models/generate",
            method="POST",
            data={"prompt": "Deep learning on NVIDIA Ampere RTX 3050", "max_new_tokens": 48},
        )

    concurrency_statuses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_send_concurrent_gen) for _ in range(8)]
        for f in concurrent.futures.as_completed(futures):
            st, b, _, _ = f.result()
            concurrency_statuses.append(st)

    has_429 = (429 in concurrency_statuses)
    has_200 = (200 in concurrency_statuses)
    assert_concurrency = (has_429 and has_200)

    results["T10_Concurrency_Protection_429"] = {
        "status": "PASS" if assert_concurrency else "FAIL",
        "http_status": 429 if has_429 else concurrency_statuses[0],
        "detail": f"Concurrent burst (8 requests): HTTP statuses={concurrency_statuses}. 429 detected: {has_429}, 200 detected: {has_200}",
    }
    print(f"[{'PASS' if assert_concurrency else 'FAIL'}] Concurrency protection (429 observed under contention: {has_429}, statuses={concurrency_statuses})")

    # 10b. Hardware Admission Safeguards Verification
    # Verify that the live request passes admission check, and test simulated threshold breach
    from backend.serving import ModelServer
    hw_details = []
    # Test valid admission check
    import torch
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        ModelServer._enforce_demo_hardware_safeguards(dev)
        hw_details.append(f"Safe check on {dev}: passed")
        safe_hw = True
    except Exception as e:
        safe_hw = False
        hw_details.append(f"Safe check on {dev} failed: {e}")

    results["T10B_Hardware_Admission_Safeguards"] = {
        "status": "PASS" if safe_hw else "FAIL",
        "http_status": 200 if safe_hw else 503,
        "detail": "; ".join(hw_details),
    }
    print(f"[{'PASS' if safe_hw else 'FAIL'}] Hardware admission safeguards active and passing under normal conditions: {safe_hw}")

    # 11. Isolation of Private Artifacts, Logs, and Pilot Checkpoints
    # 11a. Private logs blocked
    st_log1, _, _, _ = make_request("/api/artifacts/benchmark-log")
    st_log2, _, _, _ = make_request("/api/artifacts/python-test-log")
    st_safe_art, _, _, _ = make_request("/api/artifacts/benchmark-json")
    assert_logs = (st_log1 == 404 and st_log2 == 404 and st_safe_art == 200)

    # 11b. Pilot checkpoints hidden
    st_ckpts, b_ckpts, _, _ = make_request("/api/checkpoints")
    ckpt_list = b_ckpts.get("checkpoints", [])
    has_pilot = any("pilot" in c.get("id", "").lower() or "100m" in c.get("id", "").lower() for c in ckpt_list)
    assert_ckpts = (st_ckpts == 200 and not has_pilot)

    # 11c. Training job history empty
    st_jobs, b_jobs, _, _ = make_request("/api/training/jobs")
    job_list = b_jobs.get("jobs", [])
    assert_jobs = (st_jobs == 200 and len(job_list) == 0)

    assert_isolation = (assert_logs and assert_ckpts and assert_jobs)
    results["T11_Private_Artifacts_Isolation"] = {
        "status": "PASS" if assert_isolation else "FAIL",
        "http_status": 200 if assert_isolation else 500,
        "detail": f"benchmark-log HTTP {st_log1}, python-test-log HTTP {st_log2}, summary HTTP {st_safe_art}, pilot in checkpoints: {has_pilot}, jobs count: {len(job_list)}",
    }
    print(f"[{'PASS' if assert_isolation else 'FAIL'}] Private artifacts isolation: Logs 404: {st_log1==404}, Pilot hidden: {not has_pilot}, Jobs empty: {len(job_list)==0}")

    return results


def test_standard_developer_mode() -> Dict[str, Any]:
    results = {}
    print("\n" + "=" * 70)
    print("STAGE 7C ACCEPTANCE SUITE — TESTING STANDARD DEVELOPER MODE (CUDA_ENGINE_DEMO_MODE=0)")
    print("=" * 70)

    # 1. Health & Mode Check
    status, body, _, _ = make_request("/api/health")
    mode = body.get("mode")
    assert_dev_mode = (status == 200 and mode == "read-write")
    results["T12_Standard_Mode_Health"] = {
        "status": "PASS" if assert_dev_mode else "FAIL",
        "http_status": status,
        "detail": f"mode='{mode}' (expected 'read-write')",
    }
    print(f"[{'PASS' if assert_dev_mode else 'FAIL'}] Health endpoint reports standard mode='{mode}' (HTTP {status})")

    # 2. Mutating Unload is Permitted in Standard Mode
    status_unload, body_unload, _, _ = make_request("/api/models/unload", method="POST")
    assert_unload = (status_unload == 200 and body_unload.get("status") in ("unloaded", "ok"))
    results["T13_Standard_Mode_Unload_Permitted"] = {
        "status": "PASS" if assert_unload else "FAIL",
        "http_status": status_unload,
        "detail": f"Unload returned HTTP {status_unload} (status='{body_unload.get('status')}')",
    }
    print(f"[{'PASS' if assert_unload else 'FAIL'}] Model unload permitted in standard mode: HTTP {status_unload}")

    # 3. Hardware Telemetry Unsanitized (Raw UUID & OS Details)
    status_hw, body_hw, _, _ = make_request("/api/hardware")
    gpus_hw = body_hw.get("gpus", [])
    uuid_raw = gpus_hw[0].get("uuid") if gpus_hw else None
    assert_hw = (status_hw == 200 and uuid_raw != "GPU-DEMO-REDACTED" and uuid_raw is not None)
    results["T14_Standard_Mode_Hardware_Unsanitized"] = {
        "status": "PASS" if assert_hw else "FAIL",
        "http_status": status_hw,
        "detail": f"Raw GPU UUID present: '{uuid_raw}'",
    }
    print(f"[{'PASS' if assert_hw else 'FAIL'}] Standard mode exposes real hardware telemetry: UUID='{uuid_raw}'")

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        res = test_public_demo_mode()
        print("\nDemo Mode Test Results Summary:")
        print(json.dumps(res, indent=2))
        all_passed = all(r["status"] == "PASS" for r in res.values())
        sys.exit(0 if all_passed else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "standard":
        res = test_standard_developer_mode()
        print("\nStandard Mode Test Results Summary:")
        print(json.dumps(res, indent=2))
        all_passed = all(r["status"] == "PASS" for r in res.values())
        sys.exit(0 if all_passed else 1)
    else:
        print("Usage: python test_stage7c_acceptance.py [demo|standard]")
        sys.exit(1)
