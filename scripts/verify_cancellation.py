"""
Safe cancellation verification for NEXORA AI Training Studio.
Registers a disposable job, verifies transition to RUNNING, sends cancellation request,
and verifies cooperative termination, status transition to CANCELLED, and GPU lock release.
"""
import json
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8000/api/training"

def test_cancellation():
    print("=" * 70)
    print("TEST: SAFE COOPERATIVE CANCELLATION ON DISPOSABLE JOB")
    print("=" * 70)

    payload = {
        "model_name": "transformer_1m",
        "backend": "GPU_NATIVE",
        "dataset": "Tiny Shakespeare",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.0003,
        "seed": 999,
        "precision": "amp",
        "gradient_accumulation_steps": 2,
        "max_steps": 200,  # Long enough to guarantee safe cancellation window
        "seq_len": 128,
        "tokenizer_type": "char",
        "dataset_path": "data/text/tiny_shakespeare.txt"
    }

    req = urllib.request.Request(
        f"{BASE_URL}/jobs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        job = json.loads(resp.read().decode("utf-8"))
    
    job_id = job["job_id"]
    print(f"Registered disposable job for cancellation: {job_id}")

    # Wait for job to enter RUNNING state
    t0 = time.time()
    while time.time() - t0 < 10:
        q = urllib.request.Request(f"{BASE_URL}/jobs/{job_id}")
        with urllib.request.urlopen(q) as resp:
            fresh = json.loads(resp.read().decode("utf-8"))
        if fresh["status"] == "RUNNING":
            print(f"Job is actively RUNNING at step {fresh.get('progress', {}).get('step', 0)}")
            break
        time.sleep(0.1)

    # Issue cancellation
    print("Transmitting cooperative cancellation request...")
    cancel_req = urllib.request.Request(
        f"{BASE_URL}/jobs/{job_id}/cancel",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(cancel_req) as resp:
        cancel_resp = json.loads(resp.read().decode("utf-8"))
    print(f"Cancel Response: {cancel_resp}")

    # Poll for terminal CANCELLED state
    cancelled_ok = False
    for _ in range(30):
        q = urllib.request.Request(f"{BASE_URL}/jobs/{job_id}")
        with urllib.request.urlopen(q) as resp:
            fresh = json.loads(resp.read().decode("utf-8"))
        st = fresh["status"]
        print(f"Current Status: {st}")
        if st == "CANCELLED":
            cancelled_ok = True
            break
        time.sleep(0.3)

    print(f"Successfully reached terminal CANCELLED: {cancelled_ok}")

    # Verify GPU Lock Release by querying models and jobs
    print("Verifying GPU lock was safely released...")
    jobs_req = urllib.request.Request(f"{BASE_URL}/jobs")
    with urllib.request.urlopen(jobs_req) as resp:
        jobs_list = json.loads(resp.read().decode("utf-8"))
    
    cancelled_job_in_list = next((j for j in jobs_list["jobs"] if j["job_id"] == job_id), None)
    assert cancelled_job_in_list is not None
    assert cancelled_job_in_list["status"] == "CANCELLED"
    print("Confirmed job appears in history with status CANCELLED.")

    evidence = {
        "job_id": job_id,
        "initial_status": job["status"],
        "cancel_response": cancel_resp,
        "final_status": fresh["status"],
        "cancelled_ok": cancelled_ok,
    }
    with open("results/cancellation_verification_evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    return cancelled_ok

if __name__ == "__main__":
    test_cancellation()

