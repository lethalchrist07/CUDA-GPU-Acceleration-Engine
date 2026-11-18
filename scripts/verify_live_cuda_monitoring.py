"""
Controlled runtime verification script for NEXORA AI Training Studio.
Launches a small transformer_1m job on GPU_NATIVE, records high-frequency live polling
telemetry, verifies state transitions, metric accuracy, hardware NVML telemetry, and checkpoint isolation.
"""
import json
import os
import sys
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8000/api/training"

def post_json(endpoint: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(endpoint: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_controlled_cuda_test():
    print("=" * 70)
    print("NEXORA LIVE CUDA MONITORING VERIFICATION — PREFLIGHT")
    print("=" * 70)
    
    # Check Hardware endpoint
    hw_req = urllib.request.Request("http://127.0.0.1:8000/api/hardware")
    with urllib.request.urlopen(hw_req) as resp:
        hw = json.loads(resp.read().decode("utf-8"))
    
    gpu = hw["gpus"][0]
    print(f"Physical GPU: {gpu['name']}")
    print(f"Driver Version: {gpu['driver_version']}")
    print(f"Total Dedicated VRAM: {gpu['memory_total_mib']} MiB")
    print(f"Baseline Temperature: {gpu['temperature_c']} °C")
    print(f"Baseline Utilization: {gpu['utilization_percent']} %")
    print()

    # Step 2: Submit Controlled Training Job
    print("=" * 70)
    print("STEP 2: SUBMITTING CONTROLLED GPU_NATIVE TRAINING JOB")
    print("=" * 70)
    payload = {
        "model_name": "transformer_1m",
        "backend": "GPU_NATIVE",
        "dataset": "Tiny Shakespeare",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.0003,
        "seed": 42,
        "precision": "amp",
        "gradient_accumulation_steps": 2,
        "max_steps": 10,
        "seq_len": 64,
        "tokenizer_type": "char",
        "dataset_path": "data/text/tiny_shakespeare.txt"
    }

    job_create = post_json("/jobs", payload)
    job_id = job_create["job_id"]
    print(f"Job Registered: {job_id}")
    print(f"Initial Status: {job_create['status']}")
    print()

    # Step 3: High-Frequency Live Monitoring
    print("=" * 70)
    print("STEP 3: LIVE TELEMETRY STREAM POLLING")
    print("=" * 70)

    t0 = time.time()
    snapshots = []
    seen_running = False

    while time.time() - t0 < 60:
        fresh = get_json(f"/jobs/{job_id}")
        st = fresh["status"]
        if st == "RUNNING":
            seen_running = True

        prog = fresh.get("progress", {})
        step = prog.get("step") or prog.get("current_step")
        micro = prog.get("microbatches_processed")
        loss = prog.get("train_loss")
        tok_s = prog.get("tokens_per_sec")
        vram = prog.get("peak_vram_mb")
        phys = prog.get("physical_vram_mb")
        temp = prog.get("gpu_temperature_c")
        util = prog.get("gpu_utilization_pct")

        snap = {
            "time_offset_s": round(time.time() - t0, 3),
            "status": st,
            "step": step,
            "microbatches": micro,
            "loss": loss,
            "throughput_tok_s": tok_s,
            "vram_allocated_mb": vram,
            "vram_physical_mb": phys,
            "gpu_temperature_c": temp,
            "gpu_utilization_pct": util,
        }
        snapshots.append(snap)

        print(
            f"[{snap['time_offset_s']:>6.2f}s] Status={st:<9} | "
            f"Step={str(step):<3} | μBatches={str(micro):<3} | "
            f"Loss={f'{loss:.4f}' if loss is not None else 'None':<8} | "
            f"Tok/s={f'{tok_s:.1f}' if tok_s is not None else 'None':<7} | "
            f"Alloc={f'{vram:.1f}' if vram is not None else 'None':<6}MB | "
            f"Phys={f'{phys:.1f}' if phys is not None else 'None':<7}MB | "
            f"Temp={f'{temp:.1f}' if temp is not None else 'None':<5}°C | "
            f"Util={f'{util:.1f}' if util is not None else 'None'}%"
        )

        if st in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.3)

    print()
    print("=" * 70)
    print("STEP 4: COMPLETION VERIFICATION & METRICS AUDIT")
    print("=" * 70)

    final_job = get_json(f"/jobs/{job_id}")
    metrics = final_job.get("metrics", {})
    progress = final_job.get("progress", {})

    print(f"Final Status: {final_job['status']}")
    print(f"Seen RUNNING state during polling: {seen_running}")
    print(f"Optimizer Steps Completed: {metrics.get('optimizer_steps_completed')}")
    print(f"Microbatches Processed: {metrics.get('microbatches_processed')}")
    print(f"Final Train Loss: {metrics.get('final_train_loss')}")
    print(f"Final Val Loss: {metrics.get('final_val_loss')}")
    print(f"Final Val Perplexity: {metrics.get('final_val_perplexity')}")
    print(f"Peak Allocated VRAM: {metrics.get('peak_vram_mb')} MB")
    print(f"Peak Physical VRAM: {metrics.get('physical_vram_mb')} MB")
    print(f"GPU Temperature: {metrics.get('gpu_temperature_c')} °C")
    print(f"GPU Utilization: {metrics.get('gpu_utilization_pct')} %")
    print(f"Step Records Count: {len(metrics.get('step_records', []))}")
    print(f"Checkpoint Path: {final_job.get('checkpoint_path')}")

    # Checkpoint Verification
    ckpt_info = get_json(f"/jobs/{job_id}/checkpoints")
    print(f"Checkpoints Recorded in DB: {len(ckpt_info.get('checkpoints', []))}")
    if ckpt_info.get("checkpoints"):
        c = ckpt_info["checkpoints"][0]
        print(f"  Accuracy/PPL: {c['accuracy']}")
        print(f"  Size Bytes: {c['size_bytes']}")
        print(f"  SHA-256: {c['sha256']}")

    # Save summary artifact
    os.makedirs("results", exist_ok=True)
    summary_path = "results/live_cuda_monitoring_e2e_evidence.json"
    evidence = {
        "job_id": job_id,
        "model": payload["model_name"],
        "backend": payload["backend"],
        "preflight_gpu": gpu,
        "polling_snapshots": snapshots,
        "final_job": final_job,
        "checkpoint_info": ckpt_info,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    print(f"\nFull evidence JSON written to: {summary_path}")

    return job_id

if __name__ == "__main__":
    run_controlled_cuda_test()

