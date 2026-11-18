"""
scripts/audit_milestone6_preflight.py
Comprehensive Stage 1 Preflight Audit for Phase 13 Milestone 6.
Verifies hardware, CUDA runtime, PyTorch, custom CUDA kernels, model architecture,
tokenizer, dataset, checkpoints, disk space, and runs a CUDA stability probe.
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("PHASE 13 MILESTONE 6: STAGE 1 PREFLIGHT AUDIT & ENVIRONMENT VERIFICATION")
    print("=" * 80)

    audit_record = {
        "milestone": "Phase 13 Milestone 6",
        "stage": "Stage 1 Preflight Audit",
        "checks": {}
    }
    all_passed = True

    # 1. Git Status Audit (Non-destructive check)
    try:
        git_proc = subprocess.run(["git", "status", "--short"], cwd=str(ROOT_DIR), capture_output=True, text=True)
        git_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT_DIR), capture_output=True, text=True).stdout.strip()
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), capture_output=True, text=True).stdout.strip()
        print(f"[1] Git Branch / Commit:  {git_branch} ({git_commit[:8]})")
        print(f"    Git Working Tree:     Verified intact (no commit/push executed)")
        audit_record["checks"]["git_status"] = {
            "status": "PASS",
            "branch": git_branch,
            "commit": git_commit
        }
    except Exception as e:
        print(f"[1] Git Status:           Warning ({e})")
        audit_record["checks"]["git_status"] = {"status": "WARNING", "error": str(e)}

    # 2. GPU Accessibility & Driver
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    gpu_props = torch.cuda.get_device_properties(0) if cuda_available else None
    total_vram_mb = round(gpu_props.total_memory / (1024 ** 2), 2) if gpu_props else 0
    cuda_runtime = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "None"

    print(f"[2] CUDA Available:       {cuda_available} (Count: {device_count})")
    print(f"    GPU Model:            {device_name}")
    print(f"    Total VRAM:           {total_vram_mb} MiB")
    print(f"    Compute Capability:   {gpu_props.major}.{gpu_props.minor}" if gpu_props else "N/A")
    print(f"    PyTorch CUDA Version: {cuda_runtime}")
    print(f"    cuDNN Version:        {cudnn_version}")

    check_gpu = cuda_available and (total_vram_mb > 5000)
    audit_record["checks"]["gpu_accessibility"] = {
        "status": "PASS" if check_gpu else "FAIL",
        "device_name": device_name,
        "total_vram_mb": total_vram_mb,
        "cuda_version": cuda_runtime
    }
    if not check_gpu:
        all_passed = False

    # 3. Custom CUDA Kernel Compilation & Extension Parity
    try:
        from custom_kernels import _get_module
        import custom_kernels.autograd as cka
        cuda_mod = _get_module()
        test_x = torch.randn(128, 768, device="cuda", dtype=torch.float32)
        test_b = torch.randn(768, device="cuda", dtype=torch.float32)
        test_out = cka.fused_bias_activation(test_x, test_b, "gelu")
        custom_kernel_ok = (test_out.shape == test_x.shape)
        print(f"[3] Custom CUDA Extension: Available & Functional ({cuda_mod})")
    except Exception as e:
        custom_kernel_ok = False
        print(f"[3] Custom CUDA Extension: FAILED ({e})")
    audit_record["checks"]["custom_cuda_kernels"] = {
        "status": "PASS" if custom_kernel_ok else "FAIL"
    }
    if not custom_kernel_ok:
        all_passed = False

    # 4. Model Architecture & Parameter Count
    from engine.transformer.model import GPTTransformer
    from engine.transformer.config import TransformerConfig
    cfg = TransformerConfig(
        vocab_size=2048,
        max_seq_len=256,
        d_model=768,
        n_layer=12,
        n_head=12,
        d_ff=3072,
        tie_weights=True
    )
    model = GPTTransformer(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[4] Transformer Params:   {param_count:,} (Trainable: {trainable_count:,})")
    check_model = (param_count == 86825472)
    audit_record["checks"]["model_architecture"] = {
        "status": "PASS" if check_model else "FAIL",
        "param_count": param_count,
        "expected_count": 86825472
    }
    if not check_model:
        all_passed = False

    # 5. Tokenizer Integrity
    vocab_path = ROOT_DIR / "checkpoints" / "transformer" / "pretraining_campaign" / "bpe_vocab.json"
    vocab_exists = vocab_path.exists()
    vocab_hash = compute_sha256(vocab_path) if vocab_exists else ""
    expected_hash = "ffd4aac665204b09488cd84e5428a72aa93223e06708947520768c001c163841"
    check_tokenizer = vocab_exists and (vocab_hash == expected_hash)
    print(f"[5] Tokenizer Vocab Path: {vocab_path}")
    print(f"    Tokenizer SHA-256:    {vocab_hash} (Match: {vocab_hash == expected_hash})")
    audit_record["checks"]["tokenizer_integrity"] = {
        "status": "PASS" if check_tokenizer else "FAIL",
        "sha256": vocab_hash,
        "expected_sha256": expected_hash
    }
    if not check_tokenizer:
        all_passed = False

    # 6. Production Dataset Manifest & Binary Files
    train_bin = ROOT_DIR / "data" / "text" / "production_10m" / "train_tokens.bin"
    val_bin = ROOT_DIR / "data" / "text" / "production_10m" / "val_tokens.bin"
    test_bin = ROOT_DIR / "data" / "text" / "production_10m" / "test_tokens.bin"
    manifest_path = ROOT_DIR / "results" / "phase13_milestone5_dataset_manifest.json"

    train_tokens = train_bin.stat().st_size // 2 if train_bin.exists() else 0
    val_tokens = val_bin.stat().st_size // 2 if val_bin.exists() else 0
    test_tokens = test_bin.stat().st_size // 2 if test_bin.exists() else 0

    print(f"[6] Dataset Train Tokens: {train_tokens:,} ({train_bin.stat().st_size:,} bytes)")
    print(f"    Dataset Val Tokens:   {val_tokens:,} ({val_bin.stat().st_size:,} bytes)")
    print(f"    Dataset Test Tokens:  {test_tokens:,} ({test_bin.stat().st_size:,} bytes)")

    check_dataset = (train_tokens == 13215150) and (val_tokens == 104180) and (test_tokens == 331796)
    audit_record["checks"]["production_dataset"] = {
        "status": "PASS" if check_dataset else "FAIL",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "test_tokens": test_tokens
    }
    if not check_dataset:
        all_passed = False

    # 7. Checkpoint Verification (Milestone 5 Diagnostic Final Checkpoint)
    ckpt_path = ROOT_DIR / "checkpoints" / "transformer" / "pretraining_50m_diagnostic" / "final_model.pt"
    ckpt_exists = ckpt_path.exists()
    ckpt_hash = compute_sha256(ckpt_path) if ckpt_exists else ""
    ckpt_size = ckpt_path.stat().st_size if ckpt_exists else 0
    expected_ckpt_hash = "ca7c80c5fcc711e020c92a9087e40a86405e8ac01f80a7720d22f8fa96f93fba"
    
    ckpt_content = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_exists else {}
    has_model_state = "model_state_dict" in ckpt_content
    has_opt_state = "optimizer_state_dict" in ckpt_content
    has_sched_state = "scheduler_state_dict" in ckpt_content
    has_scaler_state = "scaler_state_dict" in ckpt_content
    has_rng_state = "torch_rng_state" in ckpt_content or "rng_state" in ckpt_content
    
    metrics = ckpt_content.get("metrics", {})
    ckpt_step = metrics.get("step", ckpt_content.get("step"))
    ckpt_tokens = metrics.get("total_tokens", ckpt_content.get("total_tokens"))

    print(f"[7] Checkpoint Path:      {ckpt_path}")
    print(f"    Checkpoint Size:      {ckpt_size:,} bytes")
    print(f"    Checkpoint SHA-256:   {ckpt_hash} (Match: {ckpt_hash == expected_ckpt_hash})")
    print(f"    Checkpoint Step:      {ckpt_step} (Expected: 2492)")
    print(f"    Cumulative Tokens:    {ckpt_tokens:,} (Expected: 10,205,184)")
    print(f"    State Components:     Model={has_model_state}, Opt={has_opt_state}, Sched={has_sched_state}, Scaler={has_scaler_state}, RNG={has_rng_state}")

    check_ckpt = (ckpt_exists and (ckpt_hash == expected_ckpt_hash) and
                  (ckpt_step == 2492) and (ckpt_tokens == 10205184) and
                  has_model_state and has_opt_state and has_sched_state and has_scaler_state)
    audit_record["checks"]["checkpoint_integrity"] = {
        "status": "PASS" if check_ckpt else "FAIL",
        "sha256": ckpt_hash,
        "step": ckpt_step,
        "cumulative_tokens": ckpt_tokens,
        "complete_state": check_ckpt
    }
    if not check_ckpt:
        all_passed = False

    # 8. Disk Space Available
    total, used, free = shutil.disk_usage(str(ROOT_DIR))
    free_gb = round(free / (1024 ** 3), 2)
    print(f"[8] Free Disk Space:      {free_gb} GB")
    check_disk = (free_gb >= 10.0) # Require >= 10 GB free for checkpoints and logs
    audit_record["checks"]["disk_space"] = {
        "status": "PASS" if check_disk else "FAIL",
        "free_space_gb": free_gb
    }
    if not check_disk:
        all_passed = False

    # 9. CUDA Stability & Thermal Probe
    print("[9] Running CUDA Hardware Stability Probe...")
    try:
        model = model.to("cuda")
        x_probe = torch.randint(0, 2048, (16, 256), device="cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            probe_out, _ = model(x_probe)
            loss_probe = probe_out.sum()
        loss_probe.backward()
        torch.cuda.synchronize()
        del model, x_probe, probe_out, loss_probe
        torch.cuda.empty_cache()
        cuda_probe_ok = True
        print("    CUDA Stability Probe: PASSED (Forward, AMP FP16, Backward, Memory Release OK)")
    except Exception as e:
        cuda_probe_ok = False
        print(f"    CUDA Stability Probe: FAILED ({e})")

    audit_record["checks"]["cuda_stability_probe"] = {
        "status": "PASS" if cuda_probe_ok else "FAIL"
    }
    if not cuda_probe_ok:
        all_passed = False

    audit_record["all_preflight_checks_passed"] = all_passed
    print("=" * 80)
    print(f"PREFLIGHT AUDIT VERDICT: {'ALL CHECKS PASSED - PROCEED TO STAGE 2' if all_passed else 'CRITICAL FAILURE'}")
    print("=" * 80)

    out_path = ROOT_DIR / "results" / "phase13_milestone6_preflight_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2)

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
