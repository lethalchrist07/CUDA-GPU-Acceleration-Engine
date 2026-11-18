"""
Milestone 6 Artifact Verification Script
Phase 13 Milestone 7 - Stage 1
Audits checkpoints, hashes, parameters, tokenizers, optimizer/scheduler recovery, and historical integrity.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.transformer.model import GPTTransformer, TransformerConfig


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_artifacts():
    print("=" * 70)
    print("STAGE 1: MILESTONE 6 ARTIFACT & CHECKPOINT INTEGRITY AUDIT")
    print("=" * 70)

    audit_result = {
        "audit_name": "Milestone 6 Artifact Audit",
        "milestone": "Phase 13 Milestone 7",
        "status": "IN_PROGRESS",
        "checks": {},
        "metadata": {},
        "historical_integrity": {},
    }

    # 1. Hardware & Environment
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if torch.cuda.is_available() else 0
    audit_result["metadata"]["hardware"] = {
        "device_name": device_name,
        "cuda_capability": f"{cuda_capability[0]}.{cuda_capability[1]}",
        "total_vram_mb": round(total_vram_mb, 2),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    print(f"[Hardware] Device: {device_name}, VRAM: {total_vram_mb:.1f} MB, CC: {cuda_capability[0]}.{cuda_capability[1]}")

    # 2. Paths
    checkpoint_dir = REPO_ROOT / "checkpoints" / "transformer" / "pretraining_50m_campaign"
    final_ckpt_path = checkpoint_dir / "final_model.pt"
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    vocab_path = REPO_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "bpe_vocab.json"
    manifest_path = REPO_ROOT / "results" / "phase13_milestone6_50m_campaign_summary.json"

    # Check existence
    audit_result["checks"]["final_model_exists"] = final_ckpt_path.exists()
    audit_result["checks"]["best_model_exists"] = best_ckpt_path.exists()
    audit_result["checks"]["vocab_exists"] = vocab_path.exists()
    audit_result["checks"]["m6_summary_exists"] = manifest_path.exists()
    print(f"[Files] final_model.pt: {final_ckpt_path.exists()}, best_model.pt: {best_ckpt_path.exists()}")

    # 3. Checkpoint hashes and manifest matching
    print("[Hashing] Computing SHA-256 for checkpoints...")
    final_sha256 = compute_sha256(final_ckpt_path)
    best_sha256 = compute_sha256(best_ckpt_path)
    vocab_sha256 = compute_sha256(vocab_path)

    audit_result["metadata"]["checkpoint_hashes"] = {
        "final_model_sha256": final_sha256,
        "final_model_size_bytes": final_ckpt_path.stat().st_size,
        "best_model_sha256": best_sha256,
        "best_model_size_bytes": best_ckpt_path.stat().st_size,
        "vocab_sha256": vocab_sha256,
    }

    m6_data = json.load(open(manifest_path, "r", encoding="utf-8"))
    recorded_checkpoints = m6_data.get("checkpoints", [])
    
    # Find recorded hash for final and best
    final_recorded = [c for c in recorded_checkpoints if "final_model.pt" in c["filepath"]]
    best_recorded = [c for c in recorded_checkpoints if "best_model.pt" in c["filepath"] and c["step"] == 12000]

    final_hash_match = bool(final_recorded and final_recorded[-1]["sha256"] == final_sha256)
    best_hash_match = bool(best_recorded and best_recorded[-1]["sha256"] == best_sha256)

    audit_result["checks"]["final_hash_matches_manifest"] = final_hash_match
    audit_result["checks"]["best_hash_matches_manifest"] = best_hash_match
    print(f"[Hash Match] final_model: {final_hash_match} ({final_sha256[:16]}...)")
    print(f"[Hash Match] best_model:  {best_hash_match} ({best_sha256[:16]}...)")

    # 4. Checkpoint Loading & Architecture Verification
    print("[Loading] Loading checkpoints into CPU/GPU...")
    final_ckpt = torch.load(final_ckpt_path, map_location="cpu", weights_only=False)
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)

    # Tokenizer verification
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data.get("vocab_size", len(vocab_data.get("token_to_id", {})))
    audit_result["checks"]["tokenizer_vocab_size_is_2048"] = (vocab_size == 2048)
    audit_result["metadata"]["tokenizer"] = {
        "vocab_size": vocab_size,
        "sha256": vocab_sha256,
        "matches_expected": vocab_size == 2048
    }

    # Model architecture verification
    config = TransformerConfig(
        vocab_size=2048,
        n_layer=12,
        n_head=12,
        d_model=768,
        d_ff=3072,
        max_seq_len=256,
        dropout=0.0,
        tie_weights=True,
    )
    model = GPTTransformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    audit_result["checks"]["param_count_exactly_86825472"] = (param_count == 86825472)
    audit_result["metadata"]["parameter_count"] = {
        "total_parameters": param_count,
        "trainable_parameters": trainable_param_count,
        "target": 86825472
    }
    print(f"[Architecture] Parameters: {param_count:,} (Trainable: {trainable_param_count:,})")

    # Load weights into model
    missing_f, unexpected_f = model.load_state_dict(final_ckpt["model_state_dict"], strict=True)
    audit_result["checks"]["final_model_loads_strictly"] = (len(missing_f) == 0 and len(unexpected_f) == 0)

    # 5. Token Accounting & Step Verification
    final_step = final_ckpt.get("metrics", {}).get("step", final_ckpt.get("step"))
    final_tokens = final_ckpt.get("metrics", {}).get("total_tokens", final_ckpt.get("tokens"))
    best_step = best_ckpt.get("metrics", {}).get("step", best_ckpt.get("step"))
    best_tokens = best_ckpt.get("metrics", {}).get("total_tokens", best_ckpt.get("tokens"))

    audit_result["checks"]["final_step_is_12208"] = (final_step == 12208)
    audit_result["checks"]["cumulative_tokens_is_50001920"] = (final_tokens == 50001920)
    audit_result["checks"]["best_step_is_12000"] = (best_step == 12000)

    audit_result["metadata"]["training_progress"] = {
        "final_step": final_step,
        "final_tokens": final_tokens,
        "best_step": best_step,
        "best_tokens": best_tokens,
        "final_val_loss": final_ckpt.get("metrics", {}).get("val_loss"),
        "best_val_loss": best_ckpt.get("metrics", {}).get("val_loss"),
    }
    print(f"[Training Progress] Final Step: {final_step}, Cumulative Tokens: {final_tokens:,}")
    print(f"[Training Progress] Best Step:  {best_step}, Best Val Loss: {best_ckpt.get('metrics', {}).get('val_loss')}")

    # 6. Optimizer & Scheduler State Recovery
    print("[Recovery] Testing optimizer and scheduler recovery from checkpoint...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-5, weight_decay=0.01)
    has_optimizer_state = "optimizer_state_dict" in final_ckpt
    has_scheduler_state = "scheduler_state_dict" in final_ckpt
    has_scaler_state = "scaler_state_dict" in final_ckpt

    optimizer_recovered = False
    scheduler_recovered = False
    scaler_recovered = False

    if has_optimizer_state:
        try:
            optimizer.load_state_dict(final_ckpt["optimizer_state_dict"])
            # Verify learning rate in param groups
            recovered_lr = optimizer.param_groups[0]["lr"]
            optimizer_recovered = True
            print(f"[Recovery] Optimizer state loaded successfully. Param group LR: {recovered_lr:.6e}")
        except Exception as e:
            print(f"[Recovery Error] Optimizer load failed: {e}")

    if has_scheduler_state:
        try:
            from torch.optim.lr_scheduler import LambdaLR
            # Create dummy lambda lr matching cosine
            scheduler = LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
            scheduler.load_state_dict(final_ckpt["scheduler_state_dict"])
            scheduler_recovered = True
            print(f"[Recovery] Scheduler state loaded successfully. Last epoch: {scheduler.last_epoch}")
        except Exception as e:
            print(f"[Recovery Error] Scheduler load failed: {e}")

    if has_scaler_state:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
            scaler.load_state_dict(final_ckpt["scaler_state_dict"])
            scaler_recovered = True
            print(f"[Recovery] GradScaler state loaded successfully. Scale: {scaler.get_scale()}")
        except Exception as e:
            print(f"[Recovery Error] Scaler load failed: {e}")

    audit_result["checks"]["optimizer_state_recoverable"] = optimizer_recovered
    audit_result["checks"]["scheduler_state_recoverable"] = scheduler_recovered
    audit_result["checks"]["scaler_state_recoverable"] = scaler_recovered

    # 7. Historical Checkpoint Isolation Check
    print("[Historical] Verifying historical checkpoint integrity...")
    historical_paths = {
        "pretraining_campaign_final": REPO_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "final_model.pt",
        "pretraining_10m_extended_best": REPO_ROOT / "checkpoints" / "transformer" / "pretraining_10m_extended" / "best_model.pt",
        "pretraining_50m_diagnostic_final": REPO_ROOT / "checkpoints" / "transformer" / "pretraining_50m_diagnostic" / "final_model.pt",
        "resnet18_cifar10_best": REPO_ROOT / "checkpoints" / "resnet18_cifar10" / "best_model.pt",
    }

    historical_status = {}
    for name, p in historical_paths.items():
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        historical_status[name] = {
            "exists": exists,
            "size_bytes": size,
            "path": str(p)
        }
        print(f"[Historical] {name}: exists={exists}, size={size:,} bytes")

    audit_result["historical_integrity"] = historical_status
    audit_result["checks"]["historical_checkpoints_preserved"] = all(v["exists"] and v["size_bytes"] > 0 for v in historical_status.values())

    # Overall Status
    all_passed = all(audit_result["checks"].values())
    audit_result["status"] = "PASSED" if all_passed else "FAILED"
    print(f"\n[Audit Verdict] Overall Status: {audit_result['status']}")
    for check_name, passed in audit_result["checks"].items():
        print(f"  - {check_name}: {'PASS' if passed else 'FAIL'}")

    # Save artifact
    output_path = REPO_ROOT / "results" / "phase13_milestone7_artifact_audit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)
    print(f"\nArtifact saved to: {output_path}")
    return audit_result


if __name__ == "__main__":
    audit_artifacts()
