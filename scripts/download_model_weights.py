"""
scripts/download_model_weights.py
Automated Model Weights Discovery, Placement, and Cryptographic Verification Utility.
Inspects checkpoints/MODEL_MANIFEST.json, validates model presence and SHA-256 hashes,
and assists with zero-configuration deployment of model weights.
"""
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()

def inspect_and_verify_models(manifest_path: Path, verbose: bool = True) -> Dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    models: List[Dict[str, Any]] = manifest.get("models", [])
    results = {
        "total_manifest_models": len(models),
        "present_verified": 0,
        "present_mismatch": 0,
        "missing": 0,
        "details": []
    }

    if verbose:
        print("=" * 70)
        print("  NEXORA MODEL WEIGHTS DISCOVERY & VERIFICATION UTILITY")
        print("=" * 70)
        print(f"[*] Manifest: {manifest_path.relative_to(PROJECT_ROOT).as_posix()}")
        print(f"[*] Release Version: {manifest.get('release_version', 'Unknown')}")
        print(f"[*] Hardware Target: {manifest.get('hardware_target', 'Unknown')}\n")

    for m in models:
        model_id = m.get("model_id")
        model_name = m.get("model_name")
        rel_path = m.get("relative_path")
        expected_sha = m.get("sha256", "").upper()
        expected_size = m.get("size_bytes", 0)

        full_path = PROJECT_ROOT / rel_path
        status = "MISSING"
        actual_sha = None
        actual_size = None

        if full_path.is_file():
            actual_size = full_path.stat().st_size
            actual_sha = compute_sha256(full_path)
            if actual_sha == expected_sha:
                status = "VERIFIED_OK"
                results["present_verified"] += 1
            else:
                status = "HASH_MISMATCH"
                results["present_mismatch"] += 1
        else:
            results["missing"] += 1

        detail = {
            "model_id": model_id,
            "model_name": model_name,
            "path": rel_path,
            "status": status,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
        }
        results["details"].append(detail)

        if verbose:
            print(f"[{status}] {model_name} ({model_id})")
            print(f"    Target Path: {rel_path}")
            if status == "VERIFIED_OK":
                print(f"    Size: {actual_size:,} bytes | SHA-256: {actual_sha}")
            elif status == "HASH_MISMATCH":
                print(f"    Size: {actual_size:,} bytes (Expected: {expected_size:,})")
                print(f"    Actual SHA:   {actual_sha}")
                print(f"    Expected SHA: {expected_sha}")
            else:
                print(f"    Status: Checkpoint file is absent from disk.")
                print(f"    Expected Size: {expected_size:,} bytes (~{expected_size / (1024**2):.2f} MB)")
                print(f"    Expected SHA-256: {expected_sha}")
            print()

    # Also check lightweight test fixture
    test_seed = PROJECT_ROOT / "checkpoints/test_checkpoint.pt"
    if test_seed.is_file():
        seed_sha = compute_sha256(test_seed)
        if verbose:
            print(f"[FOUND] Baseline Seed Fixture: checkpoints/test_checkpoint.pt")
            print(f"    Size: {test_seed.stat().st_size:,} bytes | SHA-256: {seed_sha}")
            print()

    # Also check ResNet-18
    resnet_path = PROJECT_ROOT / "checkpoints/resnet18_cifar10/best_model.pt"
    if resnet_path.is_file():
        resnet_sha = compute_sha256(resnet_path)
        if verbose:
            print(f"[FOUND] Vision Classifier: checkpoints/resnet18_cifar10/best_model.pt")
            print(f"    Size: {resnet_path.stat().st_size:,} bytes | SHA-256: {resnet_sha}")
            print()

    if verbose:
        print("=" * 70)
        print(f"  VERIFICATION SUMMARY: {results['present_verified']} Verified, {results['present_mismatch']} Mismatch, {results['missing']} Missing")
        if results["missing"] > 0:
            print("\n[!] NOTICE REGARDING MODEL WEIGHTS REDISTRIBUTION:")
            print("    Model weights are computationally derived artifacts distinct from the MIT-licensed code.")
            print("    Public distribution/hosting is paused pending Project Owner authorization (see MODEL_CARD.md Section 9).")
            print("    To deploy models manually:")
            print("      1. Acquire authorized best_model.pt checkpoints.")
            print("      2. Place files at the target paths indicated above.")
            print("      3. Re-run: python scripts/download_model_weights.py --verify")
        print("=" * 70)

    return results

def main():
    parser = argparse.ArgumentParser(description="NEXORA Model Weights Discovery & Verification Utility")
    parser.add_argument("--manifest", type=str, default="checkpoints/MODEL_MANIFEST.json", help="Path to MODEL_MANIFEST.json")
    parser.add_argument("--verify", action="store_true", help="Perform strict hash verification of present checkpoints")
    args = parser.parse_args()

    manifest_file = PROJECT_ROOT / args.manifest
    inspect_and_verify_models(manifest_file, verbose=True)

if __name__ == "__main__":
    main()
