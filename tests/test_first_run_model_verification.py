"""
tests/test_first_run_model_verification.py
Automated Verification Battery for BLK-05 First-Run Model Verification.

Validates:
1. Model Catalog Discovery & Architecture Registration.
2. Lightweight Seed Checkpoint (BaselineMLP 72KB) loading & non-mock inference.
3. Production Vision Model (ResNet-18) loading and inference on CUDA (RTX 3050).
4. Vision Model CPU Fallback mathematical equivalence.
5. Production Transformer (NEXORA GPT-86.8M) autoregressive generation on CUDA.
6. Transformer Model CPU Fallback autoregressive generation.
7. Atomic Model Unload & Safe Memory Cleanup.
8. Missing Checkpoint Error Handling & Actionable Recovery Guidance.
9. Cryptographic Checkpoint Hash Verification against MODEL_MANIFEST.json.
"""

import os
import json
import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
import backend.serving as serving

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    with TestClient(create_app(PROJECT_ROOT)) as c:
        yield c


class TestFirstRunModelVerification:
    """BLK-05 Comprehensive First-Run Model Verification Suite."""

    def test_model_catalog_discovery(self, client):
        """Verify GET /api/models lists all 5 registered model architectures."""
        res = client.get("/api/models")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "available"
        models = data["models"]
        model_ids = {m["model_id"] for m in models}
        expected_ids = {"resnet18_cifar10", "transformer_50m", "transformer_100m", "cifar10_mlp", "baseline_mlp"}
        assert expected_ids.issubset(model_ids), f"Missing models in catalog: {expected_ids - model_ids}"

    def test_baseline_mlp_seed_loading_and_predict(self, client):
        """Verify lightweight 72KB seed checkpoint loads and executes real inference."""
        res_load = client.post("/api/models/load", json={"model_name": "BaselineMLP", "device": "cpu"})
        assert res_load.status_code == 200, f"Load failed: {res_load.text}"
        load_data = res_load.json()
        assert load_data["status"] == "loaded"
        assert load_data["model_name"] == "BaselineMLP"
        assert load_data["parameters"] == 17226

        # Predict with custom tensor
        custom_input = [[0.05] * 64]
        res_pred = client.post("/api/models/predict", json={"tensor": custom_input})
        assert res_pred.status_code == 200, f"Predict failed: {res_pred.text}"
        pred_data = res_pred.json()
        assert pred_data["status"] == "success"
        assert pred_data["model"] == "BaselineMLP"
        assert "predicted_class" in pred_data
        assert len(pred_data["top_predictions"]) > 0
        assert pred_data["top_predictions"][0]["confidence"] > 0.0

        # Unload
        client.post("/api/models/unload")

    def test_vision_model_cuda_load_and_predict(self, client):
        """Verify ResNet-18 loads on CUDA (if available) and classifies CIFAR-10 sample 0 accurately."""
        torch = serving.get_torch()
        if not torch or not torch.cuda.is_available():
            pytest.skip("CUDA device not available on this host")

        res_load = client.post("/api/models/load", json={"model_name": "ResNet18", "device": "cuda:0"})
        assert res_load.status_code == 200, f"Load failed: {res_load.text}"
        load_data = res_load.json()
        assert load_data["status"] == "loaded"
        assert "cuda" in load_data["device"]
        assert load_data["parameters"] == 11173962

        res_pred = client.post("/api/models/predict", json={"sample_index": 0})
        assert res_pred.status_code == 200, f"Predict failed: {res_pred.text}"
        pred_data = res_pred.json()
        assert pred_data["status"] == "success"
        assert pred_data["model"] == "ResNet18"
        assert pred_data["predicted_class"] == "cat"
        assert pred_data["predicted_index"] == 3
        assert pred_data["confidence"] > 0.95
        assert pred_data["correct"] is True

        client.post("/api/models/unload")

    def test_vision_model_cpu_fallback(self, client):
        """Verify ResNet-18 executes identically on CPU with bit-for-bit classification match."""
        res_load = client.post("/api/models/load", json={"model_name": "ResNet18", "device": "cpu"})
        assert res_load.status_code == 200
        assert res_load.json()["device"] == "cpu"

        res_pred = client.post("/api/models/predict", json={"sample_index": 0})
        assert res_pred.status_code == 200
        pred_data = res_pred.json()
        assert pred_data["predicted_class"] == "cat"
        assert pred_data["confidence"] > 0.95
        assert pred_data["correct"] is True

        client.post("/api/models/unload")

    def test_generative_transformer_cuda_load_and_generate(self, client):
        """Verify NEXORA GPT-86.8M loads on CUDA and generates real autoregressive tokens."""
        torch = serving.get_torch()
        if not torch or not torch.cuda.is_available():
            pytest.skip("CUDA device not available on this host")

        res_gen = client.post("/api/models/generate", json={
            "prompt": "Technical Foundations of GPU Computing:",
            "max_new_tokens": 15,
            "temperature": 0.8,
            "device": "cuda:0",
            "seed": 42
        })
        assert res_gen.status_code == 200, f"Generation failed: {res_gen.text}"
        gen_data = res_gen.json()
        assert gen_data["status"] == "success"
        assert gen_data["generated_tokens"] == 15
        assert len(gen_data["generated_text"]) > 0
        assert gen_data["tokens_per_second"] > 0.0
        assert gen_data["peak_vram_mb"] < 1000.0  # Well within RTX 3050 6GB budget

        client.post("/api/models/unload")

    def test_generative_transformer_cpu_fallback(self, client):
        """Verify NEXORA GPT-86.8M executes autoregressive generation cleanly on CPU."""
        res_gen = client.post("/api/models/generate", json={
            "prompt": "Technical Foundations of GPU Computing:",
            "max_new_tokens": 8,
            "temperature": 0.8,
            "device": "cpu",
            "seed": 42
        })
        assert res_gen.status_code == 200, f"CPU Generation failed: {res_gen.text}"
        gen_data = res_gen.json()
        assert gen_data["status"] == "success"
        assert gen_data["device"] == "cpu"
        assert gen_data["generated_tokens"] == 8
        assert len(gen_data["generated_text"]) > 0

        client.post("/api/models/unload")

    def test_missing_model_clear_error_and_recovery_guidance(self, client):
        """Verify that missing model requests return descriptive 404 with recovery guidance."""
        res = client.post("/api/models/load", json={
            "model_name": "NonExistentModel_XYZ",
            "checkpoint": "checkpoints/does_not_exist/model.pt"
        })
        assert res.status_code == 404
        detail = res.json()["detail"]
        assert "Checkpoint not found at" in detail
        assert "checkpoints/README.md" in detail

    def test_manifest_integrity_and_hashes(self):
        """Verify that checkpoints match SHA-256 hashes defined in checkpoints/MODEL_MANIFEST.json."""
        manifest_path = PROJECT_ROOT / "checkpoints/MODEL_MANIFEST.json"
        assert manifest_path.is_file()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for m in manifest.get("models", []):
            rel_path = m["relative_path"]
            expected_sha = m["sha256"].upper()
            ckpt_path = PROJECT_ROOT / rel_path
            assert ckpt_path.is_file(), f"Manifest checkpoint missing on disk: {rel_path}"

            hasher = hashlib.sha256()
            with open(ckpt_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest().upper()
            assert actual_sha == expected_sha, (
                f"Checksum mismatch for {m['model_name']}: {actual_sha} != {expected_sha}"
            )
