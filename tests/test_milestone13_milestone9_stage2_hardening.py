"""
tests/test_milestone13_milestone9_stage2_hardening.py
Automated Verification Suite for Phase 13 Milestone 9 Stage 2: Backend & Engine Hardening.

Covers:
1. Model Server State Isolation (alternating vision & transformer inference without state corruption)
2. Thread-Safe Concurrency & Independent Model Unloading
3. Checkpoint Catalog In-Memory Caching, Cache Hits, Invalidation, and Measured Latency
4. Comparative Laboratory Contracts (Configs A, B, C, D, input validation, concurrency guard)
5. Production 50M-Token Checkpoint Cryptographic & Architectural Integrity
"""

import concurrent.futures
import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.data import (
    ArtifactStore,
    clear_catalog_cache,
    clear_digest_cache,
    digest,
    get_digest_cache_stats,
)
from backend.serving import (
    ModelServer,
    _sync_legacy_pointers,
)
from engine.transformer import BPETokenizer, GPTTransformer, TransformerConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Pytest Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    app = create_app(PROJECT_ROOT)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_serving_state():
    """Ensure clean serving state before and after each test."""
    ModelServer.unload_model("all")
    yield
    ModelServer.unload_model("all")


# -----------------------------------------------------------------------------
# TASK 1: Model Server State Isolation & Concurrency Safety
# -----------------------------------------------------------------------------
def test_alternating_vision_and_transformer_inference():
    """
    Verify that alternating between text generation and vision classification
    does not cause model state collision, dimension mismatch, or HTTP 400 crashes.
    """
    import backend.serving as serving

    # 1. First Text Generation (loads Transformer)
    gen1 = ModelServer.generate(prompt="The GPU engine", max_new_tokens=4, seed=42)
    assert gen1["status"] == "success"
    assert "Transformer" in gen1["model_name"]
    assert serving._TRANSFORMER_MODEL is not None
    assert serving._ACTIVE_MODEL is not None

    # 2. Vision Inference (must auto-load ResNet18 into vision slot without colliding)
    pred1 = ModelServer.predict(sample_index=0)
    assert pred1["status"] == "success"
    assert pred1["model"] == "ResNet18"
    assert serving._VISION_MODEL is not None
    assert serving._TRANSFORMER_MODEL is not None  # Transformer remains safely cached!

    # 3. Second Text Generation (must reuse Transformer slot without re-crashing)
    gen2 = ModelServer.generate(prompt="CUDA acceleration", max_new_tokens=4, seed=42)
    assert gen2["status"] == "success"
    assert "Transformer" in gen2["model_name"]

    # 4. Second Vision Inference (must reuse ResNet18 slot cleanly)
    pred2 = ModelServer.predict(sample_index=1)
    assert pred2["status"] == "success"
    assert pred2["model"] == "ResNet18"


def test_independent_model_unloading():
    """
    Verify that unloading the vision model does not destroy the transformer model,
    and vice-versa.
    """
    import backend.serving as serving

    # Load both models
    ModelServer.load_model("ResNet18", device_str="cpu")
    ModelServer.load_model("NEXORA GPT-86.8M", device_str="cpu")

    assert serving._VISION_MODEL is not None
    assert serving._TRANSFORMER_MODEL is not None

    # Unload only vision
    un_vis = ModelServer.unload_model(target="vision")
    assert un_vis["status"] == "unloaded"
    assert serving._VISION_MODEL is None
    assert serving._TRANSFORMER_MODEL is not None  # Transformer still alive!

    # Unload only transformer
    un_trans = ModelServer.unload_model(target="transformer")
    assert un_trans["status"] == "unloaded"
    assert serving._TRANSFORMER_MODEL is None


def test_multithreaded_inference_isolation():
    """
    Verify that concurrent calls to generate and predict across multiple threads
    are serialized safely by _SERVING_LOCK without corrupting state.
    """
    results = []

    def run_generate():
        return ModelServer.generate(prompt="Deep learning", max_new_tokens=4, seed=42)

    def run_predict():
        return ModelServer.predict(sample_index=0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_generate),
            executor.submit(run_predict),
            executor.submit(run_generate),
            executor.submit(run_predict),
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    assert len(results) == 4
    assert all(r["status"] == "success" for r in results)


# -----------------------------------------------------------------------------
# TASK 2: Checkpoint Catalog In-Memory Caching & Latency
# -----------------------------------------------------------------------------
def test_digest_cache_hits_and_invalidation(tmp_path):
    """
    Verify that digest() returns cached values on hit, invalidates on mtime change,
    and respects force_recompute.
    """
    test_file = tmp_path / "test_weight.bin"
    test_file.write_bytes(b"initial_weight_payload_1234567890")

    clear_digest_cache()
    stats_init = get_digest_cache_stats()

    # First digest: cache miss
    h1 = digest(test_file)
    stats1 = get_digest_cache_stats()
    assert stats1["misses"] == stats_init["misses"] + 1

    # Second digest: cache hit
    h2 = digest(test_file)
    stats2 = get_digest_cache_stats()
    assert stats2["hits"] == stats1["hits"] + 1
    assert h1 == h2

    # Forced recompute: bypasses cache
    h3 = digest(test_file, force_recompute=True)
    assert h3 == h1

    # Invalidation on file modification
    time.sleep(0.01)
    test_file.write_bytes(b"modified_weight_payload_0987654321")
    h4 = digest(test_file)
    assert h4 != h1

    # Restore persistent cache for subsequent tests
    clear_digest_cache(reload_persistent=True)


def test_checkpoint_catalog_in_memory_speedup(tmp_path):
    """
    Measure in-memory catalog response time and verify sub-10 ms warm latency
    using isolated test checkpoints.
    """
    ckpt_dir = tmp_path / "checkpoints" / "transformer" / "pretraining_50m_campaign"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best_model.pt").write_bytes(b"mock_50m_weights_best" * 1000)
    (ckpt_dir / "final_model.pt").write_bytes(b"mock_50m_weights_final" * 1000)
    (tmp_path / "checkpoints" / "resnet18_cifar10").mkdir(parents=True)
    (tmp_path / "checkpoints" / "resnet18_cifar10" / "best_model.pt").write_bytes(b"mock_resnet_weights" * 1000)

    store = ArtifactStore(tmp_path)
    clear_catalog_cache()

    # Call 1: cold traversal and cache population
    t0 = time.perf_counter()
    res1 = store.checkpoints_catalog()
    t_cold = (time.perf_counter() - t0) * 1000.0

    assert res1["status"] == "available"
    assert len(res1["checkpoints"]) == 3

    # Call 2: warm in-memory catalog cache lookup
    t0 = time.perf_counter()
    res2 = store.checkpoints_catalog()
    t_warm = (time.perf_counter() - t0) * 1000.0

    assert res2["status"] == "available"
    assert len(res2["checkpoints"]) == len(res1["checkpoints"])
    assert t_warm < 10.0, f"Expected warm catalog latency < 10.0 ms, measured: {t_warm:.3f} ms"
    clear_catalog_cache()


def test_checkpoint_catalog_endpoint_via_fastapi(client):
    """
    Test the FastAPI /api/checkpoints endpoint and force_rehash query param.
    """
    # Warm GET request
    resp = client.get("/api/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"
    assert "checkpoints" in data
    assert len(data["checkpoints"]) > 0

    # Verify 50M campaign checkpoint is classified with 86.83M parameters
    ckpt_50m = next((c for c in data["checkpoints"] if "50m" in c["path"]), None)
    assert ckpt_50m is not None
    assert ckpt_50m["parameters"] == 86825472
    assert "86.83M" in ckpt_50m["parameters_display"]


# -----------------------------------------------------------------------------
# TASK 3: Harden CUDA Laboratory Backend Contracts
# -----------------------------------------------------------------------------
def test_laboratory_configurations_api(client):
    """
    Verify /api/laboratory/configurations returns all four comparative configs.
    """
    resp = client.get("/api/laboratory/configurations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_configurations"] >= 4
    config_ids = [c["id"] for c in data["configurations"]]
    assert "config_a" in config_ids
    assert "config_b" in config_ids
    assert "config_c" in config_ids
    assert "config_d" in config_ids


def test_laboratory_experiments_history_api(client):
    """
    Verify /api/laboratory/experiments returns comparative benchmark data
    prioritizing Milestone 8 certified final performance.
    """
    resp = client.get("/api/laboratory/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"
    assert "experiment_data" in data


def test_create_experiment_input_validation(client):
    """
    Verify POST /api/laboratory/experiments rejects unknown configuration strings
    with HTTP 422 Unprocessable Entity.
    """
    # Invalid config name: should be rejected
    bad_req = {
        "name": "Invalid Config Test",
        "configurations": ["Config A", "Config Unknown 99"],
        "steps": 10,
        "batch_size": 16,
        "seq_len": 256,
        "seed": 42,
    }
    resp = client.post("/api/laboratory/experiments", json=bad_req)
    assert resp.status_code == 422
    assert "Invalid configuration" in resp.json()["detail"]

    # Valid configs: should succeed with 201 Created
    good_req = {
        "name": "Valid 4-Way Comparative Test",
        "configurations": ["Config A", "Config B", "Config C", "Config D"],
        "steps": 10,
        "batch_size": 16,
        "seq_len": 256,
        "seed": 42,
    }
    resp_good = client.post("/api/laboratory/experiments", json=good_req)
    assert resp_good.status_code == 201
    assert resp_good.json()["status"] == "created"


def test_cuda_benchmarks_endpoint(client):
    """
    Verify GET /api/laboratory/cuda-benchmarks returns kernel bottleneck metrics.
    """
    resp = client.get("/api/laboratory/cuda-benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"
    assert "benchmarks" in data


# -----------------------------------------------------------------------------
# TASK 4: Production 50M Checkpoint Integrity & Parameter Verification
# -----------------------------------------------------------------------------
def test_production_50m_checkpoint_integrity():
    """
    Verify the cryptographic hash, architecture parameters, and tokenizer
    for checkpoints/transformer/pretraining_50m_campaign/best_model.pt.
    """
    ckpt_path = PROJECT_ROOT / "checkpoints/transformer/pretraining_50m_campaign/best_model.pt"
    assert ckpt_path.is_file(), "50M production checkpoint must exist"

    # 1. Cryptographic SHA-256 Digest Verification
    computed_sha256 = digest(ckpt_path)
    expected_sha256 = "abcd0fe7b305cd1cef7809359ca7128f0ac94bed592a188f8432329ca1406aac"
    assert computed_sha256 == expected_sha256, "Cryptographic hash mismatch on best_model.pt"

    # 2. Architecture Hyperparameters
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    cfg_dict = meta.get("model_config", {})
    cfg = TransformerConfig.from_dict(cfg_dict)

    assert cfg.vocab_size == 2048
    assert cfg.d_model == 768
    assert cfg.n_layer == 12
    assert cfg.n_head == 12
    assert cfg.d_ff == 3072

    # 3. Parameter Count Verification
    model = GPTTransformer(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 86825472, f"Expected 86,825,472 parameters, got {total_params}"

    # 4. Tokenizer Compatibility
    tok_path = ckpt_path.parent / "bpe_vocab.json"
    assert tok_path.is_file(), "BPE vocab file must exist in checkpoint directory"
    tok = BPETokenizer.load(tok_path)
    assert tok.vocab_size == 2048

    test_str = "NVIDIA CUDA GPU Acceleration Engine"
    encoded = tok.encode(test_str)
    decoded = tok.decode(encoded)
    assert decoded == test_str, "Tokenizer encoding/decoding must be lossless"
