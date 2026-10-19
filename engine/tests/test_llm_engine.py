import os
import pytest
import torch
from engine.llm.types import InferenceRequest, InferenceResponse, InferenceMetrics
from engine.llm.safety import sanitize_prompt, validate_inference_parameters, safe_display_output
from engine.llm.mock_backend import MockLLMBackend
from engine.llm.ollama_backend import OllamaBackend
from engine.llm.engine import LLMEngine
from engine.llm.tracker import LLMTracker


def test_prompt_sanitization():
    """Verify safety sanitization cleans control codes and respects max limits."""
    raw = "Hello\x00\x07 World!\t\n"
    clean = sanitize_prompt(raw)
    assert clean == "Hello World!\t\n" or "Hello World!" in clean
    assert "\x00" not in clean

    with pytest.raises(ValueError):
        sanitize_prompt("   \x00\x07  ")

    long_prompt = "A" * 10000
    bounded = sanitize_prompt(long_prompt, max_chars=100)
    assert len(bounded) == 100


def test_parameter_validation():
    """Verify inference hyperparameter clamping within safe bounds."""
    temp, max_tok, ctx = validate_inference_parameters(temperature=5.0, max_tokens=10000, context_length=100)
    assert temp == 2.0
    assert max_tok == 2048
    assert ctx == 512

    temp_neg, max_neg, ctx_neg = validate_inference_parameters(temperature=-1.0, max_tokens=0, context_length=50000)
    assert temp_neg == 0.0
    assert max_neg == 1
    assert ctx_neg == 8192


def test_mock_backend_generation():
    """Verify MockLLMBackend generates responses with all metrics populated."""
    backend = MockLLMBackend()
    assert backend.is_available() is True
    assert "mock-qwen-4b" in backend.list_models()

    req = InferenceRequest(
        prompt="Tell me about CUDA.",
        model_name="mock-qwen-4b",
        device="cuda",
        max_tokens=32,
    )
    resp = backend.generate(req)

    assert isinstance(resp, InferenceResponse)
    assert len(resp.text) > 0
    assert resp.device == "cuda"
    assert resp.metrics.ttft_ms > 0.0
    assert resp.metrics.tokens_per_second > 0.0
    assert resp.metrics.backend_vram_mb == 512.0
    assert resp.metrics.pytorch_vram_mb == 0.0


def test_llm_engine_with_mock(tmp_path):
    """Verify LLMEngine orchestration using mock backend."""
    mock = MockLLMBackend()
    engine = LLMEngine(backend=mock, results_dir=str(tmp_path))

    resp = engine.generate(
        prompt="Explain GPU acceleration.",
        model_name="mock-qwen-4b",
        device="cuda",
        max_tokens=16,
    )

    assert resp is not None
    assert resp.metrics.output_tokens > 0

    # Verify tracker created artifacts
    assert os.path.exists(tmp_path / "inference_experiments.csv")
    assert os.path.exists(tmp_path / "inference_experiments.json")


def test_ollama_backend_detection():
    """Verify OllamaBackend detects running local Ollama service and models."""
    ollama = OllamaBackend()
    # Check if local Ollama daemon is reachable
    if ollama.is_available():
        models = ollama.list_models()
        assert len(models) >= 1
        # At least one local model must be registered (e.g. qwen3.5:4b)
        assert any("qwen" in m for m in models)
        details = ollama.get_model_details(models[0])
        assert "quantization_level" in details or "format" in details
    else:
        pytest.skip("Local Ollama runtime not running on standard port 11434")


def test_vram_distinction():
    """Verify memory metrics distinguish backend VRAM from PyTorch allocator VRAM."""
    m = InferenceMetrics(
        backend_vram_mb=3130.0,
        pytorch_vram_mb=25.0,
        global_vram_used_mb=3700.0,
    )
    assert m.backend_vram_mb != m.pytorch_vram_mb
    assert m.backend_vram_mb > 3000.0
    assert m.pytorch_vram_mb < 100.0


def test_llm_tracker(tmp_path):
    """Verify LLMTracker persists experiment records to JSON and CSV."""
    tracker = LLMTracker(results_dir=str(tmp_path))
    metrics = InferenceMetrics(
        ttft_ms=120.5,
        generation_duration_ms=450.0,
        total_duration_ms=580.0,
        prompt_tokens=15,
        output_tokens=30,
        tokens_per_second=66.6,
        backend_vram_mb=3100.0,
        pytorch_vram_mb=0.0,
        global_vram_used_mb=3500.0,
    )
    resp = InferenceResponse(
        text="Test inference output",
        model_name="qwen3.5:4b",
        device="cuda:0",
        device_name="NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        metrics=metrics,
        metadata={"quantization_level": "Q4_K_M", "parameter_size": "4.2B"},
    )

    rec = tracker.log_inference(resp, experiment_id="test_exp_01")
    assert rec["tokens_per_second"] == 66.6
    assert os.path.exists(tracker.json_file)
    assert os.path.exists(tracker.csv_file)
