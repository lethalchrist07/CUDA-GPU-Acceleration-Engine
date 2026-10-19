import time
from typing import List, Dict, Any
from .base_backend import BaseLLMBackend
from .types import InferenceRequest, InferenceResponse, InferenceMetrics
from .safety import sanitize_prompt, validate_inference_parameters


class MockLLMBackend(BaseLLMBackend):
    """
    Lightweight deterministic mock backend for automated CI/PyTest suites.
    Simulates realistic generation and latency metrics without VRAM consumption.
    """

    def __init__(self, model_name: str = "mock-qwen-4b") -> None:
        self.default_model = model_name

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["mock-qwen-4b", "mock-coder-7b"]

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        return {
            "model_name": model_name,
            "format": "gguf",
            "family": "mock",
            "parameter_size": "4.0B",
            "quantization_level": "Q4_K_M",
            "context_length": 2048,
        }

    def get_backend_vram_usage(self) -> float:
        return 512.0  # Simulated 512 MB VRAM

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        sanitized_prompt = sanitize_prompt(request.prompt)
        temp, max_tok, ctx_len = validate_inference_parameters(
            request.temperature, request.max_tokens, request.context_length
        )

        simulated_text = f"Mock response for prompt: '{sanitized_prompt[:30]}...' on {request.device}"
        out_tokens = min(max_tok, 16)
        prompt_tokens = len(sanitized_prompt.split())

        metrics = InferenceMetrics(
            ttft_ms=15.5,
            generation_duration_ms=45.0,
            total_duration_ms=62.0,
            prompt_tokens=prompt_tokens,
            output_tokens=out_tokens,
            tokens_per_second=35.5,
            backend_vram_mb=self.get_backend_vram_usage(),
            pytorch_vram_mb=0.0,
            global_vram_used_mb=1200.0,
        )

        return InferenceResponse(
            text=simulated_text,
            model_name=request.model_name or self.default_model,
            device=request.device,
            device_name="Mock Acceleration Engine",
            metrics=metrics,
            metadata=self.get_model_details(request.model_name),
        )
