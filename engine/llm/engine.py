import logging
from typing import List, Dict, Any, Optional
import torch

from .base_backend import BaseLLMBackend
from .ollama_backend import OllamaBackend
from .mock_backend import MockLLMBackend
from .types import InferenceRequest, InferenceResponse
from .tracker import LLMTracker

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Unified LLM inference engine orchestrating backend selection,
    hardware device routing (CPU vs GPU), resource monitoring, and experiment logging.
    """

    def __init__(
        self,
        backend: Optional[BaseLLMBackend] = None,
        results_dir: str = "results",
    ) -> None:
        self.tracker = LLMTracker(results_dir=results_dir)

        if backend is not None:
            self.backend = backend
        else:
            # Auto-detect available local runtime
            ollama = OllamaBackend()
            if ollama.is_available():
                self.backend = ollama
                logger.info("LLMEngine: Initialized with native local Ollama / llama.cpp runtime.")
            else:
                self.backend = MockLLMBackend()
                logger.warning("LLMEngine: Local Ollama runtime not reachable. Using Mock backend for tests.")

    def list_available_models(self) -> List[str]:
        """List all models currently installed in the active backend."""
        return self.backend.list_models()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Retrieve model metadata."""
        return self.backend.get_model_details(model_name)

    def generate(
        self,
        prompt: str,
        model_name: str = "qwen3.5:4b",
        device: str = "cuda",
        max_tokens: int = 128,
        temperature: float = 0.7,
        context_length: int = 2048,
        system_prompt: Optional[str] = None,
        timeout_seconds: int = 60,
        log_experiment: bool = True,
        experiment_id: str = "",
    ) -> InferenceResponse:
        """
        Execute an inference request on the chosen device with full metrics capture.
        """
        req = InferenceRequest(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            context_length=context_length,
            device=device,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds,
        )

        response = self.backend.generate(req)

        if log_experiment:
            self.tracker.log_inference(response, experiment_id=experiment_id)

        return response

    def compare_cpu_vs_gpu(
        self,
        prompt: str,
        model_name: str = "qwen3.5:4b",
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Dict[str, InferenceResponse]:
        """
        Run identical prompt and model on both CPU and GPU to compare TTFT, throughput, and VRAM.
        """
        logger.info(f"Comparing inference performance for {model_name} on CPU vs GPU...")

        # 1. CPU Run
        cpu_resp = self.generate(
            prompt=prompt,
            model_name=model_name,
            device="cpu",
            max_tokens=max_tokens,
            temperature=temperature,
            log_experiment=True,
            experiment_id=f"infer_cpu_{model_name.replace(':', '_')}",
        )

        # 2. GPU Run
        gpu_resp = self.generate(
            prompt=prompt,
            model_name=model_name,
            device="cuda",
            max_tokens=max_tokens,
            temperature=temperature,
            log_experiment=True,
            experiment_id=f"infer_gpu_{model_name.replace(':', '_')}",
        )

        return {"cpu": cpu_resp, "gpu": gpu_resp}
