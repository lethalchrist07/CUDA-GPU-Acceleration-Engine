import time
import logging
from typing import List, Dict, Any, Optional
import requests
import torch

from .base_backend import BaseLLMBackend
from .types import InferenceRequest, InferenceResponse, InferenceMetrics
from .safety import sanitize_prompt, validate_inference_parameters

logger = logging.getLogger(__name__)


class OllamaBackend(BaseLLMBackend):
    """
    Local LLM backend communicating with the native Ollama / llama.cpp runtime.
    Supports physical GPU acceleration (CUDA) and CPU-only inference modes.
    """

    def __init__(self, host: str = "http://127.0.0.1:11434") -> None:
        self.host = host.rstrip("/")

    def is_available(self) -> bool:
        """Check if local Ollama service is reachable."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Query all locally installed models."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to query Ollama models: {e}")
        return []

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Retrieve model metadata including quantization, parameter count, and format."""
        try:
            resp = requests.post(
                f"{self.host}/api/show",
                json={"name": model_name},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                details = data.get("details", {})
                return {
                    "model_name": model_name,
                    "format": details.get("format", "gguf"),
                    "family": details.get("family", "unknown"),
                    "parameter_size": details.get("parameter_size", "unknown"),
                    "quantization_level": details.get("quantization_level", "unknown"),
                    "context_length": details.get("context_length", 2048),
                }
        except Exception as e:
            logger.warning(f"Could not fetch details for {model_name}: {e}")
        return {
            "model_name": model_name,
            "format": "gguf",
            "family": "unknown",
            "parameter_size": "unknown",
            "quantization_level": "unknown",
            "context_length": 2048,
        }

    def get_backend_vram_usage(self) -> float:
        """Query VRAM dynamically held by active models in the Ollama process."""
        try:
            resp = requests.get(f"{self.host}/api/ps", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                total_bytes = sum(m.get("size_vram", 0) for m in models)
                return round(total_bytes / (1024.0 * 1024.0), 2)
        except Exception:
            pass
        return 0.0

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Execute prompt generation on the specified device (CPU or CUDA).
        """
        sanitized_prompt = sanitize_prompt(request.prompt)
        temp, max_tok, ctx_len = validate_inference_parameters(
            request.temperature, request.max_tokens, request.context_length
        )

        use_cuda = request.device.lower().startswith("cuda") and torch.cuda.is_available()
        # num_gpu: -1 offloads all model layers to CUDA GPU; 0 forces CPU execution
        num_gpu = -1 if use_cuda else 0
        device_label = "cuda:0" if use_cuda else "cpu"
        device_name = torch.cuda.get_device_name(0) if use_cuda else "Intel Core i5-13450HX (CPU)"

        payload = {
            "model": request.model_name,
            "prompt": sanitized_prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
                "num_ctx": ctx_len,
                "num_gpu": num_gpu,
            },
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        t_start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=request.timeout_seconds,
            )
            t_end = time.perf_counter()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Inference request timed out after {request.timeout_seconds} seconds")
        except Exception as e:
            raise RuntimeError(f"Ollama inference failed: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama server error (HTTP {resp.status_code}): {resp.text}")

        res_data = resp.json()

        # Handle both direct response text and reasoning/thinking tokens
        resp_text = res_data.get("response", "").strip()
        thinking_text = res_data.get("thinking", "").strip()

        if resp_text and thinking_text:
            generated_text = f"[Thinking Process]\n{thinking_text}\n\n[Answer]\n{resp_text}"
        elif resp_text:
            generated_text = resp_text
        elif thinking_text:
            generated_text = thinking_text
        else:
            generated_text = "(Model produced empty response token stream)"

        # Timing extraction
        # Duration metrics are reported by llama.cpp/Ollama in nanoseconds
        prompt_eval_ns = res_data.get("prompt_eval_duration", 0)
        eval_ns = res_data.get("eval_duration", 0)
        total_ns = res_data.get("total_duration", 0)

        prompt_tokens = res_data.get("prompt_eval_count", 0)
        output_tokens = res_data.get("eval_count", 0)

        ttft_ms = prompt_eval_ns / 1e6
        gen_duration_ms = eval_ns / 1e6
        total_ms = (t_end - t_start) * 1000.0

        throughput_tok_per_sec = (output_tokens / (eval_ns / 1e9)) if eval_ns > 0 else 0.0

        # Memory disaggregation (Requirement 11)
        backend_vram = self.get_backend_vram_usage()
        pytorch_vram = 0.0
        global_vram = 0.0

        if torch.cuda.is_available():
            pytorch_vram = round(torch.cuda.memory_allocated(0) / (1024.0 * 1024.0), 2)
            free_b, total_b = torch.cuda.mem_get_info(0)
            global_vram = round((total_b - free_b) / (1024.0 * 1024.0), 2)

        # If backend VRAM from /api/ps was already unmapped or reported 0 for GPU mode,
        # calculate active VRAM delta from global VRAM
        if use_cuda and backend_vram == 0.0 and global_vram > 1000.0:
            backend_vram = global_vram

        metrics = InferenceMetrics(
            ttft_ms=round(ttft_ms, 2),
            generation_duration_ms=round(gen_duration_ms, 2),
            total_duration_ms=round(total_ms, 2),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            tokens_per_second=round(throughput_tok_per_sec, 2),
            backend_vram_mb=backend_vram,
            pytorch_vram_mb=pytorch_vram,
            global_vram_used_mb=global_vram,
        )

        model_meta = self.get_model_details(request.model_name)

        return InferenceResponse(
            text=generated_text,
            model_name=request.model_name,
            device=device_label,
            device_name=device_name,
            metrics=metrics,
            metadata=model_meta,
        )
