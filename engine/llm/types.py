from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class InferenceMetrics:
    """Detailed performance and latency metrics for an inference request."""
    ttft_ms: float = 0.0                    # Time to first token (prompt evaluation latency)
    generation_duration_ms: float = 0.0     # Active token generation duration
    total_duration_ms: float = 0.0          # Total request duration (including loading/network)
    prompt_tokens: int = 0                  # Number of prompt tokens processed
    output_tokens: int = 0                  # Number of output tokens generated
    tokens_per_second: float = 0.0          # Generation throughput (tokens/sec)
    backend_vram_mb: float = 0.0            # VRAM allocated by LLM backend process (e.g. llama-server)
    pytorch_vram_mb: float = 0.0            # VRAM tracked by PyTorch process (torch.cuda.memory_allocated)
    global_vram_used_mb: float = 0.0        # Total VRAM currently utilized across the GPU hardware


@dataclass
class InferenceRequest:
    """Configuration and parameters for an LLM generation request."""
    prompt: str
    model_name: str = "qwen3.5:4b"
    temperature: float = 0.7
    max_tokens: int = 128
    context_length: int = 2048
    device: str = "cuda"                    # "cuda" or "cpu"
    system_prompt: Optional[str] = None
    timeout_seconds: int = 60


@dataclass
class InferenceResponse:
    """Result of an LLM generation request including text and metrics."""
    text: str
    model_name: str
    device: str
    device_name: str
    metrics: InferenceMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
