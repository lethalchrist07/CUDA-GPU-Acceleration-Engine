from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .types import InferenceRequest, InferenceResponse


class BaseLLMBackend(ABC):
    """Abstract interface defining the contract for local LLM inference engines."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend runtime is reachable and operational."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return the list of available local models."""
        pass

    @abstractmethod
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Retrieve architecture, parameter count, quantization, and format metadata."""
        pass

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference generation on the requested device (CPU or GPU)."""
        pass

    @abstractmethod
    def get_backend_vram_usage(self) -> float:
        """Query VRAM actively held by this inference runtime in MB."""
        pass
