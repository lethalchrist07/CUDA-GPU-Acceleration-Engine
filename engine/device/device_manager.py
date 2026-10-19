import logging
from typing import Dict, Tuple, Optional
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages CPU and CUDA hardware devices, VRAM allocations, and safety guards.
    """

    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is functional in the active PyTorch environment."""
        return torch.cuda.is_available()

    @staticmethod
    def get_device_count() -> int:
        """Return the number of accessible CUDA GPUs."""
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        """Return the brand/model name of the specified GPU."""
        if not torch.cuda.is_available():
            return "CPU (CUDA Unavailable)"
        return torch.cuda.get_device_name(device_id)

    @staticmethod
    def get_compute_capability(device_id: int = 0) -> Tuple[int, int]:
        """Return (major, minor) compute capability of the GPU."""
        if not torch.cuda.is_available():
            return (0, 0)
        return torch.cuda.get_device_capability(device_id)

    @staticmethod
    def get_memory_info(device_id: int = 0) -> Dict[str, float]:
        """
        Return memory metrics in megabytes (MB) for the specified GPU.
        """
        if not torch.cuda.is_available():
            return {
                "total_mb": 0.0,
                "free_mb": 0.0,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
            }

        total_bytes = torch.cuda.get_device_properties(device_id).total_memory
        allocated_bytes = torch.cuda.memory_allocated(device_id)
        reserved_bytes = torch.cuda.memory_reserved(device_id)
        free_bytes = total_bytes - reserved_bytes

        return {
            "total_mb": round(total_bytes / (1024.0 * 1024.0), 2),
            "free_mb": round(free_bytes / (1024.0 * 1024.0), 2),
            "allocated_mb": round(allocated_bytes / (1024.0 * 1024.0), 2),
            "reserved_mb": round(reserved_bytes / (1024.0 * 1024.0), 2),
        }

    @staticmethod
    def check_memory_headroom(required_mb: float, device_id: int = 0, safety_margin_mb: float = 512.0) -> bool:
        """
        Guard against out-of-memory errors by checking free VRAM against requirements.
        """
        if not torch.cuda.is_available():
            return True  # Host RAM handles CPU allocation

        mem_info = DeviceManager.get_memory_info(device_id)
        available_mb = mem_info["free_mb"]
        needed_mb = required_mb + safety_margin_mb
        if needed_mb > available_mb:
            logger.warning(
                f"Memory headroom insufficient: need {needed_mb:.1f} MB (including {safety_margin_mb:.1f} MB margin), "
                f"but only {available_mb:.1f} MB available on GPU {device_id}."
            )
            return False
        return True

    @staticmethod
    def empty_cache() -> None:
        """Release cached GPU memory blocks back to the OS."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_device(preferred: str = "cuda", device_id: int = 0, fallback_to_cpu: bool = True) -> torch.device:
        """
        Resolve and return the appropriate torch.device.
        Falls back safely to CPU if CUDA is requested but unavailable.
        """
        if preferred.lower().startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(f"cuda:{device_id}")
            elif fallback_to_cpu:
                logger.warning("CUDA was requested but is not available. Falling back to CPU.")
                return torch.device("cpu")
            else:
                raise RuntimeError("CUDA was explicitly requested, but no functional CUDA device was found.")
        return torch.device("cpu")


def get_device(preferred: str = "cuda", device_id: int = 0, fallback_to_cpu: bool = True) -> torch.device:
    """Convenience helper to obtain a torch.device."""
    return DeviceManager.get_device(preferred, device_id, fallback_to_cpu)
