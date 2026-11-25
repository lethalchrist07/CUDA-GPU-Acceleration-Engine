"""
backend/config.py
Centralized Configuration and Security Settings for CUDA GPU Acceleration Engine.
Manages environment variables, CORS policies, logging infrastructure, and hardware limits.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("cuda_engine")


@dataclass
class EngineSettings:
    """Production configuration settings with environment variable overrides."""

    host: str = field(
        default_factory=lambda: os.getenv("CUDA_ENGINE_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("CUDA_ENGINE_PORT", "8000"))
    )
    cors_origins: List[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "CUDA_ENGINE_CORS_ORIGINS",
                "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173,http://127.0.0.1:8000,http://localhost:8000",
            ).split(",")
            if origin.strip()
        ]
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("CUDA_ENGINE_LOG_LEVEL", "INFO").upper()
    )
    environment: str = field(
        default_factory=lambda: os.getenv("CUDA_ENGINE_ENV", "production").lower()
    )
    device: str = field(
        default_factory=lambda: os.getenv("CUDA_ENGINE_DEVICE", "auto").lower()
    )
    max_vram_reserve_mb: float = field(
        default_factory=lambda: float(os.getenv("CUDA_ENGINE_MAX_VRAM_RESERVE_MB", "2000.0"))
    )
    thermal_throttle_threshold_c: float = field(
        default_factory=lambda: float(os.getenv("CUDA_ENGINE_THERMAL_LIMIT_C", "82.0"))
    )
    demo_mode: bool = field(
        default_factory=lambda: os.getenv("CUDA_ENGINE_DEMO_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
    )
    demo_max_request_bytes: int = 32768
    demo_min_vram_reserve_mb: float = 1024.0
    demo_max_temperature_c: float = 75.0
    demo_max_prompt_chars: int = 256
    demo_max_gen_tokens: int = 48
    demo_max_tensor_floats: int = 128


_SETTINGS: Optional[EngineSettings] = None


def get_settings() -> EngineSettings:
    """Retrieve or initialize singleton application settings."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = EngineSettings()
    return _SETTINGS


def reload_settings() -> EngineSettings:
    """Reinitialize application settings from environment (used for testing and lifecycle reloads)."""
    global _SETTINGS
    _SETTINGS = EngineSettings()
    return _SETTINGS


def setup_logging(level: Optional[str] = None) -> None:
    """Configure structured logging across backend components."""
    settings = get_settings()
    log_lvl = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(
        level=log_lvl,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Silence excessively verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)


def log_hardware_diagnostic_banner() -> None:
    """
    Log an explicit diagnostic banner clearly distinguishing actual CUDA GPU execution
    from CPU fallback or emulation mode.
    """
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            vram_mb = props.total_memory / (1024 ** 2)
            logger.info("======================================================================")
            logger.info("  CUDA GPU ACCELERATION ENGINE — HARDWARE INITIALIZED")
            logger.info(f"  Physical Device: {dev_name}")
            logger.info(f"  Compute Capability: {cc[0]}.{cc[1]} (Ampere SM 8.6)")
            logger.info(f"  Total Dedicated VRAM: {vram_mb:.2f} MB")
            logger.info(f"  SM Count: {props.multi_processor_count} multiprocessors")
            logger.info("  Execution Mode: BARE-METAL CUDA HARDWARE ACCELERATION (ACTIVE)")
            logger.info("======================================================================")
        else:
            logger.warning("======================================================================")
            logger.warning("  [WARNING] CPU FALLBACK / EMULATION MODE ACTIVE")
            logger.warning("  PyTorch did not detect an accessible CUDA-enabled GPU.")
            logger.warning("  Workloads will execute on the host CPU with reduced throughput.")
            logger.warning("======================================================================")

        settings = get_settings()
        if settings.demo_mode:
            logger.info("======================================================================")
            logger.info("  [SECURITY] PUBLIC DEMO MODE ACTIVE (CUDA_ENGINE_DEMO_MODE=1)")
            logger.info("  Mutating training & laboratory endpoints: BLOCKED (403)")
            logger.info("  Public model whitelist active: resnet18_cifar10, transformer_50m, baseline_mlp")
            logger.info("  Request limit: 32 KB | Max Gen Tokens: 48 | Max Prompt: 256 chars")
            logger.info("  Hardware limits: >=1024 MB free VRAM reserve, <=75.0 C GPU temperature")
            logger.info("======================================================================")
    except Exception as e:
        logger.error(f"Error querying hardware during startup: {e}")

