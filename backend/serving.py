"""
Model Serving and Live Inference Engine for FastAPI Backend (Phase 9).
Manages loaded neural network models in GPU/CPU memory, handles atomic loading/unloading,
runs fast inference with confidence scoring, and integrates with CIFAR-10 test set.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Global serving state - Vision Models (ResNet18, CIFAR10MLP, BaselineMLP)
_VISION_MODEL: Optional[Any] = None
_VISION_MODEL_NAME: Optional[str] = None
_VISION_DEVICE: str = "cpu"
_VISION_CHECKPOINT: Optional[str] = None

# Global serving state - Transformer Models (NEXORA GPT-86.8M / 100M)
_TRANSFORMER_MODEL: Optional[Any] = None
_TRANSFORMER_MODEL_NAME: Optional[str] = None
_TRANSFORMER_DEVICE: str = "cpu"
_TRANSFORMER_CHECKPOINT: Optional[str] = None
_TRANSFORMER_TOKENIZER: Optional[Any] = None

# Legacy active model pointers (maintained in sync for 100% backwards compatibility)
_ACTIVE_MODEL: Optional[Any] = None
_ACTIVE_MODEL_NAME: Optional[str] = None
_ACTIVE_DEVICE: str = "cpu"
_ACTIVE_CHECKPOINT: Optional[str] = None
_ACTIVE_TOKENIZER: Optional[Any] = None

# Reentrant lock for thread-safe model serving and atomic load/unload
import shutil
import subprocess
import threading
from fastapi import HTTPException, status
from backend.config import get_settings

_SERVING_LOCK = threading.RLock()
# Shared non-blocking atomic concurrency lock for public demo mode inference
_INFERENCE_LOCK = threading.Lock()

PUBLIC_DEMO_MODELS = {
    "resnet18_cifar10": {
        "model_name": "ResNet18",
        "relative_path": "checkpoints/resnet18_cifar10/best_model.pt",
        "task": "Image Classification",
    },
    "transformer_50m": {
        "model_name": "NEXORA GPT-86.8M",
        "relative_path": "checkpoints/transformer/pretraining_50m_campaign/best_model.pt",
        "task": "Causal Language Modeling",
    },
    "baseline_mlp": {
        "model_name": "BaselineMLP",
        "relative_path": "checkpoints/test_checkpoint.pt",
        "task": "Synthetic Classification",
    },
}


def _sync_legacy_pointers(model_type: str = "vision") -> None:
    """Synchronize legacy global pointers with the most recently active model."""
    global _ACTIVE_MODEL, _ACTIVE_MODEL_NAME, _ACTIVE_DEVICE, _ACTIVE_CHECKPOINT, _ACTIVE_TOKENIZER
    if model_type == "vision":
        _ACTIVE_MODEL = _VISION_MODEL
        _ACTIVE_MODEL_NAME = _VISION_MODEL_NAME
        _ACTIVE_DEVICE = _VISION_DEVICE
        _ACTIVE_CHECKPOINT = _VISION_CHECKPOINT
        _ACTIVE_TOKENIZER = None
    elif model_type == "transformer":
        _ACTIVE_MODEL = _TRANSFORMER_MODEL
        _ACTIVE_MODEL_NAME = _TRANSFORMER_MODEL_NAME
        _ACTIVE_DEVICE = _TRANSFORMER_DEVICE
        _ACTIVE_CHECKPOINT = _TRANSFORMER_CHECKPOINT
        _ACTIVE_TOKENIZER = _TRANSFORMER_TOKENIZER


def get_torch():
    """Import torch safely."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def get_class_names() -> List[str]:
    """Return CIFAR-10 category names."""
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]


class ModelServer:
    """
    Manages in-memory models for real-time inference serving with thread-safe isolation.
    """

    @classmethod
    def list_available_models(cls) -> List[Dict[str, Any]]:
        """List registered model architectures and available checkpoints."""
        torch = get_torch()
        models = [
            {
                "model_id": "resnet18_cifar10",
                "name": "ResNet18",
                "task": "Image Classification",
                "dataset": "CIFAR-10 (32x32 RGB)",
                "classes": 10,
                "parameters": 11173962,
                "status": "loaded" if _VISION_MODEL_NAME == "ResNet18" or _ACTIVE_MODEL_NAME == "ResNet18" else "available",
                "device": (_VISION_DEVICE if _VISION_MODEL_NAME == "ResNet18" else _ACTIVE_DEVICE) if (_VISION_MODEL_NAME == "ResNet18" or _ACTIVE_MODEL_NAME == "ResNet18") else None,
                "active_checkpoint": (_VISION_CHECKPOINT if _VISION_MODEL_NAME == "ResNet18" else _ACTIVE_CHECKPOINT) if (_VISION_MODEL_NAME == "ResNet18" or _ACTIVE_MODEL_NAME == "ResNet18") else None,
                "checkpoints": cls._find_checkpoints("checkpoints/resnet18_cifar10"),
            },
            {
                "model_id": "transformer_50m",
                "name": "NEXORA GPT-86.8M",
                "task": "Causal Language Modeling",
                "dataset": "MultiDocumentCorpus (50M Campaign)",
                "classes": 2048,
                "parameters": 86825472,
                "parameters_pilot": 86825472,
                "layers": 12,
                "hidden_dim": 768,
                "heads": 12,
                "context_length": 256,
                "vocab_size": 2048,
                "tokenizer": "BPE (Byte-Pair Encoding)",
                "tokenizer_display": "BPE (Byte-Pair Encoding, 2048 Vocab)",
                "tokenizer_type": "BPE (Byte-Pair Encoding)",
                "supported_backends": ["CPU_NATIVE", "GPU_NATIVE", "CUSTOM_CUDA_FUSED", "ADAPTIVE_ENGINE"],
                "training_capability": True,
                "inference_capability": True,
                "training_status": "Production Pretrained (50M Campaign Certified — 12,208 Steps, Val Loss 1.9383)",
                "status": "loaded" if _TRANSFORMER_MODEL_NAME in ("NEXORA GPT-86.8M", "NEXORA GPT-100M") or "transformer" in str(_ACTIVE_MODEL_NAME).lower() else "available",
                "device": (_TRANSFORMER_DEVICE if _TRANSFORMER_MODEL is not None else _ACTIVE_DEVICE) if (_TRANSFORMER_MODEL is not None or "transformer" in str(_ACTIVE_MODEL_NAME).lower()) else None,
                "active_checkpoint": (_TRANSFORMER_CHECKPOINT if _TRANSFORMER_MODEL is not None else _ACTIVE_CHECKPOINT) if (_TRANSFORMER_MODEL is not None or "transformer" in str(_ACTIVE_MODEL_NAME).lower()) else None,
                "checkpoints": cls._find_checkpoints("checkpoints/transformer"),
            },
            {
                "model_id": "transformer_100m",
                "name": "NEXORA GPT-100M",
                "task": "Causal Language Modeling",
                "dataset": "MultiDocumentCorpus (100M Flagship Pilot)",
                "classes": 2048,
                "parameters": 86825472,
                "parameters_pilot": 85547520,
                "layers": 12,
                "hidden_dim": 768,
                "heads": 12,
                "context_length": 256,
                "vocab_size": 2048,
                "tokenizer": "BPE (Byte-Pair Encoding)",
                "tokenizer_type": "BPE (Byte-Pair Encoding)",
                "supported_backends": ["CPU_NATIVE", "GPU_NATIVE"],
                "training_capability": True,
                "inference_capability": True,
                "training_status": "Pilot Trained",
                "status": "loaded" if _TRANSFORMER_MODEL_NAME in ("NEXORA GPT-86.8M", "NEXORA GPT-100M") or "transformer" in str(_ACTIVE_MODEL_NAME).lower() else "available",
                "device": (_TRANSFORMER_DEVICE if _TRANSFORMER_MODEL is not None else _ACTIVE_DEVICE) if (_TRANSFORMER_MODEL is not None or "transformer" in str(_ACTIVE_MODEL_NAME).lower()) else None,
                "active_checkpoint": (_TRANSFORMER_CHECKPOINT if _TRANSFORMER_MODEL is not None else _ACTIVE_CHECKPOINT) if (_TRANSFORMER_MODEL is not None or "transformer" in str(_ACTIVE_MODEL_NAME).lower()) else None,
                "checkpoints": cls._find_checkpoints("checkpoints/transformer/flagship_100m_pilot"),
            },
            {
                "model_id": "cifar10_mlp",
                "name": "CIFAR10MLP",
                "task": "Image Classification",
                "dataset": "CIFAR-10",
                "classes": 10,
                "parameters": 3426250,
                "status": "loaded" if _VISION_MODEL_NAME == "CIFAR10MLP" else "available",
                "device": _VISION_DEVICE if _VISION_MODEL_NAME == "CIFAR10MLP" else None,
                "active_checkpoint": _VISION_CHECKPOINT if _VISION_MODEL_NAME == "CIFAR10MLP" else None,
                "checkpoints": cls._find_checkpoints("checkpoints/training_jobs"),
            },
            {
                "model_id": "baseline_mlp",
                "name": "BaselineMLP",
                "task": "Synthetic Classification",
                "dataset": "Synthetic [128-dim]",
                "classes": 20,
                "parameters": 17226,
                "status": "loaded" if _VISION_MODEL_NAME == "BaselineMLP" else "available",
                "device": _VISION_DEVICE if _VISION_MODEL_NAME == "BaselineMLP" else None,
                "active_checkpoint": _VISION_CHECKPOINT if _VISION_MODEL_NAME == "BaselineMLP" else None,
                "checkpoints": cls._find_checkpoints("checkpoints"),
            },
        ]
        settings = get_settings()
        if settings.demo_mode:
            # Permit only approved public demo models and remove pilot metadata
            models = [m for m in models if m.get("model_id") in ("resnet18_cifar10", "transformer_50m", "baseline_mlp")]
            for m in models:
                ckpts = m.get("checkpoints", [])
                m["checkpoints"] = [
                    c for c in ckpts
                    if "pilot" not in c.get("relative_path", "").lower()
                    and "flagship" not in c.get("relative_path", "").lower()
                    and "training_jobs" not in c.get("relative_path", "").lower()
                ]
        return models

    @classmethod
    def _find_checkpoints(cls, sub_dir: str) -> List[Dict[str, Any]]:
        dir_path = PROJECT_ROOT / sub_dir
        if not dir_path.is_dir():
            return []
        items = []
        for p in sorted(dir_path.rglob("*.pt")):
            items.append({
                "filename": p.name,
                "relative_path": p.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": p.stat().st_size,
            })
        return items

    @classmethod
    def _get_gpu_temperature(cls) -> Optional[float]:
        """Query GPU core temperature in Celsius via nvidia-smi."""
        try:
            cmd = shutil.which("nvidia-smi")
            if not cmd:
                return None
            out = subprocess.run(
                [cmd, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            val = out.stdout.strip().split("\n")[0].strip()
            return float(val)
        except Exception as e:
            logger.warning(f"Failed to query GPU temperature: {e}")
            return None

    @classmethod
    def _enforce_demo_hardware_safeguards(cls, device: Any) -> None:
        """
        Enforce 1,024 MB free VRAM reserve and 75°C temperature threshold in Demo Mode.
        Fails closed with HTTP 503 if GPU temperature cannot be verified on GPU paths.
        CPU-only operation safely bypasses GPU-specific checks.
        """
        settings = get_settings()
        if not settings.demo_mode:
            return

        torch = get_torch()
        if torch is None:
            return

        dev_type = getattr(device, "type", str(device).split(":")[0].lower())
        if dev_type != "cuda":
            # CPU-only operation safely bypasses GPU-specific checks
            return

        if not torch.cuda.is_available():
            return

        # 1. Enforce minimum 1,024 MB free VRAM reserve
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            free_mb = free_bytes / (1024 * 1024)
            if free_mb < settings.demo_min_vram_reserve_mb:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Insufficient GPU VRAM available for safe demo execution. Minimum {settings.demo_min_vram_reserve_mb:.0f} MB free VRAM required, but only {free_mb:.1f} MB is free.",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Could not query CUDA mem_get_info: {e}")

        # 2. Enforce 75°C GPU thermal limit (fail closed if temperature cannot be verified)
        temp_c = cls._get_gpu_temperature()
        if temp_c is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GPU thermal telemetry unavailable; cannot verify safe operating temperature in demo mode.",
            )
        if temp_c >= settings.demo_max_temperature_c:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"GPU thermal safety threshold exceeded ({temp_c:.1f}°C >= {settings.demo_max_temperature_c:.1f}°C). Inference paused for cooldown.",
            )

    @classmethod
    def load_model(
        cls,
        model_name: str = "ResNet18",
        checkpoint_path: Optional[str] = None,
        device_str: Optional[str] = None,
        device: Optional[str] = None,
        is_owner: bool = False,
    ) -> Dict[str, Any]:
        """
        Load weights into memory and assign to requested device with isolated model slots.
        In demo mode, strictly resticts public callers to approved public models, while
        allowing verified owner to load any valid repository checkpoint.
        """
        global _VISION_MODEL, _VISION_MODEL_NAME, _VISION_DEVICE, _VISION_CHECKPOINT
        global _TRANSFORMER_MODEL, _TRANSFORMER_MODEL_NAME, _TRANSFORMER_DEVICE, _TRANSFORMER_CHECKPOINT, _TRANSFORMER_TOKENIZER

        torch = get_torch()
        if torch is None:
            raise RuntimeError("PyTorch is not available in the active environment.")

        target_dev_str = device or device_str

        # Resolve device
        if target_dev_str is None or target_dev_str == "auto":
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif target_dev_str.startswith("cuda"):
            dev = torch.device(target_dev_str if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device("cpu")

        # GPU Contention Guard: Prevent inference collision with active training job
        if dev.type == "cuda":
            from engine.training.orchestrator import TrainingOrchestrator
            orch = TrainingOrchestrator.get_instance()
            if orch.active_gpu_job_id is not None:
                raise RuntimeError(
                    f"GPU is actively reserved by training job '{orch.active_gpu_job_id}'. "
                    "To prevent VRAM contention on the 6GB RTX 3050, run inference on CPU or wait for training to complete."
                )

        with _SERVING_LOCK:
            settings = get_settings()
            if settings.demo_mode and not is_owner:
                # 1. Reject arbitrary checkpoint paths
                if checkpoint_path is not None and str(checkpoint_path).strip() != "":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Arbitrary checkpoint paths are forbidden in demo mode. Only approved public model IDs are permitted.",
                    )

                # 2. Whitelist resolution
                norm = model_name.strip().lower()
                if norm in ("resnet18_cifar10", "resnet18", "resnet-18"):
                    model_name = "ResNet18"
                    candidates = [
                        PROJECT_ROOT / "checkpoints/resnet18_cifar10/best_model.pt",
                        PROJECT_ROOT / "checkpoints/resnet18_cifar10/latest_checkpoint.pt",
                    ]
                elif norm in ("transformer_50m", "nexora gpt-86.8m", "transformer", "gpt", "nexora") and not ("100m" in norm or "pilot" in norm):
                    model_name = "NEXORA GPT-86.8M"
                    candidates = [
                        PROJECT_ROOT / "checkpoints/transformer/pretraining_50m_campaign/best_model.pt",
                        PROJECT_ROOT / "checkpoints/transformer/pretraining_50m_campaign/final_model.pt",
                    ]
                elif norm in ("baseline_mlp", "baselinemlp", "mlp"):
                    model_name = "BaselineMLP"
                    candidates = [
                        PROJECT_ROOT / "checkpoints/test_checkpoint.pt",
                        PROJECT_ROOT / "checkpoints/latest_checkpoint.pt",
                    ]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Model '{model_name}' is not in the approved public demo whitelist. Permitted models: resnet18_cifar10, transformer_50m, baseline_mlp.",
                    )

                resolved_path = None
                for c in candidates:
                    if c.is_file():
                        resolved_path = c
                        break
                if resolved_path is None:
                    raise FileNotFoundError(f"No checkpoint file found for approved demo model '{model_name}'.")

                # 3. Resolve symlinks, verify final path containment, and reject pilot/private checkpoints
                resolved_path = resolved_path.resolve()
                checkpoints_dir = (PROJECT_ROOT / "checkpoints").resolve()
                if not resolved_path.is_relative_to(checkpoints_dir):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Path traversal outside checkpoints directory detected.",
                    )
                rel_check = resolved_path.relative_to(PROJECT_ROOT).as_posix().lower()
                if "pilot" in rel_check or "flagship_100m" in rel_check:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access to unreleased pilot checkpoints is forbidden in demo mode.",
                    )

                # 4. Enforce hardware safety before loading weights on GPU
                cls._enforce_demo_hardware_safeguards(dev)
            else:
                # Standard Mode or Verified Owner Checkpoint Resolution
                if checkpoint_path is None:
                    if model_name.lower() in ("resnet18", "resnet-18"):
                        expected_loc = "checkpoints/resnet18_cifar10/best_model.pt"
                        candidates = [
                            PROJECT_ROOT / "checkpoints/resnet18_cifar10/best_model.pt",
                            PROJECT_ROOT / "checkpoints/resnet18_cifar10/latest_checkpoint.pt",
                        ]
                    elif any(t in model_name.lower() for t in ("transformer", "gpt", "nexora")):
                        expected_loc = "checkpoints/transformer/pretraining_50m_campaign/best_model.pt"
                        candidates = [
                            PROJECT_ROOT / "checkpoints/transformer/pretraining_50m_campaign/best_model.pt",
                            PROJECT_ROOT / "checkpoints/transformer/pretraining_50m_campaign/final_model.pt",
                            PROJECT_ROOT / "checkpoints/transformer/pretraining_campaign/best_model.pt",
                            PROJECT_ROOT / "checkpoints/transformer/pretraining_campaign/final_model.pt",
                            PROJECT_ROOT / "checkpoints/transformer/flagship_100m_pilot/best_model.pt",
                            PROJECT_ROOT / "checkpoints/transformer/pilot_1m/best_model.pt",
                        ]
                    else:
                        expected_loc = "checkpoints/test_checkpoint.pt"
                        candidates = [
                            PROJECT_ROOT / "checkpoints/test_checkpoint.pt",
                            PROJECT_ROOT / "checkpoints/latest_checkpoint.pt",
                        ]

                    resolved_path = None
                    for c in candidates:
                        if c.is_file():
                            resolved_path = c
                            break
                    if resolved_path is None:
                        raise FileNotFoundError(
                            f"No checkpoint file found for '{model_name}'. "
                            f"Please place the model checkpoint at '{expected_loc}' according to checkpoints/README.md "
                            f"and checkpoints/MODEL_MANIFEST.json, or train a model in AI Training Studio."
                        )
                else:
                    resolved_path = (PROJECT_ROOT / checkpoint_path).resolve()
                    # Verify path traversal containment within project root
                    if not resolved_path.is_relative_to(PROJECT_ROOT.resolve()):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Path traversal outside project boundaries detected.",
                        )
                    if not resolved_path.is_file():
                        raise FileNotFoundError(
                            f"Checkpoint not found at '{checkpoint_path}'. "
                            f"Please verify the file path or refer to checkpoints/README.md for model placement."
                        )
                # Enforce hardware safety before loading weights on GPU
                cls._enforce_demo_hardware_safeguards(dev)

            # Instantiate architecture
            is_vision = False
            tokenizer = None
            if model_name.lower() in ("resnet18", "resnet-18"):
                from engine.models.resnet import create_resnet18
                model = create_resnet18(num_classes=10, cifar_stem=True)
                active_name = "ResNet18"
                is_vision = True
            elif model_name.lower() in ("baselinemlp", "mlp"):
                from engine.models.baseline_mlp import BaselineMLP
                # Dynamic architecture resolution matching checkpoint state_dict
                ckpt_temp = torch.load(resolved_path, map_location="cpu", weights_only=True)
                temp_sd = ckpt_temp.get("model_state_dict", ckpt_temp)
                linear_keys = [k for k in sorted(temp_sd.keys()) if k.endswith(".weight")]
                if linear_keys:
                    in_dim = temp_sd[linear_keys[0]].shape[1]
                    out_dim = temp_sd[linear_keys[-1]].shape[0]
                    h_dims = [temp_sd[k].shape[0] for k in linear_keys[:-1]]
                    has_dropout = any("network.3" in k for k in linear_keys)
                    model = BaselineMLP(input_dim=in_dim, hidden_dims=h_dims, output_dim=out_dim, dropout=0.1 if has_dropout else 0.0)
                else:
                    model = BaselineMLP()
                active_name = "BaselineMLP"
                is_vision = True
            elif any(t in model_name.lower() for t in ("transformer", "gpt", "nexora")):
                from engine.transformer import GPTTransformer, TransformerConfig, get_preset_config
                from engine.transformer import BPETokenizer, ByteTokenizer

                checkpoint = torch.load(resolved_path, map_location=dev, weights_only=True)
                meta = checkpoint.get("metadata", {})
                model_cfg_dict = meta.get("model_config")

                if model_cfg_dict:
                    cfg = TransformerConfig.from_dict(model_cfg_dict)
                else:
                    cfg = get_preset_config("flagship_100m", vocab_size=512)
                    cfg.max_seq_len = 128

                model = GPTTransformer(cfg)
                active_name = "NEXORA GPT-100M" if "100m" in str(resolved_path) else "NEXORA GPT-86.8M (Transformer)"

                # Resolve tokenizer
                tok_candidates = [
                    resolved_path.parent / "bpe_vocab.json",
                    resolved_path.parent / "tokenizer" / "bpe_vocab.json",
                    PROJECT_ROOT / "checkpoints/transformer/pretraining_50m_campaign/bpe_vocab.json",
                    PROJECT_ROOT / "checkpoints/transformer/pretraining_campaign/tokenizer/bpe_vocab.json",
                    PROJECT_ROOT / "checkpoints/transformer/pretraining_campaign/bpe_vocab.json",
                    PROJECT_ROOT / "checkpoints/transformer/flagship_100m_pilot/bpe_vocab.json",
                    PROJECT_ROOT / "checkpoints/transformer/flagship_100m_tokenizer/bpe_vocab.json",
                    PROJECT_ROOT / "checkpoints/transformer/tokenizer/vocab.json",
                ]
                fallback_tokenizer = None
                for tc in tok_candidates:
                    if tc.is_file():
                        try:
                            cand_tok = BPETokenizer.load(tc)
                            if fallback_tokenizer is None:
                                fallback_tokenizer = cand_tok
                            if cand_tok.vocab_size == cfg.vocab_size:
                                tokenizer = cand_tok
                                break
                        except Exception:
                            continue
                if tokenizer is None:
                    tokenizer = fallback_tokenizer or ByteTokenizer()
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            # Load weights with CUDA OOM protection
            try:
                checkpoint = torch.load(resolved_path, map_location=dev, weights_only=True)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state_dict)
                model.to(dev)
                model.eval()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.error(f"CUDA OutOfMemory while loading {active_name}: {e}")
                    raise RuntimeError(f"GPU out of memory loading model {active_name}: {e}") from e
                raise

            rel_checkpoint = resolved_path.relative_to(PROJECT_ROOT).as_posix()
            if is_vision:
                _VISION_MODEL = model
                _VISION_MODEL_NAME = active_name
                _VISION_DEVICE = str(dev)
                _VISION_CHECKPOINT = rel_checkpoint
                _sync_legacy_pointers("vision")
            else:
                _TRANSFORMER_MODEL = model
                _TRANSFORMER_MODEL_NAME = active_name
                _TRANSFORMER_DEVICE = str(dev)
                _TRANSFORMER_CHECKPOINT = rel_checkpoint
                _TRANSFORMER_TOKENIZER = tokenizer
                _sync_legacy_pointers("transformer")

            logger.info(f"Model {active_name} loaded successfully on {dev} from {rel_checkpoint}")
            total_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None
            return {
                "status": "loaded",
                "model_name": active_name,
                "device": str(dev),
                "checkpoint": rel_checkpoint,
                "parameters": total_params,
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
            }

    @classmethod
    def unload_model(cls, target: str = "all", is_owner: bool = False) -> Dict[str, Any]:
        """Unload active model(s) from memory and release VRAM."""
        settings = get_settings()
        if settings.demo_mode and not is_owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Model unloading is forbidden in public demo mode.",
            )

        global _VISION_MODEL, _VISION_MODEL_NAME, _VISION_DEVICE, _VISION_CHECKPOINT
        global _TRANSFORMER_MODEL, _TRANSFORMER_MODEL_NAME, _TRANSFORMER_DEVICE, _TRANSFORMER_CHECKPOINT, _TRANSFORMER_TOKENIZER
        global _ACTIVE_MODEL, _ACTIVE_MODEL_NAME, _ACTIVE_DEVICE, _ACTIVE_CHECKPOINT, _ACTIVE_TOKENIZER

        with _SERVING_LOCK:
            unloaded = []
            target_lower = (target or "all").lower()

            if target_lower in ("all", "vision", "resnet18") and _VISION_MODEL is not None:
                unloaded.append(_VISION_MODEL_NAME or "VisionModel")
                _VISION_MODEL = None
                _VISION_MODEL_NAME = None
                _VISION_DEVICE = "cpu"
                _VISION_CHECKPOINT = None

            if target_lower in ("all", "transformer", "gpt", "nexora") and _TRANSFORMER_MODEL is not None:
                unloaded.append(_TRANSFORMER_MODEL_NAME or "TransformerModel")
                _TRANSFORMER_MODEL = None
                _TRANSFORMER_MODEL_NAME = None
                _TRANSFORMER_DEVICE = "cpu"
                _TRANSFORMER_CHECKPOINT = None
                _TRANSFORMER_TOKENIZER = None

            prev = _ACTIVE_MODEL_NAME or (", ".join(unloaded) if unloaded else "None")
            _ACTIVE_MODEL = _VISION_MODEL or _TRANSFORMER_MODEL
            _ACTIVE_MODEL_NAME = _VISION_MODEL_NAME or _TRANSFORMER_MODEL_NAME
            _ACTIVE_DEVICE = _VISION_DEVICE if _VISION_MODEL is not None else (_TRANSFORMER_DEVICE if _TRANSFORMER_MODEL is not None else "cpu")
            _ACTIVE_CHECKPOINT = _VISION_CHECKPOINT or _TRANSFORMER_CHECKPOINT
            _ACTIVE_TOKENIZER = _TRANSFORMER_TOKENIZER

            torch = get_torch()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Model(s) {unloaded or prev} unloaded from memory.")
            return {"status": "unloaded", "previous_model": prev, "unloaded_models": unloaded}

    @classmethod
    def generate(
        cls,
        prompt: str = "",
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        seed: Optional[int] = 42,
        device_str: Optional[str] = None,
        is_owner: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute real autoregressive next-token generation on loaded Transformer model.
        Thread-safe and completely isolated from vision model state.
        In demo mode, enforces token/prompt clamps and concurrency locks for public users,
        while allowing verified owner unclamped generation under full hardware safeguards.
        """
        settings = get_settings()
        if settings.demo_mode and not is_owner:
            # Clamp generation to 48 tokens and prompts to 256 characters
            clamped_tokens = min(max(1, max_new_tokens), settings.demo_max_gen_tokens)
            clamped_prompt = (prompt[:settings.demo_max_prompt_chars]) if prompt and len(prompt) > settings.demo_max_prompt_chars else prompt

            acquired = _INFERENCE_LOCK.acquire(blocking=False)
            if not acquired:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="The NEXORA demo GPU is currently serving another request. Concurrency is limited to 1 active inference job. Please retry in a few seconds.",
                )
            try:
                torch = get_torch()
                effective_device_str = device_str or _TRANSFORMER_DEVICE or ("cuda:0" if (torch and torch.cuda.is_available() and settings.device != "cpu") else "cpu")
                if effective_device_str == "auto":
                    effective_device_str = _TRANSFORMER_DEVICE if (_TRANSFORMER_DEVICE and _TRANSFORMER_DEVICE != "auto") else ("cuda:0" if (torch and torch.cuda.is_available() and settings.device != "cpu") else "cpu")
                cls._enforce_demo_hardware_safeguards(torch.device(effective_device_str))
                return cls._execute_generate(clamped_prompt, clamped_tokens, temperature, top_k, top_p, seed, device_str)
            finally:
                _INFERENCE_LOCK.release()
        else:
            with _SERVING_LOCK:
                torch = get_torch()
                effective_device_str = device_str or _TRANSFORMER_DEVICE or ("cuda:0" if (torch and torch.cuda.is_available() and settings.device != "cpu") else "cpu")
                if effective_device_str == "auto":
                    effective_device_str = _TRANSFORMER_DEVICE if (_TRANSFORMER_DEVICE and _TRANSFORMER_DEVICE != "auto") else ("cuda:0" if (torch and torch.cuda.is_available() and settings.device != "cpu") else "cpu")
                cls._enforce_demo_hardware_safeguards(torch.device(effective_device_str))
                return cls._execute_generate(prompt, max_new_tokens, temperature, top_k, top_p, seed, device_str)

    @classmethod
    def _execute_generate(
        cls,
        prompt: str = "",
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        seed: Optional[int] = 42,
        device_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        global _TRANSFORMER_MODEL, _TRANSFORMER_MODEL_NAME, _TRANSFORMER_DEVICE, _TRANSFORMER_CHECKPOINT, _TRANSFORMER_TOKENIZER

        torch = get_torch()
        if torch is None:
            raise RuntimeError("PyTorch is not available.")

        # GPU Contention Guard: Prevent inference collision with active training job
        effective_device_str = device_str or _TRANSFORMER_DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(effective_device_str).startswith("cuda"):
            from engine.training.orchestrator import TrainingOrchestrator
            orch = TrainingOrchestrator.get_instance()
            if orch.active_gpu_job_id is not None:
                raise RuntimeError(
                    f"GPU is actively reserved by training job '{orch.active_gpu_job_id}'. "
                    "To prevent VRAM contention on the 6GB RTX 3050, run inference on CPU or wait for training to complete."
                )

        # If transformer model is not loaded, auto-load NEXORA GPT-86.8M
        if _TRANSFORMER_MODEL is None or not hasattr(_TRANSFORMER_MODEL, "generate"):
            cls.load_model("NEXORA GPT-86.8M", device_str=device_str)

        device = torch.device(_TRANSFORMER_DEVICE)

        gen = None
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            gen = torch.Generator(device=device.type if device.type == "cpu" else "cuda")
            gen.manual_seed(seed)

        tokenizer = _TRANSFORMER_TOKENIZER
        if tokenizer is None:
            from engine.transformer import ByteTokenizer
            tokenizer = ByteTokenizer()

        # Tokenize prompt
        raw_prompt = prompt or ""
        tokens = tokenizer.encode(raw_prompt)
        if len(tokens) == 0:
            tokens = [0]

        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        # Measure generation duration
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()

        try:
            with torch.inference_mode():
                out_idx = _TRANSFORMER_MODEL.generate(
                    idx=idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=gen,
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.error(f"CUDA OutOfMemory during text generation: {e}")
                raise RuntimeError(f"GPU out of memory during text generation: {e}") from e
            raise

        gen_time = time.perf_counter() - t0
        all_tokens = out_idx[0].tolist()
        new_token_ids = all_tokens[len(tokens):]

        full_text = tokenizer.decode(all_tokens)
        generated_text = tokenizer.decode(new_token_ids)

        tokens_per_sec = len(new_token_ids) / max(gen_time, 1e-6)

        peak_vram = 0.0
        if device.type == "cuda":
            peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)

        _sync_legacy_pointers("transformer")

        ckpt_str = str(_TRANSFORMER_CHECKPOINT or "")
        if "50m" in ckpt_str:
            status_label = "Production Pretrained Campaign (12,208 Steps, 50M Tokens, Val PPL 6.95, Val Loss 1.9383)"
        elif "pretraining_campaign" in ckpt_str:
            status_label = "Experimental / Pretrained Campaign (125 Steps, 512k Tokens, Val PPL 42.25)"
        else:
            status_label = "Experimental / Partially Trained (Pilot Checkpoint — 15 Optimizer Steps)"

        return {
            "status": "success",
            "model_name": _TRANSFORMER_MODEL_NAME,
            "device": _TRANSFORMER_DEVICE,
            "checkpoint": _TRANSFORMER_CHECKPOINT,
            "prompt": raw_prompt,
            "prompt_tokens": len(tokens),
            "generated_tokens": len(new_token_ids),
            "total_tokens": len(all_tokens),
            "generation_duration_ms": round(gen_time * 1000.0, 2),
            "tokens_per_second": round(tokens_per_sec, 1),
            "peak_vram_mb": peak_vram,
            "is_partially_trained": False if "50m" in ckpt_str else True,
            "status_label": status_label,
            "generated_text": generated_text,
            "full_text": full_text,
        }

    @classmethod
    def predict(
        cls,
        sample_index: Optional[int] = None,
        tensor_data: Optional[List[Any]] = None,
        is_owner: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute prediction on active vision model with confidence outputs and measured latency.
        Thread-safe and completely isolated from transformer text model state.
        In demo mode, enforces tensor float limits and concurrency locks for public users,
        while allowing verified owner unclamped prediction under full hardware safeguards.
        """
        settings = get_settings()
        if settings.demo_mode and not is_owner:
            if tensor_data is not None:
                def _count_floats(obj):
                    if isinstance(obj, (list, tuple)):
                        return sum(_count_floats(x) for x in obj)
                    return 1
                if _count_floats(tensor_data) > settings.demo_max_tensor_floats:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Input tensor exceeds maximum allowed {settings.demo_max_tensor_floats} floats in demo mode.",
                    )

            acquired = _INFERENCE_LOCK.acquire(blocking=False)
            if not acquired:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="The NEXORA demo GPU is currently serving another request. Concurrency is limited to 1 active inference job. Please retry in a few seconds.",
                )
            try:
                torch = get_torch()
                effective_dev_str = _VISION_DEVICE or ("cuda:0" if (torch and torch.cuda.is_available() and settings.device != "cpu") else "cpu")
                if effective_dev_str == "auto":
                    effective_dev_str = "cuda:0" if (torch and torch.cuda.is_available() and settings.device != "cpu") else "cpu"
                cls._enforce_demo_hardware_safeguards(torch.device(effective_dev_str))
                return cls._execute_predict(sample_index, tensor_data)
            finally:
                _INFERENCE_LOCK.release()
        else:
            with _SERVING_LOCK:
                torch = get_torch()
                effective_dev_str = _VISION_DEVICE or ("cuda:0" if (torch and torch.cuda.is_available() and settings.device != "cpu") else "cpu")
                if effective_dev_str == "auto":
                    effective_dev_str = "cuda:0" if (torch and torch.cuda.is_available() and settings.device != "cpu") else "cpu"
                cls._enforce_demo_hardware_safeguards(torch.device(effective_dev_str))
                return cls._execute_predict(sample_index, tensor_data)

    @classmethod
    def _execute_predict(
        cls,
        sample_index: Optional[int] = None,
        tensor_data: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        global _VISION_MODEL, _VISION_MODEL_NAME, _VISION_DEVICE, _VISION_CHECKPOINT

        torch = get_torch()
        if torch is None:
            raise RuntimeError("PyTorch is not available.")

        # Auto-load ResNet18 if no vision model currently active
        if _VISION_MODEL is None:
            cls.load_model("ResNet18")

        device = torch.device(_VISION_DEVICE)

        # Prepare input tensor
        if sample_index is not None:
            from engine.data.cifar10 import load_cifar10_raw
            extracted = PROJECT_ROOT / "data/cifar-10-batches-py"
            if not extracted.is_dir():
                raise FileNotFoundError(
                    "CIFAR-10 dataset directory not found at 'data/cifar-10-batches-py'. "
                    "Please download CIFAR-10 via engine.data.cifar10 or provide custom tensor data."
                )
            _, _, test_imgs, test_lbls, class_names = load_cifar10_raw(extracted)
            idx = max(0, min(sample_index, len(test_imgs) - 1))
            img_np = test_imgs[idx]
            actual_label_idx = int(test_lbls[idx])
            actual_label_name = class_names[actual_label_idx] if actual_label_idx < len(class_names) else str(actual_label_idx)

            # Normalization
            from engine.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD
            mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(3, 1, 1)
            raw_t = torch.from_numpy(img_np).float() / 255.0
            norm_t = (raw_t - mean) / std
            input_tensor = norm_t.unsqueeze(0).to(device)
            metadata = {
                "source": "cifar10_test_set",
                "sample_index": idx,
                "ground_truth_class": actual_label_name,
                "ground_truth_index": actual_label_idx,
            }
        elif tensor_data is not None:
            input_tensor = torch.tensor(tensor_data, dtype=torch.float32).to(device)
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)
            metadata = {"source": "custom_tensor"}
            class_names = get_class_names()
        else:
            # Default to sample 0
            return cls.predict(sample_index=0)

        # Adapt tensor for flat MLP architectures if needed
        if hasattr(_VISION_MODEL, "input_dim") and input_tensor.ndim > 2:
            input_tensor = input_tensor.flatten(1)[:, :_VISION_MODEL.input_dim]

        # Inference Timing with CUDA Sync
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        try:
            with torch.no_grad():
                logits = _VISION_MODEL(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.error(f"CUDA OutOfMemory during image prediction: {e}")
                raise RuntimeError(f"GPU out of memory during vision inference: {e}") from e
            raise

        latency_ms = (time.perf_counter() - t0) * 1000.0

        top_k = min(5, probs.size(0))
        top_probs, top_indices = torch.topk(probs, k=top_k)

        class_names = get_class_names()
        top_predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            c_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            top_predictions.append({
                "class_name": c_name,
                "class_index": idx,
                "confidence": round(prob, 4),
                "confidence_percent": round(prob * 100, 2),
            })

        pred_idx = top_predictions[0]["class_index"]
        pred_class = top_predictions[0]["class_name"]
        pred_conf = top_predictions[0]["confidence"]

        correct = None
        if "ground_truth_index" in metadata:
            correct = (pred_idx == metadata["ground_truth_index"])

        _sync_legacy_pointers("vision")

        return {
            "status": "success",
            "model": _VISION_MODEL_NAME,
            "device": _VISION_DEVICE,
            "latency_ms": round(latency_ms, 3),
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": pred_conf,
            "correct": correct,
            "top_predictions": top_predictions,
            "metadata": metadata,
        }
