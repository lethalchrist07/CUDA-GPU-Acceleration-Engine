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

# Global serving state
_ACTIVE_MODEL: Optional[Any] = None
_ACTIVE_MODEL_NAME: Optional[str] = None
_ACTIVE_DEVICE: str = "cpu"
_ACTIVE_CHECKPOINT: Optional[str] = None


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
    Manages in-memory models for real-time inference serving.
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
                "status": "loaded" if _ACTIVE_MODEL_NAME == "ResNet18" else "available",
                "device": _ACTIVE_DEVICE if _ACTIVE_MODEL_NAME == "ResNet18" else None,
                "active_checkpoint": _ACTIVE_CHECKPOINT if _ACTIVE_MODEL_NAME == "ResNet18" else None,
                "checkpoints": cls._find_checkpoints("checkpoints/resnet18_cifar10"),
            },
            {
                "model_id": "baseline_mlp",
                "name": "BaselineMLP",
                "task": "Synthetic Classification",
                "dataset": "Synthetic [128-dim]",
                "classes": 20,
                "parameters": 465172,
                "status": "loaded" if _ACTIVE_MODEL_NAME == "BaselineMLP" else "available",
                "device": _ACTIVE_DEVICE if _ACTIVE_MODEL_NAME == "BaselineMLP" else None,
                "active_checkpoint": _ACTIVE_CHECKPOINT if _ACTIVE_MODEL_NAME == "BaselineMLP" else None,
                "checkpoints": cls._find_checkpoints("checkpoints"),
            },
        ]
        return models

    @classmethod
    def _find_checkpoints(cls, sub_dir: str) -> List[Dict[str, Any]]:
        dir_path = PROJECT_ROOT / sub_dir
        if not dir_path.is_dir():
            return []
        items = []
        for p in sorted(dir_path.glob("*.pt")):
            items.append({
                "filename": p.name,
                "relative_path": p.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": p.stat().st_size,
            })
        return items

    @classmethod
    def load_model(
        cls,
        model_name: str = "ResNet18",
        checkpoint_path: Optional[str] = None,
        device_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load weights into memory and assign to requested device.
        """
        global _ACTIVE_MODEL, _ACTIVE_MODEL_NAME, _ACTIVE_DEVICE, _ACTIVE_CHECKPOINT

        torch = get_torch()
        if torch is None:
            raise RuntimeError("PyTorch is not available in the active environment.")

        # Resolve device
        if device_str is None or device_str == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device_str.startswith("cuda"):
            device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Resolve checkpoint
        if checkpoint_path is None:
            if model_name.lower() == "resnet18":
                candidates = [
                    PROJECT_ROOT / "checkpoints/resnet18_cifar10/best_model.pt",
                    PROJECT_ROOT / "checkpoints/resnet18_cifar10/latest_checkpoint.pt",
                ]
            else:
                candidates = [PROJECT_ROOT / "checkpoints/test_checkpoint.pt"]

            resolved_path = None
            for c in candidates:
                if c.is_file():
                    resolved_path = c
                    break
            if resolved_path is None:
                raise FileNotFoundError(f"No checkpoint file found for {model_name}")
        else:
            resolved_path = (PROJECT_ROOT / checkpoint_path).resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        # Instantiate architecture
        if model_name.lower() in ("resnet18", "resnet-18"):
            from engine.models.resnet import create_resnet18
            model = create_resnet18(num_classes=10, cifar_stem=True)
            active_name = "ResNet18"
        elif model_name.lower() in ("baselinemlp", "mlp"):
            from engine.models.baseline_mlp import BaselineMLP
            model = BaselineMLP(input_dim=128, hidden_dims=[512, 512, 256], output_dim=20)
            active_name = "BaselineMLP"
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Load weights
        checkpoint = torch.load(resolved_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _ACTIVE_MODEL = model
        _ACTIVE_MODEL_NAME = active_name
        _ACTIVE_DEVICE = str(device)
        _ACTIVE_CHECKPOINT = resolved_path.relative_to(PROJECT_ROOT).as_posix()

        logger.info(f"Model {active_name} loaded successfully on {_ACTIVE_DEVICE} from {_ACTIVE_CHECKPOINT}")
        return {
            "status": "loaded",
            "model_name": active_name,
            "device": _ACTIVE_DEVICE,
            "checkpoint": _ACTIVE_CHECKPOINT,
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    @classmethod
    def unload_model(cls) -> Dict[str, Any]:
        """Unload active model from memory and release VRAM."""
        global _ACTIVE_MODEL, _ACTIVE_MODEL_NAME, _ACTIVE_DEVICE, _ACTIVE_CHECKPOINT

        prev = _ACTIVE_MODEL_NAME
        _ACTIVE_MODEL = None
        _ACTIVE_MODEL_NAME = None
        _ACTIVE_DEVICE = "cpu"
        _ACTIVE_CHECKPOINT = None

        torch = get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Model {prev} unloaded from memory.")
        return {"status": "unloaded", "previous_model": prev}

    @classmethod
    def predict(
        cls,
        sample_index: Optional[int] = None,
        tensor_data: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute prediction on active model with confidence outputs and measured latency.
        """
        global _ACTIVE_MODEL, _ACTIVE_MODEL_NAME, _ACTIVE_DEVICE

        torch = get_torch()
        if torch is None:
            raise RuntimeError("PyTorch is not available.")

        # Auto-load ResNet18 if no model currently active
        if _ACTIVE_MODEL is None:
            cls.load_model("ResNet18")

        device = torch.device(_ACTIVE_DEVICE)

        # Prepare input tensor
        if sample_index is not None:
            from engine.data.cifar10 import load_cifar10_raw
            extracted = PROJECT_ROOT / "data/cifar-10-batches-py"
            if not extracted.is_dir():
                raise FileNotFoundError("CIFAR-10 dataset directory not found.")
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

        # Inference Timing with CUDA Sync
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = _ACTIVE_MODEL(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
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

        return {
            "status": "success",
            "model": _ACTIVE_MODEL_NAME,
            "device": _ACTIVE_DEVICE,
            "latency_ms": round(latency_ms, 3),
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": pred_conf,
            "correct": correct,
            "top_predictions": top_predictions,
            "metadata": metadata,
        }
