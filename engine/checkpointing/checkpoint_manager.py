import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Set
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Canonical protected artifacts that MUST NEVER be overwritten by runtime jobs
PROTECTED_PATTERNS = [
    "resnet18_cifar10",
    "phase10_stage2_milestone4",
    "phase9",
    "phase2_verification",
    "training_jobs_diagnostic_backup",
    "pretraining_50m_campaign",
    "pretraining_campaign",
    "flagship_100m_pilot",
    "production_10m",
    "pilot_1m",
    "expanded_pretraining",
]


class CheckpointManager:
    """
    Manages atomic saving and loading of model checkpoints, optimizer state,
    scheduler state, scaler state, RNG states, and training metadata across devices.
    Enforces baseline protection and integrity verification.
    """

    @staticmethod
    def is_protected_path(filepath: str) -> bool:
        """Check if a target path matches any protected baseline artifact pattern."""
        resolved = str(Path(filepath).resolve()).replace("\\", "/").lower()
        for pat in PROTECTED_PATTERNS:
            if pat in resolved:
                return True
        return False

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_protected_override: bool = False,
    ) -> str:
        """
        Save model parameters, optimizer state, scheduler state, scaler state,
        RNG states, and training metadata to disk atomically.
        Refuses to overwrite protected baseline artifacts.
        """
        abs_path = os.path.abspath(filepath)
        if not force_protected_override and CheckpointManager.is_protected_path(abs_path):
            # If the file already exists and is protected, do not overwrite!
            if os.path.exists(abs_path):
                raise PermissionError(
                    f"Protected baseline artifact cannot be overwritten: {filepath}"
                )

        target_dir = os.path.dirname(abs_path)
        os.makedirs(target_dir, exist_ok=True)

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "metadata": metadata or {},
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            payload["scaler_state_dict"] = scaler.state_dict()

        # Atomic write: write to unique temporary file in same directory, then replace
        unique_id = uuid.uuid4().hex[:8]
        tmp_filepath = f"{abs_path}.tmp_{os.getpid()}_{unique_id}"
        try:
            torch.save(payload, tmp_filepath)
            os.replace(tmp_filepath, abs_path)
        except Exception:
            if os.path.exists(tmp_filepath):
                try:
                    os.remove(tmp_filepath)
                except OSError:
                    pass
            raise

        logger.info(f"Checkpoint successfully saved atomically: {abs_path}")
        return abs_path

    @staticmethod
    def load_checkpoint(
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        map_location: Optional[torch.device] = None,
        restore_rng: bool = False,
    ) -> Dict[str, Any]:
        """
        Load model weights, optimizer, scheduler, scaler, and RNG states.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        if map_location is None:
            map_location = torch.device("cpu")

        try:
            try:
                checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
            except Exception:
                checkpoint = torch.load(filepath, map_location=map_location)
        except Exception as load_err:
            raise ValueError(f"Corrupted or unreadable checkpoint file: {filepath} ({load_err})") from load_err

        state_key = "model_state_dict" if "model_state_dict" in checkpoint else ("state_dict" if "state_dict" in checkpoint else None)
        if state_key is None:
            raise KeyError(f"Checkpoint at {filepath} does not contain 'model_state_dict' or 'state_dict'")

        model.load_state_dict(checkpoint[state_key])

        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if restore_rng:
            try:
                if "torch_rng_state" in checkpoint and checkpoint["torch_rng_state"] is not None:
                    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
                if "cuda_rng_state" in checkpoint and checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
            except Exception as rng_err:
                logger.warning(f"Could not restore RNG state: {rng_err}")

        logger.info(f"Checkpoint successfully restored from: {filepath} (device: {map_location})")
        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "metadata": checkpoint.get("metadata", {}),
            "checkpoint": checkpoint,
        }

    @staticmethod
    def validate_checkpoint(filepath: str) -> Dict[str, Any]:
        """
        Validate checkpoint readability, verify state dict presence, and compute checksum.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise ValueError(f"Checkpoint file is empty: {filepath}")

        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        digest = sha256_hash.hexdigest()

        try:
            try:
                data = torch.load(filepath, map_location="cpu", weights_only=False)
            except Exception:
                data = torch.load(filepath, map_location="cpu")
        except Exception as load_err:
            raise ValueError(f"Corrupted or unreadable checkpoint file: {filepath} ({load_err})") from load_err

        if not isinstance(data, dict):
            raise ValueError(f"Checkpoint payload must be a dictionary, got {type(data)}")

        state_key = "model_state_dict" if "model_state_dict" in data else ("state_dict" if "state_dict" in data else None)
        if state_key is None:
            raise ValueError("Checkpoint missing required key 'model_state_dict' or 'state_dict'")

        epoch = data.get("epoch", 0)
        metrics = data.get("metrics", {})
        metadata = data.get("metadata", {})
        has_optimizer = "optimizer_state_dict" in data and data["optimizer_state_dict"] is not None
        has_scheduler = "scheduler_state_dict" in data and data["scheduler_state_dict"] is not None
        has_scaler = "scaler_state_dict" in data and data["scaler_state_dict"] is not None
        num_tensors = len(data[state_key])

        return {
            "valid": True,
            "filepath": filepath,
            "size_bytes": file_size,
            "sha256": digest,
            "epoch": epoch,
            "metrics": metrics,
            "metadata": metadata,
            "has_optimizer": has_optimizer,
            "has_scheduler": has_scheduler,
            "has_scaler": has_scaler,
            "num_tensors": num_tensors,
        }


