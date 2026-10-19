import os
import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages atomic saving and loading of model checkpoints, optimizer state,
    and training metadata across devices.
    """

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save model parameters, optimizer state, and training metadata to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "metadata": metadata or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(payload, filepath)
        logger.info(f"Checkpoint successfully saved: {filepath}")
        return filepath

    @staticmethod
    def load_checkpoint(
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load model weights and optional optimizer state from disk with device remapping.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        if map_location is None:
            map_location = torch.device("cpu")

        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Checkpoint successfully restored from: {filepath} (device: {map_location})")
        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "metadata": checkpoint.get("metadata", {}),
        }
