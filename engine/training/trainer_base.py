import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract Base Trainer establishing a modular lifecycle contract for
    CPU, GPU, and experimental hybrid model training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}

        self.model.to(self.device)

    @abstractmethod
    def train_epoch(self) -> float:
        """Execute one complete training epoch and return average training loss."""
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Evaluate validation metrics."""
        pass

    def fit(self, epochs: int) -> Dict[str, Any]:
        """Standard execution loop across epochs."""
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        logger.info(f"Starting training run on device: {self.device} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics.get("loss", 0.0))
            history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))

            logger.info(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics.get('loss', 0.0):.4f} | "
                f"Val Acc: {val_metrics.get('accuracy', 0.0) * 100:.2f}%"
            )

        return history
