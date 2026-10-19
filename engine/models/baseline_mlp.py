from typing import List, Optional
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Extensible multi-layer perceptron neural network supporting dynamic depth,
    various activation functions, dropout regularization, and execution across CPU and CUDA.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Select activation
        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        act_fn = act_map.get(activation.lower(), nn.ReLU())

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Final projection head
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total or trainable parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        """Detect the device on which model parameters currently reside."""
        return next(self.parameters()).device
