"""
Phase 10 Stage 2 Milestone 4: CIFAR-10 Classification MLP
Provides identical 4-layer architectures across:
- CPU_NATIVE (PyTorch CPU)
- GPU_NATIVE (PyTorch CUDA cuBLAS/cuDNN)
- GPU_CUSTOM (Custom CUDA autograd forward and backward kernels)

Parameters: 3,426,250
Input: Flattened CIFAR-10 (B, 3072)
Output: 10 Class Logits (B, 10)
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_kernels.autograd as ck_auto


class CIFAR10MLP(nn.Module):
    """
    Unified 4-layer Multi-Layer Perceptron for CIFAR-10 Classification.
    Supports CPU_NATIVE, GPU_NATIVE, and GPU_CUSTOM execution backends with
    100% identical parameter names and weight layouts.
    """

    def __init__(
        self,
        backend: str = "GPU_CUSTOM",
        input_dim: int = 3072,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 10,
    ):
        super().__init__()
        self.backend = backend.upper()
        if self.backend not in ("CPU_NATIVE", "GPU_NATIVE", "GPU_CUSTOM"):
            raise ValueError(f"Unknown backend: {backend}. Must be CPU_NATIVE, GPU_NATIVE, or GPU_CUSTOM.")

        if hidden_dims is None:
            hidden_dims = [1024, 256, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        h1, h2, h3 = hidden_dims

        if self.backend == "GPU_CUSTOM":
            # Genuine custom CUDA autograd layers
            self.fc1 = ck_auto.CustomLinear(input_dim, h1, activation="gelu")
            self.fc2 = ck_auto.CustomLinear(h1, h2, activation="gelu")
            self.fc3 = ck_auto.CustomLinear(h2, h3, activation="relu")
            self.fc4 = ck_auto.CustomLinear(h3, num_classes, activation="none")
        else:
            # Native PyTorch layers (for CPU_NATIVE and GPU_NATIVE)
            self.fc1 = nn.Linear(input_dim, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, h3)
            self.fc4 = nn.Linear(h3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten CIFAR-10 images: (B, 3, 32, 32) -> (B, 3072)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        if self.backend == "GPU_CUSTOM":
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
        else:
            x = F.gelu(self.fc1(x), approximate="tanh")
            x = F.gelu(self.fc2(x), approximate="tanh")
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cifar10_model(
    backend: str,
    canonical_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    seed: int = 42,
) -> CIFAR10MLP:
    """
    Instantiate CIFAR10MLP for the requested backend.
    If canonical_state_dict is provided, loads exact weights to ensure identical initial model parameters.
    """
    model = CIFAR10MLP(backend=backend)

    if canonical_state_dict is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in canonical_state_dict:
                    param.copy_(canonical_state_dict[name].to(param.device))
    else:
        torch.manual_seed(seed)
        model.fc1.reset_parameters()
        model.fc2.reset_parameters()
        model.fc3.reset_parameters()
        model.fc4.reset_parameters()

    return model


def get_canonical_initial_weights(seed: int = 42) -> Dict[str, torch.Tensor]:
    """Generate canonical initial weights using a fixed seed on CPU."""
    torch.manual_seed(seed)
    base_model = CIFAR10MLP(backend="CPU_NATIVE")
    return {k: v.clone().detach().cpu() for k, v in base_model.state_dict().items()}
