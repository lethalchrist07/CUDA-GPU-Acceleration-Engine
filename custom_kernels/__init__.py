"""
Custom CUDA Kernel Library for CUDA GPU Acceleration Engine
Target: NVIDIA GeForce RTX 3050 Laptop GPU (sm_86)
Toolchain: CUDA 12.4.1 + MSVC 19.38

All operations in this package support mixed precision:
- float32 (FP32)
- float16 (FP16)
- bfloat16 (BF16)

The top-level functional operators (e.g., custom_kernels.vector_add) are forward-only
inference operators that do not attach PyTorch autograd computation graphs.
For end-to-end PyTorch autograd training with backward gradient propagation, use
custom_kernels.autograd (e.g., custom_kernels.autograd.vector_add, custom_kernels.autograd.matmul).
"""
from typing import Optional
import torch

from .loader import load_custom_kernels_module

_cuda_module = None

def _get_module():
    global _cuda_module
    if _cuda_module is None:
        _cuda_module = load_custom_kernels_module()
    return _cuda_module

def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise vector addition: c = a + b
    Forward-only operator. Does not attach autograd backward graph.
    Inputs must be contiguous CUDA tensors of identical shape and dtype (float32, float16, or bfloat16).
    """
    return _get_module().vector_add(a, b)

def scale(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Vector scaling: out = alpha * x
    Forward-only operator. Does not attach autograd backward graph.
    Input must be a contiguous CUDA tensor (float32, float16, or bfloat16).
    """
    return _get_module().vector_scale(x, float(alpha))

def fused_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fused elementwise addition + ReLU: out = max(0, a + b)
    Forward-only operator. Does not attach autograd backward graph.
    Eliminates intermediate global memory roundtrip.
    Supports float32, float16, and bfloat16.
    """
    return _get_module().fused_add_relu(a, b)

def fused_bias_activation(x: torch.Tensor, bias: torch.Tensor, activation: str = "relu") -> torch.Tensor:
    """
    Fused linear layer bias addition + activation: out = act(x + bias)
    Forward-only operator. Does not attach autograd backward graph.
    x: 2D matrix (rows x cols)
    bias: 1D vector (cols)
    activation: 'relu', 'gelu', or 'none'
    Supports float32, float16, and bfloat16.
    """
    return _get_module().fused_bias_activation(x, bias, activation.lower())

def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Rectified Linear Unit: out = max(0, x)
    Forward-only operator. Does not attach autograd backward graph.
    Supports float32, float16, and bfloat16.
    """
    return _get_module().relu(x)

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Error Linear Unit (fast tanh approximation):
    out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Forward-only operator. Does not attach autograd backward graph.
    Supports float32, float16, and bfloat16.
    """
    return _get_module().gelu(x)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid activation: out = 1 / (1 + exp(-x))
    Forward-only operator. Does not attach autograd backward graph.
    Supports float32, float16, and bfloat16.
    """
    return _get_module().sigmoid(x)

def reduce_sum(x: torch.Tensor, output_fp32: Optional[bool] = None) -> torch.Tensor:
    """
    Sum reduction: returns scalar tensor containing sum of all elements.
    Forward-only operator. Does not attach autograd backward graph.
    Accumulated in FP32 for numerical stability.
    If output_fp32 is None: defaults to True for FP16/BF16 when numel > 65536 to prevent overflow,
    otherwise matches input dtype.
    Supports float32, float16, and bfloat16.
    """
    if output_fp32 is None:
        output_fp32 = (x.numel() > 65536 and x.dtype in (torch.float16, torch.bfloat16))
    return _get_module().reduce_sum(x, bool(output_fp32))

def reduce_max(x: torch.Tensor) -> torch.Tensor:
    """
    Max reduction: returns scalar tensor containing max of all elements.
    Forward-only operator. Does not attach autograd backward graph.
    Supports float32, float16, and bfloat16.
    """
    return _get_module().reduce_max(x)

def matmul_wmma(a: torch.Tensor, b: torch.Tensor, output_fp32: bool = False) -> torch.Tensor:
    """
    Ampere Tensor Core accelerated matrix multiplication using CUDA WMMA: out = a @ b
    Forward-only operator. Does not attach autograd backward graph.
    Inputs must be contiguous 2D CUDA float16 tensors.
    Accumulation is performed in FP32 registers.
    If output_fp32 is True, returns float32 tensor; otherwise returns float16 tensor.
    """
    return _get_module().matmul_wmma(a, b, bool(output_fp32))

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication operator: out = a @ b
    Forward-only operator. Does not attach autograd backward graph.
    - If inputs are float16: dispatches to Ampere Tensor Core WMMA kernel.
    - If inputs are float32: dispatches to Shared-Memory Tiled SGEMM kernel.
    """
    if a.dtype == torch.float16:
        return _get_module().matmul_wmma(a, b, False)
    return _get_module().matmul_tiled(a, b)

__all__ = [
    "vector_add",
    "scale",
    "fused_add_relu",
    "fused_bias_activation",
    "relu",
    "gelu",
    "sigmoid",
    "reduce_sum",
    "reduce_max",
    "matmul",
    "matmul_wmma",
    "autograd",
]

from . import autograd

