"""
PyTorch Autograd Integration for Custom CUDA Kernels
Phase 10 — Stage 2 — Milestone 3

Provides torch.autograd.Function wrappers for custom CUDA kernels:
- VectorAddFunction / vector_add
- VectorScaleFunction / scale
- ReLUFunction / relu
- SigmoidFunction / sigmoid
- GELUFunction / gelu
- FusedAddReLUFunction / fused_add_relu
- FusedBiasActivationFunction / fused_bias_activation
- MatmulFunction / matmul, matmul_wmma
- CustomLinear (nn.Module)

Supports FP32, FP16, and BF16 with correct gradient propagation.
"""
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from . import _get_module


# ============================================================================
# 1. Vector Addition Autograd
# ============================================================================
class VectorAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("All inputs must be CUDA tensors")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Shape and dtype mismatch: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        
        a_contig = a.contiguous()
        b_contig = b.contiguous()
        out = _get_module().vector_add(a_contig, b_contig)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        
        grad_output_contig = grad_output.contiguous()
        grad_a = None
        grad_b = None

        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_a, grad_b = _get_module().vector_add_backward(grad_output_contig)
        elif ctx.needs_input_grad[0]:
            grad_a = grad_output_contig.clone()
        elif ctx.needs_input_grad[1]:
            grad_b = grad_output_contig.clone()

        return grad_a, grad_b


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled custom CUDA vector addition."""
    return VectorAddFunction.apply(a, b)


# ============================================================================
# 2. Vector Scaling Autograd
# ============================================================================
class VectorScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: Union[float, torch.Tensor]) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        
        is_tensor_alpha = isinstance(alpha, torch.Tensor)
        alpha_val = float(alpha.item()) if is_tensor_alpha else float(alpha)
        
        ctx.save_for_backward(x if is_tensor_alpha and alpha.requires_grad else None)
        ctx.alpha_val = alpha_val
        ctx.is_tensor_alpha = is_tensor_alpha

        x_contig = x.contiguous()
        out = _get_module().vector_scale(x_contig, alpha_val)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        
        grad_output_contig = grad_output.contiguous()
        grad_x = None
        grad_alpha = None

        if ctx.needs_input_grad[0]:
            grad_x = _get_module().vector_scale_backward(grad_output_contig, ctx.alpha_val)

        if ctx.is_tensor_alpha and ctx.needs_input_grad[1]:
            (saved_x,) = ctx.saved_tensors
            grad_alpha = (grad_output_contig * saved_x).sum()

        return grad_x, grad_alpha


def scale(x: torch.Tensor, alpha: Union[float, torch.Tensor]) -> torch.Tensor:
    """Autograd-enabled custom CUDA vector scaling."""
    return VectorScaleFunction.apply(x, alpha)


# ============================================================================
# 3. ReLU Autograd
# ============================================================================
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        x_contig = x.contiguous()
        ctx.save_for_backward(x_contig)
        return _get_module().relu(x_contig)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
        if not ctx.needs_input_grad[0]:
            return None
        if not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        (x,) = ctx.saved_tensors
        return _get_module().relu_backward(grad_output.contiguous(), x)


def relu(x: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled custom CUDA ReLU."""
    return ReLUFunction.apply(x)


# ============================================================================
# 4. Sigmoid Autograd
# ============================================================================
class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        out = _get_module().sigmoid(x.contiguous())
        # Numerically stable: save forward output Y, dX = dY * Y * (1 - Y)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
        if not ctx.needs_input_grad[0]:
            return None
        if not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        (out,) = ctx.saved_tensors
        return _get_module().sigmoid_backward(grad_output.contiguous(), out)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled custom CUDA Sigmoid."""
    return SigmoidFunction.apply(x)


# ============================================================================
# 5. GELU Autograd
# ============================================================================
class GELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        x_contig = x.contiguous()
        ctx.save_for_backward(x_contig)
        return _get_module().gelu(x_contig)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
        if not ctx.needs_input_grad[0]:
            return None
        if not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        (x,) = ctx.saved_tensors
        return _get_module().gelu_backward(grad_output.contiguous(), x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled custom CUDA GELU (tanh-approximation)."""
    return GELUFunction.apply(x)


# ============================================================================
# 6. Fused Add + ReLU Autograd
# ============================================================================
class FusedAddReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("All inputs must be CUDA tensors")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Shape/dtype mismatch: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        
        out = _get_module().fused_add_relu(a.contiguous(), b.contiguous())
        # Saving forward out Y enables single-pass gradient calculation without saving both A and B
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        (out,) = ctx.saved_tensors
        grad_out_contig = grad_output.contiguous()

        grad_a = None
        grad_b = None

        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_a, grad_b = _get_module().fused_add_relu_backward(grad_out_contig, out)
        elif ctx.needs_input_grad[0]:
            grad_a, _ = _get_module().fused_add_relu_backward(grad_out_contig, out)
        elif ctx.needs_input_grad[1]:
            _, grad_b = _get_module().fused_add_relu_backward(grad_out_contig, out)

        return grad_a, grad_b


def fused_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled custom CUDA fused Add + ReLU."""
    return FusedAddReLUFunction.apply(a, b)


# ============================================================================
# 7. Fused Bias + Activation Autograd
# ============================================================================
class FusedBiasActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bias: torch.Tensor, activation: str = "relu") -> torch.Tensor:
        if not x.is_cuda or not bias.is_cuda:
            raise ValueError("All inputs must be CUDA tensors")
        if x.dim() != 2 or bias.dim() != 1 or x.size(1) != bias.size(0):
            raise ValueError(f"Dimension mismatch: x={x.shape}, bias={bias.shape}")
        
        act = activation.lower()
        if act not in ("none", "relu", "gelu"):
            raise ValueError(f"Unsupported activation: {activation}")

        x_contig = x.contiguous()
        bias_contig = bias.contiguous()
        out = _get_module().fused_bias_activation(x_contig, bias_contig, act)

        ctx.save_for_backward(x_contig, bias_contig)
        ctx.activation = act
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        if not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        x, bias = ctx.saved_tensors
        grad_out_contig = grad_output.contiguous()

        grad_x, grad_bias = _get_module().fused_bias_activation_backward(
            grad_out_contig, x, bias, ctx.activation
        )

        res_x = grad_x if ctx.needs_input_grad[0] else None
        res_bias = grad_bias if ctx.needs_input_grad[1] else None
        return res_x, res_bias, None


def fused_bias_activation(x: torch.Tensor, bias: torch.Tensor, activation: str = "relu") -> torch.Tensor:
    """Autograd-enabled custom CUDA fused Bias + Activation."""
    return FusedBiasActivationFunction.apply(x, bias, activation)


# ============================================================================
# 8. Matrix Multiplication Autograd (FP32 Tiled & FP16 Ampere Tensor Core WMMA)
# ============================================================================
class MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, output_fp32: bool = False) -> torch.Tensor:
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        if a.dim() != 2 or b.dim() != 2 or a.size(1) != b.size(0):
            raise ValueError(f"Incompatible matrix shapes: {a.shape} and {b.shape}")
        if a.dtype != b.dtype:
            raise ValueError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")
        
        a_contig = a.contiguous()
        b_contig = b.contiguous()

        if a.dtype == torch.float16:
            out = _get_module().matmul_wmma(a_contig, b_contig, output_fp32)
        elif a.dtype == torch.float32:
            out = _get_module().matmul_tiled(a_contig, b_contig)
        else:
            raise ValueError(f"Unsupported dtype for custom matmul: {a.dtype}. Must be float16 or float32.")

        ctx.save_for_backward(a_contig, b_contig)
        ctx.dtype = a.dtype
        ctx.output_fp32 = output_fp32
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        if not grad_output.is_cuda:
            raise ValueError("grad_output must be a CUDA tensor")
        a, b = ctx.saved_tensors
        dtype = ctx.dtype
        grad_out_contig = grad_output.contiguous()

        # If forward produced FP32 output from FP16 inputs, cast grad_output back to FP16 for backward WMMA
        if dtype == torch.float16 and grad_out_contig.dtype != torch.float16:
            grad_out_contig = grad_out_contig.to(torch.float16)

        grad_a = None
        grad_b = None

        # dA = dC @ B^T  (M x N) @ (N x K) -> (M x K)
        if ctx.needs_input_grad[0]:
            b_t = b.t().contiguous()
            if dtype == torch.float16:
                grad_a = _get_module().matmul_wmma(grad_out_contig, b_t, False)
            else:
                grad_a = _get_module().matmul_tiled(grad_out_contig, b_t)

        # dB = A^T @ dC  (K x M) @ (M x N) -> (K x N)
        if ctx.needs_input_grad[1]:
            a_t = a.t().contiguous()
            if dtype == torch.float16:
                grad_b = _get_module().matmul_wmma(a_t, grad_out_contig, False)
            else:
                grad_b = _get_module().matmul_tiled(a_t, grad_out_contig)

        return grad_a, grad_b, None


def matmul(a: torch.Tensor, b: torch.Tensor, output_fp32: bool = False) -> torch.Tensor:
    """Autograd-enabled custom CUDA matrix multiplication (Tiled SGEMM or WMMA)."""
    return MatmulFunction.apply(a, b, output_fp32)


def matmul_wmma(a: torch.Tensor, b: torch.Tensor, output_fp32: bool = False) -> torch.Tensor:
    """Autograd-enabled Ampere Tensor Core WMMA matrix multiplication."""
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul_wmma requires float16 inputs")
    return MatmulFunction.apply(a, b, output_fp32)


# ============================================================================
# 9. Neural Network Module: CustomLinear
# ============================================================================
class CustomLinear(nn.Module):
    """
    Fully-connected linear layer powered by custom CUDA forward and backward kernels:
    - Matrix multiplication: Custom Tiled SGEMM (FP32) or Ampere WMMA (FP16)
    - Bias + Activation: Custom fused CUDA kernel with integrated batch-reduction backward
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "none",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation.lower()
        if self.activation not in ("none", "relu", "gelu"):
            raise ValueError(f"Unsupported activation: {activation}")

        factory_kwargs = {"device": device, "dtype": dtype}
        # PyTorch Linear weight is (out_features, in_features). We store weight as (in_features, out_features)
        # for direct matmul X @ W, or (out_features, in_features) for standard PyTorch weight compatibility.
        # Let's use (out_features, in_features) to match standard nn.Linear parameter layout!
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, in_features). Weight is (out_features, in_features).
        # We compute x @ W^T using custom matmul:
        w_t = self.weight.t().contiguous()
        y = matmul(x, w_t)
        if self.bias is not None:
            if self.activation in ("relu", "gelu"):
                return fused_bias_activation(y, self.bias, self.activation)
            else:
                return fused_bias_activation(y, self.bias, "none")
        else:
            if self.activation == "relu":
                return relu(y)
            elif self.activation == "gelu":
                return gelu(y)
            return y


__all__ = [
    "VectorAddFunction",
    "vector_add",
    "VectorScaleFunction",
    "scale",
    "ReLUFunction",
    "relu",
    "SigmoidFunction",
    "sigmoid",
    "GELUFunction",
    "gelu",
    "FusedAddReLUFunction",
    "fused_add_relu",
    "FusedBiasActivationFunction",
    "fused_bias_activation",
    "MatmulFunction",
    "matmul",
    "matmul_wmma",
    "CustomLinear",
]
