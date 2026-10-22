"""
Phase 10 Stage 2 Milestone 3: Real Training Smoke Test Suite
Verifies genuine neural network training using custom CUDA autograd backward kernels.

Compares side-by-side:
A. Native PyTorch reference layers (nn.Linear, F.gelu, F.relu, SGD / AdamW).
B. Custom CUDA autograd extension layers (CustomLinear, WMMA / SGEMM matmul, fused bias+activation).

Validates:
- Identical initial weights and identical batches.
- Correct gradient propagation (loss.backward()).
- Non-zero parameter updates (weights actually change).
- Sensible monotonic/converging loss trajectory.
- Strict absence of NaN or Inf values.
- Consistent loss and weight alignment between PyTorch and Custom CUDA.
- Confirmed execution of actual custom CUDA kernels during backward passes.
"""
from typing import Dict, List, Tuple
from unittest.mock import patch
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_kernels as ck
import custom_kernels.autograd as ck_auto


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for training smoke test")
    torch.cuda.empty_cache()


# =========================================================================
# Model Definitions
# =========================================================================

class PyTorchReferenceMLP(nn.Module):
    def __init__(self, in_dim=32, h1=64, h2=16, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x), approximate="tanh")
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CustomCUDAAutogradMLP(nn.Module):
    def __init__(self, in_dim=32, h1=64, h2=16, out_dim=4):
        super().__init__()
        self.fc1 = ck_auto.CustomLinear(in_dim, h1, activation="gelu")
        self.fc2 = ck_auto.CustomLinear(h1, h2, activation="relu")
        self.fc3 = ck_auto.CustomLinear(h2, out_dim, activation="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


def copy_weights(src: PyTorchReferenceMLP, dst: CustomCUDAAutogradMLP):
    """Deep copy weights and biases from PyTorch model to Custom CUDA model."""
    with torch.no_grad():
        dst.fc1.weight.copy_(src.fc1.weight)
        dst.fc1.bias.copy_(src.fc1.bias)
        dst.fc2.weight.copy_(src.fc2.weight)
        dst.fc2.bias.copy_(src.fc2.bias)
        dst.fc3.weight.copy_(src.fc3.weight)
        dst.fc3.bias.copy_(src.fc3.bias)


# =========================================================================
# Training Smoke Tests
# =========================================================================

class TestTrainingSmokeTest:
    """Verifies that custom CUDA autograd enables real neural network training."""

    def test_training_smoke_fp32_sgd(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        batch_size = 64
        in_dim = 32
        out_dim = 4
        num_steps = 30
        lr = 0.01

        # Fixed dataset for training convergence
        x = torch.randn(batch_size, in_dim, device="cuda", dtype=torch.float32)
        target = torch.randn(batch_size, out_dim, device="cuda", dtype=torch.float32)

        model_ref = PyTorchReferenceMLP(in_dim=in_dim, out_dim=out_dim).cuda()
        model_ck = CustomCUDAAutogradMLP(in_dim=in_dim, out_dim=out_dim).cuda()
        copy_weights(model_ref, model_ck)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=lr)
        opt_ck = torch.optim.Adam(model_ck.parameters(), lr=lr)

        losses_ref: List[float] = []
        losses_ck: List[float] = []

        initial_w1_ck = model_ck.fc1.weight.clone()

        for step in range(num_steps):
            # --- PyTorch Step ---
            opt_ref.zero_grad()
            out_ref = model_ref(x)
            loss_ref = F.mse_loss(out_ref, target)
            loss_ref.backward()
            opt_ref.step()
            losses_ref.append(loss_ref.item())

            # --- Custom CUDA Autograd Step ---
            opt_ck.zero_grad()
            out_ck = model_ck(x)
            loss_ck = F.mse_loss(out_ck, target)
            loss_ck.backward()

            # Verify gradients exist, are finite, and not NaN
            for name, param in model_ck.named_parameters():
                assert param.grad is not None, f"Gradient is None for {name} at step {step}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name} at step {step}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name} at step {step}"

            opt_ck.step()
            losses_ck.append(loss_ck.item())

            # Assert step-by-step close alignment
            torch.testing.assert_close(out_ck, out_ref, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(loss_ck, loss_ref, atol=1e-3, rtol=1e-3)

        # 1. Parameter update verification: weights have actually changed
        delta_w = torch.norm(model_ck.fc1.weight - initial_w1_ck).item()
        assert delta_w > 0.05, f"Weights did not update significantly: delta={delta_w}"

        # 2. Sensible loss trajectory: loss decreased substantially
        assert losses_ck[-1] < losses_ck[0] * 0.2, f"Loss did not decrease: start={losses_ck[0]}, end={losses_ck[-1]}"

        # 3. Final model weights match PyTorch reference
        for (name_ref, p_ref), (name_ck, p_ck) in zip(model_ref.named_parameters(), model_ck.named_parameters()):
            torch.testing.assert_close(p_ck, p_ref, atol=2e-3, rtol=2e-3)

    def test_training_smoke_fp16_wmma(self):
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        batch_size = 64
        in_dim = 32
        out_dim = 4
        num_steps = 30
        lr = 0.01

        # Fixed dataset for training convergence
        x = torch.randn(batch_size, in_dim, device="cuda", dtype=torch.float16)
        target = torch.randn(batch_size, out_dim, device="cuda", dtype=torch.float16)

        model_ref = PyTorchReferenceMLP(in_dim=in_dim, out_dim=out_dim).cuda().half()
        model_ck = CustomCUDAAutogradMLP(in_dim=in_dim, out_dim=out_dim).cuda().half()
        copy_weights(model_ref, model_ck)

        opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.05, momentum=0.9)
        opt_ck = torch.optim.SGD(model_ck.parameters(), lr=0.05, momentum=0.9)

        losses_ref: List[float] = []
        losses_ck: List[float] = []

        initial_w = model_ck.fc1.weight.clone()

        for step in range(num_steps):
            opt_ref.zero_grad()
            out_ref = model_ref(x)
            loss_ref = F.mse_loss(out_ref, target)
            loss_ref.backward()
            opt_ref.step()
            losses_ref.append(loss_ref.item())

            opt_ck.zero_grad()
            out_ck = model_ck(x)
            loss_ck = F.mse_loss(out_ck, target)
            loss_ck.backward()

            for name, param in model_ck.named_parameters():
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

            opt_ck.step()
            losses_ck.append(loss_ck.item())

        delta = torch.norm(model_ck.fc1.weight - initial_w).item()
        assert delta > 0.05, f"FP16 weights did not update: delta={delta}"
        assert losses_ck[-1] < losses_ck[0] * 0.8, f"FP16 loss did not decrease: start={losses_ck[0]}, end={losses_ck[-1]}"

        # In FP16, Tensor Core accumulation differences can cause minor drift over 30 steps
        assert abs(losses_ck[-1] - losses_ref[-1]) < 0.05

    def test_kernel_dispatch_confirmation(self):
        """Confirms that backward execution actually dispatches custom C++/CUDA kernels."""
        x = torch.randn(16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
        bias = torch.randn(32, device="cuda", dtype=torch.float32, requires_grad=True)

        mod = ck._get_module()
        called_ops = []

        orig_bias_bwd = mod.fused_bias_activation_backward
        def spy_bias_bwd(*args, **kwargs):
            called_ops.append("fused_bias_activation_backward")
            return orig_bias_bwd(*args, **kwargs)

        mod.fused_bias_activation_backward = spy_bias_bwd
        try:
            out = ck_auto.fused_bias_activation(x, bias, "relu")
            loss = out.sum()
            loss.backward()

            assert "fused_bias_activation_backward" in called_ops, "Custom backward kernel was not dispatched!"
            assert x.grad is not None
            assert bias.grad is not None
        finally:
            mod.fused_bias_activation_backward = orig_bias_bwd
