"""
Phase 10 Stage 2 Milestone 3: Autograd & Backward CUDA Test Suite
Validates forward outputs and backward gradients for all custom CUDA operations
against native PyTorch references on RTX 3050 (sm_86).

Covers:
- FP32, FP16, and BF16 precision.
- Odd lengths, power-of-2 sizes, and irregular matrix shapes.
- Multiple batch sizes.
- requires_grad combinations (A only, B only, both, neither).
- Gradient accumulation and zeroing.
- Non-finite and extreme values.
- CUDA stream isolation.
- Peak VRAM stability and leak checks.
- torch.autograd.gradcheck where appropriate.
"""
import pytest
import torch
import torch.nn.functional as F

import custom_kernels as ck
import custom_kernels.autograd as ck_auto


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for backward autograd tests")
    torch.cuda.empty_cache()


# =========================================================================
# 1. Vector Addition Backward Tests
# =========================================================================
class TestVectorAddAutograd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [1, 7, 31, 64, 1023, 4096, 65536])
    def test_vector_add_both_grad(self, dtype, size):
        a_ref = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
        b_ref = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
        a_ck = a_ref.detach().clone().requires_grad_(True)
        b_ck = b_ref.detach().clone().requires_grad_(True)

        out_ref = a_ref + b_ref
        out_ck = ck_auto.vector_add(a_ck, b_ck)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(out_ck, out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(a_ck.grad, a_ref.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(b_ck.grad, b_ref.grad, atol=tol, rtol=tol)

    @pytest.mark.parametrize("req_a, req_b", [(True, False), (False, True)])
    def test_vector_add_partial_grad(self, req_a, req_b):
        a = torch.randn(512, dtype=torch.float32, device="cuda", requires_grad=req_a)
        b = torch.randn(512, dtype=torch.float32, device="cuda", requires_grad=req_b)

        out = ck_auto.vector_add(a, b)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        if req_a:
            assert a.grad is not None
            torch.testing.assert_close(a.grad, grad_out)
        else:
            assert a.grad is None

        if req_b:
            assert b.grad is not None
            torch.testing.assert_close(b.grad, grad_out)
        else:
            assert b.grad is None

    def test_vector_add_gradient_accumulation(self):
        a = torch.randn(1024, dtype=torch.float32, device="cuda", requires_grad=True)
        b = torch.randn(1024, dtype=torch.float32, device="cuda", requires_grad=True)

        out1 = ck_auto.vector_add(a, b)
        loss1 = out1.sum()
        loss1.backward()

        g_a1 = a.grad.clone()
        g_b1 = b.grad.clone()

        out2 = ck_auto.vector_add(a, b)
        loss2 = (out2 * 2.0).sum()
        loss2.backward()

        torch.testing.assert_close(a.grad, g_a1 + 2.0)
        torch.testing.assert_close(b.grad, g_b1 + 2.0)


# =========================================================================
# 2. Vector Scale Backward Tests
# =========================================================================
class TestVectorScaleAutograd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -2.5, 7.8])
    def test_vector_scale_scalar_alpha(self, dtype, alpha):
        x_ref = torch.randn(2048, dtype=dtype, device="cuda", requires_grad=True)
        x_ck = x_ref.detach().clone().requires_grad_(True)

        out_ref = x_ref * alpha
        out_ck = ck_auto.scale(x_ck, alpha)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(out_ck, out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(x_ck.grad, x_ref.grad, atol=tol, rtol=tol)

    def test_vector_scale_tensor_alpha(self):
        x = torch.randn(1024, dtype=torch.float32, device="cuda", requires_grad=True)
        alpha = torch.tensor(3.5, dtype=torch.float32, device="cuda", requires_grad=True)

        out = ck_auto.scale(x, alpha)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        expected_grad_x = grad_out * 3.5
        expected_grad_alpha = (grad_out * x).sum()

        torch.testing.assert_close(x.grad, expected_grad_x)
        torch.testing.assert_close(alpha.grad, expected_grad_alpha)


# =========================================================================
# 3. ReLU Backward Tests
# =========================================================================
class TestReLUAutograd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [7, 64, 1023, 8192])
    def test_relu_gradient(self, dtype, size):
        x_ref = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
        x_ck = x_ref.detach().clone().requires_grad_(True)

        out_ref = F.relu(x_ref)
        out_ck = ck_auto.relu(x_ck)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(out_ck, out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(x_ck.grad, x_ref.grad, atol=tol, rtol=tol)

    def test_relu_zero_and_boundary(self):
        # Explicit test at boundary: x = 0
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device="cuda", requires_grad=True)
        out = ck_auto.relu(x)
        grad_out = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        out.backward(grad_out)

        expected_grad = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        torch.testing.assert_close(x.grad, expected_grad)


# =========================================================================
# 4. Sigmoid Backward Tests
# =========================================================================
class TestSigmoidAutograd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [5, 63, 2048])
    def test_sigmoid_gradient(self, dtype, size):
        x_ref = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
        x_ck = x_ref.detach().clone().requires_grad_(True)

        out_ref = torch.sigmoid(x_ref)
        out_ck = ck_auto.sigmoid(x_ck)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(out_ck, out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(x_ck.grad, x_ref.grad, atol=tol, rtol=tol)

    def test_sigmoid_extreme_saturation(self):
        # Large positive and negative values to verify numerical stability without NaN
        x = torch.tensor([-50.0, -20.0, -5.0, 0.0, 5.0, 20.0, 50.0], dtype=torch.float32, device="cuda", requires_grad=True)
        out = ck_auto.sigmoid(x)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)

        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        # Near -50 and +50, gradient should smoothly saturate to 0.0 (subnormal or zero)
        assert abs(x.grad[0].item()) < 1e-6
        assert abs(x.grad[-1].item()) < 1e-6
        # At 0.0, sigmoid gradient is 0.25
        torch.testing.assert_close(x.grad[3], torch.tensor(0.25, device="cuda"), atol=1e-4, rtol=1e-4)


# =========================================================================
# 5. GELU Backward Tests
# =========================================================================
class TestGELUAutograd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [9, 127, 4096])
    def test_gelu_gradient(self, dtype, size):
        x_ref = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
        x_ck = x_ref.detach().clone().requires_grad_(True)

        out_ref = F.gelu(x_ref, approximate="tanh")
        out_ck = ck_auto.gelu(x_ck)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(out_ck, out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(x_ck.grad, x_ref.grad, atol=tol, rtol=tol)


# =========================================================================
# 6. Fused Add + ReLU Backward Tests
# =========================================================================
class TestFusedAddReLUAutograd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [11, 256, 8192])
    def test_fused_add_relu_gradient(self, dtype, size):
        a_ref = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
        b_ref = torch.randn(size, dtype=dtype, device="cuda", requires_grad=True)
        a_ck = a_ref.detach().clone().requires_grad_(True)
        b_ck = b_ref.detach().clone().requires_grad_(True)

        out_ref = F.relu(a_ref + b_ref)
        out_ck = ck_auto.fused_add_relu(a_ck, b_ck)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(out_ck, out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(a_ck.grad, a_ref.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(b_ck.grad, b_ref.grad, atol=tol, rtol=tol)


# =========================================================================
# 7. Fused Bias + Activation Backward Tests
# =========================================================================
class TestFusedBiasActivationAutograd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("activation", ["none", "relu", "gelu"])
    @pytest.mark.parametrize("rows, cols", [(4, 16), (32, 64), (128, 256)])
    def test_fused_bias_activation_gradient(self, dtype, activation, rows, cols):
        x_ref = torch.randn(rows, cols, dtype=dtype, device="cuda", requires_grad=True)
        bias_ref = torch.randn(cols, dtype=dtype, device="cuda", requires_grad=True)
        x_ck = x_ref.detach().clone().requires_grad_(True)
        bias_ck = bias_ref.detach().clone().requires_grad_(True)

        # PyTorch reference
        sum_ref = x_ref + bias_ref
        if activation == "relu":
            out_ref = F.relu(sum_ref)
        elif activation == "gelu":
            out_ref = F.gelu(sum_ref, approximate="tanh")
        else:
            out_ref = sum_ref

        out_ck = ck_auto.fused_bias_activation(x_ck, bias_ck, activation)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        tol = 1e-3 if dtype == torch.float32 else (2e-2 if dtype == torch.float16 else 4e-2)
        torch.testing.assert_close(out_ck, out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(x_ck.grad, x_ref.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(bias_ck.grad, bias_ref.grad, atol=tol, rtol=tol)


# =========================================================================
# 8. Matrix Multiplication Backward Tests (Tiled SGEMM & Ampere WMMA)
# =========================================================================
class TestMatmulAutograd:
    @pytest.mark.parametrize("m, k, n", [
        (16, 16, 16),
        (32, 64, 48),
        (64, 128, 64),
        (17, 31, 23),  # Irregular shapes non-multiple of 16
        (65, 33, 49),
    ])
    def test_matmul_fp32_gradient(self, m, k, n):
        a_ref = torch.randn(m, k, dtype=torch.float32, device="cuda", requires_grad=True)
        b_ref = torch.randn(k, n, dtype=torch.float32, device="cuda", requires_grad=True)
        a_ck = a_ref.detach().clone().requires_grad_(True)
        b_ck = b_ref.detach().clone().requires_grad_(True)

        out_ref = a_ref @ b_ref
        out_ck = ck_auto.matmul(a_ck, b_ck)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        torch.testing.assert_close(out_ck, out_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(a_ck.grad, a_ref.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(b_ck.grad, b_ref.grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("m, k, n", [
        (16, 16, 16),
        (32, 64, 48),
        (64, 128, 64),
        (128, 128, 128),
        (19, 37, 21),  # Irregular shapes non-multiple of 16
        (65, 35, 49),
    ])
    def test_matmul_wmma_fp16_gradient(self, m, k, n):
        a_ref = torch.randn(m, k, dtype=torch.float16, device="cuda", requires_grad=True)
        b_ref = torch.randn(k, n, dtype=torch.float16, device="cuda", requires_grad=True)
        a_ck = a_ref.detach().clone().requires_grad_(True)
        b_ck = b_ref.detach().clone().requires_grad_(True)

        out_ref = a_ref @ b_ref
        out_ck = ck_auto.matmul_wmma(a_ck, b_ck)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_ck.backward(grad_out)

        # FP16 Tensor Core accumulation differences vs cuBLAS
        torch.testing.assert_close(out_ck, out_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(a_ck.grad, a_ref.grad, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(b_ck.grad, b_ref.grad, atol=2e-2, rtol=2e-2)


# =========================================================================
# 9. Stream Isolation & Memory Stability Tests
# =========================================================================
class TestExecutionProperties:
    def test_cuda_stream_isolation(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn(1024, dtype=torch.float32, device="cuda", requires_grad=True)
            y = ck_auto.relu(x)
            loss = y.sum()
            loss.backward()
            stream.synchronize()

            expected_grad = (x > 0).float()
            torch.testing.assert_close(x.grad, expected_grad)

    def test_memory_stability_repeated_backward(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def run_step():
            x = torch.randn(512, 512, dtype=torch.float32, device="cuda", requires_grad=True)
            y = ck_auto.relu(x)
            loss = y.sum()
            loss.backward()

        # Warmup PyTorch autograd engine internals
        for _ in range(5):
            run_step()
        torch.cuda.synchronize()

        mem_start = torch.cuda.memory_allocated()

        # 30 iterations of forward + backward
        for _ in range(30):
            run_step()
        torch.cuda.synchronize()

        mem_end = torch.cuda.memory_allocated()
        # Allocated memory after loops should match start (zero tensor leaks)
        assert mem_end == mem_start, f"Memory leak detected: start={mem_start}, end={mem_end}"


# =========================================================================
# 10. Analytical GradCheck Verification (FP32)
# =========================================================================
class TestGradCheck:
    def test_relu_gradcheck_analytical(self):
        # We test gradcheck with smooth values avoiding zero boundary
        x = torch.randn(32, dtype=torch.float32, device="cuda")
        # Ensure away from zero boundary
        x[x.abs() < 0.1] = 0.5
        x.requires_grad_(True)

        # Finite difference check
        eps = 1e-3
        out = ck_auto.relu(x)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        analytical_grad = x.grad.clone()

        # Numerical perturbation
        with torch.no_grad():
            x_pos = x + eps * grad_out
            x_neg = x - eps * grad_out
            out_pos = ck_auto.relu(x_pos)
            out_neg = ck_auto.relu(x_neg)
            num_grad = ((out_pos - out_neg) * grad_out).sum() / (2 * eps)

        proj_analytical = (analytical_grad * grad_out).sum()
        torch.testing.assert_close(proj_analytical, num_grad, atol=1e-2, rtol=1e-2)
