"""
Tests for Phase 10 Stage 1: Custom CUDA Kernel Library
Repository: CUDA GPU Acceleration Engine
Target: NVIDIA GeForce RTX 3050 6GB Laptop GPU (sm_86)

Mandatory Engineering Validations:
1. Autograd Contract: Forward-only verification (no graph attachment, grad_fn=None).
2. PyTorch CUDA Stream Integration: Correct execution on non-default streams.
3. Numerical Correctness: Operation-specific atol/rtol tolerances vs PyTorch references.
4. Memory Alignment: Vectorized float4 and safe unaligned/tail fallbacks.
5. Edge Cases: Empty tensors, irregular sizes, primes, NaNs, infinities.
6. Negative & Error Handling: Type, device, contiguity, and shape mismatches.
"""

import math
import pytest
import torch
import torch.nn.functional as F

import custom_kernels as ck


# Device availability check
@pytest.fixture(autouse=True)
def ensure_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for custom kernel tests")


# ============================================================================
# 1. AUTOGRAD CONTRACT TESTS
# ============================================================================

class TestAutogradContract:
    """Verifies that Stage 1 kernels follow the strict forward-only contract."""

    def test_vector_add_no_grad(self):
        a = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
        b = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
        c = ck.vector_add(a, b)
        assert not c.requires_grad
        assert c.grad_fn is None
        with pytest.raises(RuntimeError, match="does not require grad"):
            c.backward(torch.ones_like(c))

    def test_scale_no_grad(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
        out = ck.scale(x, 2.5)
        assert not out.requires_grad
        assert out.grad_fn is None

    def test_activations_no_grad(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
        for fn in [ck.relu, ck.gelu, ck.sigmoid]:
            out = fn(x)
            assert not out.requires_grad
            assert out.grad_fn is None

    def test_reductions_no_grad(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
        s = ck.reduce_sum(x)
        assert not s.requires_grad
        assert s.grad_fn is None
        m = ck.reduce_max(x)
        assert not m.requires_grad
        assert m.grad_fn is None

    def test_matmul_no_grad(self):
        a = torch.randn(32, 32, device="cuda", dtype=torch.float32, requires_grad=True)
        b = torch.randn(32, 32, device="cuda", dtype=torch.float32, requires_grad=True)
        c = ck.matmul(a, b)
        assert not c.requires_grad
        assert c.grad_fn is None


# ============================================================================
# 2. PYTORCH CUDA STREAM INTEGRATION TESTS
# ============================================================================

class TestCUDAStreamIntegration:
    """Verifies kernels execute asynchronously on active PyTorch CUDA streams."""

    def test_execution_on_custom_stream(self):
        stream = torch.cuda.Stream()
        a = torch.randn(1024, device="cuda", dtype=torch.float32)
        b = torch.randn(1024, device="cuda", dtype=torch.float32)
        torch.cuda.current_stream().synchronize()

        with torch.cuda.stream(stream):
            c = ck.vector_add(a, b)
            act = ck.relu(c)
            s = ck.reduce_sum(act)

        stream.synchronize()
        ref = torch.sum(torch.relu(a + b))
        assert torch.allclose(s, ref, atol=1e-4, rtol=1e-4)

    def test_multi_stream_isolation(self):
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        a1 = torch.ones(512, device="cuda", dtype=torch.float32) * 2.0
        a2 = torch.ones(512, device="cuda", dtype=torch.float32) * 5.0
        torch.cuda.current_stream().synchronize()

        with torch.cuda.stream(s1):
            out1 = ck.scale(a1, 3.0)  # 6.0

        with torch.cuda.stream(s2):
            out2 = ck.scale(a2, 4.0)  # 20.0

        s1.synchronize()
        s2.synchronize()

        assert torch.allclose(out1, torch.full_like(out1, 6.0))
        assert torch.allclose(out2, torch.full_like(out2, 20.0))

    def test_event_synchronization(self):
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        event = torch.cuda.Event()

        a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float32)
        torch.cuda.current_stream().synchronize()

        with torch.cuda.stream(s1):
            c = ck.matmul(a, b)
            event.record(s1)

        with torch.cuda.stream(s2):
            s2.wait_event(event)
            out = ck.gelu(c)

        s2.synchronize()
        ref = F.gelu(torch.matmul(a, b), approximate="tanh")
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


# ============================================================================
# 3. NUMERICAL CORRECTNESS & TOLERANCE TESTS
# ============================================================================

class TestNumericalCorrectness:
    """Verifies numerical accuracy against PyTorch references."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 16, 31, 33, 256, 1024, 65537, 100_000])
    def test_vector_add_accuracy(self, n):
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        out = ck.vector_add(a, b)
        ref = a + b
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("n", [1, 3, 4, 15, 32, 33, 1000, 100_000])
    def test_vector_scale_accuracy(self, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        alpha = 3.14159
        out = ck.scale(x, alpha)
        ref = x * alpha
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("n", [1, 7, 32, 127, 256, 4096, 65535])
    def test_fused_add_relu_accuracy(self, n):
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        out = ck.fused_add_relu(a, b)
        ref = torch.relu(a + b)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 64), (64, 1), (37, 53), (128, 256), (512, 128)])
    def test_fused_bias_activation_accuracy(self, rows, cols):
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        bias = torch.randn(cols, device="cuda", dtype=torch.float32)

        # None
        out_none = ck.fused_bias_activation(x, bias, "none")
        assert torch.allclose(out_none, x + bias, atol=1e-5, rtol=1e-5)

        # ReLU
        out_relu = ck.fused_bias_activation(x, bias, "relu")
        assert torch.allclose(out_relu, torch.relu(x + bias), atol=1e-5, rtol=1e-5)

        # GELU (tanh approx)
        out_gelu = ck.fused_bias_activation(x, bias, "gelu")
        ref_gelu = F.gelu(x + bias, approximate="tanh")
        assert torch.allclose(out_gelu, ref_gelu, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("n", [1, 4, 15, 33, 256, 1024, 65537])
    def test_relu_accuracy(self, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out = ck.relu(x)
        ref = torch.relu(x)
        assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("n", [1, 4, 15, 33, 256, 1024, 65537])
    def test_gelu_tanh_accuracy(self, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out = ck.gelu(x)
        ref = F.gelu(x, approximate="tanh")
        # Compare specifically against PyTorch tanh approximation
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("n", [1, 4, 15, 33, 256, 1024, 65537])
    def test_sigmoid_accuracy(self, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out = ck.sigmoid(x)
        ref = torch.sigmoid(x)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("n", [1, 2, 10, 32, 100, 256, 512, 1024, 4096, 65536, 1_000_000])
    def test_reduce_sum_accuracy(self, n):
        # Test integer values where summation order does not introduce float drift
        if n <= 1024:
            x = torch.randint(-10, 10, (n,), device="cuda", dtype=torch.float32)
            out = ck.reduce_sum(x)
            ref = torch.sum(x)
            assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
        else:
            x = torch.randn(n, device="cuda", dtype=torch.float32)
            out = ck.reduce_sum(x)
            ref = torch.sum(x)
            # Parallel block reduction has different addition order than PyTorch pairwise summation
            assert torch.allclose(out, ref, atol=1e-2, rtol=1e-3)

    @pytest.mark.parametrize("n", [1, 2, 7, 32, 100, 256, 512, 1024, 4096, 65536, 1_000_000])
    def test_reduce_max_accuracy(self, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out = ck.reduce_max(x)
        ref = torch.max(x)
        assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("M,K,N", [
        (1, 1, 1),
        (32, 32, 32),
        (37, 53, 19),      # Non-power-of-two, non-square
        (64, 128, 96),     # Tile multiples
        (65, 33, 97),      # Tile + 1 tails
        (128, 64, 256),
        (256, 256, 256),
    ])
    def test_matmul_accuracy(self, M, K, N):
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        out = ck.matmul(a, b)
        ref = torch.matmul(a, b)
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


# ============================================================================
# 4. MEMORY ALIGNMENT & SAFE SCALAR FALLBACK TESTS
# ============================================================================

class TestMemoryAlignment:
    """Verifies that non-16-byte aligned pointers fall back safely to scalar execution."""

    def test_unaligned_vector_add(self):
        # Base tensor aligned, but slice x[1:] has offset of 4 bytes (not 16-byte aligned)
        raw_a = torch.randn(129, device="cuda", dtype=torch.float32)
        raw_b = torch.randn(129, device="cuda", dtype=torch.float32)
        a_unaligned = raw_a[1:]  # length 128, unaligned pointer
        b_unaligned = raw_b[1:]  # length 128, unaligned pointer

        out = ck.vector_add(a_unaligned, b_unaligned)
        ref = a_unaligned + b_unaligned
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_unaligned_scale(self):
        raw = torch.randn(131, device="cuda", dtype=torch.float32)
        x_unaligned = raw[3:]  # length 128, offset 12 bytes
        out = ck.scale(x_unaligned, 2.0)
        ref = x_unaligned * 2.0
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_unaligned_activations(self):
        raw = torch.randn(257, device="cuda", dtype=torch.float32)
        x_unaligned = raw[1:]
        # ReLU
        out_relu = ck.relu(x_unaligned)
        assert torch.allclose(out_relu, torch.relu(x_unaligned), atol=1e-6, rtol=1e-6)
        # GELU
        out_gelu = ck.gelu(x_unaligned)
        assert torch.allclose(out_gelu, F.gelu(x_unaligned, approximate="tanh"), atol=1e-4, rtol=1e-4)
        # Sigmoid
        out_sig = ck.sigmoid(x_unaligned)
        assert torch.allclose(out_sig, torch.sigmoid(x_unaligned), atol=1e-4, rtol=1e-4)


# ============================================================================
# 5. EDGE CASES & SPECIAL VALUES
# ============================================================================

class TestEdgeCases:
    """Verifies empty tensors, NaNs, infinities, and extreme bounds."""

    def test_empty_tensor_elementwise(self):
        empty = torch.empty(0, device="cuda", dtype=torch.float32)
        assert ck.vector_add(empty, empty).numel() == 0
        assert ck.scale(empty, 2.0).numel() == 0
        assert ck.relu(empty).numel() == 0
        assert ck.gelu(empty).numel() == 0
        assert ck.sigmoid(empty).numel() == 0
        assert ck.fused_add_relu(empty, empty).numel() == 0

    def test_empty_tensor_reduction(self):
        empty = torch.empty(0, device="cuda", dtype=torch.float32)
        s = ck.reduce_sum(empty)
        assert s.item() == 0.0

        with pytest.raises(RuntimeError, match="Cannot compute reduce_max on an empty tensor"):
            ck.reduce_max(empty)

    def test_empty_tensor_matmul(self):
        a = torch.empty(0, 32, device="cuda", dtype=torch.float32)
        b = torch.empty(32, 16, device="cuda", dtype=torch.float32)
        c = ck.matmul(a, b)
        assert c.shape == (0, 16)

        a2 = torch.empty(16, 0, device="cuda", dtype=torch.float32)
        b2 = torch.empty(0, 32, device="cuda", dtype=torch.float32)
        c2 = ck.matmul(a2, b2)
        assert c2.shape == (16, 32)

    def test_special_floats_nan_inf(self):
        x = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1e20, -1e20],
                         device="cuda", dtype=torch.float32)
        # ReLU: relu(inf)=inf, relu(-inf)=0
        out_relu = ck.relu(x)
        assert torch.isinf(out_relu[0]) and out_relu[0] > 0
        assert out_relu[1] == 0.0
        assert torch.isnan(out_relu[2])

        # Sigmoid: sigmoid(inf)=1, sigmoid(-inf)=0
        out_sig = ck.sigmoid(x)
        assert out_sig[0] == 1.0
        assert out_sig[1] == 0.0
        assert torch.isnan(out_sig[2])

        # Max reduction with infinities
        finite_and_inf = torch.tensor([-100.0, 50.0, float("inf"), 20.0], device="cuda", dtype=torch.float32)
        assert ck.reduce_max(finite_and_inf).item() == float("inf")

        neg_infs = torch.tensor([float("-inf"), -1e10], device="cuda", dtype=torch.float32)
        assert ck.reduce_max(neg_infs).item() == -1e10


# ============================================================================
# 6. NEGATIVE & ERROR HANDLING TESTS
# ============================================================================

class TestNegativeAndErrorHandling:
    """Verifies that invalid inputs raise descriptive RuntimeErrors."""

    def test_reject_cpu_tensors(self):
        a_cpu = torch.randn(10, dtype=torch.float32)
        b_cpu = torch.randn(10, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            ck.vector_add(a_cpu, b_cpu)

        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            ck.relu(a_cpu)

        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            ck.reduce_sum(a_cpu)

        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            ck.matmul(a_cpu, b_cpu)

    def test_reject_non_float32(self):
        a_f64 = torch.randn(10, device="cuda", dtype=torch.float64)
        b_f64 = torch.randn(10, device="cuda", dtype=torch.float64)
        with pytest.raises(RuntimeError, match="must be float32"):
            ck.vector_add(a_f64, b_f64)

        x_int = torch.randint(0, 10, (10,), device="cuda", dtype=torch.int32)
        with pytest.raises(RuntimeError, match="must be float32"):
            ck.relu(x_int)

    def test_reject_non_contiguous(self):
        a = torch.randn(16, 16, device="cuda", dtype=torch.float32)
        a_non_contig = a.t()  # Transpose is non-contiguous
        assert not a_non_contig.is_contiguous()

        with pytest.raises(RuntimeError, match="must be contiguous"):
            ck.relu(a_non_contig)

    def test_reject_shape_mismatch(self):
        a = torch.randn(10, device="cuda", dtype=torch.float32)
        b = torch.randn(20, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="must have the same shape"):
            ck.vector_add(a, b)

    def test_reject_matmul_dimension_mismatch(self):
        a = torch.randn(16, 32, device="cuda", dtype=torch.float32)
        b = torch.randn(64, 16, device="cuda", dtype=torch.float32)  # 32 != 64
        with pytest.raises(RuntimeError, match="Matrix dimension mismatch"):
            ck.matmul(a, b)

    def test_reject_invalid_activation_name(self):
        x = torch.randn(16, 16, device="cuda", dtype=torch.float32)
        bias = torch.randn(16, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Unsupported activation type"):
            ck.fused_bias_activation(x, bias, "unsupported_activation")
