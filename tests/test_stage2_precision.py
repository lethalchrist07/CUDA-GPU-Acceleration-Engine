"""
Phase 10 Stage 2 Milestone 1: Mixed-Precision CUDA Test Suite
Validates FP16 and BF16 precision, vectorization, odd lengths, alignment fallbacks,
special values, stream execution, numerical stability, and memory safety on RTX 3050 (sm_86).
"""
import pytest
import torch
import torch.nn.functional as F
import math

import custom_kernels as ck

# Device verification fixture
@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for custom kernel precision tests")
    torch.cuda.empty_cache()

# =========================================================================
# 1. FP16 Elementwise & Fused Kernels
# =========================================================================

class TestFP16Elementwise:
    """Correctness and alignment tests for FP16 elementwise operations."""

    @pytest.mark.parametrize("size", [1, 3, 7, 8, 15, 16, 63, 64, 1023, 1024, 65535, 65536, 131072])
    def test_vector_add_fp16(self, size):
        a = torch.randn(size, dtype=torch.float16, device="cuda")
        b = torch.randn(size, dtype=torch.float16, device="cuda")
        expected = a + b
        actual = ck.vector_add(a, b)
        assert actual.dtype == torch.float16
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    def test_vector_add_fp16_unaligned(self):
        # Slice to induce non-16-byte memory alignment
        base_a = torch.randn(2048, dtype=torch.float16, device="cuda")
        base_b = torch.randn(2048, dtype=torch.float16, device="cuda")
        for offset in [1, 2, 3, 5, 7]:
            a = base_a[offset : offset + 1024].contiguous()
            b = base_b[offset : offset + 1024].contiguous()
            expected = a + b
            actual = ck.vector_add(a, b)
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -1.0, 2.5, -0.125, 13.75])
    @pytest.mark.parametrize("size", [5, 16, 127, 2048])
    def test_vector_scale_fp16(self, alpha, size):
        x = torch.randn(size, dtype=torch.float16, device="cuda")
        expected = x * alpha
        actual = ck.scale(x, alpha)
        assert actual.dtype == torch.float16
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("size", [1, 9, 32, 255, 1024, 32768])
    def test_fused_add_relu_fp16(self, size):
        a = torch.randn(size, dtype=torch.float16, device="cuda")
        b = torch.randn(size, dtype=torch.float16, device="cuda")
        expected = F.relu(a + b)
        actual = ck.fused_add_relu(a, b)
        assert actual.dtype == torch.float16
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("activation", ["none", "relu", "gelu"])
    @pytest.mark.parametrize("shape", [(16, 10), (32, 64), (128, 512), (17, 33), (1, 10)])
    def test_fused_bias_activation_fp16(self, activation, shape):
        rows, cols = shape
        x = torch.randn(rows, cols, dtype=torch.float16, device="cuda")
        bias = torch.randn(cols, dtype=torch.float16, device="cuda")

        lin = x + bias
        if activation == "relu":
            expected = F.relu(lin)
        elif activation == "gelu":
            expected = F.gelu(lin, approximate="tanh")
        else:
            expected = lin

        actual = ck.fused_bias_activation(x, bias, activation=activation)
        assert actual.dtype == torch.float16
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


# =========================================================================
# 2. BF16 Elementwise & Fused Kernels
# =========================================================================

class TestBF16Elementwise:
    """Correctness and alignment tests for BF16 elementwise operations."""

    @pytest.mark.parametrize("size", [1, 3, 7, 8, 15, 16, 63, 64, 1023, 1024, 65535, 65536, 131072])
    def test_vector_add_bf16(self, size):
        a = torch.randn(size, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(size, dtype=torch.bfloat16, device="cuda")
        expected = a + b
        actual = ck.vector_add(a, b)
        assert actual.dtype == torch.bfloat16
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)

    def test_vector_add_bf16_unaligned(self):
        base_a = torch.randn(2048, dtype=torch.bfloat16, device="cuda")
        base_b = torch.randn(2048, dtype=torch.bfloat16, device="cuda")
        for offset in [1, 2, 3, 5, 7]:
            a = base_a[offset : offset + 1024].contiguous()
            b = base_b[offset : offset + 1024].contiguous()
            expected = a + b
            actual = ck.vector_add(a, b)
            torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -1.0, 2.5, -0.125, 13.75])
    @pytest.mark.parametrize("size", [5, 16, 127, 2048])
    def test_vector_scale_bf16(self, alpha, size):
        x = torch.randn(size, dtype=torch.bfloat16, device="cuda")
        expected = x * alpha
        actual = ck.scale(x, alpha)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("size", [1, 9, 32, 255, 1024, 32768])
    def test_fused_add_relu_bf16(self, size):
        a = torch.randn(size, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(size, dtype=torch.bfloat16, device="cuda")
        expected = F.relu(a + b)
        actual = ck.fused_add_relu(a, b)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("activation", ["none", "relu", "gelu"])
    @pytest.mark.parametrize("shape", [(16, 10), (32, 64), (128, 512), (17, 33), (1, 10)])
    def test_fused_bias_activation_bf16(self, activation, shape):
        rows, cols = shape
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(cols, dtype=torch.bfloat16, device="cuda")

        lin = x + bias
        if activation == "relu":
            expected = F.relu(lin)
        elif activation == "gelu":
            expected = F.gelu(lin, approximate="tanh")
        else:
            expected = lin

        actual = ck.fused_bias_activation(x, bias, activation=activation)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


# =========================================================================
# 3. Activations (FP16 & BF16)
# =========================================================================

class TestMixedPrecisionActivations:
    """Correctness tests for activations in FP16 and BF16."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [1, 5, 8, 17, 64, 1023, 2048, 65536])
    def test_relu(self, dtype, size):
        x = torch.randn(size, dtype=dtype, device="cuda") * 5.0
        expected = F.relu(x)
        actual = ck.relu(x)
        assert actual.dtype == dtype
        tol = 1e-3 if dtype == torch.float16 else 5e-3
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [1, 5, 8, 17, 64, 1023, 2048, 65536])
    def test_gelu(self, dtype, size):
        x = torch.randn(size, dtype=dtype, device="cuda") * 3.0
        expected = F.gelu(x, approximate="tanh")
        actual = ck.gelu(x)
        assert actual.dtype == dtype
        tol = 2e-3 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [1, 5, 8, 17, 64, 1023, 2048, 65536])
    def test_sigmoid(self, dtype, size):
        x = torch.randn(size, dtype=dtype, device="cuda") * 4.0
        expected = torch.sigmoid(x)
        actual = ck.sigmoid(x)
        assert actual.dtype == dtype
        tol = 2e-3 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


# =========================================================================
# 4. Reductions & FP32 Accumulation Stability
# =========================================================================

class TestMixedPrecisionReductions:
    """Reductions testing with emphasis on FP32 accumulator stability and overflow prevention."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [1, 10, 100, 1024, 10000])
    def test_reduce_sum_small(self, dtype, size):
        x = torch.randn(size, dtype=dtype, device="cuda")
        # Compare against FP32 accumulation
        expected = torch.sum(x.float())
        actual = ck.reduce_sum(x, output_fp32=True)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-3)

    def test_reduce_sum_fp16_overflow_prevention(self):
        """Verify that summing 100,000 ones in FP16 does not overflow to inf (65504 limit)."""
        N = 100000
        x = torch.ones(N, dtype=torch.float16, device="cuda")

        # Native FP16 sum without accumulator would overflow/saturate:
        native_half_sum = torch.sum(x)
        assert torch.isinf(native_half_sum), "Expected naive half sum of 100k ones to overflow FP16 range"

        # Custom kernel with FP32 accumulator returning FP32:
        acc_sum = ck.reduce_sum(x, output_fp32=True)
        assert acc_sum.dtype == torch.float32
        assert not torch.isinf(acc_sum)
        assert math.isclose(acc_sum.item(), 100000.0, rel_tol=1e-4)

        # Auto-promotion when output_fp32=None for large tensor:
        auto_sum = ck.reduce_sum(x)
        assert auto_sum.dtype == torch.float32
        assert math.isclose(auto_sum.item(), 100000.0, rel_tol=1e-4)

    def test_reduce_sum_bf16_large(self):
        """Verify BF16 large reduction with FP32 accumulator."""
        N = 100000
        x = torch.ones(N, dtype=torch.bfloat16, device="cuda")
        acc_sum = ck.reduce_sum(x, output_fp32=True)
        assert acc_sum.dtype == torch.float32
        assert math.isclose(acc_sum.item(), 100000.0, rel_tol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [1, 7, 64, 1023, 4096, 65536])
    def test_reduce_max_mixed(self, dtype, size):
        x = torch.randn(size, dtype=dtype, device="cuda")
        expected = torch.max(x)
        actual = ck.reduce_max(x)
        assert actual.dtype == dtype
        tol = 1e-3 if dtype == torch.float16 else 5e-3
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_reduce_max_all_negative(self, dtype):
        x = -torch.rand(1000, dtype=dtype, device="cuda") - 10.0
        expected = torch.max(x)
        actual = ck.reduce_max(x)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected)


# =========================================================================
# 5. Special Floating-Point Values & Boundary Conditions
# =========================================================================

class TestSpecialFloatingPointValues:
    """Tests handling of NaN, Inf, signed zeros, subnormals, and empty inputs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_nan_handling_relu(self, dtype):
        x = torch.tensor([1.0, float("nan"), -2.0, float("nan")], dtype=dtype, device="cuda")
        out = ck.relu(x)
        assert out[0] == 1.0
        assert torch.isnan(out[1])
        assert out[2] == 0.0
        assert torch.isnan(out[3])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_inf_handling_relu(self, dtype):
        x = torch.tensor([float("inf"), float("-inf"), 0.0], dtype=dtype, device="cuda")
        out = ck.relu(x)
        assert torch.isinf(out[0]) and out[0] > 0
        assert out[1] == 0.0
        assert out[2] == 0.0

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_signed_zeros_scale(self, dtype):
        x = torch.tensor([0.0, -0.0], dtype=dtype, device="cuda")
        out = ck.scale(x, -1.0)
        # Verify no kernel exception or NaN creation
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_empty_tensor_handling(self, dtype):
        empty = torch.empty(0, dtype=dtype, device="cuda")
        assert ck.vector_add(empty, empty).numel() == 0
        assert ck.scale(empty, 2.0).numel() == 0
        assert ck.fused_add_relu(empty, empty).numel() == 0
        assert ck.relu(empty).numel() == 0
        assert ck.gelu(empty).numel() == 0
        assert ck.sigmoid(empty).numel() == 0
        assert ck.reduce_sum(empty).item() == 0.0

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_empty_reduce_max_raises(self, dtype):
        empty = torch.empty(0, dtype=dtype, device="cuda")
        with pytest.raises(RuntimeError, match="Cannot compute reduce_max on an empty tensor"):
            ck.reduce_max(empty)


# =========================================================================
# 6. CUDA Stream Integration
# =========================================================================

class TestCUDAStreamIntegration:
    """Validates that operations execute correctly on active non-default CUDA streams."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_default_stream_execution(self, dtype):
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        with torch.cuda.stream(s1):
            a = torch.randn(10000, dtype=dtype, device="cuda")
            b = torch.randn(10000, dtype=dtype, device="cuda")
            out1 = ck.vector_add(a, b)

        with torch.cuda.stream(s2):
            c = torch.randn(10000, dtype=dtype, device="cuda")
            out2 = ck.relu(c)

        s1.synchronize()
        s2.synchronize()

        tol = 1e-3 if dtype == torch.float16 else 5e-3
        torch.testing.assert_close(out1, a + b, atol=tol, rtol=tol)
        torch.testing.assert_close(out2, F.relu(c), atol=tol, rtol=tol)


# =========================================================================
# 7. Input Validation & Forward-Only Contract
# =========================================================================

class TestInputValidationAndContract:
    """Verifies strict error handling for mismatched types, devices, and autograd contract."""

    def test_dtype_mismatch_raises(self):
        a = torch.randn(100, dtype=torch.float16, device="cuda")
        b = torch.randn(100, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError, match="must have the same dtype"):
            ck.vector_add(a, b)

    def test_cpu_tensor_raises(self):
        a = torch.randn(100, dtype=torch.float16, device="cpu")
        b = torch.randn(100, dtype=torch.float16, device="cpu")
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            ck.vector_add(a, b)

    def test_non_contiguous_raises(self):
        base = torch.randn(10, 10, dtype=torch.float16, device="cuda")
        non_contig = base.t()  # Transposed tensor is non-contiguous
        with pytest.raises(RuntimeError, match="must be contiguous"):
            ck.relu(non_contig)

    def test_shape_mismatch_raises(self):
        a = torch.randn(100, dtype=torch.float16, device="cuda")
        b = torch.randn(50, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="must have the same shape"):
            ck.vector_add(a, b)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_only_contract(self, dtype):
        """Confirm that kernels operate without constructing autograd backward graph."""
        x = torch.randn(100, dtype=dtype, device="cuda", requires_grad=True)
        out = ck.relu(x)
        assert out.grad_fn is None, "Custom kernel must not attach autograd graph in forward-only Stage 2 Milestone 1"


# =========================================================================
# 8. Memory Safety & Leak Verification
# =========================================================================

class TestMemorySafety:
    """Verifies that no VRAM is leaked over repeated invocations."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_zero_vram_leak_over_iterations(self, dtype):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        a = torch.randn(50000, dtype=dtype, device="cuda")
        b = torch.randn(50000, dtype=dtype, device="cuda")

        # Warmup and cleanly delete warmup temporary
        warmup_out = ck.vector_add(a, b)
        del warmup_out
        torch.cuda.synchronize()

        initial_allocated = torch.cuda.memory_allocated()

        # Run 100 iterations of various mixed-precision operations
        for _ in range(100):
            c = ck.vector_add(a, b)
            d = ck.fused_add_relu(c, a)
            e = ck.scale(d, 0.5)
            f = ck.relu(e)
            g = ck.gelu(f)
            h = ck.sigmoid(g)
            s = ck.reduce_sum(h)
            m = ck.reduce_max(h)
            del c, d, e, f, g, h, s, m

        torch.cuda.synchronize()
        final_allocated = torch.cuda.memory_allocated()

        # Verify memory returned exactly to baseline
        assert final_allocated == initial_allocated, (
            f"VRAM leak detected! Initial: {initial_allocated} bytes, Final: {final_allocated} bytes"
        )

