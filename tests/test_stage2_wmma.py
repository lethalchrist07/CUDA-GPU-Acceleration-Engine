"""
Phase 10 Stage 2 Milestone 2: Ampere Tensor Core WMMA Matrix Multiplication Tests
Validates real NVIDIA Ampere Tensor Core accelerated matrix multiplication using CUDA WMMA intrinsics.
Target: NVIDIA GeForce RTX 3050 Laptop GPU (sm_86).
"""
import pytest
import torch
import math

import custom_kernels as ck

# Device verification fixture
@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for custom kernel WMMA tests")
    torch.cuda.empty_cache()

# =========================================================================
# 1. Square Matrix Correctness (Powers of 2 and Multiples of 16)
# =========================================================================

class TestWMMASquare:
    """Correctness on square matrices using Tensor Core WMMA."""

    @pytest.mark.parametrize("size", [16, 32, 48, 64, 128, 256, 512, 1024])
    def test_matmul_wmma_fp16_out(self, size):
        a = torch.randn(size, size, dtype=torch.float16, device="cuda")
        b = torch.randn(size, size, dtype=torch.float16, device="cuda")
        expected = torch.matmul(a, b)
        actual = ck.matmul_wmma(a, b, output_fp32=False)

        assert actual.dtype == torch.float16
        assert actual.shape == (size, size)
        # FP16 tolerance: atol=5e-2 accommodates FP32 accumulation vs cuBLAS tree reduction
        # (for dot product of 512-1024 elements, 1 FP16 ULP at magnitude ~32 is 0.03125)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)

    @pytest.mark.parametrize("size", [16, 32, 48, 64, 128, 256, 512, 1024])
    def test_matmul_wmma_fp32_out(self, size):
        a = torch.randn(size, size, dtype=torch.float16, device="cuda")
        b = torch.randn(size, size, dtype=torch.float16, device="cuda")
        # Reference in full float32 precision
        expected = torch.matmul(a.float(), b.float())
        actual = ck.matmul_wmma(a, b, output_fp32=True)

        assert actual.dtype == torch.float32
        assert actual.shape == (size, size)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_generic_matmul_dispatch(self, size):
        # Test generic ck.matmul dispatch
        a_fp16 = torch.randn(size, size, dtype=torch.float16, device="cuda")
        b_fp16 = torch.randn(size, size, dtype=torch.float16, device="cuda")
        out_fp16 = ck.matmul(a_fp16, b_fp16)
        assert out_fp16.dtype == torch.float16
        torch.testing.assert_close(out_fp16, torch.matmul(a_fp16, b_fp16), atol=1e-2, rtol=1e-2)

        # Stage 1 FP32 tiled matmul dispatch preserved
        a_fp32 = torch.randn(size, size, dtype=torch.float32, device="cuda")
        b_fp32 = torch.randn(size, size, dtype=torch.float32, device="cuda")
        out_fp32 = ck.matmul(a_fp32, b_fp32)
        assert out_fp32.dtype == torch.float32
        torch.testing.assert_close(out_fp32, torch.matmul(a_fp32, b_fp32), atol=1e-4, rtol=1e-4)

# =========================================================================
# 2. Rectangular Matrix Correctness
# =========================================================================

class TestWMMARectangular:
    """Correctness on various rectangular matrix dimensions."""

    @pytest.mark.parametrize("M,K,N", [
        (64, 128, 256),
        (256, 64, 128),
        (512, 128, 64),
        (128, 512, 256),
        (16, 256, 16),
        (128, 16, 128),
        (256, 1024, 64),
        (64, 256, 512),
    ])
    def test_rectangular_shapes_fp16(self, M, K, N):
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        expected = torch.matmul(a, b)
        actual = ck.matmul_wmma(a, b, output_fp32=False)

        assert actual.shape == (M, N)
        assert actual.dtype == torch.float16
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("M,K,N", [
        (64, 128, 256),
        (256, 64, 128),
        (512, 128, 64),
        (128, 512, 256),
    ])
    def test_rectangular_shapes_fp32(self, M, K, N):
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        expected = torch.matmul(a.float(), b.float())
        actual = ck.matmul_wmma(a, b, output_fp32=True)

        assert actual.shape == (M, N)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

# =========================================================================
# 3. Boundary Handling, Nonmultiples of 16, and Irregular Shapes
# =========================================================================

class TestWMMABoundaryAndIrregular:
    """Tests zero-padding and boundary handling for arbitrary dimensions."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 1, 1),
        (3, 5, 7),
        (7, 11, 13),
        (15, 15, 15),
        (17, 31, 19),
        (33, 49, 65),
        (63, 63, 63),
        (65, 65, 65),
        (100, 200, 300),
        (300, 500, 200),
        (500, 250, 125),
        (99, 101, 103),
    ])
    def test_irregular_dimensions_fp16(self, M, K, N):
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        expected = torch.matmul(a, b)
        actual = ck.matmul_wmma(a, b, output_fp32=False)

        assert actual.shape == (M, N)
        assert actual.dtype == torch.float16
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("M,K,N", [
        (1, 1, 1),
        (7, 11, 13),
        (33, 49, 65),
        (100, 200, 300),
    ])
    def test_irregular_dimensions_fp32(self, M, K, N):
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        expected = torch.matmul(a.float(), b.float())
        actual = ck.matmul_wmma(a, b, output_fp32=True)

        assert actual.shape == (M, N)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    def test_empty_matrices(self):
        # M = 0
        a = torch.empty(0, 32, dtype=torch.float16, device="cuda")
        b = torch.randn(32, 64, dtype=torch.float16, device="cuda")
        out = ck.matmul_wmma(a, b)
        assert out.shape == (0, 64)

        # N = 0
        a = torch.randn(64, 32, dtype=torch.float16, device="cuda")
        b = torch.empty(32, 0, dtype=torch.float16, device="cuda")
        out = ck.matmul_wmma(a, b)
        assert out.shape == (64, 0)

        # K = 0: mathematically yields all zeros of shape (M, N)
        a = torch.empty(32, 0, dtype=torch.float16, device="cuda")
        b = torch.empty(0, 64, dtype=torch.float16, device="cuda")
        out = ck.matmul_wmma(a, b)
        assert out.shape == (32, 64)
        assert (out == 0).all()

# =========================================================================
# 4. Special Numerical Values & Mathematical Properties
# =========================================================================

class TestWMMANumericalProperties:
    """Special values, identity, zeros, and dynamic range."""

    def test_identity_multiplication(self):
        for size in [16, 64, 100, 128]:
            a = torch.randn(size, size, dtype=torch.float16, device="cuda")
            eye = torch.eye(size, dtype=torch.float16, device="cuda")
            out = ck.matmul_wmma(a, eye)
            torch.testing.assert_close(out, a, atol=1e-3, rtol=1e-3)

    def test_zero_matrix_multiplication(self):
        a = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        zero = torch.zeros(64, 64, dtype=torch.float16, device="cuda")
        out = ck.matmul_wmma(a, zero)
        assert (out == 0).all()

    @pytest.mark.parametrize("scale_factor", [1e-3, 1e-2, 0.1, 1.0, 5.0, 10.0])
    def test_dynamic_range(self, scale_factor):
        a = torch.randn(64, 64, dtype=torch.float16, device="cuda") * scale_factor
        b = torch.randn(64, 64, dtype=torch.float16, device="cuda") * scale_factor
        expected = torch.matmul(a, b)
        actual = ck.matmul_wmma(a, b)
        torch.testing.assert_close(actual, expected, atol=1e-2 * scale_factor**2, rtol=1e-2)

    def test_nan_propagation(self):
        a = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        b = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        a[10, 20] = float("nan")
        out = ck.matmul_wmma(a, b)
        # Entire row 10 of output should be NaN
        assert torch.isnan(out[10]).all()
        # Other rows should not be NaN
        assert not torch.isnan(out[0]).any()

    def test_inf_propagation(self):
        a = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        b = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        a[5, 5] = float("inf")
        out = ck.matmul_wmma(a, b)
        assert torch.isinf(out[5]).any() or torch.isnan(out[5]).any()

# =========================================================================
# 5. Input Validation & Error Handling
# =========================================================================

class TestWMMAValidation:
    """Input validation, dimension matching, device checks, and dtype safety."""

    def test_dimension_mismatch(self):
        a = torch.randn(64, 32, dtype=torch.float16, device="cuda")
        b = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="Matrix dimension mismatch"):
            ck.matmul_wmma(a, b)

    def test_1d_tensor_rejected(self):
        a = torch.randn(64, dtype=torch.float16, device="cuda")
        b = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="must be a 2D matrix"):
            ck.matmul_wmma(a, b)

    def test_3d_tensor_rejected(self):
        a = torch.randn(2, 64, 64, dtype=torch.float16, device="cuda")
        b = torch.randn(2, 64, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="must be a 2D matrix"):
            ck.matmul_wmma(a, b)

    def test_cpu_tensor_rejected(self):
        a = torch.randn(64, 64, dtype=torch.float16, device="cpu")
        b = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            ck.matmul_wmma(a, b)

    def test_non_fp16_input_rejected(self):
        # FP32 inputs to matmul_wmma should raise clear error instructing user
        a_f32 = torch.randn(64, 64, dtype=torch.float32, device="cuda")
        b_f32 = torch.randn(64, 64, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="must be float16"):
            ck.matmul_wmma(a_f32, b_f32)

        # BF16 inputs to matmul_wmma should also raise clear error
        a_bf16 = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError, match="must be float16"):
            ck.matmul_wmma(a_bf16, b_bf16)

# =========================================================================
# 6. CUDA Streams Integration
# =========================================================================

class TestWMMACUDAStreams:
    """Verification of proper PyTorch CUDA stream handling."""

    def test_custom_cuda_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a = torch.randn(128, 128, dtype=torch.float16, device="cuda")
            b = torch.randn(128, 128, dtype=torch.float16, device="cuda")
            out = ck.matmul_wmma(a, b)
            expected = torch.matmul(a, b)

        stream.synchronize()
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_multiple_streams_concurrent(self):
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        with torch.cuda.stream(s1):
            a1 = torch.randn(128, 128, dtype=torch.float16, device="cuda")
            b1 = torch.randn(128, 128, dtype=torch.float16, device="cuda")
            out1 = ck.matmul_wmma(a1, b1)
            ref1 = torch.matmul(a1, b1)

        with torch.cuda.stream(s2):
            a2 = torch.randn(128, 128, dtype=torch.float16, device="cuda")
            b2 = torch.randn(128, 128, dtype=torch.float16, device="cuda")
            out2 = ck.matmul_wmma(a2, b2)
            ref2 = torch.matmul(a2, b2)

        s1.synchronize()
        s2.synchronize()

        torch.testing.assert_close(out1, ref1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out2, ref2, atol=1e-2, rtol=1e-2)

# =========================================================================
# 7. Memory Safety & Execution Stability
# =========================================================================

class TestWMMAMemorySafety:
    """Validates absence of memory leaks and numerical determinism."""

    def test_zero_vram_leak(self):
        import gc
        a = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        b = torch.randn(256, 256, dtype=torch.float16, device="cuda")

        # Warmup
        for _ in range(5):
            _ = ck.matmul_wmma(a, b)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        base_mem = torch.cuda.memory_allocated()
        for _ in range(50):
            res = ck.matmul_wmma(a, b)
            del res
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        post_mem = torch.cuda.memory_allocated()
        assert post_mem <= base_mem, f"VRAM leak detected: baseline={base_mem}, post={post_mem}"


    def test_repeated_execution_determinism(self):
        a = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        b = torch.randn(256, 256, dtype=torch.float16, device="cuda")

        first = ck.matmul_wmma(a, b)
        for _ in range(20):
            subsequent = ck.matmul_wmma(a, b)
            torch.testing.assert_close(first, subsequent, atol=0.0, rtol=0.0)

# =========================================================================
# 8. Forward-Only Autograd Contract
# =========================================================================

class TestWMMAAutogradContract:
    """Verifies that matmul_wmma is forward-only and does not attach backward graph."""

    def test_forward_only_no_grad(self):
        a = torch.randn(64, 64, dtype=torch.float16, device="cuda", requires_grad=True)
        b = torch.randn(64, 64, dtype=torch.float16, device="cuda", requires_grad=True)

        out = ck.matmul_wmma(a, b)
        # Output must NOT have a grad_fn attached
        assert out.grad_fn is None
        assert not out.requires_grad

        # Attempting backward must fail because no autograd graph was built
        with pytest.raises(RuntimeError):
            out.sum().backward()
