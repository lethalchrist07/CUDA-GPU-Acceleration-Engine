"""
tests/test_cuda_portability.py
Comprehensive Test Suite for BLK-02 CUDA Portability & Toolchain Remediation.

Validates:
1. CUDA_PATH present detection.
2. CUDA_PATH absent fallback to CUDA_PATH_V* / nvcc in PATH / standard directories.
3. CUDA Toolkit unavailable graceful handling (no crash, safe diagnostics).
4. MSVC compiler unavailable graceful handling (vswhere absent, cl.exe absent).
5. PyTorch CUDA runtime & architecture detection (Ampere sm_86, Ada sm_89, Blackwell).
6. CPU-only environment forward & backward execution with PyTorch ops.
7. Native compilation failure recovery & fallback activation.
8. Numerical accuracy of all 18 fallback operations vs PyTorch references.
9. Unrelated module imports (cifar10_mlp, orchestrator, training routes) succeed cleanly
   even when compiler toolchains are completely absent.
"""

import os
import shutil
import unittest.mock as mock
from pathlib import Path
from typing import Dict, Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_kernels as ck
import custom_kernels.loader as loader
import custom_kernels.autograd as ck_auto
from engine.training.dispatch_tracker import KernelDispatchTracker


@pytest.fixture(autouse=True)
def clean_loader_cache():
    """Ensure clean loader cache for each test run."""
    orig_cached = loader._CACHED_MODULE
    orig_status = loader._TOOLCHAIN_STATUS
    yield
    loader._CACHED_MODULE = orig_cached
    loader._TOOLCHAIN_STATUS = orig_status
    ck._cuda_module = orig_cached


# ============================================================================
# 1. Toolchain Detection Tests
# ============================================================================

class TestCUDAToolchainDetection:
    """Verifies portable toolchain discovery across diverse Windows configurations."""

    def test_cuda_path_present(self, tmp_path):
        """Verify find_cuda_toolkit detects valid CUDA_PATH."""
        fake_cuda = tmp_path / "fake_cuda_12"
        (fake_cuda / "bin").mkdir(parents=True)
        (fake_cuda / "include").mkdir(parents=True)
        (fake_cuda / "bin" / "nvcc.exe").write_text("fake nvcc")
        (fake_cuda / "include" / "cuda.h").write_text("// fake cuda.h")

        with mock.patch.dict(os.environ, {"CUDA_PATH": str(fake_cuda)}, clear=False):
            found = loader.find_cuda_toolkit()
            assert found is not None
            assert Path(found) == fake_cuda

    def test_cuda_path_absent_fallback_to_version_env(self, tmp_path):
        """Verify discovery falls back to CUDA_PATH_V* when CUDA_PATH is absent."""
        fake_cuda_v13 = tmp_path / "cuda_v13_4"
        (fake_cuda_v13 / "bin").mkdir(parents=True)
        (fake_cuda_v13 / "include").mkdir(parents=True)
        (fake_cuda_v13 / "bin" / "nvcc.exe").write_text("fake nvcc")
        (fake_cuda_v13 / "include" / "cuda.h").write_text("// fake cuda.h")

        env_mock = {
            "CUDA_PATH_V13_4": str(fake_cuda_v13),
        }
        with mock.patch.dict(os.environ, env_mock, clear=True), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch.object(torch.version, "cuda", "13.4"):
            found = loader.find_cuda_toolkit()
            assert found is not None
            assert Path(found) == fake_cuda_v13

    def test_cuda_toolkit_unavailable_graceful(self):
        """Verify find_cuda_toolkit returns None without throwing when no CUDA is present."""
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch("pathlib.Path.is_dir", return_value=False):
            found = loader.find_cuda_toolkit()
            assert found is None

            loader.reset_cached_module()
            status = loader.configure_process_toolchain()
            assert status["is_compilation_ready"] is False
            assert status["cuda_dir"] is None
            assert any("not detected" in d for d in status["diagnostics"])

    def test_msvc_unavailable_graceful(self):
        """Verify find_msvc_environment returns (None, None, None) without throwing."""
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(loader, "query_vswhere", return_value=[]), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch("pathlib.Path.is_dir", return_value=False):
            msvc_bin, vcvars_bat, env_dict = loader.find_msvc_environment()
            assert msvc_bin is None
            assert vcvars_bat is None
            assert env_dict is None

            loader.reset_cached_module()
            status = loader.configure_process_toolchain()
            assert status["is_compilation_ready"] is False
            assert status["msvc_bin"] is None


# ============================================================================
# 2. Hardware Architecture & PyTorch CUDA Runtime Tests
# ============================================================================

class TestHardwareArchitectureDetection:
    """Verifies GPU architecture targeting across Ampere, Ada Lovelace, and Blackwell."""

    def test_ampere_rtx3050_targeting(self):
        """Verify sm_86 architecture generation for RTX 3050 / RTX 3060."""
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)):
            arch = loader.detect_gpu_architecture()
            assert arch["arch_list"] == "8.6"
            assert arch["arch_code"] == "compute_86"
            assert arch["sm_code"] == "sm_86"
            assert "Ampere" in arch["family"]
            assert any("-gencode=arch=compute_86,code=sm_86" in f for f in arch["extra_cuda_cflags"])

    def test_ada_rtx4050_targeting(self):
        """Verify sm_89 architecture generation for RTX 4050 / RTX 4060."""
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_capability", return_value=(8, 9)):
            arch = loader.detect_gpu_architecture()
            assert arch["arch_list"] == "8.9"
            assert arch["arch_code"] == "compute_89"
            assert arch["sm_code"] == "sm_89"
            assert "Ada Lovelace" in arch["family"]
            assert any("-gencode=arch=compute_89,code=sm_89" in f for f in arch["extra_cuda_cflags"])

    def test_blackwell_rtx5090_targeting(self):
        """Verify sm_100 / sm_120 architecture generation for Blackwell."""
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            arch = loader.detect_gpu_architecture()
            assert arch["arch_list"] == "10.0"
            assert arch["arch_code"] == "compute_100"
            assert arch["sm_code"] == "sm_100"
            assert "Blackwell" in arch["family"]

    def test_cuda_unavailable_defaults_safely(self):
        """Verify safe default targeting when CUDA device is absent."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            arch = loader.detect_gpu_architecture()
            assert arch["arch_list"] == "8.6"
            assert "Ampere" in arch["family"]


# ============================================================================
# 3. Native Compilation Failure & Fallback Activation Tests
# ============================================================================

class TestNativeCompilationFailureRecovery:
    """Verifies that compilation or toolchain failures gracefully activate PyTorch fallbacks."""

    def test_native_compilation_failure_activates_fallback(self):
        """Simulate torch.utils.cpp_extension failure; verify fallback module returns."""
        loader.reset_cached_module()

        # Mock toolchain as ready, but cpp_extension.load raising an error
        fake_status = {
            "is_compilation_ready": True,
            "cuda_dir": "C:/fake/cuda",
            "msvc_bin": "C:/fake/msvc",
            "vcvars_bat": "C:/fake/vcvars64.bat",
            "architecture": loader.detect_gpu_architecture(),
            "diagnostics": [],
        }

        with mock.patch.object(loader, "configure_process_toolchain", return_value=fake_status), \
             mock.patch("torch.utils.cpp_extension.load", side_effect=RuntimeError("fatal error C1189: MSVC compiler error")):
            mod = loader.load_custom_kernels_module(allow_fallback=True)
            assert isinstance(mod, loader.FallbackCUDAKernelModule)
            assert mod.is_fallback is True
            assert mod.is_native is False

    def test_native_compilation_failure_raises_when_fallback_disallowed(self):
        """Verify allow_fallback=False re-raises compilation exception."""
        loader.reset_cached_module()
        fake_status = {
            "is_compilation_ready": True,
            "cuda_dir": "C:/fake/cuda",
            "msvc_bin": "C:/fake/msvc",
            "vcvars_bat": "C:/fake/vcvars64.bat",
            "architecture": loader.detect_gpu_architecture(),
            "diagnostics": [],
        }

        with mock.patch.object(loader, "configure_process_toolchain", return_value=fake_status), \
             mock.patch("torch.utils.cpp_extension.load", side_effect=RuntimeError("Compilation error")):
            with pytest.raises(RuntimeError, match="Compilation error"):
                loader.load_custom_kernels_module(allow_fallback=False)


# ============================================================================
# 4. Pure PyTorch Fallback Numerical Accuracy Tests (18 Operators)
# ============================================================================

class TestFallbackNumericalAccuracy:
    """Verifies that FallbackCUDAKernelModule accurately implements all 18 operations."""

    @pytest.fixture
    def fb(self):
        return loader.FallbackCUDAKernelModule()

    def test_fallback_elementwise_accuracy(self, fb):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(64, 64, device=dev, dtype=torch.float32)
        b = torch.randn(64, 64, device=dev, dtype=torch.float32)

        # vector_add
        out_add = fb.vector_add(a, b)
        assert torch.allclose(out_add, a + b, atol=1e-6)

        # vector_add_backward
        grad_out = torch.randn_like(out_add)
        ga, gb = fb.vector_add_backward(grad_out)
        assert torch.allclose(ga, grad_out) and torch.allclose(gb, grad_out)

        # vector_scale
        out_scale = fb.vector_scale(a, 2.5)
        assert torch.allclose(out_scale, a * 2.5, atol=1e-6)

        # vector_scale_backward
        gx = fb.vector_scale_backward(grad_out, 2.5)
        assert torch.allclose(gx, grad_out * 2.5, atol=1e-6)

        # fused_add_relu
        out_fused = fb.fused_add_relu(a, b)
        assert torch.allclose(out_fused, torch.relu(a + b), atol=1e-6)

        # fused_add_relu_backward
        gfa, gfb = fb.fused_add_relu_backward(grad_out, out_fused)
        expected_g = grad_out * (out_fused > 0).float()
        assert torch.allclose(gfa, expected_g) and torch.allclose(gfb, expected_g)

    def test_fallback_activations_accuracy(self, fb):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(32, 32, device=dev, dtype=torch.float32)
        grad_out = torch.randn_like(x)

        # relu & backward
        out_relu = fb.relu(x)
        assert torch.allclose(out_relu, torch.relu(x))
        gr = fb.relu_backward(grad_out, x)
        assert torch.allclose(gr, grad_out * (x > 0).float())

        # sigmoid & backward
        out_sig = fb.sigmoid(x)
        assert torch.allclose(out_sig, torch.sigmoid(x))
        gs = fb.sigmoid_backward(grad_out, out_sig)
        assert torch.allclose(gs, grad_out * out_sig * (1.0 - out_sig))

        # gelu & backward
        out_gelu = fb.gelu(x)
        expected_gelu = F.gelu(x, approximate="tanh")
        assert torch.allclose(out_gelu, expected_gelu, atol=1e-6)

        x_det = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            ref_y = F.gelu(x_det, approximate="tanh")
            ref_grad = torch.autograd.grad(ref_y, x_det, grad_out)[0]
        gg = fb.gelu_backward(grad_out, x)
        assert torch.allclose(gg, ref_grad, atol=1e-6)

    def test_fallback_fused_bias_activation(self, fb):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(16, 32, device=dev, dtype=torch.float32)
        bias = torch.randn(32, device=dev, dtype=torch.float32)
        grad_out = torch.randn_like(x)

        # ReLU forward & backward
        out_relu = fb.fused_bias_activation(x, bias, "relu")
        assert torch.allclose(out_relu, torch.relu(x + bias), atol=1e-6)
        gx, gb = fb.fused_bias_activation_backward(grad_out, x, bias, "relu")
        expected_gx = grad_out * ((x + bias) > 0).float()
        expected_gb = expected_gx.sum(dim=0)
        assert torch.allclose(gx, expected_gx, atol=1e-6)
        assert torch.allclose(gb, expected_gb, atol=1e-6)

        # GELU forward & backward
        out_gelu = fb.fused_bias_activation(x, bias, "gelu")
        assert torch.allclose(out_gelu, F.gelu(x + bias, approximate="tanh"), atol=1e-6)

    def test_fallback_matmul_and_reductions(self, fb):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(16, 32, device=dev, dtype=torch.float32)
        b = torch.randn(32, 24, device=dev, dtype=torch.float32)

        # matmul_tiled
        out_mm = fb.matmul_tiled(a, b)
        assert torch.allclose(out_mm, a @ b, atol=1e-5)

        # reduce_sum & reduce_max
        x = torch.randn(100, device=dev, dtype=torch.float32)
        assert torch.allclose(fb.reduce_sum(x), x.sum(), atol=1e-5)
        assert torch.allclose(fb.reduce_max(x), x.max(), atol=1e-5)


# ============================================================================
# 5. CPU-Only Environment Full Pipeline Execution Tests
# ============================================================================

class TestCPUOnlyEnvironmentExecution:
    """Verifies that in a CPU-only environment, models train and infer without CUDA."""

    def test_cpu_only_autograd_training_step(self):
        """Simulate non-CUDA system; verify CustomLinear runs full forward and backward pass."""
        loader.reset_cached_module()

        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch.dict(os.environ, {"NEXORA_FORCE_FALLBACK": "1"}):
            # Model with custom autograd layers
            linear = ck_auto.CustomLinear(16, 8, activation="gelu")
            x = torch.randn(4, 16, requires_grad=True)

            out = linear(x)
            assert out.shape == (4, 8)

            loss = out.sum()
            loss.backward()

            assert linear.weight.grad is not None
            assert linear.bias.grad is not None
            assert not torch.isnan(linear.weight.grad).any()


# ============================================================================
# 6. Fallback Accounting & No-False-Claim Verification
# ============================================================================

class TestFallbackAccountingIntegrity:
    """Verifies that fallback usage is never falsely claimed as custom CUDA execution."""

    def test_tracker_marks_is_custom_dispatched_false_on_fallback(self):
        """Verify KernelDispatchTracker reports is_custom_dispatched=False when in fallback mode."""
        loader.reset_cached_module()

        with mock.patch.dict(os.environ, {"NEXORA_FORCE_FALLBACK": "1"}):
            fb_mod = loader.load_custom_kernels_module()
            assert fb_mod.is_fallback is True

            tracker = KernelDispatchTracker()
            with tracker:
                a = torch.randn(10, device="cuda" if torch.cuda.is_available() else "cpu")
                b = torch.randn(10, device="cuda" if torch.cuda.is_available() else "cpu")
                _ = ck.vector_add(a, b)

            summary = tracker.get_summary()
            assert summary["is_fallback_active"] is True
            assert summary["is_native_custom"] is False
            # Crucial requirement: Must NOT claim custom kernel executed when fallback was used
            assert summary["is_custom_dispatched"] is False


# ============================================================================
# 7. Unrelated Module Import Isolation Tests
# ============================================================================

class TestUnrelatedModuleImportIsolation:
    """Verifies that if toolchains are absent, unrelated application modules import cleanly."""

    def test_import_cifar10_mlp_and_orchestrator_without_toolchain(self):
        """Verify CIFAR10MLP and TrainingOrchestrator import and initialize without toolchains."""
        loader.reset_cached_module()

        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch.object(loader, "find_cuda_toolkit", return_value=None), \
             mock.patch.object(loader, "find_msvc_environment", return_value=(None, None, None)):
            
            # 1. Import CIFAR10MLP
            from engine.models.cifar10_mlp import CIFAR10MLP
            mlp = CIFAR10MLP(backend="CPU_NATIVE", input_dim=32, hidden_dims=[16, 8, 4], num_classes=2)
            x = torch.randn(2, 32)
            out = mlp(x)
            assert out.shape == (2, 2)

            # 2. Import TrainingOrchestrator
            from engine.training.orchestrator import TrainingOrchestrator
            models = TrainingOrchestrator.list_models()
            assert len(models) > 0

            # 3. Import backend training router
            from backend.routes.training import router
            assert router.prefix == "/api/training"
