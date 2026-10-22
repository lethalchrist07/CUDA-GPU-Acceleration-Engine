#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "include/kernel_common.cuh"
#include "include/elementwise.cuh"
#include "include/activations.cuh"
#include "include/reductions.cuh"
#include "include/matmul_tiled.cuh"
#include "include/matmul_wmma.cuh"

// Helper macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_SUPPORTED_DTYPE(x) TORCH_CHECK( \
    (x).scalar_type() == at::kFloat || (x).scalar_type() == at::kHalf || (x).scalar_type() == at::kBFloat16, \
    #x " must be float32, float16, or bfloat16")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_SUPPORTED_DTYPE(x); } while (0)
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK((x).device() == (y).device(), #x " and " #y " must be on the same CUDA device")
#define CHECK_SAME_DTYPE(x, y) TORCH_CHECK((x).scalar_type() == (y).scalar_type(), #x " and " #y " must have the same dtype")

// =========================================================================
// Elementwise Operations
// =========================================================================

torch::Tensor vector_add_forward(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_SAME_DEVICE(a, b);
    CHECK_SAME_DTYPE(a, b);
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors a and b must have the same shape for vector addition");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    auto out = torch::empty_like(a);
    size_t n = a.numel();
    if (n == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (a.scalar_type() == at::kFloat) {
        launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    } else if (a.scalar_type() == at::kHalf) {
        launch_vector_add_fp16(
            reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            n, stream
        );
    } else if (a.scalar_type() == at::kBFloat16) {
        launch_vector_add_bf16(
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor vector_scale_forward(const torch::Tensor& x, float alpha) {
    CHECK_INPUT(x);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty_like(x);
    size_t n = x.numel();
    if (n == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_vector_scale(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_vector_scale_fp16(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            alpha, n, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_vector_scale_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            alpha, n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor fused_add_relu_forward(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_SAME_DEVICE(a, b);
    CHECK_SAME_DTYPE(a, b);
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors a and b must have the same shape for fused add_relu");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    auto out = torch::empty_like(a);
    size_t n = a.numel();
    if (n == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (a.scalar_type() == at::kFloat) {
        launch_fused_add_relu(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    } else if (a.scalar_type() == at::kHalf) {
        launch_fused_add_relu_fp16(
            reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            n, stream
        );
    } else if (a.scalar_type() == at::kBFloat16) {
        launch_fused_add_relu_bf16(
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor fused_bias_activation_forward(const torch::Tensor& x,
                                            const torch::Tensor& bias,
                                            const std::string& activation) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    CHECK_SAME_DEVICE(x, bias);
    CHECK_SAME_DTYPE(x, bias);
    TORCH_CHECK(x.dim() == 2, "Input x must be a 2D matrix (rows x cols)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
    TORCH_CHECK(bias.size(0) == x.size(1), "Bias size must match second dimension of x");

    int act_type = 0;
    if (activation == "relu") {
        act_type = 1;
    } else if (activation == "gelu") {
        act_type = 2;
    } else if (activation == "none" || activation.empty()) {
        act_type = 0;
    } else {
        TORCH_CHECK(false, "Unsupported activation type: " + activation + ". Expected 'relu', 'gelu', or 'none'.");
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty_like(x);
    size_t rows = x.size(0);
    size_t cols = x.size(1);
    if (rows == 0 || cols == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_fused_bias_activation(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols, act_type, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_fused_bias_activation_fp16(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            rows, cols, act_type, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_fused_bias_activation_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            rows, cols, act_type, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

// =========================================================================
// Activations
// =========================================================================

torch::Tensor relu_forward(const torch::Tensor& x) {
    CHECK_INPUT(x);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty_like(x);
    size_t n = x.numel();
    if (n == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_relu(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_relu_fp16(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            n, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_relu_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor gelu_forward(const torch::Tensor& x) {
    CHECK_INPUT(x);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty_like(x);
    size_t n = x.numel();
    if (n == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_gelu(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_gelu_fp16(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            n, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_gelu_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor sigmoid_forward(const torch::Tensor& x) {
    CHECK_INPUT(x);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty_like(x);
    size_t n = x.numel();
    if (n == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_sigmoid(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_sigmoid_fp16(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            n, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_sigmoid_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

// =========================================================================
// Reductions
// =========================================================================

torch::Tensor reduce_sum_forward(const torch::Tensor& x, bool output_fp32 = false) {
    CHECK_INPUT(x);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    size_t n = x.numel();

    auto out_options = x.options();
    if (output_fp32 || x.scalar_type() == at::kFloat) {
        out_options = out_options.dtype(at::kFloat);
    }
    auto out = torch::zeros({}, out_options);
    if (n == 0) return out;

    int threads = 256;
    int blocks = static_cast<int>((n + threads * 2 - 1) / (threads * 2));
    if (blocks > 1024) blocks = 1024;
    // Workspace is always FP32 for numerical stability
    auto workspace = torch::empty({blocks > 1 ? blocks : 1}, x.options().dtype(at::kFloat));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_reduce_sum(x.data_ptr<float>(), out.data_ptr<float>(), workspace.data_ptr<float>(), n, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_reduce_sum_fp16(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            output_fp32 ? static_cast<void*>(out.data_ptr<float>()) : static_cast<void*>(out.data_ptr<at::Half>()),
            workspace.data_ptr<float>(),
            n, output_fp32, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_reduce_sum_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            output_fp32 ? static_cast<void*>(out.data_ptr<float>()) : static_cast<void*>(out.data_ptr<at::BFloat16>()),
            workspace.data_ptr<float>(),
            n, output_fp32, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor reduce_max_forward(const torch::Tensor& x) {
    CHECK_INPUT(x);
    size_t n = x.numel();
    TORCH_CHECK(n > 0, "Cannot compute reduce_max on an empty tensor");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty({}, x.options());

    int threads = 256;
    int blocks = static_cast<int>((n + threads * 2 - 1) / (threads * 2));
    if (blocks > 1024) blocks = 1024;
    auto workspace = torch::empty({blocks > 1 ? blocks : 1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_reduce_max(x.data_ptr<float>(), out.data_ptr<float>(), workspace.data_ptr<float>(), n, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_reduce_max_fp16(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            reinterpret_cast<half*>(workspace.data_ptr<at::Half>()),
            n, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_reduce_max_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return out;
}

// =========================================================================
// Matrix Multiplication (Preserved Stage 1 Float32 Tiled SGEMM)
// =========================================================================

torch::Tensor matmul_tiled_forward(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_CUDA(a);
    CHECK_CONTIGUOUS(a);
    CHECK_FLOAT32(a);

    CHECK_CUDA(b);
    CHECK_CONTIGUOUS(b);
    CHECK_FLOAT32(b);

    CHECK_SAME_DEVICE(a, b);
    TORCH_CHECK(a.dim() == 2, "Tensor a must be a 2D matrix");
    TORCH_CHECK(b.dim() == 2, "Tensor b must be a 2D matrix");
    TORCH_CHECK(a.size(1) == b.size(0),
                "Matrix dimension mismatch: a.shape[1] (" + std::to_string(a.size(1)) +
                ") must match b.shape[0] (" + std::to_string(b.size(0)) + ")");

    int M = static_cast<int>(a.size(0));
    int K = static_cast<int>(a.size(1));
    int N = static_cast<int>(b.size(1));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    auto out = torch::empty({M, N}, a.options());
    if (M == 0 || K == 0 || N == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_tiled_matmul(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, K, N, stream);
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor matmul_wmma_forward(const torch::Tensor& a, const torch::Tensor& b, bool output_fp32) {
    CHECK_CUDA(a);
    CHECK_CUDA(b);
    CHECK_CONTIGUOUS(a);
    CHECK_CONTIGUOUS(b);
    TORCH_CHECK(a.scalar_type() == at::kHalf, "matmul_wmma: Tensor a must be float16 (half)");
    TORCH_CHECK(b.scalar_type() == at::kHalf, "matmul_wmma: Tensor b must be float16 (half)");
    CHECK_SAME_DEVICE(a, b);

    TORCH_CHECK(a.dim() == 2, "Tensor a must be a 2D matrix");
    TORCH_CHECK(b.dim() == 2, "Tensor b must be a 2D matrix");
    TORCH_CHECK(a.size(1) == b.size(0),
                "Matrix dimension mismatch: a.shape[1] (" + std::to_string(a.size(1)) +
                ") must match b.shape[0] (" + std::to_string(b.size(0)) + ")");

    int M = static_cast<int>(a.size(0));
    int K = static_cast<int>(a.size(1));
    int N = static_cast<int>(b.size(1));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (output_fp32) {
        auto out = torch::empty({M, N}, a.options().dtype(torch::kFloat32));
        if (M == 0 || N == 0) return out;
        launch_wmma_matmul_fp32_out(
            reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
            out.data_ptr<float>(),
            M, K, N, stream
        );
        CUDA_KERNEL_CHECK();
        return out;
    } else {
        auto out = torch::empty({M, N}, a.options());
        if (M == 0 || N == 0) return out;
        launch_wmma_matmul_fp16_out(
            reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            M, K, N, stream
        );
        CUDA_KERNEL_CHECK();
        return out;
    }
}

// =========================================================================
// PyBind11 Module Definition
// =========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Phase 10 Stage 2: Custom CUDA Kernel Library C++ Bindings";

    m.def("vector_add", &vector_add_forward, "Vector addition forward (CUDA)");
    m.def("vector_scale", &vector_scale_forward, "Vector scaling forward (CUDA)");
    m.def("fused_add_relu", &fused_add_relu_forward, "Fused add + ReLU forward (CUDA)");
    m.def("fused_bias_activation", &fused_bias_activation_forward, "Fused bias + activation forward (CUDA)");
    m.def("relu", &relu_forward, "ReLU activation forward (CUDA)");
    m.def("gelu", &gelu_forward, "Fast GELU (tanh approx) activation forward (CUDA)");
    m.def("sigmoid", &sigmoid_forward, "Sigmoid activation forward (CUDA)");
    m.def("reduce_sum", &reduce_sum_forward, py::arg("x"), py::arg("output_fp32") = false, "Sum reduction forward (CUDA)");
    m.def("reduce_max", &reduce_max_forward, "Max reduction forward (CUDA)");
    m.def("matmul_tiled", &matmul_tiled_forward, "2D Shared Memory Tiled SGEMM forward (CUDA)");
    m.def("matmul_wmma", &matmul_wmma_forward, py::arg("a"), py::arg("b"), py::arg("output_fp32") = false, "Ampere Tensor Core WMMA matrix multiplication forward (CUDA)");
}
