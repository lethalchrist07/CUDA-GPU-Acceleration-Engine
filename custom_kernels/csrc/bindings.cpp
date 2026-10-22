#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "include/kernel_common.cuh"
#include "include/elementwise.cuh"
#include "include/activations.cuh"
#include "include/reductions.cuh"
#include "include/matmul_tiled.cuh"
#include "include/matmul_wmma.cuh"
#include "include/backward.cuh"

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
// Backward Operations
// =========================================================================

std::vector<torch::Tensor> vector_add_backward_forward(
    const torch::Tensor& grad_out, bool need_a, bool need_b
) {
    CHECK_INPUT(grad_out);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out));
    size_t n = grad_out.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    torch::Tensor grad_a = need_a ? torch::empty_like(grad_out) : torch::Tensor();
    torch::Tensor grad_b = need_b ? torch::empty_like(grad_out) : torch::Tensor();

    if (n == 0 || (!need_a && !need_b)) return {grad_a, grad_b};

    if (grad_out.scalar_type() == at::kFloat) {
        launch_vector_add_backward(
            grad_out.data_ptr<float>(),
            need_a ? grad_a.data_ptr<float>() : nullptr,
            need_b ? grad_b.data_ptr<float>() : nullptr,
            n, need_a, need_b, stream
        );
    } else if (grad_out.scalar_type() == at::kHalf) {
        launch_vector_add_backward_fp16(
            reinterpret_cast<const half*>(grad_out.data_ptr<at::Half>()),
            need_a ? reinterpret_cast<half*>(grad_a.data_ptr<at::Half>()) : nullptr,
            need_b ? reinterpret_cast<half*>(grad_b.data_ptr<at::Half>()) : nullptr,
            n, need_a, need_b, stream
        );
    } else if (grad_out.scalar_type() == at::kBFloat16) {
        launch_vector_add_backward_bf16(
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
            need_a ? reinterpret_cast<__nv_bfloat16*>(grad_a.data_ptr<at::BFloat16>()) : nullptr,
            need_b ? reinterpret_cast<__nv_bfloat16*>(grad_b.data_ptr<at::BFloat16>()) : nullptr,
            n, need_a, need_b, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return {grad_a, grad_b};
}

torch::Tensor vector_scale_backward_forward(const torch::Tensor& grad_out, float alpha) {
    CHECK_INPUT(grad_out);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out));
    auto grad_x = torch::empty_like(grad_out);
    size_t n = grad_out.numel();
    if (n == 0) return grad_x;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (grad_out.scalar_type() == at::kFloat) {
        launch_vector_scale_backward(grad_out.data_ptr<float>(), grad_x.data_ptr<float>(), alpha, n, stream);
    } else if (grad_out.scalar_type() == at::kHalf) {
        launch_vector_scale_backward_fp16(
            reinterpret_cast<const half*>(grad_out.data_ptr<at::Half>()),
            reinterpret_cast<half*>(grad_x.data_ptr<at::Half>()),
            alpha, n, stream
        );
    } else if (grad_out.scalar_type() == at::kBFloat16) {
        launch_vector_scale_backward_bf16(
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr<at::BFloat16>()),
            alpha, n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return grad_x;
}

torch::Tensor relu_backward_forward(const torch::Tensor& grad_out, const torch::Tensor& x) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(x);
    CHECK_SAME_DEVICE(grad_out, x);
    CHECK_SAME_DTYPE(grad_out, x);
    TORCH_CHECK(grad_out.sizes() == x.sizes(), "grad_out and x must have the same shape for relu_backward");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out));
    auto grad_x = torch::empty_like(x);
    size_t n = x.numel();
    if (n == 0) return grad_x;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_relu_backward(grad_out.data_ptr<float>(), x.data_ptr<float>(), grad_x.data_ptr<float>(), n, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_relu_backward_fp16(
            reinterpret_cast<const half*>(grad_out.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(grad_x.data_ptr<at::Half>()),
            n, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_relu_backward_bf16(
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return grad_x;
}

torch::Tensor sigmoid_backward_forward(const torch::Tensor& grad_out, const torch::Tensor& forward_out) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(forward_out);
    CHECK_SAME_DEVICE(grad_out, forward_out);
    CHECK_SAME_DTYPE(grad_out, forward_out);
    TORCH_CHECK(grad_out.sizes() == forward_out.sizes(), "grad_out and forward_out must have the same shape for sigmoid_backward");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out));
    auto grad_x = torch::empty_like(forward_out);
    size_t n = forward_out.numel();
    if (n == 0) return grad_x;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (forward_out.scalar_type() == at::kFloat) {
        launch_sigmoid_backward(grad_out.data_ptr<float>(), forward_out.data_ptr<float>(), grad_x.data_ptr<float>(), n, stream);
    } else if (forward_out.scalar_type() == at::kHalf) {
        launch_sigmoid_backward_fp16(
            reinterpret_cast<const half*>(grad_out.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(forward_out.data_ptr<at::Half>()),
            reinterpret_cast<half*>(grad_x.data_ptr<at::Half>()),
            n, stream
        );
    } else if (forward_out.scalar_type() == at::kBFloat16) {
        launch_sigmoid_backward_bf16(
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(forward_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return grad_x;
}

torch::Tensor gelu_backward_forward(const torch::Tensor& grad_out, const torch::Tensor& x) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(x);
    CHECK_SAME_DEVICE(grad_out, x);
    CHECK_SAME_DTYPE(grad_out, x);
    TORCH_CHECK(grad_out.sizes() == x.sizes(), "grad_out and x must have the same shape for gelu_backward");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out));
    auto grad_x = torch::empty_like(x);
    size_t n = x.numel();
    if (n == 0) return grad_x;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kFloat) {
        launch_gelu_backward(grad_out.data_ptr<float>(), x.data_ptr<float>(), grad_x.data_ptr<float>(), n, stream);
    } else if (x.scalar_type() == at::kHalf) {
        launch_gelu_backward_fp16(
            reinterpret_cast<const half*>(grad_out.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(grad_x.data_ptr<at::Half>()),
            n, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_gelu_backward_bf16(
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr<at::BFloat16>()),
            n, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return grad_x;
}

std::vector<torch::Tensor> fused_add_relu_backward_forward(
    const torch::Tensor& grad_out, const torch::Tensor& forward_out, bool need_a, bool need_b
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(forward_out);
    CHECK_SAME_DEVICE(grad_out, forward_out);
    CHECK_SAME_DTYPE(grad_out, forward_out);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out));
    size_t n = grad_out.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    torch::Tensor grad_a = need_a ? torch::empty_like(grad_out) : torch::Tensor();
    torch::Tensor grad_b = need_b ? torch::empty_like(grad_out) : torch::Tensor();

    if (n == 0 || (!need_a && !need_b)) return {grad_a, grad_b};

    if (grad_out.scalar_type() == at::kFloat) {
        launch_fused_add_relu_backward(
            grad_out.data_ptr<float>(), forward_out.data_ptr<float>(),
            need_a ? grad_a.data_ptr<float>() : nullptr,
            need_b ? grad_b.data_ptr<float>() : nullptr,
            n, need_a, need_b, stream
        );
    } else if (grad_out.scalar_type() == at::kHalf) {
        launch_fused_add_relu_backward_fp16(
            reinterpret_cast<const half*>(grad_out.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(forward_out.data_ptr<at::Half>()),
            need_a ? reinterpret_cast<half*>(grad_a.data_ptr<at::Half>()) : nullptr,
            need_b ? reinterpret_cast<half*>(grad_b.data_ptr<at::Half>()) : nullptr,
            n, need_a, need_b, stream
        );
    } else if (grad_out.scalar_type() == at::kBFloat16) {
        launch_fused_add_relu_backward_bf16(
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(forward_out.data_ptr<at::BFloat16>()),
            need_a ? reinterpret_cast<__nv_bfloat16*>(grad_a.data_ptr<at::BFloat16>()) : nullptr,
            need_b ? reinterpret_cast<__nv_bfloat16*>(grad_b.data_ptr<at::BFloat16>()) : nullptr,
            n, need_a, need_b, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return {grad_a, grad_b};
}

std::vector<torch::Tensor> fused_bias_activation_backward_forward(
    const torch::Tensor& grad_out, const torch::Tensor& x, const torch::Tensor& bias,
    const std::string& act_str, bool need_x, bool need_bias
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    CHECK_SAME_DEVICE(grad_out, x);
    CHECK_SAME_DEVICE(grad_out, bias);
    CHECK_SAME_DTYPE(grad_out, x);
    CHECK_SAME_DTYPE(grad_out, bias);

    TORCH_CHECK(x.dim() == 2, "x must be a 2D matrix (M x N)");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be a 2D matrix (M x N)");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D vector (N)");
    TORCH_CHECK(x.sizes() == grad_out.sizes(), "x and grad_out must have identical shapes");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias size must match x.shape[1]");

    int M = static_cast<int>(x.size(0));
    int N = static_cast<int>(x.size(1));

    int act_type = 0; // None
    if (act_str == "relu") act_type = 1;
    else if (act_str == "gelu") act_type = 2;
    else if (act_str != "none") {
        TORCH_CHECK(false, "Unsupported activation type for fused_bias_activation_backward: " + act_str);
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_out));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    torch::Tensor grad_x = need_x ? torch::empty_like(x) : torch::Tensor();
    torch::Tensor grad_bias = need_bias ? torch::empty_like(bias) : torch::Tensor();

    if (M == 0 || N == 0 || (!need_x && !need_bias)) return {grad_x, grad_bias};

    if (x.scalar_type() == at::kFloat) {
        launch_fused_bias_activation_backward(
            grad_out.data_ptr<float>(), x.data_ptr<float>(), bias.data_ptr<float>(),
            need_x ? grad_x.data_ptr<float>() : nullptr,
            need_bias ? grad_bias.data_ptr<float>() : nullptr,
            M, N, act_type, need_x, need_bias, stream
        );
    } else if (x.scalar_type() == at::kHalf) {
        launch_fused_bias_activation_backward_fp16(
            reinterpret_cast<const half*>(grad_out.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
            need_x ? reinterpret_cast<half*>(grad_x.data_ptr<at::Half>()) : nullptr,
            need_bias ? reinterpret_cast<half*>(grad_bias.data_ptr<at::Half>()) : nullptr,
            M, N, act_type, need_x, need_bias, stream
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        launch_fused_bias_activation_backward_bf16(
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
            need_x ? reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr<at::BFloat16>()) : nullptr,
            need_bias ? reinterpret_cast<__nv_bfloat16*>(grad_bias.data_ptr<at::BFloat16>()) : nullptr,
            M, N, act_type, need_x, need_bias, stream
        );
    }
    CUDA_KERNEL_CHECK();
    return {grad_x, grad_bias};
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

    // Backward operators
    m.def("vector_add_backward", &vector_add_backward_forward, py::arg("grad_out"), py::arg("need_a") = true, py::arg("need_b") = true, "Vector addition backward (CUDA)");
    m.def("vector_scale_backward", &vector_scale_backward_forward, py::arg("grad_out"), py::arg("alpha"), "Vector scale backward (CUDA)");
    m.def("relu_backward", &relu_backward_forward, py::arg("grad_out"), py::arg("x"), "ReLU activation backward (CUDA)");
    m.def("sigmoid_backward", &sigmoid_backward_forward, py::arg("grad_out"), py::arg("forward_out"), "Sigmoid activation backward (CUDA)");
    m.def("gelu_backward", &gelu_backward_forward, py::arg("grad_out"), py::arg("x"), "Fast GELU activation backward (CUDA)");
    m.def("fused_add_relu_backward", &fused_add_relu_backward_forward, py::arg("grad_out"), py::arg("forward_out"), py::arg("need_a") = true, py::arg("need_b") = true, "Fused add + ReLU backward (CUDA)");
    m.def("fused_bias_activation_backward", &fused_bias_activation_backward_forward, py::arg("grad_out"), py::arg("x"), py::arg("bias"), py::arg("activation") = "relu", py::arg("need_x") = true, py::arg("need_bias") = true, "Fused bias + activation backward (CUDA)");
}

