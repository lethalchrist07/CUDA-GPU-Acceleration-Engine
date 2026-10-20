#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "include/kernel_common.cuh"
#include "include/elementwise.cuh"
#include "include/activations.cuh"
#include "include/reductions.cuh"
#include "include/matmul_tiled.cuh"

// Helper macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x); } while (0)
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK((x).device() == (y).device(), #x " and " #y " must be on the same CUDA device")

// =========================================================================
// Elementwise Operations
// =========================================================================

torch::Tensor vector_add_forward(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_SAME_DEVICE(a, b);
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors a and b must have the same shape for vector addition");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    auto out = torch::empty_like(a);
    size_t n = a.numel();
    if (n == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n, stream);
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
    launch_vector_scale(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n, stream);
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor fused_add_relu_forward(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_SAME_DEVICE(a, b);
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors a and b must have the same shape for fused add_relu");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    auto out = torch::empty_like(a);
    size_t n = a.numel();
    if (n == 0) return out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_fused_add_relu(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor fused_bias_activation_forward(const torch::Tensor& x,
                                            const torch::Tensor& bias,
                                            const std::string& activation) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    CHECK_SAME_DEVICE(x, bias);
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
    launch_fused_bias_activation(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols, act_type, stream);
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
    launch_relu(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
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
    launch_gelu(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
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
    launch_sigmoid(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    CUDA_KERNEL_CHECK();
    return out;
}

// =========================================================================
// Reductions
// =========================================================================

torch::Tensor reduce_sum_forward(const torch::Tensor& x) {
    CHECK_INPUT(x);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    size_t n = x.numel();
    auto out = torch::zeros({}, x.options());
    if (n == 0) return out;

    int threads = 256;
    int blocks = static_cast<int>((n + threads * 2 - 1) / (threads * 2));
    if (blocks > 1024) blocks = 1024;
    auto workspace = (blocks > 1) ? torch::empty({blocks}, x.options()) : torch::empty({1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_reduce_sum(x.data_ptr<float>(), out.data_ptr<float>(), workspace.data_ptr<float>(), n, stream);
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
    auto workspace = (blocks > 1) ? torch::empty({blocks}, x.options()) : torch::empty({1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_reduce_max(x.data_ptr<float>(), out.data_ptr<float>(), workspace.data_ptr<float>(), n, stream);
    CUDA_KERNEL_CHECK();
    return out;
}

// =========================================================================
// Matrix Multiplication
// =========================================================================

torch::Tensor matmul_tiled_forward(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
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

// =========================================================================
// PyBind11 Module Definition
// =========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Phase 10 Stage 1: Custom CUDA Kernel Library C++ Bindings";

    m.def("vector_add", &vector_add_forward, "Vector addition forward (CUDA)");
    m.def("vector_scale", &vector_scale_forward, "Vector scaling forward (CUDA)");
    m.def("fused_add_relu", &fused_add_relu_forward, "Fused add + ReLU forward (CUDA)");
    m.def("fused_bias_activation", &fused_bias_activation_forward, "Fused bias + activation forward (CUDA)");
    m.def("relu", &relu_forward, "ReLU activation forward (CUDA)");
    m.def("gelu", &gelu_forward, "Fast GELU (tanh approx) activation forward (CUDA)");
    m.def("sigmoid", &sigmoid_forward, "Sigmoid activation forward (CUDA)");
    m.def("reduce_sum", &reduce_sum_forward, "Sum reduction forward (CUDA)");
    m.def("reduce_max", &reduce_max_forward, "Max reduction forward (CUDA)");
    m.def("matmul_tiled", &matmul_tiled_forward, "2D Shared Memory Tiled SGEMM forward (CUDA)");
}
