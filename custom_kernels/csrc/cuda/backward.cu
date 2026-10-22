#include "../include/backward.cuh"
#include "../include/kernel_common.cuh"

#include <cmath>

namespace {

// =========================================================================
// 1. Vector Add Backward Kernels
// =========================================================================

template <typename T>
__global__ void vector_add_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_a,
    T* __restrict__ grad_b,
    size_t n,
    bool need_a,
    bool need_b
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T go = grad_out[idx];
        if (need_a) grad_a[idx] = go;
        if (need_b) grad_b[idx] = go;
    }
}

// Vectorized FP32
__global__ void vector_add_backward_kernel_vec4(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_a,
    float* __restrict__ grad_b,
    size_t vec_n,
    bool need_a,
    bool need_b
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_n) {
        const float4* go_v = reinterpret_cast<const float4*>(grad_out);
        float4 v = go_v[idx];
        if (need_a) {
            float4* ga_v = reinterpret_cast<float4*>(grad_a);
            ga_v[idx] = v;
        }
        if (need_b) {
            float4* gb_v = reinterpret_cast<float4*>(grad_b);
            gb_v[idx] = v;
        }
    }
}

// Vectorized FP16
__global__ void vector_add_backward_kernel_fp16_vec8(
    const half* __restrict__ grad_out,
    half* __restrict__ grad_a,
    half* __restrict__ grad_b,
    size_t vec_n,
    bool need_a,
    bool need_b
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_n) {
        const float4* go_v = reinterpret_cast<const float4*>(grad_out);
        float4 v = go_v[idx];
        if (need_a) {
            float4* ga_v = reinterpret_cast<float4*>(grad_a);
            ga_v[idx] = v;
        }
        if (need_b) {
            float4* gb_v = reinterpret_cast<float4*>(grad_b);
            gb_v[idx] = v;
        }
    }
}

// Vectorized BF16
__global__ void vector_add_backward_kernel_bf16_vec8(
    const __nv_bfloat16* __restrict__ grad_out,
    __nv_bfloat16* __restrict__ grad_a,
    __nv_bfloat16* __restrict__ grad_b,
    size_t vec_n,
    bool need_a,
    bool need_b
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_n) {
        const float4* go_v = reinterpret_cast<const float4*>(grad_out);
        float4 v = go_v[idx];
        if (need_a) {
            float4* ga_v = reinterpret_cast<float4*>(grad_a);
            ga_v[idx] = v;
        }
        if (need_b) {
            float4* gb_v = reinterpret_cast<float4*>(grad_b);
            gb_v[idx] = v;
        }
    }
}

// =========================================================================
// 2. Vector Scale Backward Kernels
// =========================================================================

template <typename T>
__global__ void vector_scale_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x,
    float alpha,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, float>) {
            grad_x[idx] = grad_out[idx] * alpha;
        } else if constexpr (std::is_same_v<T, half>) {
            float v = __half2float(grad_out[idx]) * alpha;
            grad_x[idx] = __float2half(v);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float v = __bfloat162float(grad_out[idx]) * alpha;
            grad_x[idx] = __float2bfloat16(v);
        }
    }
}

// =========================================================================
// 3. ReLU Backward Kernels
// =========================================================================

template <typename T>
__global__ void relu_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ x,
    T* __restrict__ grad_x,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, float>) {
            grad_x[idx] = (x[idx] > 0.0f) ? grad_out[idx] : 0.0f;
        } else if constexpr (std::is_same_v<T, half>) {
            half zero = __float2half(0.0f);
            grad_x[idx] = __hgt(x[idx], zero) ? grad_out[idx] : zero;
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            __nv_bfloat16 zero = __float2bfloat16(0.0f);
            grad_x[idx] = __hgt(x[idx], zero) ? grad_out[idx] : zero;
        }
    }
}

// =========================================================================
// 4. Sigmoid Backward Kernels (Numerically stable using output y)
// =========================================================================

template <typename T>
__global__ void sigmoid_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ forward_out,
    T* __restrict__ grad_x,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, float>) {
            float y = forward_out[idx];
            grad_x[idx] = grad_out[idx] * y * (1.0f - y);
        } else if constexpr (std::is_same_v<T, half>) {
            float y = __half2float(forward_out[idx]);
            float dy = __half2float(grad_out[idx]);
            grad_x[idx] = __float2half(dy * y * (1.0f - y));
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float y = __bfloat162float(forward_out[idx]);
            float dy = __bfloat162float(grad_out[idx]);
            grad_x[idx] = __float2bfloat16(dy * y * (1.0f - y));
        }
    }
}

// =========================================================================
// 5. GELU Backward Kernels (Exact tanh approximation derivative)
// =========================================================================

__device__ __forceinline__ float gelu_tanh_bwd_scalar(float x, float dout) {
    // GELU(x) = 0.5 * x * (1 + tanh(u(x)))
    // u(x) = sqrt(2/pi) * (x + 0.044715 * x^3)
    constexpr float k = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float c = 0.044715f;
    constexpr float three_c_k = 3.0f * c * k; // 0.1070322244f

    float x2 = x * x;
    float x3 = x2 * x;
    float u = k * (x + c * x3);
    float t = tanhf(u);

    float du_dx = k + three_c_k * x2;
    float dt_dx = (1.0f - t * t) * du_dx;
    float dy_dx = 0.5f * (1.0f + t) + 0.5f * x * dt_dx;

    return dout * dy_dx;
}

template <typename T>
__global__ void gelu_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ x,
    T* __restrict__ grad_x,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, float>) {
            grad_x[idx] = gelu_tanh_bwd_scalar(x[idx], grad_out[idx]);
        } else if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x[idx]);
            float dout_f = __half2float(grad_out[idx]);
            grad_x[idx] = __float2half(gelu_tanh_bwd_scalar(x_f, dout_f));
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float x_f = __bfloat162float(x[idx]);
            float dout_f = __bfloat162float(grad_out[idx]);
            grad_x[idx] = __float2bfloat16(gelu_tanh_bwd_scalar(x_f, dout_f));
        }
    }
}

// =========================================================================
// 6. Fused Add + ReLU Backward Kernels
// =========================================================================

template <typename T>
__global__ void fused_add_relu_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ forward_out,
    T* __restrict__ grad_a,
    T* __restrict__ grad_b,
    size_t n,
    bool need_a,
    bool need_b
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T zero;
        bool mask = false;
        if constexpr (std::is_same_v<T, float>) {
            zero = 0.0f;
            mask = (forward_out[idx] > 0.0f);
        } else if constexpr (std::is_same_v<T, half>) {
            zero = __float2half(0.0f);
            mask = __hgt(forward_out[idx], zero);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            zero = __float2bfloat16(0.0f);
            mask = __hgt(forward_out[idx], zero);
        }

        T g = mask ? grad_out[idx] : zero;
        if (need_a) grad_a[idx] = g;
        if (need_b) grad_b[idx] = g;
    }
}

// =========================================================================
// 7. Fused Bias + Activation Backward Kernels
// =========================================================================

template <typename T>
__global__ void fused_bias_act_backward_dx_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ x,
    const T* __restrict__ bias,
    T* __restrict__ grad_x,
    int M,
    int N,
    int act_type // 0: None, 1: ReLU, 2: GELU
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        int col = idx % N;

        float x_f, b_f, dout_f;
        if constexpr (std::is_same_v<T, float>) {
            x_f = x[idx];
            b_f = bias[col];
            dout_f = grad_out[idx];
        } else if constexpr (std::is_same_v<T, half>) {
            x_f = __half2float(x[idx]);
            b_f = __half2float(bias[col]);
            dout_f = __half2float(grad_out[idx]);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            x_f = __bfloat162float(x[idx]);
            b_f = __bfloat162float(bias[col]);
            dout_f = __bfloat162float(grad_out[idx]);
        }

        float z = x_f + b_f;
        float dx_f = 0.0f;

        if (act_type == 0) { // None / linear
            dx_f = dout_f;
        } else if (act_type == 1) { // ReLU
            dx_f = (z > 0.0f) ? dout_f : 0.0f;
        } else if (act_type == 2) { // GELU
            dx_f = gelu_tanh_bwd_scalar(z, dout_f);
        }

        if constexpr (std::is_same_v<T, float>) {
            grad_x[idx] = dx_f;
        } else if constexpr (std::is_same_v<T, half>) {
            grad_x[idx] = __float2half(dx_f);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            grad_x[idx] = __float2bfloat16(dx_f);
        }
    }
}

// Column reduction across rows to compute d_bias[col] = sum_{m=0}^{M-1} grad_x[m, col]
template <typename T>
__global__ void column_reduce_bias_kernel(
    const T* __restrict__ grad_x,
    T* __restrict__ grad_bias,
    int M,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < M; ++row) {
            int idx = row * N + col;
            if constexpr (std::is_same_v<T, float>) {
                sum += grad_x[idx];
            } else if constexpr (std::is_same_v<T, half>) {
                sum += __half2float(grad_x[idx]);
            } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                sum += __bfloat162float(grad_x[idx]);
            }
        }

        if constexpr (std::is_same_v<T, float>) {
            grad_bias[col] = sum;
        } else if constexpr (std::is_same_v<T, half>) {
            grad_bias[col] = __float2half(sum);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            grad_bias[col] = __float2bfloat16(sum);
        }
    }
}

} // namespace

// =========================================================================
// Host Launch Functions
// =========================================================================

void launch_vector_add_backward(
    const float* grad_out, float* grad_a, float* grad_b,
    size_t n, bool need_a, bool need_b, cudaStream_t stream
) {
    if (n == 0 || (!need_a && !need_b)) return;
    int threads = 256;
    if (n % 4 == 0 && is_aligned_16(grad_out) &&
        (!need_a || is_aligned_16(grad_a)) && (!need_b || is_aligned_16(grad_b))) {
        size_t vec_n = n / 4;
        int blocks = static_cast<int>((vec_n + threads - 1) / threads);
        vector_add_backward_kernel_vec4<<<blocks, threads, 0, stream>>>(grad_out, grad_a, grad_b, vec_n, need_a, need_b);
    } else {
        int blocks = static_cast<int>((n + threads - 1) / threads);
        vector_add_backward_kernel<float><<<blocks, threads, 0, stream>>>(grad_out, grad_a, grad_b, n, need_a, need_b);
    }
}

void launch_vector_add_backward_fp16(
    const half* grad_out, half* grad_a, half* grad_b,
    size_t n, bool need_a, bool need_b, cudaStream_t stream
) {
    if (n == 0 || (!need_a && !need_b)) return;
    int threads = 256;
    if (n % 8 == 0 && is_aligned_16(grad_out) &&
        (!need_a || is_aligned_16(grad_a)) && (!need_b || is_aligned_16(grad_b))) {
        size_t vec_n = n / 8;
        int blocks = static_cast<int>((vec_n + threads - 1) / threads);
        vector_add_backward_kernel_fp16_vec8<<<blocks, threads, 0, stream>>>(grad_out, grad_a, grad_b, vec_n, need_a, need_b);
    } else {
        int blocks = static_cast<int>((n + threads - 1) / threads);
        vector_add_backward_kernel<half><<<blocks, threads, 0, stream>>>(grad_out, grad_a, grad_b, n, need_a, need_b);
    }
}

void launch_vector_add_backward_bf16(
    const __nv_bfloat16* grad_out, __nv_bfloat16* grad_a, __nv_bfloat16* grad_b,
    size_t n, bool need_a, bool need_b, cudaStream_t stream
) {
    if (n == 0 || (!need_a && !need_b)) return;
    int threads = 256;
    if (n % 8 == 0 && is_aligned_16(grad_out) &&
        (!need_a || is_aligned_16(grad_a)) && (!need_b || is_aligned_16(grad_b))) {
        size_t vec_n = n / 8;
        int blocks = static_cast<int>((vec_n + threads - 1) / threads);
        vector_add_backward_kernel_bf16_vec8<<<blocks, threads, 0, stream>>>(grad_out, grad_a, grad_b, vec_n, need_a, need_b);
    } else {
        int blocks = static_cast<int>((n + threads - 1) / threads);
        vector_add_backward_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(grad_out, grad_a, grad_b, n, need_a, need_b);
    }
}

void launch_vector_scale_backward(
    const float* grad_out, float* grad_x, float alpha, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    vector_scale_backward_kernel<float><<<blocks, threads, 0, stream>>>(grad_out, grad_x, alpha, n);
}

void launch_vector_scale_backward_fp16(
    const half* grad_out, half* grad_x, float alpha, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    vector_scale_backward_kernel<half><<<blocks, threads, 0, stream>>>(grad_out, grad_x, alpha, n);
}

void launch_vector_scale_backward_bf16(
    const __nv_bfloat16* grad_out, __nv_bfloat16* grad_x, float alpha, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    vector_scale_backward_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(grad_out, grad_x, alpha, n);
}

void launch_relu_backward(
    const float* grad_out, const float* x, float* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    relu_backward_kernel<float><<<blocks, threads, 0, stream>>>(grad_out, x, grad_x, n);
}

void launch_relu_backward_fp16(
    const half* grad_out, const half* x, half* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    relu_backward_kernel<half><<<blocks, threads, 0, stream>>>(grad_out, x, grad_x, n);
}

void launch_relu_backward_bf16(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* x, __nv_bfloat16* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    relu_backward_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(grad_out, x, grad_x, n);
}

void launch_sigmoid_backward(
    const float* grad_out, const float* forward_out, float* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    sigmoid_backward_kernel<float><<<blocks, threads, 0, stream>>>(grad_out, forward_out, grad_x, n);
}

void launch_sigmoid_backward_fp16(
    const half* grad_out, const half* forward_out, half* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    sigmoid_backward_kernel<half><<<blocks, threads, 0, stream>>>(grad_out, forward_out, grad_x, n);
}

void launch_sigmoid_backward_bf16(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* forward_out, __nv_bfloat16* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    sigmoid_backward_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(grad_out, forward_out, grad_x, n);
}

void launch_gelu_backward(
    const float* grad_out, const float* x, float* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gelu_backward_kernel<float><<<blocks, threads, 0, stream>>>(grad_out, x, grad_x, n);
}

void launch_gelu_backward_fp16(
    const half* grad_out, const half* x, half* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gelu_backward_kernel<half><<<blocks, threads, 0, stream>>>(grad_out, x, grad_x, n);
}

void launch_gelu_backward_bf16(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* x, __nv_bfloat16* grad_x, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gelu_backward_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(grad_out, x, grad_x, n);
}

void launch_fused_add_relu_backward(
    const float* grad_out, const float* forward_out, float* grad_a, float* grad_b,
    size_t n, bool need_a, bool need_b, cudaStream_t stream
) {
    if (n == 0 || (!need_a && !need_b)) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    fused_add_relu_backward_kernel<float><<<blocks, threads, 0, stream>>>(grad_out, forward_out, grad_a, grad_b, n, need_a, need_b);
}

void launch_fused_add_relu_backward_fp16(
    const half* grad_out, const half* forward_out, half* grad_a, half* grad_b,
    size_t n, bool need_a, bool need_b, cudaStream_t stream
) {
    if (n == 0 || (!need_a && !need_b)) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    fused_add_relu_backward_kernel<half><<<blocks, threads, 0, stream>>>(grad_out, forward_out, grad_a, grad_b, n, need_a, need_b);
}

void launch_fused_add_relu_backward_bf16(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* forward_out, __nv_bfloat16* grad_a, __nv_bfloat16* grad_b,
    size_t n, bool need_a, bool need_b, cudaStream_t stream
) {
    if (n == 0 || (!need_a && !need_b)) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    fused_add_relu_backward_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(grad_out, forward_out, grad_a, grad_b, n, need_a, need_b);
}

void launch_fused_bias_activation_backward(
    const float* grad_out, const float* x, const float* bias, float* grad_x, float* grad_bias,
    int M, int N, int act_type, bool need_x, bool need_bias, cudaStream_t stream
) {
    if (M == 0 || N == 0) return;
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (need_x) {
        fused_bias_act_backward_dx_kernel<float><<<blocks, threads, 0, stream>>>(
            grad_out, x, bias, grad_x, M, N, act_type
        );
    }
    if (need_bias) {
        // Reduce across rows M for each of the N columns
        int col_threads = 256;
        int col_blocks = (N + col_threads - 1) / col_threads;
        const float* src = need_x ? grad_x : grad_out; // if need_x, grad_x has pre-activation gradient dZ
        column_reduce_bias_kernel<float><<<col_blocks, col_threads, 0, stream>>>(
            src, grad_bias, M, N
        );
    }
}

void launch_fused_bias_activation_backward_fp16(
    const half* grad_out, const half* x, const half* bias, half* grad_x, half* grad_bias,
    int M, int N, int act_type, bool need_x, bool need_bias, cudaStream_t stream
) {
    if (M == 0 || N == 0) return;
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (need_x) {
        fused_bias_act_backward_dx_kernel<half><<<blocks, threads, 0, stream>>>(
            grad_out, x, bias, grad_x, M, N, act_type
        );
    }
    if (need_bias) {
        int col_threads = 256;
        int col_blocks = (N + col_threads - 1) / col_threads;
        const half* src = need_x ? grad_x : grad_out;
        column_reduce_bias_kernel<half><<<col_blocks, col_threads, 0, stream>>>(
            src, grad_bias, M, N
        );
    }
}

void launch_fused_bias_activation_backward_bf16(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* grad_x, __nv_bfloat16* grad_bias,
    int M, int N, int act_type, bool need_x, bool need_bias, cudaStream_t stream
) {
    if (M == 0 || N == 0) return;
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (need_x) {
        fused_bias_act_backward_dx_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            grad_out, x, bias, grad_x, M, N, act_type
        );
    }
    if (need_bias) {
        int col_threads = 256;
        int col_blocks = (N + col_threads - 1) / col_threads;
        const __nv_bfloat16* src = need_x ? grad_x : grad_out;
        column_reduce_bias_kernel<__nv_bfloat16><<<col_blocks, col_threads, 0, stream>>>(
            src, grad_bias, M, N
        );
    }
}
