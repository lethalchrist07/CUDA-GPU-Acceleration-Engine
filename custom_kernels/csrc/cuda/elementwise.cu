#include "../include/elementwise.cuh"
#include "../include/kernel_common.cuh"
#include <cmath>

// =========================================================================
// Vector Addition
// =========================================================================
__global__ void vector_add_kernel_scalar(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_kernel_vec4(const float4* __restrict__ a,
                                       const float4* __restrict__ b,
                                       float4* __restrict__ c,
                                       size_t n_vec4) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec4; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float4 va = a[i];
        float4 vb = b[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[i] = vc;
    }
}

void launch_vector_add(const float* a, const float* b, float* c, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(a) && is_aligned_16(b) && is_aligned_16(c);
    if (aligned && n >= 4) {
        size_t n_vec4 = n / 4;
        size_t tail = n % 4;

        int threads = 256;
        int blocks = static_cast<int>((n_vec4 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_add_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            n_vec4
        );

        if (tail > 0) {
            size_t offset = n_vec4 * 4;
            vector_add_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                a + offset, b + offset, c + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_add_kernel_scalar<<<blocks, threads, 0, stream>>>(a, b, c, n);
    }
}

// =========================================================================
// Vector Scale
// =========================================================================
__global__ void vector_scale_kernel_scalar(const float* __restrict__ x,
                                           float* __restrict__ out,
                                           float alpha,
                                           size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = x[i] * alpha;
    }
}

__global__ void vector_scale_kernel_vec4(const float4* __restrict__ x,
                                         float4* __restrict__ out,
                                         float alpha,
                                         size_t n_vec4) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec4; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float4 vx = x[i];
        float4 vout;
        vout.x = vx.x * alpha;
        vout.y = vx.y * alpha;
        vout.z = vx.z * alpha;
        vout.w = vx.w * alpha;
        out[i] = vout;
    }
}

void launch_vector_scale(const float* x, float* out, float alpha, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(x) && is_aligned_16(out);
    if (aligned && n >= 4) {
        size_t n_vec4 = n / 4;
        size_t tail = n % 4;

        int threads = 256;
        int blocks = static_cast<int>((n_vec4 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_scale_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x),
            reinterpret_cast<float4*>(out),
            alpha,
            n_vec4
        );

        if (tail > 0) {
            size_t offset = n_vec4 * 4;
            vector_scale_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                x + offset, out + offset, alpha, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_scale_kernel_scalar<<<blocks, threads, 0, stream>>>(x, out, alpha, n);
    }
}

// =========================================================================
// Fused Add + ReLU
// =========================================================================
__global__ void fused_add_relu_kernel_scalar(const float* __restrict__ a,
                                             const float* __restrict__ b,
                                             float* __restrict__ out,
                                             size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float sum = a[i] + b[i];
        out[i] = (isnan(sum) || sum > 0.0f) ? sum : 0.0f;
    }
}

__global__ void fused_add_relu_kernel_vec4(const float4* __restrict__ a,
                                           const float4* __restrict__ b,
                                           float4* __restrict__ out,
                                           size_t n_vec4) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec4; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float4 va = a[i];
        float4 vb = b[i];
        float4 vout;
        float sx = va.x + vb.x;
        float sy = va.y + vb.y;
        float sz = va.z + vb.z;
        float sw = va.w + vb.w;
        vout.x = (isnan(sx) || sx > 0.0f) ? sx : 0.0f;
        vout.y = (isnan(sy) || sy > 0.0f) ? sy : 0.0f;
        vout.z = (isnan(sz) || sz > 0.0f) ? sz : 0.0f;
        vout.w = (isnan(sw) || sw > 0.0f) ? sw : 0.0f;
        out[i] = vout;
    }
}

void launch_fused_add_relu(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(a) && is_aligned_16(b) && is_aligned_16(out);
    if (aligned && n >= 4) {
        size_t n_vec4 = n / 4;
        size_t tail = n % 4;

        int threads = 256;
        int blocks = static_cast<int>((n_vec4 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        fused_add_relu_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(out),
            n_vec4
        );

        if (tail > 0) {
            size_t offset = n_vec4 * 4;
            fused_add_relu_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                a + offset, b + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        fused_add_relu_kernel_scalar<<<blocks, threads, 0, stream>>>(a, b, out, n);
    }
}

// =========================================================================
// Fused Bias + Activation (2D matrix rows x cols, 1D bias cols)
// =========================================================================
__device__ __forceinline__ float apply_act(float val, int act_type) {
    if (act_type == 1) { // ReLU
        return (isnan(val) || val > 0.0f) ? val : 0.0f;
    } else if (act_type == 2) { // Fast GELU
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float SQRT_2_OVER_PI = 0.7978845608028654f;
        const float COEFF = 0.044715f;
        float inner = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
        return 0.5f * val * (1.0f + tanhf(inner));
    }
    return val;
}

__global__ void fused_bias_activation_kernel(const float* __restrict__ x,
                                             const float* __restrict__ bias,
                                             float* __restrict__ out,
                                             size_t rows,
                                             size_t cols,
                                             int act_type) {
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        size_t idx = row * cols + col;
        float val = x[idx] + bias[col];
        out[idx] = apply_act(val, act_type);
    }
}

void launch_fused_bias_activation(const float* x,
                                  const float* bias,
                                  float* out,
                                  size_t rows,
                                  size_t cols,
                                  int act_type,
                                  cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((cols + block.x - 1) / block.x),
        static_cast<unsigned int>((rows + block.y - 1) / block.y)
    );

    fused_bias_activation_kernel<<<grid, block, 0, stream>>>(
        x, bias, out, rows, cols, act_type
    );
}

// =========================================================================
// FP16 Mixed-Precision Kernels
// =========================================================================

// Vector Add FP16
__global__ void vector_add_fp16_kernel_scalar(const half* __restrict__ a,
                                              const half* __restrict__ b,
                                              half* __restrict__ c,
                                              size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        c[i] = __hadd(a[i], b[i]);
    }
}

__global__ void vector_add_fp16_kernel_vec8(const float4* __restrict__ a,
                                            const float4* __restrict__ b,
                                            float4* __restrict__ c,
                                            size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        Half8Vec va, vb, vc;
        va.v4 = a[i];
        vb.v4 = b[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            vc.h2[k] = __hadd2(va.h2[k], vb.h2[k]);
        }
        c[i] = vc.v4;
    }
}

void launch_vector_add_fp16(const half* a, const half* b, half* c, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(a) && is_aligned_16(b) && is_aligned_16(c);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_add_fp16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            vector_add_fp16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                a + offset, b + offset, c + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_add_fp16_kernel_scalar<<<blocks, threads, 0, stream>>>(a, b, c, n);
    }
}

// Vector Scale FP16
__global__ void vector_scale_fp16_kernel_scalar(const half* __restrict__ x,
                                                half* __restrict__ out,
                                                half alpha,
                                                size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = __hmul(x[i], alpha);
    }
}

__global__ void vector_scale_fp16_kernel_vec8(const float4* __restrict__ x,
                                              float4* __restrict__ out,
                                              half2 alpha2,
                                              size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        Half8Vec vx, vout;
        vx.v4 = x[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            vout.h2[k] = __hmul2(vx.h2[k], alpha2);
        }
        out[i] = vout.v4;
    }
}

void launch_vector_scale_fp16(const half* x, half* out, float alpha, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    half alpha_h = __float2half(alpha);
    half2 alpha2;
    alpha2.x = alpha_h;
    alpha2.y = alpha_h;

    bool aligned = is_aligned_16(x) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_scale_fp16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x),
            reinterpret_cast<float4*>(out),
            alpha2,
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            vector_scale_fp16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                x + offset, out + offset, alpha_h, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_scale_fp16_kernel_scalar<<<blocks, threads, 0, stream>>>(x, out, alpha_h, n);
    }
}

// Fused Add + ReLU FP16
__global__ void fused_add_relu_fp16_kernel_scalar(const half* __restrict__ a,
                                                  const half* __restrict__ b,
                                                  half* __restrict__ out,
                                                  size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float sum = __half2float(__hadd(a[i], b[i]));
        out[i] = __float2half((isnan(sum) || sum > 0.0f) ? sum : 0.0f);
    }
}

__global__ void fused_add_relu_fp16_kernel_vec8(const float4* __restrict__ a,
                                                const float4* __restrict__ b,
                                                float4* __restrict__ out,
                                                size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        Half8Vec va, vb, vout;
        va.v4 = a[i];
        vb.v4 = b[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            half2 s2 = __hadd2(va.h2[k], vb.h2[k]);
            float f0 = __half2float(s2.x);
            float f1 = __half2float(s2.y);
            vout.h2[k].x = __float2half((isnan(f0) || f0 > 0.0f) ? f0 : 0.0f);
            vout.h2[k].y = __float2half((isnan(f1) || f1 > 0.0f) ? f1 : 0.0f);
        }
        out[i] = vout.v4;
    }
}

void launch_fused_add_relu_fp16(const half* a, const half* b, half* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(a) && is_aligned_16(b) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        fused_add_relu_fp16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(out),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            fused_add_relu_fp16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                a + offset, b + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        fused_add_relu_fp16_kernel_scalar<<<blocks, threads, 0, stream>>>(a, b, out, n);
    }
}

// Fused Bias + Activation FP16
__global__ void fused_bias_activation_fp16_kernel(const half* __restrict__ x,
                                                  const half* __restrict__ bias,
                                                  half* __restrict__ out,
                                                  size_t rows,
                                                  size_t cols,
                                                  int act_type) {
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        size_t idx = row * cols + col;
        float val = __half2float(x[idx]) + __half2float(bias[col]);
        out[idx] = __float2half(apply_act(val, act_type));
    }
}

void launch_fused_bias_activation_fp16(const half* x,
                                       const half* bias,
                                       half* out,
                                       size_t rows,
                                       size_t cols,
                                       int act_type,
                                       cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((cols + block.x - 1) / block.x),
        static_cast<unsigned int>((rows + block.y - 1) / block.y)
    );

    fused_bias_activation_fp16_kernel<<<grid, block, 0, stream>>>(
        x, bias, out, rows, cols, act_type
    );
}

// =========================================================================
// BF16 Mixed-Precision Kernels
// =========================================================================

// Vector Add BF16
__global__ void vector_add_bf16_kernel_scalar(const __nv_bfloat16* __restrict__ a,
                                              const __nv_bfloat16* __restrict__ b,
                                              __nv_bfloat16* __restrict__ c,
                                              size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        c[i] = __hadd(a[i], b[i]);
    }
}

__global__ void vector_add_bf16_kernel_vec8(const float4* __restrict__ a,
                                            const float4* __restrict__ b,
                                            float4* __restrict__ c,
                                            size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        BF16_8Vec va, vb, vc;
        va.v4 = a[i];
        vb.v4 = b[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            vc.b2[k] = __hadd2(va.b2[k], vb.b2[k]);
        }
        c[i] = vc.v4;
    }
}

void launch_vector_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(a) && is_aligned_16(b) && is_aligned_16(c);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_add_bf16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            vector_add_bf16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                a + offset, b + offset, c + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_add_bf16_kernel_scalar<<<blocks, threads, 0, stream>>>(a, b, c, n);
    }
}

// Vector Scale BF16
__global__ void vector_scale_bf16_kernel_scalar(const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out,
                                                __nv_bfloat16 alpha,
                                                size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = __hmul(x[i], alpha);
    }
}

__global__ void vector_scale_bf16_kernel_vec8(const float4* __restrict__ x,
                                              float4* __restrict__ out,
                                              __nv_bfloat162 alpha2,
                                              size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        BF16_8Vec vx, vout;
        vx.v4 = x[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            vout.b2[k] = __hmul2(vx.b2[k], alpha2);
        }
        out[i] = vout.v4;
    }
}

void launch_vector_scale_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, float alpha, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    __nv_bfloat16 alpha_b = __float2bfloat16(alpha);
    __nv_bfloat162 alpha2;
    alpha2.x = alpha_b;
    alpha2.y = alpha_b;

    bool aligned = is_aligned_16(x) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_scale_bf16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x),
            reinterpret_cast<float4*>(out),
            alpha2,
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            vector_scale_bf16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                x + offset, out + offset, alpha_b, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        vector_scale_bf16_kernel_scalar<<<blocks, threads, 0, stream>>>(x, out, alpha_b, n);
    }
}

// Fused Add + ReLU BF16
__global__ void fused_add_relu_bf16_kernel_scalar(const __nv_bfloat16* __restrict__ a,
                                                  const __nv_bfloat16* __restrict__ b,
                                                  __nv_bfloat16* __restrict__ out,
                                                  size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float sum = __bfloat162float(__hadd(a[i], b[i]));
        out[i] = __float2bfloat16((isnan(sum) || sum > 0.0f) ? sum : 0.0f);
    }
}

__global__ void fused_add_relu_bf16_kernel_vec8(const float4* __restrict__ a,
                                                const float4* __restrict__ b,
                                                float4* __restrict__ out,
                                                size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        BF16_8Vec va, vb, vout;
        va.v4 = a[i];
        vb.v4 = b[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 s2 = __hadd2(va.b2[k], vb.b2[k]);
            float f0 = __bfloat162float(s2.x);
            float f1 = __bfloat162float(s2.y);
            vout.b2[k].x = __float2bfloat16((isnan(f0) || f0 > 0.0f) ? f0 : 0.0f);
            vout.b2[k].y = __float2bfloat16((isnan(f1) || f1 > 0.0f) ? f1 : 0.0f);
        }
        out[i] = vout.v4;
    }
}

void launch_fused_add_relu_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(a) && is_aligned_16(b) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        fused_add_relu_bf16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(out),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            fused_add_relu_bf16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                a + offset, b + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        fused_add_relu_bf16_kernel_scalar<<<blocks, threads, 0, stream>>>(a, b, out, n);
    }
}

// Fused Bias + Activation BF16
__global__ void fused_bias_activation_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const __nv_bfloat16* __restrict__ bias,
                                                  __nv_bfloat16* __restrict__ out,
                                                  size_t rows,
                                                  size_t cols,
                                                  int act_type) {
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        size_t idx = row * cols + col;
        float val = __bfloat162float(x[idx]) + __bfloat162float(bias[col]);
        out[idx] = __float2bfloat16(apply_act(val, act_type));
    }
}

void launch_fused_bias_activation_bf16(const __nv_bfloat16* x,
                                       const __nv_bfloat16* bias,
                                       __nv_bfloat16* out,
                                       size_t rows,
                                       size_t cols,
                                       int act_type,
                                       cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((cols + block.x - 1) / block.x),
        static_cast<unsigned int>((rows + block.y - 1) / block.y)
    );

    fused_bias_activation_bf16_kernel<<<grid, block, 0, stream>>>(
        x, bias, out, rows, cols, act_type
    );
}

