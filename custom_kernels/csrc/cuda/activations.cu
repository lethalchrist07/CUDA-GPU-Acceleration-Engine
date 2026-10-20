#include "../include/activations.cuh"
#include "../include/kernel_common.cuh"
#include <cmath>

// =========================================================================
// ReLU
// =========================================================================
__device__ __forceinline__ float relu_scalar(float x) {
    return (isnan(x) || x > 0.0f) ? x : 0.0f;
}

__global__ void relu_kernel_scalar(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = relu_scalar(in[i]);
    }
}

__global__ void relu_kernel_vec4(const float4* __restrict__ in,
                                 float4* __restrict__ out,
                                 size_t n_vec4) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec4; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float4 vin = in[i];
        float4 vout;
        vout.x = relu_scalar(vin.x);
        vout.y = relu_scalar(vin.y);
        vout.z = relu_scalar(vin.z);
        vout.w = relu_scalar(vin.w);
        out[i] = vout;
    }
}

void launch_relu(const float* in, float* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 4) {
        size_t n_vec4 = n / 4;
        size_t tail = n % 4;

        int threads = 256;
        int blocks = static_cast<int>((n_vec4 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        relu_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec4
        );

        if (tail > 0) {
            size_t offset = n_vec4 * 4;
            relu_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        relu_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

// =========================================================================
// GELU (Fast Tanh Approximation matching PyTorch approximate='tanh')
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// =========================================================================
__device__ __forceinline__ float gelu_tanh_scalar(float x) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel_scalar(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = gelu_tanh_scalar(in[i]);
    }
}

__global__ void gelu_kernel_vec4(const float4* __restrict__ in,
                                 float4* __restrict__ out,
                                 size_t n_vec4) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec4; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float4 vin = in[i];
        float4 vout;
        vout.x = gelu_tanh_scalar(vin.x);
        vout.y = gelu_tanh_scalar(vin.y);
        vout.z = gelu_tanh_scalar(vin.z);
        vout.w = gelu_tanh_scalar(vin.w);
        out[i] = vout;
    }
}

void launch_gelu(const float* in, float* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 4) {
        size_t n_vec4 = n / 4;
        size_t tail = n % 4;

        int threads = 256;
        int blocks = static_cast<int>((n_vec4 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        gelu_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec4
        );

        if (tail > 0) {
            size_t offset = n_vec4 * 4;
            gelu_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        gelu_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

// =========================================================================
// Sigmoid
// =========================================================================
__device__ __forceinline__ float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void sigmoid_kernel_scalar(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = sigmoid_scalar(in[i]);
    }
}

__global__ void sigmoid_kernel_vec4(const float4* __restrict__ in,
                                    float4* __restrict__ out,
                                    size_t n_vec4) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec4; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float4 vin = in[i];
        float4 vout;
        vout.x = sigmoid_scalar(vin.x);
        vout.y = sigmoid_scalar(vin.y);
        vout.z = sigmoid_scalar(vin.z);
        vout.w = sigmoid_scalar(vin.w);
        out[i] = vout;
    }
}

void launch_sigmoid(const float* in, float* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 4) {
        size_t n_vec4 = n / 4;
        size_t tail = n % 4;

        int threads = 256;
        int blocks = static_cast<int>((n_vec4 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        sigmoid_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec4
        );

        if (tail > 0) {
            size_t offset = n_vec4 * 4;
            sigmoid_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        sigmoid_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

// =========================================================================
// FP16 Mixed-Precision Activations
// =========================================================================

// ReLU FP16
__global__ void relu_fp16_kernel_scalar(const half* __restrict__ in,
                                        half* __restrict__ out,
                                        size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float s = __half2float(in[i]);
        out[i] = __float2half((isnan(s) || s > 0.0f) ? s : 0.0f);
    }
}

__global__ void relu_fp16_kernel_vec8(const float4* __restrict__ in,
                                      float4* __restrict__ out,
                                      size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        Half8Vec vin, vout;
        vin.v4 = in[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f0 = __half2float(vin.h2[k].x);
            float f1 = __half2float(vin.h2[k].y);
            vout.h2[k].x = __float2half((isnan(f0) || f0 > 0.0f) ? f0 : 0.0f);
            vout.h2[k].y = __float2half((isnan(f1) || f1 > 0.0f) ? f1 : 0.0f);
        }
        out[i] = vout.v4;
    }
}

void launch_relu_fp16(const half* in, half* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        relu_fp16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            relu_fp16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        relu_fp16_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

// GELU FP16
__global__ void gelu_fp16_kernel_scalar(const half* __restrict__ in,
                                        half* __restrict__ out,
                                        size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = __float2half(gelu_tanh_scalar(__half2float(in[i])));
    }
}

__global__ void gelu_fp16_kernel_vec8(const float4* __restrict__ in,
                                      float4* __restrict__ out,
                                      size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        Half8Vec vin, vout;
        vin.v4 = in[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f0 = gelu_tanh_scalar(__half2float(vin.h2[k].x));
            float f1 = gelu_tanh_scalar(__half2float(vin.h2[k].y));
            vout.h2[k].x = __float2half(f0);
            vout.h2[k].y = __float2half(f1);
        }
        out[i] = vout.v4;
    }
}

void launch_gelu_fp16(const half* in, half* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        gelu_fp16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            gelu_fp16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        gelu_fp16_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

// Sigmoid FP16
__global__ void sigmoid_fp16_kernel_scalar(const half* __restrict__ in,
                                           half* __restrict__ out,
                                           size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = __float2half(sigmoid_scalar(__half2float(in[i])));
    }
}

__global__ void sigmoid_fp16_kernel_vec8(const float4* __restrict__ in,
                                         float4* __restrict__ out,
                                         size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        Half8Vec vin, vout;
        vin.v4 = in[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f0 = sigmoid_scalar(__half2float(vin.h2[k].x));
            float f1 = sigmoid_scalar(__half2float(vin.h2[k].y));
            vout.h2[k].x = __float2half(f0);
            vout.h2[k].y = __float2half(f1);
        }
        out[i] = vout.v4;
    }
}

void launch_sigmoid_fp16(const half* in, half* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        sigmoid_fp16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            sigmoid_fp16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        sigmoid_fp16_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

// =========================================================================
// BF16 Mixed-Precision Activations
// =========================================================================

// ReLU BF16
__global__ void relu_bf16_kernel_scalar(const __nv_bfloat16* __restrict__ in,
                                        __nv_bfloat16* __restrict__ out,
                                        size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float s = __bfloat162float(in[i]);
        out[i] = __float2bfloat16((isnan(s) || s > 0.0f) ? s : 0.0f);
    }
}

__global__ void relu_bf16_kernel_vec8(const float4* __restrict__ in,
                                      float4* __restrict__ out,
                                      size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        BF16_8Vec vin, vout;
        vin.v4 = in[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f0 = __bfloat162float(vin.b2[k].x);
            float f1 = __bfloat162float(vin.b2[k].y);
            vout.b2[k].x = __float2bfloat16((isnan(f0) || f0 > 0.0f) ? f0 : 0.0f);
            vout.b2[k].y = __float2bfloat16((isnan(f1) || f1 > 0.0f) ? f1 : 0.0f);
        }
        out[i] = vout.v4;
    }
}

void launch_relu_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        relu_bf16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            relu_bf16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        relu_bf16_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

// GELU BF16
__global__ void gelu_bf16_kernel_scalar(const __nv_bfloat16* __restrict__ in,
                                        __nv_bfloat16* __restrict__ out,
                                        size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = __float2bfloat16(gelu_tanh_scalar(__bfloat162float(in[i])));
    }
}

__global__ void gelu_bf16_kernel_vec8(const float4* __restrict__ in,
                                      float4* __restrict__ out,
                                      size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        BF16_8Vec vin, vout;
        vin.v4 = in[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f0 = gelu_tanh_scalar(__bfloat162float(vin.b2[k].x));
            float f1 = gelu_tanh_scalar(__bfloat162float(vin.b2[k].y));
            vout.b2[k].x = __float2bfloat16(f0);
            vout.b2[k].y = __float2bfloat16(f1);
        }
        out[i] = vout.v4;
    }
}

void launch_gelu_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        gelu_bf16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            gelu_bf16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        gelu_bf16_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

// Sigmoid BF16
__global__ void sigmoid_bf16_kernel_scalar(const __nv_bfloat16* __restrict__ in,
                                           __nv_bfloat16* __restrict__ out,
                                           size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[i] = __float2bfloat16(sigmoid_scalar(__bfloat162float(in[i])));
    }
}

__global__ void sigmoid_bf16_kernel_vec8(const float4* __restrict__ in,
                                         float4* __restrict__ out,
                                         size_t n_vec8) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n_vec8; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        BF16_8Vec vin, vout;
        vin.v4 = in[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f0 = sigmoid_scalar(__bfloat162float(vin.b2[k].x));
            float f1 = sigmoid_scalar(__bfloat162float(vin.b2[k].y));
            vout.b2[k].x = __float2bfloat16(f0);
            vout.b2[k].y = __float2bfloat16(f1);
        }
        out[i] = vout.v4;
    }
}

void launch_sigmoid_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    bool aligned = is_aligned_16(in) && is_aligned_16(out);
    if (aligned && n >= 8) {
        size_t n_vec8 = n / 8;
        size_t tail = n % 8;

        int threads = 256;
        int blocks = static_cast<int>((n_vec8 + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        sigmoid_bf16_kernel_vec8<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(out),
            n_vec8
        );

        if (tail > 0) {
            size_t offset = n_vec8 * 8;
            sigmoid_bf16_kernel_scalar<<<1, static_cast<int>(tail), 0, stream>>>(
                in + offset, out + offset, tail
            );
        }
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        sigmoid_bf16_kernel_scalar<<<blocks, threads, 0, stream>>>(in, out, n);
    }
}

