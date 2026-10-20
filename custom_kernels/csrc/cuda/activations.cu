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
