#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>

// ReLU
void launch_relu(
    const float* in,
    float* out,
    size_t n,
    cudaStream_t stream
);

void launch_relu_fp16(
    const half* in,
    half* out,
    size_t n,
    cudaStream_t stream
);

void launch_relu_bf16(
    const __nv_bfloat16* in,
    __nv_bfloat16* out,
    size_t n,
    cudaStream_t stream
);

// GELU
void launch_gelu(
    const float* in,
    float* out,
    size_t n,
    cudaStream_t stream
);

void launch_gelu_fp16(
    const half* in,
    half* out,
    size_t n,
    cudaStream_t stream
);

void launch_gelu_bf16(
    const __nv_bfloat16* in,
    __nv_bfloat16* out,
    size_t n,
    cudaStream_t stream
);

// Sigmoid
void launch_sigmoid(
    const float* in,
    float* out,
    size_t n,
    cudaStream_t stream
);

void launch_sigmoid_fp16(
    const half* in,
    half* out,
    size_t n,
    cudaStream_t stream
);

void launch_sigmoid_bf16(
    const __nv_bfloat16* in,
    __nv_bfloat16* out,
    size_t n,
    cudaStream_t stream
);

