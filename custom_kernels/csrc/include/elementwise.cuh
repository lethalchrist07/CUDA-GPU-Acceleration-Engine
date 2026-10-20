#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>

// Vector Addition
void launch_vector_add(
    const float* a,
    const float* b,
    float* c,
    size_t n,
    cudaStream_t stream
);

void launch_vector_add_fp16(
    const half* a,
    const half* b,
    half* c,
    size_t n,
    cudaStream_t stream
);

void launch_vector_add_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    size_t n,
    cudaStream_t stream
);

// Vector Scaling
void launch_vector_scale(
    const float* x,
    float* out,
    float alpha,
    size_t n,
    cudaStream_t stream
);

void launch_vector_scale_fp16(
    const half* x,
    half* out,
    float alpha,
    size_t n,
    cudaStream_t stream
);

void launch_vector_scale_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* out,
    float alpha,
    size_t n,
    cudaStream_t stream
);

// Fused Add + ReLU
void launch_fused_add_relu(
    const float* a,
    const float* b,
    float* out,
    size_t n,
    cudaStream_t stream
);

void launch_fused_add_relu_fp16(
    const half* a,
    const half* b,
    half* out,
    size_t n,
    cudaStream_t stream
);

void launch_fused_add_relu_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    size_t n,
    cudaStream_t stream
);

// Fused Bias + Activation
void launch_fused_bias_activation(
    const float* x,
    const float* bias,
    float* out,
    size_t rows,
    size_t cols,
    int activation_type, // 0: None, 1: ReLU, 2: GELU
    cudaStream_t stream
);

void launch_fused_bias_activation_fp16(
    const half* x,
    const half* bias,
    half* out,
    size_t rows,
    size_t cols,
    int activation_type,
    cudaStream_t stream
);

void launch_fused_bias_activation_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    __nv_bfloat16* out,
    size_t rows,
    size_t cols,
    int activation_type,
    cudaStream_t stream
);

