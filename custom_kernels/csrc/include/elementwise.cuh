#pragma once

#include <cuda_runtime.h>
#include <cstddef>

void launch_vector_add(
    const float* a,
    const float* b,
    float* c,
    size_t n,
    cudaStream_t stream
);

void launch_vector_scale(
    const float* x,
    float* out,
    float alpha,
    size_t n,
    cudaStream_t stream
);

void launch_fused_add_relu(
    const float* a,
    const float* b,
    float* out,
    size_t n,
    cudaStream_t stream
);

void launch_fused_bias_activation(
    const float* x,
    const float* bias,
    float* out,
    size_t rows,
    size_t cols,
    int activation_type, // 0: None, 1: ReLU, 2: GELU
    cudaStream_t stream
);
