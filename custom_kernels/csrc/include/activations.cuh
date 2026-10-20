#pragma once

#include <cuda_runtime.h>
#include <cstddef>

void launch_relu(
    const float* in,
    float* out,
    size_t n,
    cudaStream_t stream
);

void launch_gelu(
    const float* in,
    float* out,
    size_t n,
    cudaStream_t stream
);

void launch_sigmoid(
    const float* in,
    float* out,
    size_t n,
    cudaStream_t stream
);
