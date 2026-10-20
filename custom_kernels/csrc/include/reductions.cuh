#pragma once

#include <cuda_runtime.h>
#include <cstddef>

void launch_reduce_sum(
    const float* in,
    float* out,
    float* d_workspace,
    size_t n,
    cudaStream_t stream
);

void launch_reduce_max(
    const float* in,
    float* out,
    float* d_workspace,
    size_t n,
    cudaStream_t stream
);
