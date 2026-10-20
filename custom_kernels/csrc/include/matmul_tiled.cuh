#pragma once

#include <cuda_runtime.h>

#define TILE_DIM 32

void launch_tiled_matmul(
    const float* a,
    const float* b,
    float* c,
    int M,
    int K,
    int N,
    cudaStream_t stream
);
