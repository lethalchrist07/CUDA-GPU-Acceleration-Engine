#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * Launch Ampere Tensor Core accelerated matrix multiplication using CUDA WMMA.
 * C = A @ B
 * Accumulation is performed in FP32 for numerical stability.
 *
 * Supports arbitrary M, N, K (including non-multiples of 16 and irregular shapes)
 * through zero-padded shared memory staging and boundary-checked epilogue writes.
 *
 * Output: FP16
 */
void launch_wmma_matmul_fp16_out(
    const half* a,
    const half* b,
    half* c,
    int M,
    int K,
    int N,
    cudaStream_t stream
);

/**
 * Launch Ampere Tensor Core accelerated matrix multiplication using CUDA WMMA.
 * C = A @ B
 * Accumulation is performed in FP32 for numerical stability.
 *
 * Output: FP32
 */
void launch_wmma_matmul_fp32_out(
    const half* a,
    const half* b,
    float* c,
    int M,
    int K,
    int N,
    cudaStream_t stream
);
