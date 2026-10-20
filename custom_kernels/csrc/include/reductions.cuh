#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>

// Sum Reduction (FP32 Accumulation)
void launch_reduce_sum(
    const float* in,
    float* out,
    float* d_workspace,
    size_t n,
    cudaStream_t stream
);

void launch_reduce_sum_fp16(
    const half* in,
    void* out,
    float* d_workspace,
    size_t n,
    bool output_fp32,
    cudaStream_t stream
);

void launch_reduce_sum_bf16(
    const __nv_bfloat16* in,
    void* out,
    float* d_workspace,
    size_t n,
    bool output_fp32,
    cudaStream_t stream
);

// Max Reduction
void launch_reduce_max(
    const float* in,
    float* out,
    float* d_workspace,
    size_t n,
    cudaStream_t stream
);

void launch_reduce_max_fp16(
    const half* in,
    half* out,
    half* d_workspace,
    size_t n,
    cudaStream_t stream
);

void launch_reduce_max_bf16(
    const __nv_bfloat16* in,
    __nv_bfloat16* out,
    __nv_bfloat16* d_workspace,
    size_t n,
    cudaStream_t stream
);

