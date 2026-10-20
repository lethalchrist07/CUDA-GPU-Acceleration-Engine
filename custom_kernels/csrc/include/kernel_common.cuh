#pragma once

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <iostream>

// CUDA kernel launch check
#ifndef CUDA_KERNEL_CHECK
#define CUDA_KERNEL_CHECK()                                                    \
    do {                                                                       \
        cudaError_t __err = cudaGetLastError();                                \
        if (__err != cudaSuccess) {                                            \
            std::string __msg = std::string("CUDA kernel launch failed: ") +   \
                                cudaGetErrorString(__err);                     \
            TORCH_CHECK(false, __msg);                                         \
        }                                                                      \
    } while (0)
#endif

// Check 16-byte alignment for float4 vectorization
inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
}

// Warp shuffle down primitive
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
