#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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

// Check 16-byte alignment for 128-bit vectorized loads/stores
inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
}

// 128-bit vectorization containers (16 bytes)
union Float4Vec {
    float4 v4;
    float f[4];
};

union Half8Vec {
    float4 v4;
    half2 h2[4];
    half h[8];
};

union BF16_8Vec {
    float4 v4;
    __nv_bfloat162 b2[4];
    __nv_bfloat16 b[8];
};

// Warp shuffle down primitives
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

