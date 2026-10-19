#pragma once

#include <iostream>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "[-] CUDA Error at " << __FILE__ << ":" << __LINE__  \
                      << " -> " << cudaGetErrorString(err)                    \
                      << " (code " << static_cast<int>(err) << ")"            \
                      << std::endl;                                           \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_VOID(call)                                                 \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "[-] CUDA Error at " << __FILE__ << ":" << __LINE__  \
                      << " -> " << cudaGetErrorString(err)                    \
                      << " (code " << static_cast<int>(err) << ")"            \
                      << std::endl;                                           \
            return;                                                           \
        }                                                                     \
    } while (0)

inline bool checkGpuMemoryAvailable(size_t required_bytes, size_t safety_margin_mb = 128) {
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "[-] Failed to query GPU memory info: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    size_t safety_bytes = safety_margin_mb * 1024 * 1024;
    if (required_bytes + safety_bytes > free_mem) {
        std::cerr << "[-] Memory safety check failed! Required: " 
                  << (required_bytes / (1024.0 * 1024.0)) << " MB, Available Free: " 
                  << (free_mem / (1024.0 * 1024.0)) << " MB (Safety Margin: " 
                  << safety_margin_mb << " MB)\n";
        return false;
    }
    return true;
}
