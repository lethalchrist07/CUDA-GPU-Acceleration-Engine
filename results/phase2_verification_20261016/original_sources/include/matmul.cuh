#pragma once

#include "benchmark_types.h"
#include "cuda_utils.cuh"

// Block tile dimension for shared memory kernel
constexpr int MATMUL_TILE_DIM = 16;

// CPU Matrix Multiplication Reference (Cache-optimized i-k-j loop)
void matmulCPU(const float* a, const float* b, float* c, int n);

// Basic Naive CUDA Matrix Multiplication Kernel (Global Memory)
__global__ void matmulBasicKernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n);

// Tiled Shared Memory CUDA Matrix Multiplication Kernel
__global__ void matmulTiledSharedKernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c,
                                        int n);

// Matrix Multiplication Benchmark Runners
bool runMatmulBenchmarks(int n, 
                         int warmup_runs, 
                         int timed_runs, 
                         BenchmarkResult& outBasicResult, 
                         BenchmarkResult& outTiledResult);
