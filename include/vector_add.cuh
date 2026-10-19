#pragma once

#include "benchmark_types.h"
#include "cuda_utils.cuh"

// CPU Vector Addition Reference
void vectorAddCPU(const float* a, const float* b, float* c, size_t n);

// CUDA Vector Addition Kernel
__global__ void vectorAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                size_t n);

// Vector Addition Benchmark Runner
bool runVectorAddBenchmark(size_t n, int warmup_runs, int timed_runs, BenchmarkResult& outResult);
