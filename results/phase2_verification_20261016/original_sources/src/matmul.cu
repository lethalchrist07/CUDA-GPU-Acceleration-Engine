#include "matmul.cuh"
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

// Cache-optimized i-k-j loop order CPU reference
void matmulCPU(const float* a, const float* b, float* c, int n) {
    // Zero out c
    for (int i = 0; i < n * n; ++i) {
        c[i] = 0.0f;
    }
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_ik = a[i * n + k];
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
}

// Basic Naive CUDA Kernel (Global Memory)
__global__ void matmulBasicKernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Tiled Shared Memory CUDA Kernel
__global__ void matmulTiledSharedKernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c,
                                        int n) {
    __shared__ float s_A[MATMUL_TILE_DIM][MATMUL_TILE_DIM];
    __shared__ float s_B[MATMUL_TILE_DIM][MATMUL_TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * MATMUL_TILE_DIM + ty;
    int col = blockIdx.x * MATMUL_TILE_DIM + tx;

    float sum = 0.0f;
    int numTiles = (n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * MATMUL_TILE_DIM + tx;
        int bRow = t * MATMUL_TILE_DIM + ty;

        s_A[ty][tx] = (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;
        s_B[ty][tx] = (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MATMUL_TILE_DIM; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

bool runMatmulBenchmarks(int n, 
                         int warmup_runs, 
                         int timed_runs, 
                         BenchmarkResult& outBasicResult, 
                         BenchmarkResult& outTiledResult) {
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);
    const size_t total_required_gpu_bytes = 3 * bytes;

    if (!checkGpuMemoryAvailable(total_required_gpu_bytes, 128)) {
        std::cerr << "[-] Skipping Matmul benchmark for N=" << n << " due to VRAM limits.\n";
        return false;
    }

    // 1. Initialize Base Results Metadata
    auto setupMeta = [&](BenchmarkResult& res, const std::string& variant) {
        res.category = "Matrix Multiplication";
        res.variant = variant;
        res.dimension = static_cast<size_t>(n);
        res.workload_size = std::to_string(n) + "x" + std::to_string(n);
        res.memory_mb = static_cast<double>(total_required_gpu_bytes) / (1024.0 * 1024.0);
        res.warmup_runs = warmup_runs;
        res.timed_runs = timed_runs;
        res.tolerance = 1e-3f; // Floating-point sum accumulation tolerance for float32 GEMM
    };
    setupMeta(outBasicResult, "Basic Naive CUDA");
    setupMeta(outTiledResult, "Tiled Shared Memory CUDA");

    // 2. Host Allocations & Normalization (keep inputs within [-1.0, 1.0] to prevent overflow)
    std::vector<float> h_A(elements);
    std::vector<float> h_B(elements);
    std::vector<float> h_C_cpu(elements, 0.0f);
    std::vector<float> h_C_basic(elements, 0.0f);
    std::vector<float> h_C_tiled(elements, 0.0f);

    for (size_t i = 0; i < elements; ++i) {
        h_A[i] = std::sin(static_cast<float>(i % 360) * 0.0174533f) * 0.5f;
        h_B[i] = std::cos(static_cast<float>(i % 360) * 0.0174533f) * 0.5f;
    }

    // 3. CPU Reference Timing
    // CPU warmup
    matmulCPU(h_A.data(), h_B.data(), h_C_cpu.data(), n);

    std::vector<double> cpu_times;
    int cpu_runs = (n >= 2048) ? 2 : timed_runs; // For 2048x2048, 2 runs suffice to save time
    cpu_times.reserve(cpu_runs);
    for (int r = 0; r < cpu_runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmulCPU(h_A.data(), h_B.data(), h_C_cpu.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_times.push_back(ms);
    }
    double cpu_median_ms = computeMedian(cpu_times);
    outBasicResult.cpu_time_ms = cpu_median_ms;
    outTiledResult.cpu_time_ms = cpu_median_ms;

    // 4. Device Allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes));

    dim3 blockDim(MATMUL_TILE_DIM, MATMUL_TILE_DIM);
    dim3 gridDim((n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM,
                 (n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM);

    // Flops for GEMM: 2 * n^3
    double totalFlops = 2.0 * static_cast<double>(n) * n * n;

    // Timing Events
    cudaEvent_t evH2DStart, evH2DStop, evKStart, evKStop, evD2HStart, evD2HStop;
    CUDA_CHECK(cudaEventCreate(&evH2DStart));
    CUDA_CHECK(cudaEventCreate(&evH2DStop));
    CUDA_CHECK(cudaEventCreate(&evKStart));
    CUDA_CHECK(cudaEventCreate(&evKStop));
    CUDA_CHECK(cudaEventCreate(&evD2HStart));
    CUDA_CHECK(cudaEventCreate(&evD2HStop));

    // ==========================================
    // 5. RUN BASIC NAIVE CUDA KERNEL BENCHMARK
    // ==========================================
    // Warmup
    for (int w = 0; w < warmup_runs; ++w) {
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
        matmulBasicKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        cudaMemcpy(h_C_basic.data(), d_C, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    std::vector<double> basic_h2d, basic_kernel, basic_d2h, basic_e2e;
    for (int r = 0; r < timed_runs; ++r) {
        CUDA_CHECK(cudaEventRecord(evH2DStart));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(evH2DStop));

        CUDA_CHECK(cudaEventRecord(evKStart));
        matmulBasicKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(evKStop));

        CUDA_CHECK(cudaEventRecord(evD2HStart));
        CUDA_CHECK(cudaMemcpy(h_C_basic.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(evD2HStop));

        CUDA_CHECK(cudaEventSynchronize(evD2HStop));

        float h2d_ms = 0.0f, k_ms = 0.0f, d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, evH2DStart, evH2DStop));
        CUDA_CHECK(cudaEventElapsedTime(&k_ms, evKStart, evKStop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, evD2HStart, evD2HStop));

        basic_h2d.push_back(h2d_ms);
        basic_kernel.push_back(k_ms);
        basic_d2h.push_back(d2h_ms);
        basic_e2e.push_back(h2d_ms + k_ms + d2h_ms);
    }

    outBasicResult.gpu_h2d_ms = computeMedian(basic_h2d);
    outBasicResult.gpu_kernel_ms = computeMedian(basic_kernel);
    outBasicResult.gpu_d2h_ms = computeMedian(basic_d2h);
    outBasicResult.gpu_end_to_end_ms = computeMedian(basic_e2e);
    outBasicResult.kernel_speedup = (outBasicResult.gpu_kernel_ms > 0.0) 
        ? (cpu_median_ms / outBasicResult.gpu_kernel_ms) : 0.0;
    outBasicResult.end_to_end_speedup = (outBasicResult.gpu_end_to_end_ms > 0.0) 
        ? (cpu_median_ms / outBasicResult.gpu_end_to_end_ms) : 0.0;
    outBasicResult.gflops = (totalFlops / (outBasicResult.gpu_kernel_ms * 1e-3)) / 1e9;
    // Basic kernel reads A and B n times: (2*n + 1) * n^2 * sizeof(float)
    outBasicResult.memory_bandwidth_gb_s = ((2.0 * n + 1.0) * elements * sizeof(float) / 
                                            (outBasicResult.gpu_kernel_ms * 1e-3)) / 1e9;

    // Verify Basic
    float max_err_basic = 0.0f;
    double sum_sq_basic = 0.0;
    int mismatches_basic = 0;
    for (size_t i = 0; i < elements; ++i) {
        float diff = std::fabs(h_C_basic[i] - h_C_cpu[i]);
        if (diff > max_err_basic) max_err_basic = diff;
        sum_sq_basic += diff * diff;
        if (diff > outBasicResult.tolerance) {
            mismatches_basic++;
        }
    }
    outBasicResult.max_abs_error = max_err_basic;
    outBasicResult.rmse = std::sqrt(sum_sq_basic / elements);
    outBasicResult.mismatched_elements = mismatches_basic;
    outBasicResult.verified = (mismatches_basic == 0);

    // ==========================================
    // 6. RUN TILED SHARED MEMORY KERNEL BENCHMARK
    // ==========================================
    // Warmup
    for (int w = 0; w < warmup_runs; ++w) {
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
        matmulTiledSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        cudaMemcpy(h_C_tiled.data(), d_C, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    std::vector<double> tiled_h2d, tiled_kernel, tiled_d2h, tiled_e2e;
    for (int r = 0; r < timed_runs; ++r) {
        CUDA_CHECK(cudaEventRecord(evH2DStart));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(evH2DStop));

        CUDA_CHECK(cudaEventRecord(evKStart));
        matmulTiledSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(evKStop));

        CUDA_CHECK(cudaEventRecord(evD2HStart));
        CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(evD2HStop));

        CUDA_CHECK(cudaEventSynchronize(evD2HStop));

        float h2d_ms = 0.0f, k_ms = 0.0f, d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, evH2DStart, evH2DStop));
        CUDA_CHECK(cudaEventElapsedTime(&k_ms, evKStart, evKStop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, evD2HStart, evD2HStop));

        tiled_h2d.push_back(h2d_ms);
        tiled_kernel.push_back(k_ms);
        tiled_d2h.push_back(d2h_ms);
        tiled_e2e.push_back(h2d_ms + k_ms + d2h_ms);
    }

    outTiledResult.gpu_h2d_ms = computeMedian(tiled_h2d);
    outTiledResult.gpu_kernel_ms = computeMedian(tiled_kernel);
    outTiledResult.gpu_d2h_ms = computeMedian(tiled_d2h);
    outTiledResult.gpu_end_to_end_ms = computeMedian(tiled_e2e);
    outTiledResult.kernel_speedup = (outTiledResult.gpu_kernel_ms > 0.0) 
        ? (cpu_median_ms / outTiledResult.gpu_kernel_ms) : 0.0;
    outTiledResult.end_to_end_speedup = (outTiledResult.gpu_end_to_end_ms > 0.0) 
        ? (cpu_median_ms / outTiledResult.gpu_end_to_end_ms) : 0.0;
    outTiledResult.gflops = (totalFlops / (outTiledResult.gpu_kernel_ms * 1e-3)) / 1e9;
    // Tiled kernel reads each element once per tile phase from global memory:
    // Global loads: 2 * (n / MATMUL_TILE_DIM) * elements + elements write
    double globalTransfers = (2.0 * static_cast<double>(n) / MATMUL_TILE_DIM + 1.0) * elements * sizeof(float);
    outTiledResult.memory_bandwidth_gb_s = (globalTransfers / (outTiledResult.gpu_kernel_ms * 1e-3)) / 1e9;

    // Verify Tiled
    float max_err_tiled = 0.0f;
    double sum_sq_tiled = 0.0;
    int mismatches_tiled = 0;
    for (size_t i = 0; i < elements; ++i) {
        float diff = std::fabs(h_C_tiled[i] - h_C_cpu[i]);
        if (diff > max_err_tiled) max_err_tiled = diff;
        sum_sq_tiled += diff * diff;
        if (diff > outTiledResult.tolerance) {
            mismatches_tiled++;
        }
    }
    outTiledResult.max_abs_error = max_err_tiled;
    outTiledResult.rmse = std::sqrt(sum_sq_tiled / elements);
    outTiledResult.mismatched_elements = mismatches_tiled;
    outTiledResult.verified = (mismatches_tiled == 0);

    // Cleanup
    cudaEventDestroy(evH2DStart);
    cudaEventDestroy(evH2DStop);
    cudaEventDestroy(evKStart);
    cudaEventDestroy(evKStop);
    cudaEventDestroy(evD2HStart);
    cudaEventDestroy(evD2HStop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return true;
}
