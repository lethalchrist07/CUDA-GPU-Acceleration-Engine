#include "vector_add.cuh"
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

void vectorAddCPU(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                size_t n) {
    size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

bool runVectorAddBenchmark(size_t n, int warmup_runs, int timed_runs, BenchmarkResult& outResult) {
    const size_t bytes = n * sizeof(float);
    const size_t total_required_gpu_bytes = 3 * bytes;

    if (!checkGpuMemoryAvailable(total_required_gpu_bytes, 128)) {
        std::cerr << "[-] Skipping Vector Add benchmark for N=" << n << " due to VRAM limits.\n";
        return false;
    }

    outResult.category = "Vector Addition";
    outResult.variant = "Standard CUDA Kernel";
    outResult.dimension = n;
    outResult.workload_size = std::to_string(n) + " elements";
    outResult.memory_mb = static_cast<double>(total_required_gpu_bytes) / (1024.0 * 1024.0);
    outResult.warmup_runs = warmup_runs;
    outResult.timed_runs = timed_runs;
    outResult.tolerance = 1e-5f;

    // 1. Host Memory Allocation & Initialization
    std::vector<float> h_A(n);
    std::vector<float> h_B(n);
    std::vector<float> h_C_cpu(n, 0.0f);
    std::vector<float> h_C_gpu(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        h_A[i] = std::sin(static_cast<float>(i % 360) * 0.0174533f) * 2.5f;
        h_B[i] = std::cos(static_cast<float>(i % 360) * 0.0174533f) * 1.5f;
    }

    // 2. CPU Timing (Warmup + Repeated Runs)
    // CPU warmup
    vectorAddCPU(h_A.data(), h_B.data(), h_C_cpu.data(), n);

    std::vector<double> cpu_times;
    cpu_times.reserve(timed_runs);
    for (int r = 0; r < timed_runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vectorAddCPU(h_A.data(), h_B.data(), h_C_cpu.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_times.push_back(ms);
    }
    outResult.cpu_time_ms = computeMedian(cpu_times);

    // 3. GPU Memory Allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes));

    const int threadsPerBlock = 256;
    const unsigned int blocksPerGrid = static_cast<unsigned int>((n + threadsPerBlock - 1) / threadsPerBlock);

    // 4. GPU Warmup
    for (int w = 0; w < warmup_runs; ++w) {
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
        cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    // 5. Timed GPU Runs
    std::vector<double> h2d_times;
    std::vector<double> kernel_times;
    std::vector<double> d2h_times;
    std::vector<double> e2e_times;

    cudaEvent_t evH2DStart, evH2DStop, evKStart, evKStop, evD2HStart, evD2HStop;
    CUDA_CHECK(cudaEventCreate(&evH2DStart));
    CUDA_CHECK(cudaEventCreate(&evH2DStop));
    CUDA_CHECK(cudaEventCreate(&evKStart));
    CUDA_CHECK(cudaEventCreate(&evKStop));
    CUDA_CHECK(cudaEventCreate(&evD2HStart));
    CUDA_CHECK(cudaEventCreate(&evD2HStop));

    for (int r = 0; r < timed_runs; ++r) {
        // H2D
        CUDA_CHECK(cudaEventRecord(evH2DStart));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(evH2DStop));

        // Kernel
        CUDA_CHECK(cudaEventRecord(evKStart));
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(evKStop));

        // D2H
        CUDA_CHECK(cudaEventRecord(evD2HStart));
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(evD2HStop));

        CUDA_CHECK(cudaEventSynchronize(evD2HStop));

        float h2d_ms = 0.0f, k_ms = 0.0f, d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, evH2DStart, evH2DStop));
        CUDA_CHECK(cudaEventElapsedTime(&k_ms, evKStart, evKStop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, evD2HStart, evD2HStop));

        h2d_times.push_back(h2d_ms);
        kernel_times.push_back(k_ms);
        d2h_times.push_back(d2h_ms);
        e2e_times.push_back(h2d_ms + k_ms + d2h_ms);
    }

    outResult.gpu_h2d_ms = computeMedian(h2d_times);
    outResult.gpu_kernel_ms = computeMedian(kernel_times);
    outResult.gpu_d2h_ms = computeMedian(d2h_times);
    outResult.gpu_end_to_end_ms = computeMedian(e2e_times);

    // 6. Metrics Calculation
    outResult.kernel_speedup = (outResult.gpu_kernel_ms > 0.0) 
        ? (outResult.cpu_time_ms / outResult.gpu_kernel_ms) : 0.0;
    outResult.end_to_end_speedup = (outResult.gpu_end_to_end_ms > 0.0) 
        ? (outResult.cpu_time_ms / outResult.gpu_end_to_end_ms) : 0.0;
    outResult.gflops = (static_cast<double>(n) / (outResult.gpu_kernel_ms * 1e-3)) / 1e9;
    outResult.memory_bandwidth_gb_s = (static_cast<double>(total_required_gpu_bytes) / 
                                       (outResult.gpu_kernel_ms * 1e-3)) / 1e9;

    // 7. Numerical Verification
    float max_err = 0.0f;
    double sum_sq_err = 0.0;
    int mismatches = 0;

    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > max_err) max_err = diff;
        sum_sq_err += diff * diff;
        if (diff > outResult.tolerance) {
            mismatches++;
        }
    }

    outResult.max_abs_error = max_err;
    outResult.rmse = std::sqrt(sum_sq_err / n);
    outResult.mismatched_elements = mismatches;
    outResult.verified = (mismatches == 0);

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
