#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// Robust CUDA error-checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err)                    \
                      << " (" << err << ")" << std::endl;                     \
            return 1;                                                         \
        }                                                                     \
    } while (0)

// CUDA Vector Addition Kernel
__global__ void vectorAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU Reference Implementation
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Helper to determine cores per SM based on Compute Capability
int getCoresPerSM(int major, int minor) {
    switch (major) {
        case 7:
            return 64; // Volta (7.0, 7.2), Turing (7.5)
        case 8:
            if (minor == 0) return 64;  // Ampere GA100
            if (minor == 6 || minor == 7 || minor == 9) return 128; // Ampere GA10x / Ada Lovelace
            return 128;
        case 9:
            return 128; // Hopper
        default:
            return 128;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "  CUDA GPU Acceleration Engine - Diagnostics & Check\n";
    std::cout << "========================================================\n\n";

    // 1. Toolchain & Runtime Information
    int driverVersion = 0;
    int runtimeVersion = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    std::cout << "[SYSTEM / TOOLCHAIN]\n";
    std::cout << "  CUDA Driver Version:  " << (driverVersion / 1000) << "." << (driverVersion % 100) << "\n";
    std::cout << "  CUDA Runtime Version: " << (runtimeVersion / 1000) << "." << (runtimeVersion % 100) << "\n";

    // 2. GPU Detection & Hardware Properties
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "[-] Error: No CUDA-capable devices found.\n";
        return 1;
    }

    int devId = 0;
    CUDA_CHECK(cudaSetDevice(devId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devId));

    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    int memoryClockRateKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memoryClockRateKHz, cudaDevAttrMemoryClockRate, devId));
    int coreClockRateKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coreClockRateKHz, cudaDevAttrClockRate, devId));

    int coresPerSM = getCoresPerSM(prop.major, prop.minor);
    int totalCores = prop.multiProcessorCount * coresPerSM;

    std::cout << "\n[GPU HARDWARE SPECIFICATIONS]\n";
    std::cout << "  Device ID:                  " << devId << "\n";
    std::cout << "  Device Name:                " << prop.name << "\n";
    std::cout << "  Compute Capability:         " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Streaming Multiprocessors:  " << prop.multiProcessorCount << "\n";
    std::cout << "  CUDA Cores per SM:          " << coresPerSM << "\n";
    std::cout << "  Total CUDA Cores:           " << totalCores << "\n";
    std::cout << "  Total Global Memory:        " << std::fixed << std::setprecision(2)
              << (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB ("
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB)\n";
    std::cout << "  Free Global Memory:         "
              << (double)freeMem / (1024.0 * 1024.0 * 1024.0) << " GB ("
              << (freeMem / (1024 * 1024)) << " MB)\n";
    std::cout << "  Core Clock Rate:            " << coreClockRateKHz / 1000.0 << " MHz\n";
    std::cout << "  Memory Clock Rate:          " << memoryClockRateKHz / 1000.0 << " MHz\n";
    std::cout << "  Memory Bus Width:           " << prop.memoryBusWidth << " bits\n";
    double peakBandwidthGBs = 2.0 * (memoryClockRateKHz * 1e3) * (prop.memoryBusWidth / 8.0) / 1.0e9;
    std::cout << "  Peak Memory Bandwidth:      " << peakBandwidthGBs << " GB/s\n";
    std::cout << "  L2 Cache Size:              " << (prop.l2CacheSize / 1024) << " KB\n";
    std::cout << "  Warp Size:                  " << prop.warpSize << " threads\n";
    std::cout << "  Max Threads Per Block:      " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max Threads Per SM:         " << prop.maxThreadsPerMultiProcessor << "\n";

    // 3. Vector Addition Benchmark Configuration
    const int N = 10000000; // 10 Million elements
    const size_t bytes = N * sizeof(float);
    std::cout << "\n[BENCHMARK WORKLOAD]\n";
    std::cout << "  Operation:                  Vector Addition (C[i] = A[i] + B[i])\n";
    std::cout << "  Vector Element Count (N):   " << N << " elements\n";
    std::cout << "  Vector Memory (per array):  " << (double)bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  Total Active Memory:        " << (double)(3 * bytes) / (1024.0 * 1024.0) << " MB (A + B + C)\n";

    // 4. Host Allocation & Initialization
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C_gpu(N, 0.0f);
    std::vector<float> h_C_cpu(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        h_A[i] = std::sin(static_cast<float>(i)) * 2.5f;
        h_B[i] = std::cos(static_cast<float>(i)) * 1.5f;
    }

    // 5. CPU Reference Computation & Timing
    std::cout << "\n[EXECUTING CPU COMPUTATION]...\n";
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuDurationMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "  CPU Execution Time:         " << cpuDurationMs << " ms\n";

    // 6. Device Allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Create CUDA Timing Events
    cudaEvent_t evH2DStart, evH2DStop;
    cudaEvent_t evKernelStart, evKernelStop;
    cudaEvent_t evD2HStart, evD2HStop;
    CUDA_CHECK(cudaEventCreate(&evH2DStart));
    CUDA_CHECK(cudaEventCreate(&evH2DStop));
    CUDA_CHECK(cudaEventCreate(&evKernelStart));
    CUDA_CHECK(cudaEventCreate(&evKernelStop));
    CUDA_CHECK(cudaEventCreate(&evD2HStart));
    CUDA_CHECK(cudaEventCreate(&evD2HStop));

    // 7. Host -> Device Transfer
    std::cout << "\n[EXECUTING GPU COMPUTATION]...\n";
    CUDA_CHECK(cudaEventRecord(evH2DStart));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(evH2DStop));

    // 8. Launch Kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "  Kernel Launch Config:       " << blocksPerGrid << " blocks x "
              << threadsPerBlock << " threads/block (" << (blocksPerGrid * threadsPerBlock) << " total threads)\n";

    CUDA_CHECK(cudaEventRecord(evKernelStart));
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evKernelStop));

    // 9. Device -> Host Transfer
    CUDA_CHECK(cudaEventRecord(evD2HStart));
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(evD2HStop));

    // Synchronize stop event
    CUDA_CHECK(cudaEventSynchronize(evD2HStop));

    // Compute GPU Timing Metrics
    float h2dTimeMs = 0.0f;
    float kernelTimeMs = 0.0f;
    float d2hTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2dTimeMs, evH2DStart, evH2DStop));
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, evKernelStart, evKernelStop));
    CUDA_CHECK(cudaEventElapsedTime(&d2hTimeMs, evD2HStart, evD2HStop));
    float totalGpuTimeMs = h2dTimeMs + kernelTimeMs + d2hTimeMs;

    std::cout << "  Host -> Device Transfer:    " << h2dTimeMs << " ms\n";
    std::cout << "  Kernel Execution Time:      " << kernelTimeMs << " ms\n";
    std::cout << "  Device -> Host Transfer:    " << d2hTimeMs << " ms\n";
    std::cout << "  Total GPU End-to-End Time:  " << totalGpuTimeMs << " ms\n";

    // 10. Numerical Validation
    std::cout << "\n[NUMERICAL VERIFICATION]\n";
    float maxAbsError = 0.0f;
    double sumSquaredError = 0.0;
    int errorCount = 0;
    const float tolerance = 1e-5f;

    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > maxAbsError) {
            maxAbsError = diff;
        }
        sumSquaredError += diff * diff;
        if (diff > tolerance) {
            if (errorCount < 5) {
                std::cerr << "  [-] Mismatch at index " << i << ": GPU=" << h_C_gpu[i]
                          << ", CPU=" << h_C_cpu[i] << ", Diff=" << diff << "\n";
            }
            errorCount++;
        }
    }

    double rmse = std::sqrt(sumSquaredError / N);
    std::cout << "  Maximum Absolute Error:     " << maxAbsError << "\n";
    std::cout << "  Root Mean Square Error:     " << rmse << "\n";
    std::cout << "  Error Threshold:            " << tolerance << "\n";
    std::cout << "  Mismatched Elements:        " << errorCount << " / " << N << "\n";

    bool passVerification = (errorCount == 0);

    // 11. Performance Metrics
    // 3 memory operations (2 reads + 1 write) per element
    double effectiveBandwidthGBs = (3.0 * bytes) / (kernelTimeMs * 1e-3) / 1e9;
    // 1 FLOP per element
    double kernelGFlops = (double)N / (kernelTimeMs * 1e-3) / 1e9;
    double kernelSpeedup = cpuDurationMs / kernelTimeMs;
    double endToEndSpeedup = cpuDurationMs / totalGpuTimeMs;

    std::cout << "\n[PERFORMANCE METRICS]\n";
    std::cout << "  Kernel Throughput:          " << kernelGFlops << " GFLOPS\n";
    std::cout << "  Effective Memory Bandwidth: " << effectiveBandwidthGBs << " GB/s\n";
    std::cout << "  Speedup (Kernel vs CPU):    " << kernelSpeedup << "x\n";
    std::cout << "  Speedup (End-to-End vs CPU):" << endToEndSpeedup << "x\n";

    // 12. Resource Cleanup
    CUDA_CHECK(cudaEventDestroy(evH2DStart));
    CUDA_CHECK(cudaEventDestroy(evH2DStop));
    CUDA_CHECK(cudaEventDestroy(evKernelStart));
    CUDA_CHECK(cudaEventDestroy(evKernelStop));
    CUDA_CHECK(cudaEventDestroy(evD2HStart));
    CUDA_CHECK(cudaEventDestroy(evD2HStop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // 13. Final Status Report
    std::cout << "\n========================================================\n";
    if (passVerification) {
        std::cout << "  PHASE 1 STATUS: [PASS] - GPU Acceleration Engine Verified!\n";
    } else {
        std::cout << "  PHASE 1 STATUS: [FAIL] - Numerical Discrepancy Detected!\n";
    }
    std::cout << "========================================================\n";

    return passVerification ? 0 : 1;
}
