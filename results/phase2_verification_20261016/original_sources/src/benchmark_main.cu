#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "cuda_utils.cuh"
#include "benchmark_types.h"
#include "vector_add.cuh"
#include "matmul.cuh"
#include "benchmark_export.h"

void printHeader(const std::string& title) {
    std::cout << "\n========================================================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "========================================================================================\n";
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================================================\n";
    std::cout << "      CUDA GPU Acceleration Engine - Comprehensive Benchmark Suite (Phase 2)\n";
    std::cout << "========================================================================================\n\n";

    // 1. Device Diagnostics
    int deviceId = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess) {
        std::cerr << "[-] Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    std::string deviceName = prop.name;
    std::string driverVerStr = std::to_string(driverVersion / 1000) + "." + std::to_string(driverVersion % 100);
    std::string runtimeVerStr = std::to_string(runtimeVersion / 1000) + "." + std::to_string(runtimeVersion % 100);

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    std::cout << "[SYSTEM / ENVIRONMENT]\n";
    std::cout << "  Device:             " << deviceName << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total VRAM:         " << (totalMem / (1024 * 1024)) << " MB\n";
    std::cout << "  Free VRAM:          " << (freeMem / (1024 * 1024)) << " MB\n";
    std::cout << "  CUDA Driver:        " << driverVerStr << "\n";
    std::cout << "  CUDA Runtime:       " << runtimeVerStr << "\n";
    std::cout << "  Streaming Multiprocessors: " << prop.multiProcessorCount << "\n\n";

    std::vector<BenchmarkResult> allResults;

    // ========================================================
    // 2. VECTOR ADDITION BENCHMARKS ACROSS SIZES
    // ========================================================
    printHeader("BENCHMARK SUITE: Vector Addition (A[i] + B[i])");
    std::vector<size_t> vectorSizes = {
        100000,     // 100K
        1000000,    // 1M
        10000000,   // 10M
        25000000    // 25M
    };

    const int vecWarmup = 3;
    const int vecTimed = 5;

    for (size_t n : vectorSizes) {
        std::cout << "\n[*] Running Vector Addition (N = " << n << " elements)..." << std::flush;
        BenchmarkResult res;
        if (runVectorAddBenchmark(n, vecWarmup, vecTimed, res)) {
            allResults.push_back(res);
            std::cout << " [DONE]\n";
            std::cout << "    CPU Time:      " << std::fixed << std::setprecision(3) << res.cpu_time_ms << " ms\n";
            std::cout << "    GPU Kernel:    " << res.gpu_kernel_ms << " ms (Speedup: " 
                      << std::setprecision(2) << res.kernel_speedup << "x)\n";
            std::cout << "    GPU End-to-End:" << res.gpu_end_to_end_ms << " ms (Speedup: " 
                      << res.end_to_end_speedup << "x)\n";
            std::cout << "    Bandwidth:     " << res.memory_bandwidth_gb_s << " GB/s | GFLOPS: " << res.gflops << "\n";
            std::cout << "    Verification:  " << (res.verified ? "[PASS]" : "[FAIL]") 
                      << " (Max Abs Error: " << std::scientific << res.max_abs_error << ")\n";
        }
    }

    // ========================================================
    // 3. MATRIX MULTIPLICATION BENCHMARKS ACROSS SIZES
    // ========================================================
    printHeader("BENCHMARK SUITE: Matrix Multiplication (C = A x B)");
    std::vector<int> matrixSizes = {
        256,   // 256 x 256
        512,   // 512 x 512
        1024,  // 1024 x 1024
        2048   // 2048 x 2048
    };

    const int matWarmup = 2;
    const int matTimed = 3;

    for (int n : matrixSizes) {
        std::cout << "\n[*] Running Matrix Multiplication (Size = " << n << " x " << n << ")...\n";
        BenchmarkResult basicRes, tiledRes;
        if (runMatmulBenchmarks(n, matWarmup, matTimed, basicRes, tiledRes)) {
            allResults.push_back(basicRes);
            allResults.push_back(tiledRes);

            std::cout << "  [-] CPU Reference Time: " << std::fixed << std::setprecision(2) 
                      << basicRes.cpu_time_ms << " ms\n";

            std::cout << "  [+] Basic CUDA Kernel:\n";
            std::cout << "      Kernel Time:   " << basicRes.gpu_kernel_ms << " ms (Kernel Speedup: " 
                      << basicRes.kernel_speedup << "x)\n";
            std::cout << "      End-to-End:    " << basicRes.gpu_end_to_end_ms << " ms (E2E Speedup: " 
                      << basicRes.end_to_end_speedup << "x)\n";
            std::cout << "      GFLOPS:        " << basicRes.gflops << " GFLOPS\n";
            std::cout << "      Status:        " << (basicRes.verified ? "[PASS]" : "[FAIL]") 
                      << " (Max Err: " << std::scientific << basicRes.max_abs_error << ")\n";

            std::cout << "  [+] Tiled Shared-Memory Kernel:\n";
            std::cout << "      Kernel Time:   " << std::fixed << std::setprecision(2) << tiledRes.gpu_kernel_ms 
                      << " ms (Kernel Speedup: " << tiledRes.kernel_speedup << "x)\n";
            std::cout << "      End-to-End:    " << tiledRes.gpu_end_to_end_ms << " ms (E2E Speedup: " 
                      << tiledRes.end_to_end_speedup << "x)\n";
            std::cout << "      GFLOPS:        " << tiledRes.gflops << " GFLOPS\n";
            double tiledOverBasic = (basicRes.gpu_kernel_ms > 0.0) 
                ? (basicRes.gpu_kernel_ms / tiledRes.gpu_kernel_ms) : 0.0;
            std::cout << "      Tiled vs Basic:" << tiledOverBasic << "x Faster\n";
            std::cout << "      Status:        " << (tiledRes.verified ? "[PASS]" : "[FAIL]") 
                      << " (Max Err: " << std::scientific << tiledRes.max_abs_error << ")\n";
        }
    }

    // ========================================================
    // 4. SUMMARY DASHBOARD TABLE
    // ========================================================
    printHeader("PERFORMANCE SUMMARY TABLE");
    std::cout << std::left 
              << std::setw(22) << "Workload" 
              << std::setw(20) << "Variant" 
              << std::setw(11) << "CPU (ms)" 
              << std::setw(11) << "GPU K(ms)" 
              << std::setw(12) << "GPU E2E(ms)" 
              << std::setw(10) << "K-Speedup" 
              << std::setw(10) << "E2E-Spd" 
              << std::setw(11) << "GFLOPS" 
              << std::setw(8)  << "Status" << "\n";
    std::cout << std::string(105, '-') << "\n";

    bool allPassed = true;
    for (const auto& r : allResults) {
        if (!r.verified) allPassed = false;
        std::cout << std::left 
                  << std::setw(22) << (r.category + " " + r.workload_size)
                  << std::setw(20) << r.variant
                  << std::setw(11) << std::fixed << std::setprecision(2) << r.cpu_time_ms
                  << std::setw(11) << r.gpu_kernel_ms
                  << std::setw(12) << r.gpu_end_to_end_ms
                  << std::setw(10) << (std::to_string(static_cast<int>(r.kernel_speedup)) + "." + 
                                       std::to_string(static_cast<int>(r.kernel_speedup * 10) % 10) + "x")
                  << std::setw(10) << (std::to_string(static_cast<int>(r.end_to_end_speedup)) + "." + 
                                       std::to_string(static_cast<int>(r.end_to_end_speedup * 10) % 10) + "x")
                  << std::setw(11) << r.gflops
                  << std::setw(8)  << (r.verified ? "[PASS]" : "[FAIL]") << "\n";
    }
    std::cout << std::string(105, '-') << "\n";

    // ========================================================
    // 5. EXPORT RESULTS TO CSV AND JSON
    // ========================================================
    printHeader("EXPORTING BENCHMARK ARTIFACTS");
    std::string csvPath = "results/benchmark_results.csv";
    std::string jsonPath = "results/benchmark_results.json";

    exportToCSV(csvPath, allResults);
    exportToJSON(jsonPath, allResults, deviceName, driverVerStr, runtimeVerStr);

    std::cout << "\n========================================================================================\n";
    if (allPassed) {
        std::cout << "  PHASE 2 STATUS: [PASS] - All Benchmarks and Numerical Validations Succeeded!\n";
    } else {
        std::cout << "  PHASE 2 STATUS: [FAIL] - Discrepancies Detected During Benchmarking!\n";
    }
    std::cout << "========================================================================================\n\n";

    return allPassed ? 0 : 1;
}
