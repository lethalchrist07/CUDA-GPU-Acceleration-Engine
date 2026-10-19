#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
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
    std::filesystem::path outputDir = "results";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_suite.exe [--output-dir DIRECTORY] [--help]\n"
                      << "Runs device 0: vector sizes 100000, 1000000, 10000000, 25000000;\n"
                      << "basic and tiled matrix sizes 256, 512, 1024, 2048.\n"
                      << "GPU warmup/timed runs: vector 3/5, matrix 2/3. CPU: one warmup,\n"
                      << "5 vector samples, 3 matrix samples (2 at 2048).\n"
                      << "E2E: host wall time for H2D, launch, D2H and synchronization;\n"
                      << "excludes allocation, initialization and verification.\n"
                      << "Speedups compare this single-threaded CPU reference only.\n";
            return 0;
        }
        if (arg == "--output-dir" && i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            outputDir = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << " (use --help)\n";
            return 2;
        }
    }
    std::error_code directoryError;
    std::filesystem::create_directories(outputDir, directoryError);
    if (directoryError) {
        std::cerr << "Cannot create output directory: " << directoryError.message() << "\n";
        return 1;
    }
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
    if (cudaSetDevice(deviceId) != cudaSuccess ||
        cudaDriverGetVersion(&driverVersion) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtimeVersion) != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA device or query versions.\n";
        return 1;
    }

    std::string deviceName = prop.name;
    std::string driverVerStr = std::to_string(driverVersion / 1000) + "." + std::to_string((driverVersion % 1000) / 10);
    std::string runtimeVerStr = std::to_string(runtimeVersion / 1000) + "." + std::to_string((runtimeVersion % 1000) / 10);

    size_t freeMem = 0, totalMem = 0;
    if (cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess) {
        std::cerr << "Failed to query CUDA memory.\n";
        return 1;
    }

    std::cout << "[SYSTEM / ENVIRONMENT]\n";
    std::cout << "  Device:             " << deviceName << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total VRAM:         " << (totalMem / (1024 * 1024)) << " MiB\n";
    std::cout << "  Free VRAM:          " << (freeMem / (1024 * 1024)) << " MiB\n";
    std::cout << "  CUDA Driver:        " << driverVerStr << "\n";
    std::cout << "  CUDA Runtime:       " << runtimeVerStr << "\n";
    std::cout << "  Streaming Multiprocessors: " << prop.multiProcessorCount << "\n\n";

    std::cout << "Times are medians. E2E is synchronized host wall time, excluding allocation.\n"
              << "CPU baseline: single-threaded reference, not optimized BLAS.\n"
              << "Bandwidth is estimated requested global-memory traffic, not measured DRAM throughput.\n";
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
            double tiledOverBasic = (tiledRes.gpu_kernel_ms > 0.0)
                ? (basicRes.gpu_kernel_ms / tiledRes.gpu_kernel_ms) : 0.0;
            std::cout << "      Basic/Tiled kernel time ratio: " << tiledOverBasic << "x\n";
            std::cout << "      Status:        " << (tiledRes.verified ? "[PASS]" : "[FAIL]") 
                      << " (Max Err: " << std::scientific << tiledRes.max_abs_error << ")\n";
        }
    }

    // ========================================================
    // 4. SUMMARY DASHBOARD TABLE
    // ========================================================
    printHeader("PERFORMANCE SUMMARY TABLE");
    std::cout << std::left 
              << std::setw(40) << "Workload"
              << std::setw(27) << "Variant"
              << std::setw(11) << "CPU (ms)" 
              << std::setw(11) << "GPU K(ms)" 
              << std::setw(12) << "GPU E2E(ms)" 
              << std::setw(13) << "K-Speedup(x)"
              << std::setw(13) << "E2E-Spd(x)"
              << std::setw(11) << "GFLOPS" 
              << std::setw(8)  << "Status" << "\n";
    std::cout << std::string(146, '-') << "\n";

    const size_t expectedCount = vectorSizes.size() + 2 * matrixSizes.size();
    bool allPassed = benchmarksPassed(allResults, expectedCount);
    if (allResults.size() != expectedCount) {
        std::cerr << "Incomplete benchmark run: " << allResults.size() << "/" << expectedCount << " results.\n";
    }
    for (const auto& r : allResults) {
        if (!r.verified) allPassed = false;
        std::cout << std::left 
                  << std::setw(40) << (r.category + " " + r.workload_size)
                  << std::setw(27) << r.variant
                  << std::setw(11) << std::fixed << std::setprecision(4) << r.cpu_time_ms
                  << std::setw(11) << r.gpu_kernel_ms
                  << std::setw(12) << r.gpu_end_to_end_ms
                  << std::setw(13) << std::setprecision(3) << r.kernel_speedup
                  << std::setw(13) << r.end_to_end_speedup
                  << std::setw(11) << r.gflops
                  << std::setw(8)  << (r.verified ? "[PASS]" : "[FAIL]") << "\n";
    }
    std::cout << std::string(146, '-') << "\n";

    // ========================================================
    // 5. EXPORT RESULTS TO CSV AND JSON
    // ========================================================
    printHeader("EXPORTING BENCHMARK ARTIFACTS");
    std::string csvPath = (outputDir / "benchmark_results.csv").string();
    std::string jsonPath = (outputDir / "benchmark_results.json").string();

    const bool csvExported = exportToCSV(csvPath, allResults);
    const bool jsonExported = exportToJSON(jsonPath, allResults, deviceName, driverVerStr, runtimeVerStr, expectedCount);
    allPassed = allPassed && csvExported && jsonExported;

    std::cout << "\n========================================================================================\n";
    if (allPassed) {
        std::cout << "  PHASE 2 STATUS: [PASS] - All Benchmarks and Numerical Validations Succeeded!\n";
    } else {
        std::cout << "  PHASE 2 STATUS: [FAIL] - Discrepancies Detected During Benchmarking!\n";
    }
    std::cout << "========================================================================================\n\n";

    return allPassed ? 0 : 1;
}
