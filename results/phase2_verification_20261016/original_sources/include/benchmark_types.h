#pragma once

#include <string>
#include <vector>
#include <algorithm>

struct BenchmarkResult {
    std::string category;          // e.g. "Vector Addition", "Matrix Multiplication"
    std::string variant;           // e.g. "Basic CUDA", "Tiled Shared Memory"
    std::string workload_size;     // e.g. "10,000,000 elements", "1024x1024"
    size_t dimension = 0;          // N or matrix dimension
    double memory_mb = 0.0;        // Total active memory in MB
    double cpu_time_ms = 0.0;      // Median CPU time
    double gpu_h2d_ms = 0.0;       // Median Host-to-Device transfer
    double gpu_kernel_ms = 0.0;    // Median Kernel execution time
    double gpu_d2h_ms = 0.0;       // Median Device-to-Host transfer
    double gpu_end_to_end_ms = 0.0;// Median End-to-End time (H2D + Kernel + D2H)
    double kernel_speedup = 0.0;   // cpu_time_ms / gpu_kernel_ms
    double end_to_end_speedup = 0.0; // cpu_time_ms / gpu_end_to_end_ms
    double gflops = 0.0;           // GFLOPS of kernel
    double memory_bandwidth_gb_s = 0.0; // Effective GB/s
    float max_abs_error = 0.0f;
    double rmse = 0.0;
    float tolerance = 1e-4f;
    int mismatched_elements = 0;
    bool verified = false;
    int warmup_runs = 3;
    int timed_runs = 5;
};

inline double computeMedian(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t n = vals.size();
    if (n % 2 == 0) {
        return (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
    } else {
        return vals[n / 2];
    }
}
