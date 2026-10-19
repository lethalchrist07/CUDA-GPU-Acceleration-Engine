#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

struct BenchmarkResult {
    std::string category;          // e.g. "Vector Addition", "Matrix Multiplication"
    std::string variant;           // e.g. "Basic CUDA", "Tiled Shared Memory"
    std::string workload_size;     // e.g. "10,000,000 elements", "1024x1024"
    size_t dimension = 0;          // N or matrix dimension
    double memory_mb = 0.0;        // Legacy field name: total active memory in MiB
    double cpu_time_ms = 0.0;      // Median CPU time
    double gpu_h2d_ms = 0.0;       // Median Host-to-Device transfer
    double gpu_kernel_ms = 0.0;    // Median Kernel execution time
    double gpu_d2h_ms = 0.0;       // Median Device-to-Host transfer
    double gpu_end_to_end_ms = 0.0;// Host wall time: H2D through synchronized D2H; excludes allocation
    double kernel_speedup = 0.0;   // cpu_time_ms / gpu_kernel_ms
    double end_to_end_speedup = 0.0; // cpu_time_ms / gpu_end_to_end_ms
    double gflops = 0.0;           // GFLOPS of kernel
    double memory_bandwidth_gb_s = 0.0; // Estimated requested global-memory bytes / kernel seconds, not DRAM throughput
    float max_abs_error = 0.0f;
    double rmse = 0.0;
    float tolerance = 1e-4f;
    int mismatched_elements = 0;
    bool verified = false;
    int warmup_runs = 3;
    int timed_runs = 5;
    int cpu_timed_runs = 0;
    std::vector<double> cpu_samples_ms, h2d_samples_ms, kernel_samples_ms, d2h_samples_ms, e2e_samples_ms;
};

// Reject non-finite values: fabs(NaN) > tolerance alone would silently pass.
inline void verifyAgainstCPU(const float* actual, const float* reference, size_t count,
                             BenchmarkResult& result) {
    result.max_abs_error = 0.0f;
    result.mismatched_elements = 0;
    double squared_error = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(actual[i] - reference[i]);
        if (!std::isfinite(diff)) {
            ++result.mismatched_elements;
            result.max_abs_error = std::numeric_limits<float>::infinity();
            squared_error = std::numeric_limits<double>::infinity();
        } else {
            result.max_abs_error = std::max(result.max_abs_error, diff);
            squared_error += static_cast<double>(diff) * diff;
            if (diff > result.tolerance) ++result.mismatched_elements;
        }
    }
    result.rmse = count ? std::sqrt(squared_error / count) : 0.0;
    result.verified = count > 0 && result.mismatched_elements == 0;
}

inline bool benchmarksPassed(const std::vector<BenchmarkResult>& results, size_t expected_count) {
    return expected_count > 0 && results.size() == expected_count &&
        std::all_of(results.begin(), results.end(), [](const BenchmarkResult& r) { return r.verified; });
}

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
