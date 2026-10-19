#include "benchmark_export.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace {
void writeJSONNumber(std::ostream& file, double value) {
    if (std::isfinite(value)) file << value;
    else file << "null";
}

void writeSamples(std::ostream& file, const char* name, const std::vector<double>& samples, bool last) {
    file << "        \"" << name << "\": [";
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) file << ", ";
        writeJSONNumber(file, samples[i]);
    }
    file << "]" << (last ? "\n" : ",\n");
}
}

bool exportToCSV(const std::string& filepath, const std::vector<BenchmarkResult>& results) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[-] Failed to open CSV export file: " << filepath << std::endl;
        return false;
    }

    // CSV Header
    file << "Category,Variant,WorkloadSize,Dimension,MemoryMB,"
         << "CPUTimeMs,GPUH2DMs,GPUKernelMs,GPUD2HMs,GPUEndToEndMs,"
         << "KernelSpeedup,EndToEndSpeedup,GFLOPS,MemoryBandwidthGBs,"
         << "MaxAbsError,RMSE,Tolerance,MismatchedElements,Verified,WarmupRuns,TimedRuns,CPUTimedRuns\n";

    for (const auto& r : results) {
        file << "\"" << r.category << "\","
             << "\"" << r.variant << "\","
             << "\"" << r.workload_size << "\","
             << r.dimension << ","
             << std::fixed << std::setprecision(9) << r.memory_mb << ","
             << r.cpu_time_ms << ","
             << r.gpu_h2d_ms << ","
             << r.gpu_kernel_ms << ","
             << r.gpu_d2h_ms << ","
             << r.gpu_end_to_end_ms << ","
             << r.kernel_speedup << ","
             << r.end_to_end_speedup << ","
             << r.gflops << ","
             << r.memory_bandwidth_gb_s << ","
             << std::scientific << std::setprecision(9) << r.max_abs_error << ","
             << r.rmse << ","
             << r.tolerance << ","
             << std::fixed << r.mismatched_elements << ","
             << (r.verified ? "PASS" : "FAIL") << ","
             << r.warmup_runs << ","
             << r.timed_runs << "," << r.cpu_timed_runs << "\n";
    }

    file.close();
    if (!file) {
        std::cerr << "[-] Failed to write CSV export: " << filepath << std::endl;
        return false;
    }
    std::cout << "[+] Exported benchmark results to CSV: " << filepath << std::endl;
    return true;
}

bool exportToJSON(const std::string& filepath, 
                   const std::vector<BenchmarkResult>& results,
                   const std::string& device_name,
                   const std::string& driver_ver,
                   const std::string& runtime_ver,
                   size_t expected_count) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[-] Failed to open JSON export file: " << filepath << std::endl;
        return false;
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    char timeStr[64];
    ctime_s(timeStr, sizeof(timeStr), &now_time);
    std::string timeString(timeStr);
    if (!timeString.empty() && timeString.back() == '\n') timeString.pop_back();

    int passedCount = 0;
    for (const auto& r : results) {
        if (r.verified) passedCount++;
    }

    file << "{\n";
    file << "  \"schema_version\": 2,\n";
    file << "  \"system_info\": {\n";
    file << "    \"device_name\": \"" << device_name << "\",\n";
    file << "    \"cuda_driver_version\": \"" << driver_ver << "\",\n";
    file << "    \"cuda_runtime_version\": \"" << runtime_ver << "\",\n";
    file << "    \"timestamp\": \"" << timeString << "\"\n";
    file << "  },\n";
    file << "  \"measurement_notes\": {\n";
    file << "    \"time_unit\": \"ms; medians of individual samples\",\n";
    file << "    \"memory_mb_unit\": \"MiB (legacy field name)\",\n";
    file << "    \"gpu_end_to_end\": \"host steady_clock: H2D, launch, D2H, synchronization and timing instrumentation; excludes allocation, initialization, verification\",\n";
    file << "    \"gpu_events\": \"CUDA event intervals on default stream; pageable host buffers; not isolated DMA or profiler kernel durations\",\n";
    file << "    \"cpu_baseline\": \"single-threaded C++ reference; one warmup; not optimized BLAS\",\n";
    file << "    \"memory_bandwidth_gb_s\": \"estimated requested global-memory bytes / kernel seconds / 1e9; cache reuse included; NOT measured DRAM bandwidth\",\n";
    file << "    \"gflops\": \"vector N or matrix 2*N^3 operations / kernel seconds / 1e9\",\n";
    file << "    \"build\": \"CUDA sm_86; --use_fast_math; build configuration recorded in verification log\"\n";
    file << "  },\n";
    file << "  \"summary\": {\n";
    file << "    \"expected_benchmarks\": " << expected_count << ",\n";
    file << "    \"total_benchmarks\": " << results.size() << ",\n";
    file << "    \"passed\": " << passedCount << ",\n";
    file << "    \"failed\": " << (results.size() - passedCount) << ",\n";
    file << "    \"missing_benchmarks\": " << (expected_count > results.size() ? expected_count - results.size() : 0) << ",\n";
    file << "    \"status\": \"" << (benchmarksPassed(results, expected_count) ? "PASS" : "FAIL") << "\"\n";
    file << "  },\n";
    file << "  \"benchmarks\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        file << "    {\n";
        file << "      \"id\": " << (i + 1) << ",\n";
        file << "      \"category\": \"" << r.category << "\",\n";
        file << "      \"variant\": \"" << r.variant << "\",\n";
        file << "      \"workload_size\": \"" << r.workload_size << "\",\n";
        file << "      \"dimension\": " << r.dimension << ",\n";
        file << "      \"memory_mb\": " << std::fixed << std::setprecision(9) << r.memory_mb << ",\n";
        file << "      \"cpu_time_ms\": " << r.cpu_time_ms << ",\n";
        file << "      \"gpu_h2d_ms\": " << r.gpu_h2d_ms << ",\n";
        file << "      \"gpu_kernel_ms\": " << r.gpu_kernel_ms << ",\n";
        file << "      \"gpu_d2h_ms\": " << r.gpu_d2h_ms << ",\n";
        file << "      \"gpu_end_to_end_ms\": " << r.gpu_end_to_end_ms << ",\n";
        file << "      \"kernel_speedup\": " << r.kernel_speedup << ",\n";
        file << "      \"end_to_end_speedup\": " << r.end_to_end_speedup << ",\n";
        file << "      \"gflops\": " << r.gflops << ",\n";
        file << "      \"memory_bandwidth_gb_s\": " << r.memory_bandwidth_gb_s << ",\n";
        file << "      \"max_abs_error\": " << std::scientific << std::setprecision(9);
        writeJSONNumber(file, r.max_abs_error);
        file << ",\n      \"rmse\": ";
        writeJSONNumber(file, r.rmse);
        file << ",\n";
        file << "      \"tolerance\": " << r.tolerance << ",\n";
        file << "      \"mismatched_elements\": " << std::fixed << r.mismatched_elements << ",\n";
        file << "      \"verified\": " << (r.verified ? "true" : "false") << ",\n";
        file << "      \"warmup_runs\": " << r.warmup_runs << ",\n";
        file << "      \"timed_runs\": " << r.timed_runs << ",\n";
        file << "      \"cpu_timed_runs\": " << r.cpu_timed_runs << ",\n";
        file << "      \"samples_ms\": {\n" << std::setprecision(9);
        writeSamples(file, "cpu", r.cpu_samples_ms, false);
        writeSamples(file, "h2d", r.h2d_samples_ms, false);
        writeSamples(file, "kernel", r.kernel_samples_ms, false);
        writeSamples(file, "d2h", r.d2h_samples_ms, false);
        writeSamples(file, "end_to_end", r.e2e_samples_ms, true);
        file << "      }\n";
        file << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    if (!file) {
        std::cerr << "[-] Failed to write JSON export: " << filepath << std::endl;
        return false;
    }
    std::cout << "[+] Exported benchmark results to JSON: " << filepath << std::endl;
    return true;
}
