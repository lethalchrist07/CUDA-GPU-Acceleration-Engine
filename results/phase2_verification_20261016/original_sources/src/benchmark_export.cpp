#include "benchmark_export.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

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
         << "MaxAbsError,RMSE,Tolerance,MismatchedElements,Verified,WarmupRuns,TimedRuns\n";

    for (const auto& r : results) {
        file << "\"" << r.category << "\","
             << "\"" << r.variant << "\","
             << "\"" << r.workload_size << "\","
             << r.dimension << ","
             << std::fixed << std::setprecision(2) << r.memory_mb << ","
             << std::setprecision(4) << r.cpu_time_ms << ","
             << r.gpu_h2d_ms << ","
             << r.gpu_kernel_ms << ","
             << r.gpu_d2h_ms << ","
             << r.gpu_end_to_end_ms << ","
             << std::setprecision(2) << r.kernel_speedup << ","
             << r.end_to_end_speedup << ","
             << r.gflops << ","
             << r.memory_bandwidth_gb_s << ","
             << std::scientific << std::setprecision(6) << r.max_abs_error << ","
             << r.rmse << ","
             << r.tolerance << ","
             << std::fixed << r.mismatched_elements << ","
             << (r.verified ? "PASS" : "FAIL") << ","
             << r.warmup_runs << ","
             << r.timed_runs << "\n";
    }

    file.close();
    std::cout << "[+] Exported benchmark results to CSV: " << filepath << std::endl;
    return true;
}

bool exportToJSON(const std::string& filepath, 
                   const std::vector<BenchmarkResult>& results,
                   const std::string& device_name,
                   const std::string& driver_ver,
                   const std::string& runtime_ver) {
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
    file << "  \"system_info\": {\n";
    file << "    \"device_name\": \"" << device_name << "\",\n";
    file << "    \"cuda_driver_version\": \"" << driver_ver << "\",\n";
    file << "    \"cuda_runtime_version\": \"" << runtime_ver << "\",\n";
    file << "    \"timestamp\": \"" << timeString << "\"\n";
    file << "  },\n";
    file << "  \"summary\": {\n";
    file << "    \"total_benchmarks\": " << results.size() << ",\n";
    file << "    \"passed\": " << passedCount << ",\n";
    file << "    \"failed\": " << (results.size() - passedCount) << ",\n";
    file << "    \"status\": \"" << (passedCount == results.size() ? "PASS" : "FAIL") << "\"\n";
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
        file << "      \"memory_mb\": " << std::fixed << std::setprecision(2) << r.memory_mb << ",\n";
        file << "      \"cpu_time_ms\": " << std::setprecision(4) << r.cpu_time_ms << ",\n";
        file << "      \"gpu_h2d_ms\": " << r.gpu_h2d_ms << ",\n";
        file << "      \"gpu_kernel_ms\": " << r.gpu_kernel_ms << ",\n";
        file << "      \"gpu_d2h_ms\": " << r.gpu_d2h_ms << ",\n";
        file << "      \"gpu_end_to_end_ms\": " << r.gpu_end_to_end_ms << ",\n";
        file << "      \"kernel_speedup\": " << std::setprecision(2) << r.kernel_speedup << ",\n";
        file << "      \"end_to_end_speedup\": " << r.end_to_end_speedup << ",\n";
        file << "      \"gflops\": " << r.gflops << ",\n";
        file << "      \"memory_bandwidth_gb_s\": " << r.memory_bandwidth_gb_s << ",\n";
        file << "      \"max_abs_error\": " << std::scientific << std::setprecision(6) << r.max_abs_error << ",\n";
        file << "      \"rmse\": " << r.rmse << ",\n";
        file << "      \"tolerance\": " << r.tolerance << ",\n";
        file << "      \"mismatched_elements\": " << std::fixed << r.mismatched_elements << ",\n";
        file << "      \"verified\": " << (r.verified ? "true" : "false") << ",\n";
        file << "      \"warmup_runs\": " << r.warmup_runs << ",\n";
        file << "      \"timed_runs\": " << r.timed_runs << "\n";
        file << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    std::cout << "[+] Exported benchmark results to JSON: " << filepath << std::endl;
    return true;
}
