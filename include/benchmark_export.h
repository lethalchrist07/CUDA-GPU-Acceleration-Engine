#pragma once

#include <string>
#include <vector>
#include "benchmark_types.h"

// Export benchmark results to CSV format
bool exportToCSV(const std::string& filepath, const std::vector<BenchmarkResult>& results);

// Export benchmark results to formatted JSON format
bool exportToJSON(const std::string& filepath, 
                   const std::vector<BenchmarkResult>& results,
                   const std::string& device_name,
                   const std::string& driver_ver,
                   const std::string& runtime_ver,
                   size_t expected_count);
