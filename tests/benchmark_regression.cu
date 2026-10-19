#include "vector_add.cuh"
#include "matmul.cuh"
#include "benchmark_export.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>

#define REQUIRE(condition) do { if (!(condition)) { \
    std::cerr << "Failed at line " << __LINE__ << ": " #condition "\n"; return false; \
} } while (0)

bool runRegression() {
    BenchmarkResult result;
    result.tolerance = 1e-5f;
    float reference[] = {1.0f, -2.0f, 0.0f};
    float actual[] = {1.0f, -2.0f, 0.0f};
    verifyAgainstCPU(actual, reference, 3, result);
    REQUIRE(result.verified);
    actual[0] = std::numeric_limits<float>::quiet_NaN();
    actual[1] = std::numeric_limits<float>::infinity();
    verifyAgainstCPU(actual, reference, 3, result);
    REQUIRE(!result.verified && result.mismatched_elements == 2);
    REQUIRE(!benchmarksPassed({}, 12));
    BenchmarkResult passed;
    passed.verified = true;
    REQUIRE(!benchmarksPassed({passed}, 12));
    REQUIRE(!benchmarksPassed({result}, 1));
    REQUIRE(benchmarksPassed({passed}, 1));

    // A directory is a guaranteed invalid file target, independent of permissions.
    REQUIRE(!exportToCSV(".", {passed}));
    REQUIRE(!exportToJSON(".", {passed}, "test", "13.4", "13.4", 1));
    const std::string failurePath = "benchmark_regression_failure.json";
    REQUIRE(exportToJSON(failurePath, {result}, "test", "13.4", "13.4", 12));
    std::ifstream file(failurePath);
    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    REQUIRE(contents.find("\"status\": \"FAIL\"") != std::string::npos);
    REQUIRE(contents.find("\"max_abs_error\": null") != std::string::npos);
    REQUIRE(contents.find("\"missing_benchmarks\": 11") != std::string::npos);
    file.close();
    std::filesystem::remove(failurePath);

    REQUIRE(!runVectorAddBenchmark(0, 1, 1, result));
    REQUIRE(!runVectorAddBenchmark(257, 1, 0, result));
    REQUIRE(runVectorAddBenchmark(257, 1, 3, result));
    REQUIRE(result.verified && result.cpu_samples_ms.size() == 3);
    REQUIRE(result.gpu_end_to_end_ms > 0 && result.gpu_kernel_ms > 0);
    BenchmarkResult basic, tiled;
    REQUIRE(!runMatmulBenchmarks(0, 1, 1, basic, tiled));
    REQUIRE(runMatmulBenchmarks(17, 1, 3, basic, tiled));
    REQUIRE(basic.verified && tiled.verified);

    // Independent double-precision reference, mixed-sign inputs and partial tiles.
    const int n = 33;
    const size_t count = n * n, bytes = count * sizeof(float);
    std::vector<float> a(count), b(count), expected(count), output(count);
    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(static_cast<int>((i * 37 + 11) % 101) - 50) / 64.0f;
        b[i] = static_cast<float>(static_cast<int>((i * 19 + 7) % 97) - 48) / 64.0f;
    }
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0;
            for (int k = 0; k < n; ++k) sum += static_cast<double>(a[row*n+k]) * b[k*n+col];
            expected[row*n+col] = static_cast<float>(sum);
        }
    }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&da), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&db), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dc), bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    const dim3 block(MATMUL_TILE_DIM, MATMUL_TILE_DIM);
    const dim3 grid((n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM,
                    (n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM);
    for (int variant = 0; variant < 2; ++variant) {
        CUDA_CHECK(cudaMemset(dc, 0xff, bytes));
        if (variant == 0) matmulBasicKernel<<<grid, block>>>(da, db, dc, n);
        else matmulTiledSharedKernel<<<grid, block>>>(da, db, dc, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(output.data(), dc, bytes, cudaMemcpyDeviceToHost));
        verifyAgainstCPU(output.data(), expected.data(), count, result);
        REQUIRE(result.verified);
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    std::cout << "PASS: non-finite/missing/export failures, vector tail, partial matrix tiles, independent CPU reference.\n";
    return true;
}

int main() { return runRegression() ? 0 : 1; }
