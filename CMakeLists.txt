cmake_minimum_required(VERSION 3.20)

project(CudaGpuAccelerationEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target NVIDIA GeForce RTX 3050 Laptop GPU (Ampere Architecture: compute_86, sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set runtime output directories to bin/ across configurations
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_SOURCE_DIR}/bin)

# =======================================================
# Target 1: Phase 1 Diagnostics (Preserved)
# =======================================================
add_executable(gpu_check src/gpu_check.cu)

target_compile_options(gpu_check PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3>
)

target_include_directories(gpu_check PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# =======================================================
# Target 2: Phase 2 Comprehensive Benchmarking Suite
# =======================================================
add_executable(benchmark_suite
    src/benchmark_main.cu
    src/vector_add.cu
    src/matmul.cu
    src/benchmark_export.cpp
)

target_compile_options(benchmark_suite PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3>
)

target_include_directories(benchmark_suite PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

include(CTest)
if(BUILD_TESTING)
    add_executable(benchmark_regression
        tests/benchmark_regression.cu
        src/vector_add.cu
        src/matmul.cu
        src/benchmark_export.cpp
    )
    target_include_directories(benchmark_regression PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(benchmark_regression PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3>
    )
    add_test(NAME cuda_benchmark_regression COMMAND benchmark_regression)
    add_test(NAME benchmark_help COMMAND benchmark_suite --help)
    add_test(NAME benchmark_rejects_unknown_option COMMAND benchmark_suite --not-a-supported-option)
    set_tests_properties(benchmark_rejects_unknown_option PROPERTIES WILL_FAIL TRUE)
endif()
