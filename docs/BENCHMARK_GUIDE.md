# Reproducible CUDA Benchmark Guide

This document provides a comprehensive, step-by-step engineering guide for reproducing, verifying, and benchmarking the **CUDA GPU Acceleration Engine** across NVIDIA GPU architectures.

---

## 1. Environment & Hardware Specifications

The benchmark suite is validated and certified on the following reference system:

| Component | Specification |
|---|---|
| **GPU Model** | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| **Compute Capability** | SM 8.6 (Ampere Architecture, `GA107`) |
| **Streaming Multiprocessors** | 20 SMs (2,560 CUDA Cores) |
| **Dedicated VRAM** | 6,144 MiB GDDR6 (96-bit bus, 168 GB/s physical memory bandwidth) |
| **Host Processor** | Intel Core i5-13450HX (10 Cores, 16 Threads, up to 4.6 GHz) |
| **Host Memory** | 16 GB DDR5 4800 MHz |
| **Operating System** | Windows 11 Home 64-bit (Build 26100) |
| **CUDA Toolkit** | Version 13.4 (V13.4.92) / Toolkit 12.0+ compatible |
| **NVIDIA Driver** | 617.42 (Direct3D / WDDM mode) |
| **C++ Compiler** | MSVC 19.51 (Visual Studio 2026 Build Tools) / C++17 standard |
| **Build System** | CMake 4.3.1 & Ninja 1.13 |

---

## 2. Workload Architecture

The micro-benchmark suite executes **12 deterministic, serially isolated workloads**:

### A. Vector Addition ($C = A + B$)
- **Memory Access Characteristic**: High-throughput memory-bandwidth bound.
- **Precision**: Single-precision IEEE 754 float (`float32`).
- **Memory Traffic per Iteration**: $3 \times N \times 4\text{ bytes}$ ($A, B$ Host-to-Device; $C$ Device-to-Host).
- **Scale Boundaries Tested**:
  1. **100,000 elements** (~1.14 MiB buffer)
  2. **1,000,000 elements** (~11.44 MiB buffer)
  3. **10,000,000 elements** (~114.44 MiB buffer)
  4. **25,000,000 elements** (~286.10 MiB buffer)

### B. Matrix Multiplication ($C = A \times B$)
- **Compute Characteristic**: Compute-bound arithmetic intensity with $2 \times N^3$ floating-point operations.
- **Precision**: Single-precision FP32.
- **Architectural Implementations**:
  1. **Basic Naive CUDA Kernel**: Each thread computes one output element via uncoalesced global memory strided reads.
  2. **Tiled Shared Memory CUDA Kernel**: $16 \times 16$ thread block tiles leveraging on-chip SRAM (`__shared__`) and synchronization barriers (`__syncthreads()`) to reuse sub-matrix tiles from fast L1/shared memory, drastically reducing global memory requests.
- **Workload Sizes Tested**:
  - **$256 \times 256$** ($0.134\text{ GFLOPs}$, 0.75 MiB total buffer)
  - **$512 \times 512$** ($1.074\text{ GFLOPs}$, 3.00 MiB total buffer)
  - **$1024 \times 1024$** ($8.590\text{ GFLOPs}$, 12.00 MiB total buffer)
  - **$2048 \times 2048$** ($68.719\text{ GFLOPs}$, 48.00 MiB total buffer)

---

## 3. Timing & Precision Methodology

Following [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing), timings are categorized into two strictly delineated definitions:

### 1. Synchronized Host Wall Time (End-to-End / E2E)
- Measured via `std::chrono::steady_clock` on the host thread.
- **Includes**:
  - Asynchronous memory transfer launch (`cudaMemcpyAsync` H2D).
  - CUDA kernel grid launch overhead.
  - Device-to-Host memory transfer (`cudaMemcpyAsync` D2H).
  - Device synchronization barrier (`cudaStreamSynchronize`).
  - High-precision host timing instrumentation.
- **Excludes**:
  - Host RAM allocations (`malloc`) and device allocations (`cudaMalloc`).
  - Pseudo-random data generation and initialization.
  - CPU reference computation.
  - Numerical parity checks and output element validation.

### 2. CUDA Event Intervals (Kernel Time)
- Measured via hardware GPU timestamps (`cudaEventRecord` and `cudaEventElapsedTime`).
- Records the precise microsecond-level execution of the compute kernel on the default stream.

### 3. Iteration & Warmup Protocol
- **Vector Benchmarks**: 3 GPU warmup iterations, 5 timed iterations. CPU reference: 1 warmup, 5 timed iterations.
- **Matrix Benchmarks**: 2 GPU warmup iterations, 3 timed iterations. CPU reference: 1 warmup, 3 timed iterations (2 iterations at size 2048 to prevent thermal drift).
- **Reported Metric**: Exact statistical **median** of all timed samples (never arithmetic mean).

### 4. Numerical Correctness Standards
- Every output element from the final timed iteration is verified against the double-precision CPU reference.
- **Vector Addition**: Maximum absolute error $\le 10^{-5}$ (`PASS`).
- **Matrix Multiplication**: Maximum absolute error $\le 10^{-3}$ (`PASS`, accounting for non-associativity of FP32 additions).
- Zero mismatched elements allowed.

---

## 4. Step-by-Step Reproduction Instructions

### Prerequisites
Ensure Visual Studio 2022/2026 Build Tools, CUDA Toolkit (12.0+ or 13.0+), and Ninja are on your `PATH`:

```powershell
# Verify compilers
nvcc --version
cl
cmake --version
ninja --version
```

### Step 1: Build the Native CUDA Benchmark Suite
From the repository root:

```powershell
# Configure Ninja Release build
cmake -B build/reproduce-ninja -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86"

# Compile benchmark suite and regression binaries
cmake --build build/reproduce-ninja --config Release
```

### Step 2: Execute Micro-Benchmark Suite
Run the compiled executable with an output directory for CSV and JSON logs:

```powershell
# Create output directory
New-Item -ItemType Directory -Force -Path results/reproduction_run

# Execute benchmark suite
.\build\reproduce-ninja\benchmark_suite.exe --output-dir results/reproduction_run
```

### Step 3: Run Native CTest Validation
```powershell
& ctest --test-dir build/reproduce-ninja -C Release --output-on-failure
```

---

## 5. Reference Benchmark Baseline (Phase 2 Certified)

These are the reference measurements captured on the certified testbed (RTX 3050 Laptop GPU, SM 8.6, MSVC 19.51, CUDA 13.4):

| Workload | Variant | CPU ms | GPU Kernel ms | GPU E2E ms | Kernel Speedup | E2E Speedup | GFLOPS / Bandwidth |
|---|---|---:|---:|---:|---:|---:|---|
| **Vector 100K** | Standard | 0.0091 | 0.0287 | 0.3527 | 0.32× | 0.03× | 41.9 GB/s |
| **Vector 1M** | Standard | 0.5162 | 0.1065 | 2.2527 | 4.85× | 0.23× | 112.6 GB/s |
| **Vector 10M** | Standard | 6.5588 | 0.7944 | 14.8087 | 8.26× | 0.44× | 151.1 GB/s |
| **Vector 25M** | Standard | 15.9809 | 1.9304 | 45.1440 | 8.28× | 0.35× | 155.4 GB/s |
| **SGEMM 256×256** | Basic | 3.9081 | 0.0813 | 0.2513 | 48.06× | 15.55× | 412.7 GFLOPS |
| **SGEMM 256×256** | Tiled | 3.9081 | 0.0687 | 0.2306 | 56.88× | 16.95× | 488.4 GFLOPS |
| **SGEMM 512×512** | Basic | 31.7281 | 0.4784 | 1.1036 | 66.32× | 28.75× | 561.1 GFLOPS |
| **SGEMM 512×512** | Tiled | 31.7281 | 0.3707 | 0.9510 | 85.59× | 33.36× | 724.1 GFLOPS |
| **SGEMM 1024×1024** | Basic | 253.4822 | 3.5257 | 5.0719 | 71.89× | 49.98× | 609.1 GFLOPS |
| **SGEMM 1024×1024** | Tiled | 253.4822 | 2.8237 | 4.4160 | 89.77× | 57.40× | 760.5 GFLOPS |
| **SGEMM 2048×2048** | Basic | 2690.8590 | 36.4785 | 43.0002 | 73.77× | 62.58× | 471.0 GFLOPS |
| **SGEMM 2048×2048** | Tiled | 2690.8590 | 23.0480 | 29.9297 | **116.75×** | **89.91×** | 745.4 GFLOPS |

---

## 6. Important Interpretation Notes

1. **CPU Reference**: Baseline speedups compare against a clean single-threaded C++ implementation, not multithreaded AVX-512 BLAS (e.g. OpenBLAS or Intel MKL).
2. **Small Vector Transfer Penalty**: Small vector sizes ($100\text{K}$) exhibit E2E speedups $< 1.0\times$ due to PCIe transfer and kernel launch latency dominating execution. True speedup emerges at $\ge 1\text{M}$ elements.
3. **Memory Bandwidth Metric**: `memory_bandwidth_gb_s` measures estimated requested algorithmic global memory traffic divided by kernel time. For tiled multiplication, cache reuse reduces physical DRAM traffic; this number reflects algorithmic throughput, not physical pin bandwidth.
