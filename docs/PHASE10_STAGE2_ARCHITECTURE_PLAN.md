# PHASE 10 — STAGE 2: ADVANCED CUDA ACCELERATION — ARCHITECTURE & IMPLEMENTATION PLAN

**Project:** CUDA GPU Acceleration Engine  
**Workspace:** `C:\Projects\CUDA-GPU-Acceleration-Engine`  
**Target Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (`sm_86`, Ampere, 20 SMs, 80 Tensor Cores, 96-bit GDDR6 @ 7001 MHz)  
**Compilation Toolchain:** Side-by-Side CUDA Toolkit 12.4.1 (`V12.4.131`) + MSVC 19.38 (`14.38.33145`) + PyTorch `2.6.0+cu124`  
**Document Version:** 1.0.0 (Phase 10 Stage 2 Blueprint)  
**Baseline Git Commit:** `b255b60f207fc5beca25e842839022f781eb3496` (Stage 1 Working Tree Clean)

---

## 1. Executive Summary & Objectives

Phase 10 Stage 1 established a stable, mathematically verified custom CUDA kernel library (`custom_kernels/`) supporting contiguous `float32` tensors on PyTorch CUDA streams with zero regressions.

**Stage 2 Objective:**  
Elevate the library into an advanced, production-grade acceleration system by integrating:
1. **Ampere Tensor Core Acceleration:** Warp-level matrix multiply-accumulate (`nvcuda::wmma`) delivering multi-TFLOPS dense linear algebra on the RTX 3050.
2. **Mixed-Precision Primitives:** Native `FP16` (`__half`, `half2`, `half8`) and `BF16` (`__nv_bfloat16`, `__nv_bfloat162`) elementwise, activation, and reduction kernels with **FP32 accumulation** to guarantee numerical stability.
3. **Autograd Backward Gradients:** First-class PyTorch automatic differentiation support (`torch.autograd.Function`) with custom CUDA backward kernels validated via `torch.autograd.gradcheck`.
4. **ResNet-18 Inference Acceleration:** Drop-in acceleration of residual additions (`fused_add_relu`) and classification projections in the Phase 9 model pipeline.
5. **Rigorous Benchmarking & Telemetry:** Comparative evaluation against `torch.matmul`, cuBLAS FP16/TF32 Tensor Core paths, and native PyTorch autograd baselines under strict memory ceilings (< 1.5 GB VRAM).

---

## 2. Hardware Architecture & Tensor Core Mapping (NVIDIA Ampere sm_86)

### 2.1 RTX 3050 6GB Laptop GPU Specifications

| Parameter | Hardware Value | Architectural Significance for Stage 2 |
| :--- | :--- | :--- |
| **Compute Capability** | `sm_86` (Ampere Architecture) | Supports 3rd Gen Tensor Cores, TF32, native BF16, and FP16 MMA |
| **Streaming Multiprocessors (SMs)** | 20 SMs | Up to 32 active warps per SM (total 640 concurrent warps) |
| **Tensor Cores** | 80 Tensor Cores (4 per SM) | 256 FP16 FMA operations per clock cycle per SM |
| **Peak FP32 CUDA Core Throughput** | ~6.6 TFLOPS | Theoretical upper limit for Stage 1 classical SGEMM |
| **Peak FP16 Tensor Core Throughput** | **~33.8 TFLOPS** (Dense) | $5.1\times$ higher compute density than FP32 CUDA Cores |
| **Memory Bus Width** | **96 bits** (3 $\times$ 32-bit controllers) | Verified via `cudaGetDeviceProperties` |
| **Memory Clock** | 7001 MHz GDDR6 | Peak theoretical bandwidth: **168.02 GB/s** |
| **L2 Cache Size** | 1.5 MB | Tile caching for GEMM shared memory staging |
| **Shared Memory / L1 per SM** | Up to 100 KB configurable | Supports multi-warp $16 \times 16 \times 16$ WMMA fragments |

### 2.2 Which Operations Benefit From Tensor Cores vs Vectorized CUDA Cores?

A critical engineering principle in modern GPU computing is recognizing arithmetic intensity:

$$\text{Arithmetic Intensity} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Traffic (Bytes)}}$$

```
+---------------------------------------------------------------------------------------+
| ARITHMETIC INTENSITY SPECTRUM ON NVIDIA AMPERE (sm_86)                                 |
+---------------------------------------------------------------------------------------+
|  Low Intensity (< 1 FLOP/byte)                  | High Intensity (> 10 FLOPs/byte)    |
|  MEMORY-BANDWIDTH BOUND                         | COMPUTE BOUND                       |
|                                                 |                                     |
|  - vector_add (0.08 FLOP/B)                     | - Dense Matrix Multiplication       |
|  - vector_scale (0.12 FLOP/B)                   |   (GEMM: 2*M*N*K / 4*(M*K+K*N+M*N)) |
|  - relu / gelu / sigmoid (0.12 - 0.5 FLOP/B)    |   Intensity on 1024^3: ~170 FLOPs/B |
|  - reduce_sum / reduce_max (0.25 FLOP/B)        |                                     |
+-------------------------------------------------+-------------------------------------+
|  DOES NOT BENEFIT FROM TENSOR CORES             | MASSIVELY ACCELERATED BY TENSOR     |
|  - Tensor Cores sit idle waiting for memory bus |   CORES                             |
|  - Optimization: 128-bit vectorization (float4, | - Delivers 10x-20x throughput over  |
|    half2, half8) to saturate memory controllers |   classical CUDA core SGEMM         |
+---------------------------------------------------------------------------------------+
```

1. **Matrix Multiplication (GEMM):** Highly compute-bound ($> 100\text{ FLOPs/byte}$). Tensor Cores accelerate the inner MAC loop from 1 multiply-add per CUDA core to 256 operations per Tensor Core cycle.
2. **Elementwise & Activations:** Strictly memory-bandwidth bound ($< 0.5\text{ FLOPs/byte}$). Tensor Cores cannot accelerate these operations because DRAM bus latency and bandwidth dominate. Instead, **FP16/BF16 vectorization (`half2` / `half8`)** halves the memory traffic (40 MB $\rightarrow$ 20 MB for 10M elements), directly boosting memory-bound throughput by up to $2\times$.

### 2.3 Realistic Performance Expectations Against cuBLAS

In Stage 1, our basic shared-memory SGEMM achieved $0.569\text{ TFLOPS}$ ($0.16\times$ cuBLAS). For Stage 2 Tensor Core GEMM, we maintain honest benchmarking integrity:
- **What Custom WMMA Achieves:** Using `nvcuda::wmma` with $16 \times 16 \times 16$ fragments, custom WMMA achieves **8 to 15 TFLOPS** on the RTX 3050 Laptop GPU, delivering an immediate **$15\times$ to $25\times$ speedup over Stage 1 classical tiled SGEMM**.
- **Why Highly Tuned cuBLAS Still Leads:** cuBLAS uses multi-stage asynchronous global-to-shared copy pipelines (`cp.async` introduced in Ampere), register-level double-buffering, bank-conflict-free shared memory swizzling, and heuristic tile selection tailored to each matrix aspect ratio. Our custom WMMA kernel represents a transparent, readable, high-performance implementation that demonstrates mastery of Tensor Core programming without fabricating claims of beating closed-source, vendor-tuned libraries.

---

## 3. Component Architecture & File Layout

Stage 2 extends the existing `custom_kernels` package modularly without modifying Stage 1 APIs or breaking backwards compatibility:

```
CUDA-GPU-Acceleration-Engine/
├── custom_kernels/
│   ├── __init__.py                         # Updated API exposing FP16/BF16 and Autograd Functions
│   ├── loader.py                           # Extended toolchain loader with FP16/BF16 flags
│   ├── autograd.py                         # [NEW] PyTorch torch.autograd.Function wrappers
│   ├── resnet_inference.py                 # [NEW] ResNet-18 custom kernel inference acceleration module
│   └── csrc/
│       ├── bindings.cpp                    # Updated PyBind11 bindings with FP16, BF16, and Backward dispatches
│       ├── include/
│       │   ├── kernel_common.cuh           # Updated with half2/bfloat162 helpers & alignment checks
│       │   ├── elementwise.cuh             # Updated with FP16/BF16 and backward signatures
│       │   ├── activations.cuh             # Updated with FP16/BF16 and backward signatures
│       │   ├── reductions.cuh              # Updated with FP16/BF16 FP32-accumulator signatures
│       │   ├── matmul_tiled.cuh            # Preserved Stage 1 classical tiled SGEMM
│       │   └── matmul_wmma.cuh             # [NEW] Tensor Core WMMA matrix multiplication signatures
│       └── cuda/
│           ├── elementwise.cu              # Updated with half2/bfloat162 and backward kernels
│           ├── activations.cu              # Updated with half2/bfloat162 and backward kernels
│           ├── reductions.cu               # Updated with half2/bfloat162 FP32-accumulated reductions
│           ├── matmul_tiled.cu             # Preserved Stage 1 classical SGEMM
│           ├── matmul_wmma.cu              # [NEW] Ampere Tensor Core WMMA 16x16x16 kernel
│           └── backward.cu                 # [NEW] Dedicated backward gradient kernels
├── tests/
│   ├── test_custom_kernels.py              # Preserved Stage 1 test suite (104 tests)
│   ├── test_stage2_precision.py            # [NEW] FP16 & BF16 accuracy and numerical tolerance tests
│   ├── test_stage2_wmma.py                 # [NEW] Tensor Core WMMA correctness, alignment, and shapes
│   ├── test_stage2_autograd.py             # [NEW] Backward gradient validation via torch.autograd.gradcheck
│   └── test_stage2_resnet.py               # [NEW] ResNet-18 inference acceleration validation
├── scripts/
│   ├── benchmark_custom_kernels.py         # Preserved Stage 1 benchmark suite
│   └── benchmark_stage2_advanced.py        # [NEW] Advanced Stage 2 mixed-precision & Tensor Core benchmarks
├── docs/
│   ├── PHASE10_STAGE1_ARCHITECTURE.md      # Preserved Stage 1 record
│   ├── PHASE10_STAGE2_ARCHITECTURE_PLAN.md # [THIS FILE] Architecture and implementation plan
│   └── PHASE10_STAGE2_BENCHMARK_REPORT.md  # [FUTURE] Empirical Stage 2 benchmark findings
└── results/
    └── custom_kernels/
        ├── stage1_benchmark_results.json   # Preserved Stage 1 baseline
        ├── stage1_benchmark_results_audited.json # Preserved Stage 1 audited
        └── stage2_benchmark_results.json   # [FUTURE] Stage 2 benchmark records
```

---

## 4. Kernel Technical Specifications

### 4.1 Mixed-Precision Vectorization Architecture (`half2` / `half8`)

For memory-bound elementwise operations (`add`, `scale`, `relu`, `gelu`, `sigmoid`):
- **FP16 Path:** Uses `__half2` arithmetic (`__hadd2`, `__hmul2`, `__hmax2`). When a 16-byte aligned pointer is available, an LDG.E.128 loads **8 FP16 elements** (`float4` cast to `half8` or $4 \times \text{half2}$) in a single instruction.
- **BF16 Path:** Uses `__nv_bfloat162` arithmetic (`__hadd2`, `__hmul2` from `<cuda_bf16.h>`).
- **Memory Bandwidth Impact:** Transferring $10\text{M}$ elements requires only $20\text{ MB}$ (read) $+ 20\text{ MB}$ (write) $= 40\text{ MB}$ DRAM traffic (compared to $80\text{ MB}$ in FP32), doubling effective element processing throughput.

### 4.2 Tensor Core Matrix Multiplication (`matmul_wmma.cu`)

#### Warp-Level WMMA Formulation:
- Utilizes NVIDIA `nvcuda::wmma` intrinsics from `<mma.h>`.
- Fragment Configuration:
  ```cpp
  #include <mma.h>
  using namespace nvcuda;

  // Warp-level fragment dimensions: M=16, N=16, K=16
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  // Fragments: Inputs half (FP16), Accumulator float (FP32)
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  ```

#### Block-Level Tiling Structure:
- Each thread block computes a $64 \times 64$ sub-matrix of $C$.
- Block contains $4 \text{ warps}$ ($128 \text{ threads}$ arranged as a $2 \times 2$ warp grid).
- Shared memory buffers: `__shared__ half s_A[64][16]` and `__shared__ half s_B[16][64]`.
- Inner loop streams $K$ in steps of 16, executing:
  ```cpp
  wmma::load_matrix_sync(a_frag, &s_A[warp_row * 16][k_step], 16);
  wmma::load_matrix_sync(b_frag, &s_B[k_step][warp_col * 16], 64);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  ```
- Store fragment: `wmma::store_matrix_sync(&c_out[row * N + col], c_frag, N, wmma::mem_row_major)` with support for FP16 or FP32 outputs.

#### Handling Irregular and Non-Multiple-of-16 Dimensions:
1. **Dimension Check:** If $M, N, K$ are all multiples of 16, launch the high-performance Tensor Core WMMA kernel.
2. **Padding / Fallback:** If dimensions are irregular (e.g. $37 \times 53 \times 19$), execute the safe Stage 1 classical tiled SGEMM fallback or apply dynamic zero-padded boundary tiles.

### 4.3 Mixed-Precision Reductions with FP32 Accumulators

In FP16, the maximum representable float is **65,504**, and the precision step at 1000 is 1.0. Directly summing millions of FP16 numbers in half-precision causes immediate overflow and massive rounding errors.

**Stage 2 Reduction Design:**
- Load elements as `half2` or `__nv_bfloat162`.
- Immediately convert to `float` (FP32) in registers: `float v0 = __half2float(h.x); float v1 = __half2float(h.y);`.
- Perform intra-warp reduction (`__shfl_down_sync`) and inter-warp shared memory accumulation using **FP32 IEEE-754 arithmetic**.
- Write the final scalar to memory as `float` or cast to `half` if requested.
- **Guarantee:** Eliminates overflow on large tensors up to $100\text{M}$ elements.

### 4.4 Autograd Backward Kernels (`backward.cu`)

To enable custom autograd operators without violating the Stage 1 forward-only contract:

1. **`vector_add_backward`**:
   $$\text{grad}_a = \text{grad}_c, \quad \text{grad}_b = \text{grad}_c$$
   Identity pass / copy from `grad_output`.
2. **`fused_add_relu_backward`**:
   $$\text{grad}_a = \text{grad}_b = (a + b > 0) \cdot \text{grad}_{\text{out}}$$
   Computes activation mask in registers directly from saved inputs $a, b$, avoiding any auxiliary mask tensor storage.
3. **`fused_bias_activation_backward`**:
   - For $Y = \text{act}(X + \text{bias})$:
     $$\text{grad}_X = \text{act}'(X + \text{bias}) \odot \text{grad}_Y$$
     $$\text{grad}_{\text{bias}} = \sum_{\text{rows}} \left(\text{act}'(X + \text{bias}) \odot \text{grad}_Y\right)$$
   - Uses row-wise warp reduction to compute bias gradients with zero temporary DRAM allocations.
4. **`matmul_backward`**:
   $$\text{grad}_A = \text{grad}_C \times B^T, \quad \text{grad}_B = A^T \times \text{grad}_C$$
   Dispatches transposed matrix products via custom GEMM or Tensor Core WMMA.

---

## 5. Numerical Accuracy & Tolerance Strategy

Mixed-precision arithmetic has lower mantissa bit-depth than FP32. Strict operation-specific tolerance policies must be codified:

| Data Type | Sign Bits | Exponent Bits | Mantissa Bits | Machine Epsilon ($\epsilon$) | Dynamic Range |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32** | 1 | 8 | 23 | $\approx 1.19 \times 10^{-7}$ | $\pm 3.4 \times 10^{38}$ |
| **FP16** | 1 | 5 | 10 | $\approx 9.77 \times 10^{-4}$ | $\pm 65,504$ |
| **BF16** | 1 | 8 | 7 | $\approx 7.81 \times 10^{-3}$ | $\pm 3.4 \times 10^{38}$ |

### Codified Tolerance Policies for Stage 2 Testing:

1. **FP16 Elementwise & Activations:**
   - `atol = 1e-3`, `rtol = 1e-3` (matches 10-bit half precision limit).
2. **BF16 Elementwise & Activations:**
   - `atol = 5e-3`, `rtol = 5e-3` (matches 7-bit bfloat16 precision limit).
3. **Tensor Core WMMA (FP16 inputs, FP32 accumulator):**
   - Compared against `torch.matmul(A.half(), B.half())` or FP32 reference.
   - `atol = 1e-2`, `rtol = 1e-2` (accounts for FMA accumulation order differences).
4. **FP16/BF16 Reductions (with FP32 Accumulator):**
   - Compared against `torch.sum(x.float())`.
   - `atol = 1e-2`, `rtol = 1e-3`.
5. **Autograd Backward Gradients (`gradcheck`):**
   - Verified in `float64` (double precision) using perturbation step `eps = 1e-6`, `atol = 1e-4`, `rtol = 1e-3`.
   - Forward function accepts double or casts safely during gradient verification.

---

## 6. PyTorch Integration & Autograd Strategy

### 6.1 Python Autograd Wrapper (`custom_kernels/autograd.py`)

```python
import torch
from .loader import load_custom_kernels_module

_C = None

def _get_C():
    global _C
    if _C is None:
        _C = load_custom_kernels_module()
    return _C

class FusedAddReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return _get_C().fused_add_relu(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_a, grad_b = _get_C().fused_add_relu_backward(grad_output, a, b)
        return grad_a, grad_b

def fused_add_relu_autograd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused Add + ReLU with full PyTorch autograd backward gradient support."""
    return FusedAddReLUFunction.apply(a, b)
```

### 6.2 ResNet-18 Inference Pipeline Integration (`custom_kernels/resnet_inference.py`)

In Stage 2, we create an inference accelerator for the existing Phase 9 ResNet-18 model:
- **Residual Connection Acceleration:** In `BasicBlock.forward`:
  ```python
  # Standard PyTorch:
  # out += identity
  # out = self.relu(out)
  # Accelerated:
  out = custom_kernels.fused_add_relu(out, identity)
  ```
- **Classification Head Acceleration:** In `ResNet18.forward`:
  ```python
  # Standard PyTorch: out = self.fc(out) (out @ W^T + bias)
  # Accelerated:
  out = custom_kernels.matmul_wmma(out, self.fc.weight.T)
  out = custom_kernels.fused_bias_activation(out, self.fc.bias, activation="none")
  ```
- **Zero Checkpoint Disruption:** Loads weights directly from `checkpoints/resnet18_cifar10/best_model.pt` without retraining or modifying saved state dict keys.

---

## 7. Benchmark Methodology & Statistical Controls

The Stage 2 benchmark harness (`scripts/benchmark_stage2_advanced.py`) incorporates all lessons learned from the Stage 1 audit:

1. **Garbage Collection Isolation:** Explicit `gc.collect()`, `torch.cuda.empty_cache()`, and `gc.disable()` during active timing loops.
2. **Dual-Timing Paradigm:**
   - Pure GPU execution via CUDA events on the active stream (`enable_timing=True`).
   - End-to-end synchronized host time via `time.perf_counter()`.
3. **Statistical Metrics:** Minimum, median, maximum, and standard deviation over 50 iterations.
4. **Telemetry Logging:** Logs GPU temperature, graphics clock, memory clock, and power draw via `nvidia-smi` query before and after runs.
5. **Baselines:**
   - Custom WMMA FP16 vs `torch.matmul` (cuBLAS Tensor Core path).
   - Custom WMMA FP16 vs Stage 1 Tiled SGEMM FP32 (measuring Tensor Core generational speedup).
   - Custom FP16 Elementwise vs Native PyTorch FP16.

---

## 8. GPU Memory Safety Plan (RTX 3050 6GB Budget)

| Resource Category | Budget Limit | Safeguard Mechanism |
| :--- | :--- | :--- |
| **Max Allocation per Op** | $< 250\text{ MB}$ | Enforces tensor size ceilings ($N \le 10\text{M}$ elements) |
| **Peak Live VRAM Ceiling** | **$< 1.5\text{ GB}$** | Active budget guard (`verify_vram_safety`) checking physical allocation |
| **PyTorch Cache Hygiene** | $< 500\text{ MB}$ reserved | `torch.cuda.empty_cache()` called between benchmark categories |
| **Scratch Buffers** | $< 16\text{ KB}$ | Allocated via PyTorch caching allocator; zero `cudaMallocAsync` in loops |
| **Operating Headroom** | $> 4.0\text{ GB}$ free | Leaves ample VRAM for OS, display compositor, and background tasks |

---

## 9. Implementation Milestones & Verification Gates

The implementation is structured into 5 strictly sequential, validated milestones:

```
+-----------------------------------------------------------------------------------+
| STAGE 2 MILESTONE ROADMAP                                                         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [MILESTONE 1: Mixed-Precision Core]                                              |
|  - Implement half2 / bfloat162 elementwise, activations, and reductions           |
|  - Implement FP32-accumulated reductions for half/bfloat16                        |
|  - Gate: test_stage2_precision.py (all pass, verified tolerances)                 |
|                                         |                                         |
|  [MILESTONE 2: Ampere Tensor Core WMMA] v                                         |
|  - Implement matmul_wmma.cu (16x16x16 WMMA fragments, 64x64 block tiling)        |
|  - Add dimension alignment checks and irregular shape fallback                    |
|  - Gate: test_stage2_wmma.py (verify against torch.matmul across sizes)           |
|                                         |                                         |
|  [MILESTONE 3: Autograd Backward Gradients] v                                     |
|  - Implement backward.cu (add_backward, fused_relu_backward, bias_backward)       |
|  - Implement custom_kernels/autograd.py (torch.autograd.Function wrappers)        |
|  - Gate: test_stage2_autograd.py (torch.autograd.gradcheck passes)                |
|                                         |                                         |
|  [MILESTONE 4: ResNet-18 Inference Integration] v                                 |
|  - Implement custom_kernels/resnet_inference.py                                   |
|  - Verify on CIFAR-10 test batches with checkpoints/resnet18_cifar10/best_model.pt|
|  - Gate: test_stage2_resnet.py (verify 100% output parity with original model)    |
|                                         |                                         |
|  [MILESTONE 5: Full Benchmark Suite & Audit Gate] v                               |
|  - Run scripts/benchmark_stage2_advanced.py                                       |
|  - Run full regression suites (Stage 1 tests, backend, engine, manifest, native)  |
|  - Gate: Produce PHASE10_STAGE2_BENCHMARK_REPORT.md with PASS verdict             |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 10. Risk Assessment & Mitigation Strategies

| Risk | Likelihood | Impact | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **FP16 Underflow / Overflow in Reductions** | High | Critical | **Mandatory FP32 Accumulator:** Warp shuffles and shared memory use `float` registers. Conversion to `half` occurs only at final store. |
| **WMMA Alignment Faults on Odd Shapes** | Medium | High | **Boundary Guards & Safe Fallback:** Kernel checks `(M % 16 == 0 && N % 16 == 0 && K % 16 == 0)`. Irregular shapes route to Stage 1 tiled SGEMM. |
| **Autograd Graph Memory Leaks** | Medium | Medium | **Ephemeral Contexts:** `ctx.save_for_backward` stores only necessary tensors ($a, b$). Gradients are computed in-place where valid. |
| **Compiler Toolset Drift** | Low | High | **Process-Local Lock:** `loader.py` pins MSVC `14.38.33130` and CUDA 12.4 via `-ccbin`, eliminating interference from newer MSVC 14.51. |
| **Phase 2 Manifest Modification** | Low | Fatal | **Strict File Quarantine:** Zero modifications to `CMakeLists.txt` or `src/*.cu`. All Stage 2 files stay in `custom_kernels/`, `tests/`, and `docs/`. |

---

## 11. Acceptance Criteria for Stage 2 Approval

Stage 2 will be deemed complete and eligible for release gate approval only when all of the following criteria are met:
1. **Unit Tests:** 100% pass rate across Stage 1 (`test_custom_kernels.py`) and all new Stage 2 test suites (`test_stage2_precision.py`, `test_stage2_wmma.py`, `test_stage2_autograd.py`, `test_stage2_resnet.py`).
2. **Autograd Mathematical Validation:** `torch.autograd.gradcheck` passes on all backward operators without numerical divergence.
3. **Tensor Core Speedup:** Custom WMMA achieves measurable multi-TFLOPS performance ($\ge 8\text{ TFLOPS}$ on $1024^3$) on the RTX 3050 Laptop GPU, representing a $\ge 15\times$ speedup over Stage 1 classical SGEMM.
4. **ResNet-18 Parity:** Accelerated inference on `checkpoints/resnet18_cifar10/best_model.pt` produces identical classification logits within `atol=1e-4` of native PyTorch ResNet-18.
5. **Zero Regressions:** Backend and engine pytest suites (57 passed), Phase 2 manifest preservation (17/17 SHA-256 match), and native binary regression (`bin/benchmark_regression.exe`) pass completely.
6. **No Repository Pollution:** Working tree of tracked files remains clean; no checkpoints or extension binaries are tracked in Git.
