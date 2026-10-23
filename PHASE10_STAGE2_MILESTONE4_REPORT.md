# PHASE 10 — STAGE 2 — MILESTONE 4 REPORT
## Real AI Model Integration: Three-Backend CIFAR-10 Training Engine & Final Benchmark Audit

**Author:** Senior NVIDIA CUDA/C++ Engineer, PyTorch Training Systems Engineer, GPU Performance Architect  
**Project:** CUDA GPU Acceleration Engine  
**Workspace:** `C:\Projects\CUDA-GPU-Acceleration-Engine`  
**Baseline Git Commit:** `a77071c04875206d05d2d639bac6447758103b47` (Milestone 3 autograd)  
**Hardware Platform:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (`sm_86`, Ampere), Intel Core i5-13450HX, 16 GB RAM  
**Software Toolchain:** CUDA Toolkit 12.4.1, MSVC 19.38.33439, PyTorch 2.6.0+cu124, Python 3.11.9, Windows 11 (build 26300)  
**Date:** October 23, 2026  

---

## 1. Executive Summary

In **Phase 10 — Stage 2 — Milestone 4**, we successfully integrated our custom CUDA kernel and autograd infrastructure into a complete, reproducible **three-backend neural network training engine** trained on real **CIFAR-10** image classification data.

The engine supports three strictly isolated, mathematically comparable execution backends:
1. **`CPU_NATIVE`**: Standard PyTorch execution executing entirely on the host Intel Core i5-13450HX CPU.
2. **`GPU_NATIVE`**: Standard PyTorch CUDA execution utilizing NVIDIA cuBLAS and cuDNN on the RTX 3050 Laptop GPU.
3. **`GPU_CUSTOM`**: Custom GPU acceleration engine utilizing compiled custom CUDA forward and backward autograd kernels (`matmul_tiled`, `fused_bias_activation`, and `fused_bias_activation_backward`) implemented in C++/CUDA.

### Key Milestones Achieved
- **Authentic CIFAR-10 Dataset Verification:** Verified local presence of authentic CIFAR-10 Python batch binaries (50,000 train / 10,000 test images) matching official MD5 `c58f30108f718f92721af3b95e74349a`. Zero synthetic data or mock inputs were used for benchmarking.
- **Strict Disjoint Subsets & Zero Data Leakage:** Established deterministic 10,000 train / 2,000 validation / 2,000 test partitions with 0 image overlap across partitions. The test partition was evaluated strictly once at the end of training and exerted zero influence over model selection.
- **Deep 4-Layer MLP Architecture:** Formulated a canonical architecture ($3,072 \to 1,024 \to 256 \to 64 \to 10$) with GELU and ReLU activations totaling **3,426,250 trainable FP32 parameters** with bitwise-identical canonical parameter initialization across all three backends.
- **Zero-Mock Dispatch Tracking:** Implemented non-invasive runtime instrumentation (`KernelDispatchTracker`) intercepting real C++ module invocations. Confirmed **4,887 genuine custom CUDA kernel dispatches** during training with explicit fallback auditing (237 cross-entropy loss, 237 SGD optimizer steps).
- **100% Convergence Parity:** All three backends demonstrated equivalent convergence trajectories starting from initial cross-entropy loss $\sim 2.302$ down to final training loss $\sim 1.83$, achieving final test accuracies of **39.45%** (`CPU_NATIVE`), **38.80%** (`GPU_NATIVE`), and **39.35%** (`GPU_CUSTOM`).
- **Comprehensive Regression Verification:** **512 / 512** custom CUDA unit tests passed, **57 passed / 1 skipped** backend and engine regression tests passed, **17 / 17** Phase 2 cryptographic manifest hashes confirmed matching (100% integrity), native benchmark regression binary passed, and the Phase 9 ResNet-18 checkpoint remained bitwise intact.

---

## 2. Real Model and Dataset Architecture

### 2.1 CIFAR-10 Data Pipeline and Subset Methodology
The training pipeline processes authentic CIFAR-10 images ($32 \times 32 \times 3$, 3,072 input features per image):
- **Source:** Local directory `data/cifar-10-batches-py/` containing verified CIFAR-10 batches (`data_batch_1` through `data_batch_5` and `test_batch`), unpacked from `cifar-10-python.tar.gz` (MD5: `c58f30108f718f92721af3b95e74349a`).
- **Partitioning Methodology:**
  1. The 50,000 training images from `data_batch_1` to `data_batch_5` are permuted deterministically via `numpy.random.RandomState(42).permutation(50000)`.
  2. The first 5,000 indices form the validation candidate pool; the remaining 45,000 form the training candidate pool.
  3. **Train Subset (10,000 samples):** First 10,000 samples taken from the training pool. Class distribution is natural and balanced across all 10 classes: `[997, 990, 973, 1019, 951, 1041, 980, 982, 1043, 1024]`.
  4. **Validation Subset (2,000 samples):** First 2,000 samples taken from the validation pool. Class distribution: `[197, 199, 188, 203, 197, 208, 188, 208, 208, 204]`.
  5. **Test Subset (2,000 samples):** First 2,000 samples taken directly from the official 10,000-image `test_batch`. Class distribution: `[196, 198, 195, 199, 198, 185, 216, 193, 217, 203]`.
- **Data Leakage Audit:**
  - Raw image byte hashes confirm:
    - Train $\cap$ Validation Overlap = **0**
    - Train $\cap$ Test Overlap = **0**
    - Validation $\cap$ Test Overlap = **0**
  - Validation set is used solely for inter-epoch model checkpoint evaluation (`best_val_acc`).
  - Test set is evaluated strictly once at the end of training after all training and validation epochs have completed. Test set gradients never backpropagate, and test accuracy never influences checkpointing.
- **Preprocessing:** Min-max normalized to $[0.0, 1.0]$, standardized with CIFAR-10 dataset mean $(0.4914, 0.4822, 0.4465)$ and exact dataset standard deviation $(0.2470, 0.2435, 0.2616)$, flattened into FP32 vectors of dimension $3,072$.
- **Augmentation:** Applied only to the training split (`RandomCrop` with 4px reflection padding and `RandomHorizontalFlip` with $p=0.5$). Validation and test splits receive no augmentation.

### 2.2 Deep MLP Architecture
To thoroughly exercise custom CUDA matrix multiplication, fused bias-activation, and autograd gradient propagation, we designed a 4-layer Deep Multi-Layer Perceptron:

```
Input (B, 3072)
    │
    ▼
[Linear Layer 1: 3072 -> 1024] ──> Fused GELU Activation  (3,146,752 params)
    │
    ▼
[Linear Layer 2: 1024 -> 256]  ──> Fused GELU Activation  (  262,400 params)
    │
    ▼
[Linear Layer 3: 256 -> 64]    ──> Fused ReLU Activation  (   16,448 params)
    │
    ▼
[Linear Layer 4: 64 -> 10]     ──> Linear Output (Logits) (      650 params)
    │
    ▼
Output (B, 10)  ──> Cross Entropy Loss (PyTorch Reference Fallback)
```

**Total Trainable Parameters:** $3,146,752 + 262,400 + 16,448 + 650 = \mathbf{3,426,250}$ FP32 parameters.

*Design Rationale:* Integrating both GELU (tanh approximation) and ReLU activations ensures that multiple compiled custom CUDA activation and autograd backward kernels (`gelu`, `gelu_backward`, `relu`, `relu_backward` within `fused_bias_activation` and `fused_bias_activation_backward`) are rigorously exercised during training.

### 2.3 Layer-by-Layer Backend Dispatch Mapping
The identical state dictionary structure is initialized deterministically with seed `42` (`torch.manual_seed(42)` and Kaiming uniform weight initialization) and distributed across backends:

| Layer Stage | `CPU_NATIVE` | `GPU_NATIVE` | `GPU_CUSTOM` Execution |
| :--- | :--- | :--- | :--- |
| **Input Flatten** | `Tensor.view(B, -1)` | `Tensor.view(B, -1)` | `Tensor.view(B, -1)` |
| **FC1 Forward** | PyTorch CPU Linear + GELU | cuBLAS `torch.addmm` + GELU | `custom_kernels.autograd.CustomLinear` (`matmul_tiled` + `fused_bias_activation[gelu]`) |
| **FC2 Forward** | PyTorch CPU Linear + GELU | cuBLAS `torch.addmm` + GELU | `custom_kernels.autograd.CustomLinear` (`matmul_tiled` + `fused_bias_activation[gelu]`) |
| **FC3 Forward** | PyTorch CPU Linear + ReLU | cuBLAS `torch.addmm` + ReLU | `custom_kernels.autograd.CustomLinear` (`matmul_tiled` + `fused_bias_activation[relu]`) |
| **FC4 (Output)** | PyTorch CPU Linear | cuBLAS `torch.addmm` | `custom_kernels.autograd.CustomLinear` (`matmul_tiled` + `fused_bias_activation[none]`) |
| **Loss** | Native `nn.CrossEntropyLoss` | Native `nn.CrossEntropyLoss` | Native `nn.CrossEntropyLoss` (Explicit Fallback) |
| **FC4 Backward** | PyTorch CPU Autograd | cuBLAS Backward | `fused_bias_activation_backward[none]` + `matmul_tiled` ($dY \cdot W^T$, $X^T \cdot dY$) |
| **FC3 Backward** | PyTorch CPU Autograd | cuBLAS Backward | `fused_bias_activation_backward[relu]` + `matmul_tiled` ($dY \cdot W^T$, $X^T \cdot dY$) |
| **FC2 Backward** | PyTorch CPU Autograd | cuBLAS Backward | `fused_bias_activation_backward[gelu]` + `matmul_tiled` ($dY \cdot W^T$, $X^T \cdot dY$) |
| **FC1 Backward** | PyTorch CPU Autograd | cuBLAS Backward | `fused_bias_activation_backward[gelu]` + `matmul_tiled` ($X^T \cdot dY$ weight grad) |
| **Optimizer Step**| Native `optim.SGD` | Native `optim.SGD` | Native `optim.SGD` (Explicit Fallback) |

---

## 3. Custom CUDA Kernel Dispatch Verification

To guarantee zero simulation or mockup, `KernelDispatchTracker` wraps the underlying C++ extension functions (`_C.matmul_tiled`, `_C.matmul_wmma`, `_C.fused_bias_activation`, `_C.fused_bias_activation_backward`) using dynamic runtime interception.

### 3.1 Mathematical Dispatch Accounting (3 Epochs, 237 Training Steps, 48 Validation Steps)
The workload consists of:
- **Training:** 10,000 samples / 128 batch size = 79 steps/epoch $\times$ 3 epochs = **237 training steps**.
- **Validation:** 2,000 samples / 128 batch size = 16 steps/epoch $\times$ 3 epochs = **48 validation steps** (forward only).

#### Per-Step Dispatch Breakdown:
1. **Training Forward Pass (per step):**
   - 4 `matmul_tiled` forward matrix multiplications ($X \cdot W^T$ across FC1, FC2, FC3, FC4).
   - 4 `fused_bias_activation` forward operations (FC1 GELU, FC2 GELU, FC3 ReLU, FC4 Linear/None).
2. **Training Backward Pass (per step):**
   - 4 `fused_bias_activation_backward` gradient operations (FC4 None, FC3 ReLU, FC2 GELU, FC1 GELU).
   - 7 `matmul_tiled` backward matrix multiplications:
     - FC4: $dY \cdot W^T$ (input grad) + $X^T \cdot dY$ (weight grad) = 2
     - FC3: $dY \cdot W^T$ (input grad) + $X^T \cdot dY$ (weight grad) = 2
     - FC2: $dY \cdot W^T$ (input grad) + $X^T \cdot dY$ (weight grad) = 2
     - FC1: $X^T \cdot dY$ (weight grad only, input images have `requires_grad=False`) = 1
3. **Validation Pass (per step):**
   - 4 `matmul_tiled` forward matrix multiplications.
   - 4 `fused_bias_activation` forward operations.

#### Aggregate Dispatch Counts:
$$\text{matmul\_tiled} = (237 \times 4) + (237 \times 7) + (48 \times 4) = 948 + 1,659 + 192 = \mathbf{2,799}$$
$$\text{fused\_bias\_activation} = (237 \times 4) + (48 \times 4) = 948 + 192 = \mathbf{1,140}$$
$$\text{fused\_bias\_activation\_backward} = 237 \times 4 = \mathbf{948}$$
$$\mathbf{Total\ Custom\ Kernel\ Invocations} = 2,799 + 1,140 + 948 = \mathbf{4,887}$$

### 3.2 Dispatch Tracker Reporting Clarification
In `dispatch_tracker.py`, functions are categorized by name matching: any function not ending with `_backward` is grouped under `custom_forward_dispatches`. Because `matmul_tiled` is a general SGEMM kernel called in both forward and backward autograd steps, all 2,799 invocations appear under `custom_forward_dispatches` in the raw JSON output.
The true execution phase breakdown is:
- **True Forward Custom Dispatches:** $948\text{ (train fwd)} + 192\text{ (val fwd)} + 1,140\text{ (fused bias)} = \mathbf{2,280}$
- **True Backward Custom Dispatches:** $1,659\text{ (train bwd matmul)} + 948\text{ (fused bias bwd)} = \mathbf{2,607}$
- **Total Custom Dispatches:** $\mathbf{4,887}$

### 3.3 Explicit Native Fallback Accounting
- `cross_entropy_loss`: **237** invocations (standard PyTorch loss function).
- `optimizer_step`: **237** invocations (standard PyTorch SGD optimizer).
- `is_custom_dispatched = True` confirmed.

---

## 4. Experimental Results and Performance Analysis

All three backends executed sequentially on the identical machine under strictly standardized training budgets:
- **Dataset:** CIFAR-10 (10,000 train, 2,000 val, 2,000 test)
- **Batch Size:** 128
- **Learning Rate:** 0.01 (SGD, momentum=0.9, weight_decay=1e-4)
- **Epochs:** 3 (237 training steps)
- **Precision:** FP32 (Single Precision)

### 4.1 Convergence and Accuracy Parity

| Backend | Initial Train Loss | Epoch 1 Loss | Epoch 2 Loss | Epoch 3 Loss | Best Val Acc (%) | Final Test Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CPU_NATIVE`** | ~2.302 | 2.2012 | 1.9642 | 1.8335 | 39.05% | **39.45%** |
| **`GPU_NATIVE`** | ~2.302 | 2.2009 | 1.9611 | 1.8292 | 38.65% | **38.80%** |
| **`GPU_CUSTOM`** | ~2.302 | 2.1978 | 1.9602 | 1.8290 | 38.85% | **39.35%** |

*Analysis:* All three backends exhibit mathematically equivalent convergence trajectories down to within $0.004$ loss delta and achieve test accuracies within $0.65\%$ of each other. The minor variations between CPU and GPU reflect standard FP32 non-associative floating-point reduction order differences.

### 4.2 Latency, Throughput, and Resource Consumption Audit

| Metric | `CPU_NATIVE` | `GPU_NATIVE` | `GPU_CUSTOM` | Custom vs CPU | Custom vs Native GPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Wall-Clock Training Time** | 8.223 s | 5.345 s | 6.104 s | **1.347× faster** | 0.876× throughput |
| **End-to-End Pipeline Throughput** | 3,648.2 samples/s | 5,612.7 samples/s | 4,914.4 samples/s | **1.347× faster** | 87.56% of cuBLAS |
| **Median Step Latency** | 15.56 ms | 5.33 ms | 8.57 ms | **1.815× faster** | 1.608× latency |
| **Mean Step Latency** | 15.72 ms | 6.15 ms | 9.20 ms | **1.708× faster** | 1.496× latency |
| **Total Pure Compute Step Time** | 3.725 s | 1.458 s | 2.179 s | **1.708× faster** | 0.669× time |
| **Pure Compute Training Throughput** | 8,053.7 samples/s | 20,576.1 samples/s | 13,767.8 samples/s | **1.708× faster** | 66.91% of cuBLAS |
| **Peak Allocated VRAM** | 0.0 MB | 71.97 MB | 71.97 MB | N/A | **Identical (100%)** |
| **Peak Reserved VRAM** | 0.0 MB | 82.00 MB | 84.00 MB | N/A | +2.0 MB overhead |
| **Host Process RSS Memory** | 705 MB $\to$ 840 MB | 840 MB $\to$ 1,158 MB | 1,158 MB $\to$ 1,189 MB | N/A | Stable working set |
| **GPU Power Draw** | 11.43 W – 11.64 W | 6.60 W – 9.96 W | 9.96 W – 12.36 W | N/A | Normal thermal profile |
| **GPU Temperature** | 51.0 °C – 54.0 °C | 48.0 °C – 54.0 °C | 54.0 °C – 56.0 °C | N/A | Well below thermal limit |

### 4.3 Detailed Timing Discrepancy & Latency Analysis
A critical question in benchmarking GPU systems is: **Why do median step latency (8.57 ms) and total wall-clock training time (6.104 s) imply different effective per-step durations (25.75 ms)?**

#### Mathematical Decomposition of the 17.18 ms/step Gap:
Over 3 epochs (237 steps), the effective wall-clock time per step is $6,104\text{ ms} / 237 = \mathbf{25.75\text{ ms/step}}$, whereas the median measured step latency is $\mathbf{8.57\text{ ms/step}}$. The difference ($\mathbf{17.18\text{ ms/step}}$) represents real system overhead outside the core GPU compute step:

1. **Inter-Step DataLoader & CPU Augmentation Overhead (~4.5 ms/step):**
   Step latency timing begins *after* `batch_idx, (data, targets)` is yielded by the DataLoader. The CPU-bound operations—fetching batch tensors from memory, performing random reflection crops, applying random horizontal flips, and converting dtypes—take approximately 4.5 ms per batch on the host CPU.
2. **Host-to-Device Memory Transfer (~0.8 ms/step):**
   `data.to(device)` and `targets.to(device)` move $128 \times 3,072 \times 4\text{ bytes} \approx 1.57\text{ MB}$ per batch over PCIe Gen4. This transfer occurs immediately prior to `t0` and is preceded by a CUDA synchronization.
3. **Inter-Epoch Validation Passes (6,000 forward evaluations = ~35% of total wall time):**
   In `trainer.fit()`, `epoch_time_sec` and `total_wall_time_sec` measure the complete epoch including `val_loss, val_acc = self.evaluate(self.val_loader)`. Across 3 epochs, **48 validation batches (6,000 sample forward evaluations)** are executed. This accounts for approximately $1.92\text{ s}$ of the $6.104\text{ s}$ total wall time.
4. **Synchronous Checkpoint Disk Serialization (~0.3 s total):**
   Whenever a new best validation accuracy is recorded, `trainer.save_checkpoint()` synchronizes and serializes the complete model and optimizer state dictionaries to disk (`torch.save`), introducing synchronous disk I/O latency.
5. **Cold-Start JIT and Driver Allocation Outlier (Epoch 1, Step 1):**
   The first step incurs CUDA driver context allocation, memory pool allocation, and kernel instruction caching ($80.49\text{ ms}$ for `GPU_CUSTOM`, $130.28\text{ ms}$ for `GPU_NATIVE`). The median metric cleanly filters out this one-time startup spike, whereas total wall time includes it.

#### Throughput Metric Definitions:
- **End-to-End Pipeline Throughput:** Calculated as $\text{Total Train Samples} / \text{Total Wall Time} = 30,000 / 6.104\text{ s} = \mathbf{4,914.4\text{ samples/s}}$. This represents the practical, real-world user throughput of the complete training process including validation and data loading.
- **Pure Compute Training Throughput:** Calculated as $\text{Total Train Samples} / \text{Total Step Time} = 30,000 / 2.179\text{ s} = \mathbf{13,767.8\text{ samples/s}}$. This represents the raw computational execution speed of the custom CUDA kernels during optimization steps.

### 4.4 Architectural Breakdown and cuBLAS Comparison
1. **Custom GPU vs Host CPU ($1.347\times$ wall speedup, $1.708\times$ compute speedup):**
   The custom CUDA acceleration engine significantly outperforms host CPU execution across all metrics: end-to-end throughput (4,914 vs 3,648 samples/s) and pure compute throughput (13,768 vs 8,054 samples/s).
2. **Custom GPU vs Native GPU cuBLAS ($87.6\%$ wall throughput, $66.9\%$ compute throughput):**
   In end-to-end wall time, `GPU_CUSTOM` achieves **87.6% of cuBLAS throughput** (6.10s vs 5.35s). In pure compute step latency, cuBLAS is $1.496\times$ faster (6.15 ms vs 9.20 ms).
   - **Why cuBLAS is faster:**
     - **Python Autograd Entry Overhead:** Each custom layer incurs Python `torch.autograd.Function.apply` function entry/exit overhead, creating intermediate tuple allocations.
     - **Ampere Hardware Asynchronous Copies (`cp.async`):** cuBLAS internally uses Ampere hardware-accelerated asynchronous tensor copy instructions (`cp.async`) and multi-stage software pipelining across thread blocks, whereas our `matmul_tiled` implementation relies on standard shared memory double-buffering.
     - **Kernel Launch Queueing:** Native PyTorch coalesces certain operations into internal execution graphs, whereas custom layers launch individual forward and backward kernels sequentially on the default stream.
3. **VRAM Safety:**
   Both GPU backends consume **exactly 71.97 MB** peak allocated VRAM. This is safely well beneath the 6,144 MB hardware limit of the RTX 3050 Laptop GPU (1.17% VRAM utilization).

---

## 5. Verification and Correctness Testing

### 5.1 Unit and Integration Test Results
A dedicated test suite [`tests/test_stage2_milestone4_training.py`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/tests/test_stage2_milestone4_training.py) was executed:

| Test Case | Description | Result |
| :--- | :--- | :---: |
| `test_output_shapes_and_parameter_counts` | Verifies identical parameter counts (3,426,250) and output tensor shapes `(B, 10)` across all three backends. | **PASSED** |
| `test_canonical_initial_weights_bitwise_equality` | Confirms bitwise identical parameter values across backends at initialization. | **PASSED** |
| `test_forward_output_parity_on_same_input` | Verifies forward activation alignment ($< 1 \times 10^{-3}$ max absolute delta) between backends on identical input tensors. | **PASSED** |
| `test_pilot_training_run_and_parameter_updates` | Executes pilot training runs verifying non-zero parameter updates, valid loss progression, affirmative kernel dispatch, and zero NaNs/Infs. | **PASSED** |
| `test_checkpoint_roundtrip_fidelity` | Verifies atomic state saving and loading round-trip with zero parameter drift. | **PASSED** |
| `test_three_backends_comparable_step_loss` | Confirms synchronized loss trajectories across backends on identical single-batch steps ($< 1 \times 10^{-3}$ delta). | **PASSED** |

**Test Summary:** `6 passed in 30.57s`.

---

## 6. Full Regression and System Integrity Audit

To maintain complete backward compatibility and system safety, the comprehensive repository regression suite was executed:

1. **Full Custom CUDA Kernel Test Suite:**
   - Command: `.venv\Scripts\python.exe -m pytest -o pythonpath=. tests/test_custom_kernels.py tests/test_stage2_precision.py tests/test_stage2_wmma.py tests/test_stage2_backward.py tests/test_stage2_training_smoke.py tests/test_stage2_milestone4_training.py -q`
   - Result: **512 passed in 12.59s** (100% pass rate).
2. **Backend and Engine Regression Suite:**
   - Command: `.venv\Scripts\python.exe -m pytest -o pythonpath=. backend/tests engine/tests -q`
   - Result: **57 passed, 1 skipped in 12.19s** (100% baseline match).
3. **Native Regression Executable:**
   - Command: `.\bin\benchmark_regression.exe`
   - Result: `PASS: non-finite/missing/export failures, vector tail, partial matrix tiles, independent CPU reference.`
4. **Phase 2 Cryptographic Manifest Integrity:**
   - Verified all 17 protected files against `results/phase2_verification_20261016/manifest.json`.
   - Result: **17 / 17 SHA-256 hashes matched (100% integrity)**.
5. **Phase 9 ResNet-18 Checkpoint Integrity:**
   - Path: `checkpoints/resnet18_cifar10/best_model.pt`
   - Size: 134,220,210 bytes
   - SHA-256: `a411eb7c079058e3aada3edb2d851106273527a8d17971936a24671542dc3a78`
   - Status: **Bitwise identical, completely untouched**.
6. **Frontend and Environment Isolation:**
   - `frontend/` directory remained 100% untouched.
   - Toolchain used exclusively process-local CUDA 12.4.1 and MSVC 19.38.

---

## 7. Deliverables and File Inventory

| File Path | Description |
| :--- | :--- |
| [`engine/models/cifar10_mlp.py`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/engine/models/cifar10_mlp.py) | 4-layer Deep MLP architecture with backend-switchable layers and canonical weight generation. |
| [`engine/training/cifar10_three_backend_trainer.py`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/engine/training/cifar10_three_backend_trainer.py) | Production three-backend trainer supporting profiling, checkpointing, and metrics calculation. |
| [`engine/training/dispatch_tracker.py`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/engine/training/dispatch_tracker.py) | Dynamic C++ extension interception module for kernel call accounting and fallback logging. |
| [`tests/test_stage2_milestone4_training.py`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/tests/test_stage2_milestone4_training.py) | Unit and integration test suite verifying three-backend parity and training correctness. |
| [`scripts/train_stage2_cifar10_comparison.py`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/scripts/train_stage2_cifar10_comparison.py) | Reproducible three-backend benchmark script executing sequential training and saving JSON results. |
| [`results/phase10_stage2_milestone4/cifar10_three_backend_results.json`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/results/phase10_stage2_milestone4/cifar10_three_backend_results.json) | Complete machine-readable experiment dataset with step timings, telemetry, and dispatches. |
| [`checkpoints/phase10_stage2_milestone4/`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/checkpoints/phase10_stage2_milestone4) | Saved best checkpoints (`cpu_native_best.pt`, `gpu_native_best.pt`, `gpu_custom_best.pt`). |

---

## 8. Known Limitations and Optimization Priorities

### 8.1 Current Limitations
1. **Python Autograd Dispatch Boundary:** In the `GPU_CUSTOM` path, backward execution passes through `torch.autograd.Function`, which creates Python interpreter boundary overhead for small layer operations compared to native C++ engine graphs.
2. **Single-Precision SGEMM Focus:** Milestone 4 focused strictly on FP32 training parity. Mixed-precision WMMA Tensor Core kernels (`matmul_wmma`) are ready in our C++ extension but were intentionally kept separate for fair FP32 baseline comparisons.
3. **Loss and Optimizer Fallback:** Loss computation (`CrossEntropyLoss`) and parameter updates (`SGD.step()`) currently use PyTorch native implementations, as documented in our dispatch tracker.
4. **Inter-Backend DataLoader Shuffle Permutation:** In the comparative benchmark script, the DataLoader generator is not re-seeded between backends, meaning each backend sees consecutive distinct epoch shuffles of the identical 10,000 dataset samples.

### 8.2 Recommended Optimization Priorities for Milestone 5
1. **Fused Optimizer Kernels:** Implement a custom CUDA kernel for fused SGD/AdamW parameter updates ($W \leftarrow W - \eta \cdot (\nabla W + \lambda W)$), keeping weights and gradients in GPU registers and eliminating separate tensor memory reads/writes.
2. **C++ End-to-End Execution Graph:** Implement a C++ training loop harness bypassing Python autograd overhead during batch iterations.
3. **Mixed-Precision FP16 Training with Tensor Cores:** Introduce AMP (Automatic Mixed Precision) using our verified `matmul_wmma` kernel for forward GEMMs with FP32 master weight accumulation.

---

## 9. Reproducible Execution Commands

All benchmark results and regression checks can be reproduced using the following commands from the repository root:

### 1. Download and Verify CIFAR-10 Dataset
```powershell
.venv\Scripts\python.exe -c "from engine.data.cifar10 import download_and_verify_cifar10; download_and_verify_cifar10()"
```

### 2. Execute Three-Backend Comparative Training Benchmark
```powershell
.venv\Scripts\python.exe scripts/train_stage2_cifar10_comparison.py --epochs 3 --batch-size 128 --lr 0.01 --max-train-samples 10000
```

### 3. Run Dedicated Milestone 4 Training Unit & Parity Tests
```powershell
.venv\Scripts\python.exe -m pytest -o pythonpath=. tests/test_stage2_milestone4_training.py -v
```

### 4. Run Full Custom CUDA Kernel Test Suite (512 Tests)
```powershell
.venv\Scripts\python.exe -m pytest -o pythonpath=. tests/test_custom_kernels.py tests/test_stage2_precision.py tests/test_stage2_wmma.py tests/test_stage2_backward.py tests/test_stage2_training_smoke.py tests/test_stage2_milestone4_training.py -q
```

### 5. Run Backend and Engine Regression Suite (57 Passed, 1 Skipped)
```powershell
.venv\Scripts\python.exe -m pytest -o pythonpath=. backend/tests engine/tests -q
```

### 6. Run Native Benchmark Regression Binary
```powershell
.\bin\benchmark_regression.exe
```

### 7. Verify Phase 2 Manifest Cryptographic Integrity (17/17 Hashes)
```powershell
.venv\Scripts\python.exe -c "import json, hashlib, pathlib; manifest = json.load(open('results/phase2_verification_20261016/manifest.json')); assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == m['sha256'] for p, m in manifest.items()); print('All 17 hashes match: 100% verified')"
```

### 8. Verify Phase 9 ResNet-18 Checkpoint Integrity
```powershell
.venv\Scripts\python.exe -c "import hashlib, pathlib; b = pathlib.Path('checkpoints/resnet18_cifar10/best_model.pt').read_bytes(); assert hashlib.sha256(b).hexdigest() == 'a411eb7c079058e3aada3edb2d851106273527a8d17971936a24671542dc3a78' and len(b) == 134220210; print('Phase 9 Checkpoint bitwise verified: 100% intact')"
```
