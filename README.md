# CUDA GPU Acceleration Engine

[![CUDA](https://img.shields.io/badge/CUDA-12.4%20%7C%2013.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-17%20(MSVC%2014.38)-00599C?logo=c%2B%2B&logoColor=white)](https://visualstudio.microsoft.com/)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18%20(TypeScript)-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.4-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Acceptance](https://img.shields.io/badge/E2E%20Acceptance-10%2F10%20PASS-10B981)](#-acceptance-and-certification-results)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, high-performance GPU acceleration, neural network training, and real-time observability platform engineered for **NVIDIA Ampere SM 8.6+** architectures.

The engine bridges bare-metal CUDA C++ (tiled shared-memory GEMM, sub-warp FP16 WMMA Tensor Core intrinsics, and fused bias-GELU epilogues) with an asynchronous FastAPI backend and a cinematic dark-mode React 18 dashboard.

---

## 1. Full Platform Architecture

```mermaid
graph TD
    subgraph UI ["Interactive Frontend (React 18 + TypeScript + Vite + Tailwind CSS)"]
        Dashboard["GPU Acceleration Command & Control Dashboard"]
        CmdCenter["1. GPU Command Center (Live NVML Telemetry)"]
        Studio["2. AI Training Studio (Multi-Backend Orchestrator)"]
        Registry["3. Model Registry & Checkpoint Inspector"]
        Inference["4. LLM Inference Playground (KV-Cache Autoregression)"]
        Lab["5. CUDA Laboratory (Configs A, B, C, D Roofline)"]
        Dashboard --> CmdCenter & Studio & Registry & Inference & Lab
    end

    subgraph API ["Backend Service (FastAPI + Starlette Daemon • Port 8000)"]
        RouterHub["FastAPI Routing Hub"]
        Orchestrator["TrainingOrchestrator (Thread Loop + SQLite JobStore)"]
        ModelServer["ModelServer (In-Memory GPU Cache & Generation Engine)"]
        HWMonitor["HardwareMonitor (NVML & nvidia-smi Fallback)"]
        Concurrency["Concurrency Guard & Resource Lock (HTTP 409)"]
        RouterHub --> Orchestrator & ModelServer & HWMonitor & Concurrency
    end

    subgraph Engine ["High-Performance Compute & Acceleration Engine"]
        Adaptive["AdaptiveExecutionEngine (Evidence-Based Kernel Selection)"]
        CustomCUDA["Custom CUDA C++ Kernels (Shared Memory Tiled & WMMA FP16)"]
        FusedOps["Fused Epilogues (Bias + GELU / cuBLASLt)"]
        PyTorchNative["Native PyTorch SDPA & FlashAttention-2"]
        CPURef["Reference Single-Threaded CPU Engine"]
        Adaptive --> CustomCUDA & FusedOps & PyTorchNative & CPURef
    end

    subgraph Hardware ["Target Hardware Infrastructure"]
        GPU["NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere SM 8.6, 20 SMs, 2560 Cores)"]
        CPU["Intel Core i5-13450HX (10 Cores, 16 Threads)"]
    end

    UI <== "Vite Reverse Proxy (/api/*)" ==> API
    API <== "Direct In-Process Dispatch" ==> Engine
    Engine <== "CUDA Runtime API & Driver" ==> GPU
```

---

## 2. Key Engineering Highlights

- **Custom Bare-Metal CUDA & Tensor Core Kernels**:
  - **Tiled Shared-Memory GEMM**: $16\times16$ and $32\times32$ on-chip SRAM block tiling with memory barrier synchronization (`__syncthreads()`), achieving up to **$116.75\times$ kernel speedup** over CPU baseline.
  - **Sub-Warp WMMA Tensor Cores**: Hardware-accelerated FP16 matrix multiply-accumulate warp intrinsics targeting Ampere Tensor Cores.
  - **Custom Fused Epilogues**: CUDA C++ kernels combining bias addition with GELU activation in a single memory pass, eliminating intermediate activation SRAM traffic.
- **Adaptive Kernel Selection (`AdaptiveExecutionEngine`)**:
  - Dynamically profiles workload shapes and operational phases.
  - Automatically routes **training** to cuBLASLt epilogue-fused PyTorch (optimizing autograd backward pass throughput) and **inference** to custom CUDA WMMA kernels.
- **NEXORA GPT Pretrained Language Models**:
  - **NEXORA GPT-86.8M**: Production foundation model certified across 12,208 optimizer steps (50,004,608 tokens trained, final validation loss 1.9383, validation perplexity 6.95).
  - **NEXORA GPT-100M**: Target preset architecture ($92,133,888$ parameters) and exploratory pilot checkpoint (15 steps, $85.94\text{M}$ active params, truthfully marked `is_partially_trained: true`).
- **AI Training Studio & Multi-Backend Orchestrator**:
  - Supports 7 distinct architectures (`transformer_1m`, `transformer_25m`, `transformer_50m`, `transformer_100m`, `resnet18`, `cifar10_mlp`, `baseline_mlp`).
  - Strict mutual exclusion GPU locking, real-time metric streaming, cooperative step-bounded cancellation, and atomic `.pt` serialization.
- **Comparative Training Laboratory (Configs A–D)**:
  - **Config A**: Standard PyTorch Baseline (FP32, Manual Quadratic Attention).
  - **Config B**: Optimized PyTorch (AMP FP16, Native SDPA).
  - **Config C**: Optimized PyTorch + Custom CUDA (SDPA, AMP, Fused Bias+GELU).
  - **Config D**: Adaptive Execution Engine (cuBLASLt bias fusion).
- **Direct Hardware Observability**:
  - Real-time NVML hardware telemetry polling (SM load %, temperature °C, board power W, dedicated VRAM allocation) with automatic fallback and thermal safety thresholds ($75^\circ\text{C}$ warning, $83^\circ\text{C}$ critical cutoff).

---

## 3. System Requirements & Hardware Specifications

| Component | Minimum Specification | Certified Target System |
| :--- | :--- | :--- |
| **GPU** | NVIDIA Ampere SM 8.0+ (4GB VRAM) | **NVIDIA GeForce RTX 3050 6GB Laptop GPU (SM 8.6, 20 SMs)** |
| **Host CPU** | x86_64 Quad-Core CPU | **Intel Core i5-13450HX (10 Cores, 16 Threads, up to 4.6 GHz)** |
| **Host RAM** | 16 GB DDR4/DDR5 | **16 GB DDR5 4800 MHz** |
| **Operating System** | Windows 10/11 64-bit or Ubuntu 22.04+ | **Windows 11 Home 64-bit (Build 26100)** |
| **NVIDIA Driver** | >= 535.00 | **GeForce Game Ready Driver 572.16 (CUDA 12.8 compatible)** |
| **CUDA Toolkit** | CUDA 12.0+ or 13.0+ | **CUDA 12.4 (Active PyTorch runtime) / CUDA 13.4 Native** |
| **Host Compiler** | MSVC 19.38+ (C++17) or GCC 11+ | **MSVC 14.38.33130 (Visual Studio Build Tools) / C++17** |
| **Python** | Python 3.10 – 3.12 | **Python 3.11.9 (64-bit)** |
| **Node.js** | Node.js 18+ & npm 9+ | **Node.js v20+ / npm v10+** |

---

## 4. Windows Environment Setup & Clean Installation

### Option A: Standard Virtual Environment (Recommended)

```powershell
# 1. Clone repository and navigate to root
cd C:\Projects\CUDA-GPU-Acceleration-Engine

# 2. Create isolated Python 3.11 virtual environment
python -m venv .venv

# 3. Activate environment
.\.venv\Scripts\Activate.ps1

# 4. Install production dependencies (includes PyTorch 2.6.0+cu124)
pip install -r requirements.txt

# 5. Install frontend dependencies
cd frontend
npm install
cd ..
```

### Option B: Conda / Mamba Environment

```powershell
# 1. Create environment from pinned configuration
conda env create -f environment.yml

# 2. Activate environment
conda activate cuda-acceleration-engine

# 3. Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## 5. Quickstart & Service Execution

### One-Click Platform Launcher

Launch both the backend and frontend simultaneously with automated environment checks:

```powershell
# Batch launcher
.\start_platform.bat

# Or PowerShell launcher
.\start_platform.ps1
```

### Manual Individual Service Launch

**Terminal 1 — FastAPI Backend Daemon (Port 8000)**:
```powershell
cd C:\Projects\CUDA-GPU-Acceleration-Engine
$env:PYTHONPATH = "C:\Projects\CUDA-GPU-Acceleration-Engine"
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — React/Vite Frontend Dev Server (Port 5173)**:
```powershell
cd C:\Projects\CUDA-GPU-Acceleration-Engine\frontend
npm run dev
```

- **Interactive Dashboard**: `http://127.0.0.1:5173/`
- **Swagger REST API Documentation**: `http://127.0.0.1:8000/docs`
- **API Health Check**: `http://127.0.0.1:8000/api/health`

### Production Static Build Preview

```powershell
cd frontend
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

---

## 6. Pretrained Models & Scientific Lineage Truthfulness

The platform maintains absolute scientific integrity by strictly distinguishing certified pretrained foundation models from exploratory architecture pilots:

```mermaid
graph TD
    subgraph M1 ["Certified Foundation Model (Intended Public Release — CC-BY-4.0; Release Auth Pending)"]
        A1["NEXORA GPT-86.8M"]
        A2["Weights: checkpoints/transformer/pretraining_50m_campaign/best_model.pt"]
        A3["Pretraining: 12,208 Steps | 50,004,608 Tokens"]
        A4["Validation: Loss 1.9383 | Perplexity 6.95"]
        A5["Status: Production Ready | CC-BY-4.0 Selected (Release Auth Pending)"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph M2 ["Exploratory Architecture Pilot (Internal Engineering Material)"]
        B1["NEXORA GPT-100M Pilot"]
        B2["Weights: checkpoints/transformer/flagship_100m_pilot/best_model.pt"]
        B3["Pretraining: 15 Steps (Exploratory Smoke Run)"]
        B4["Validation: Step Loss 5.3174"]
        B5["Status: Internal Fixture | Withheld from Public Distribution"]
        B1 --> B2 --> B3 --> B4 --> B5
    end
```

### Model Architectural Comparison
| Parameter / Metric | NEXORA GPT-86.8M (Production Campaign) | NEXORA GPT-100M (Preset Target) | NEXORA GPT-100M (Pilot Checkpoint) |
| :--- | :--- | :--- | :--- |
| **Model Role** | Certified Pretrained Foundation Model (Intended Public — CC-BY-4.0; Release Auth Pending) | Full Flagship Architecture Specification | Exploratory Architecture Verification (Internal Only) |
| **Exact Parameters** | **86,825,472** | **92,133,888** | **85,940,736** (pilot vocab subset) |
| **Layers / Heads / Hidden** | 12 Layers / 12 Heads / $d=768$ | 12 Layers / 12 Heads / $d=768$ | 12 Layers / 12 Heads / $d=768$ |
| **Feed-Forward Dimension** | $d_{\text{ff}} = 3072$ | $d_{\text{ff}} = 3072$ | $d_{\text{ff}} = 3072$ |
| **Vocabulary Size ($V$)** | 2048 | 8192 | 512 |
| **Context Length ($T$)** | 256 | 1024 | 128 |
| **Weight Tying** | Enabled | Enabled | Enabled |
| **Steps Trained** | **12,208 steps** | — | **15 steps** |
| **Tokens Processed** | **50,004,608 tokens** | — | ~15,360 tokens |
| **Loss / Perplexity** | Val Loss: **1.9383** / PPL: **6.95** | — | Step Loss: **5.3174** |
| **Registry Flag** | `is_partially_trained: false` | — | `is_partially_trained: true` |
| **Generation Latency** | **17.68–18.49 ms/token** | Estimated ~20 ms/token | **20.95 ms/token** (47.7 tok/s) |
| **VRAM Footprint** | **344.8 MB** (Inference) | Estimated ~520 MB | **342.1 MB** (Inference) |

---

## 7. Verified CUDA Benchmarks & Comparative Laboratory

### Dense GEMM Compute Scaling (RTX 3050 Laptop GPU)
Measurements captured over 20 synchronized runs after 5 warm-up iterations:

| Dimension | Mode / Engine | Mean Latency (ms) | TFLOPS | Speedup vs. CPU | Speedup vs. FP32 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$1024 \times 1024$** | CPU Single-Thread Ref (FP32) | 15.44 ms | 0.139 | 1.0x | — |
| | GPU Native cuBLAS (FP32) | 0.909 ms | 2.36 | 17.0x | 1.0x |
| | GPU Native AMP (FP16) | 0.549 ms | 3.91 | **28.1x** | **1.65x** |
| **$2048 \times 2048$** | CPU Single-Thread Ref (FP32) | 327.17 ms | 0.053 | 1.0x | — |
| | GPU Native cuBLAS (FP32) | 13.163 ms | 1.31 | 24.9x | 1.0x |
| | GPU Native AMP (FP16) | 1.920 ms | **8.95** | **170.4x** | **6.86x** |

### Matched 4-Way Comparative Training Laboratory (Configs A–D)
Synchronized gradient step evaluation on micro-transformer ($d=768, L=4, H=12$, Batch Size 4, Sequence Length 128):

| Configuration | Architectural Pipeline | Step Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) | Speedup vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Config A** | Standard PyTorch Baseline (FP32, Quadratic Attention) | 55.80 ms | 9,175.0 | 577.86 MB | 1.00x (Baseline) |
| **Config B** | Optimized PyTorch (Native SDPA + AMP FP16) | 46.21 ms | 11,080.2 | 578.36 MB | **1.21x** |
| **Config C** | Optimized PyTorch + Custom CUDA (Fused Bias+GELU) | **44.49 ms** | **11,509.1** | **489.83 MB** | **1.25x** |
| **Config D** | Adaptive Execution Engine (cuBLASLt Epilogue Fusion) | 46.34 ms | 11,049.9 | 633.46 MB | **1.20x** |

> [!TIP]
> **Configuration C** yields both the fastest step execution (**44.49 ms**) and the lowest memory footprint (**489.83 MB**, saving **88.53 MB** over Config B) due to kernel epilogue fusion eliminating intermediate activation tensors in global memory.

---

## 8. REST API Usage & Examples

### Health & Telemetry
```bash
# Query system health
curl -s http://127.0.0.1:8000/api/health

# Query live NVML GPU metrics
curl -s http://127.0.0.1:8000/api/hardware
```

### Model Inference
```bash
# Generate text using NEXORA GPT-86.8M
curl -X POST http://127.0.0.1:8000/api/models/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "NEXORA GPT-86.8M",
    "prompt": "The CUDA GPU architecture",
    "max_new_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9,
    "device": "cuda:0"
  }'
```

### AI Training Job Dispatch
```bash
# Submit a step-bounded training job
curl -X POST http://127.0.0.1:8000/api/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "transformer_50m",
    "backend": "GPU_NATIVE",
    "dataset": "data/text/tiny_shakespeare.txt",
    "epochs": 1,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "max_steps": 10
  }'

# Cooperatively cancel a job
curl -X POST http://127.0.0.1:8000/api/training/jobs/{job_id}/cancel
```

---

## 9. Automated Testing & Release Certification

The repository includes a comprehensive 787-test catalog covering unit, integration, numerical parity, autograd correctness, and full system acceptance:

```powershell
# 1. Run Milestone 9 End-to-End Acceptance Suite (10 passed in 17.5s)
.\.venv\Scripts\python.exe -m pytest tests/test_milestone9_e2e_acceptance.py -v

# 2. Run Cumulative Regression & Production Suite (34 passed in 41.4s)
.\.venv\Scripts\python.exe -m pytest tests/test_milestone13_milestone9_stage2_hardening.py tests/test_milestone13_milestone9_stage4_integration.py tests/test_milestone13_milestone9_stage5_production_readiness.py tests/test_milestone9_e2e_acceptance.py -v

# 3. Run Performance, Resource & Stability Validation Suite (5/5 passed in 90.7s)
.\.venv\Scripts\python.exe scripts/validate_milestone9_performance_resources.py
```

### Certification Summary
```
========================================================================================
   CUDA GPU ACCELERATION ENGINE — FINAL RELEASE CERTIFICATION
========================================================================================
   End-to-End Acceptance Domains:      10 / 10 PASS (100.0%)
   Performance & Stability Domains:    5 / 5 PASS (100.0%)
   Cumulative Hardening Tests:         34 / 34 PASS (100.0%)
   Total Repository Test Suite:        787 Tests Collected
   Target Hardware Certified:          NVIDIA GeForce RTX 3050 6GB Laptop GPU (SM 8.6)
   Measured VRAM Leakage:              0.0000 MB across all test batteries
   Thermal Safety Boundary:            SAFE (Peak 59.09°C vs 75°C warning / 83°C cutoff)
   Release Target Date:                November 30, 2026
   Final Release Status:               PRODUCTION CERTIFIED FOR PACKAGING & RELEASE
========================================================================================
```

---

## 10. Repository Packaging & Exclusion Rules

The platform follows strict packaging rules configured via [`.gitignore`](.gitignore):
- **Included in Package**: Core CUDA C++ sources, PyTorch engines, FastAPI backend, React/Vite frontend source, production static build (`frontend/dist`), verified benchmark artifacts (`results/`), and master configuration files.
- **Excluded from Package**: Local Python virtual environments (`.venv/`, `.venv*/`), Node dependency trees (`node_modules/`), intermediate build objects (`build/`, `*.obj`, `*.o`), raw pretraining datasets (`data/*.bin`, `data/cifar-10-batches-py/`), temporary scratch files (`scratch/`, `*.tmp`), and raw exploratory checkpoint dumps.
- **Protected Checkpoints**: Production certified checkpoint (`checkpoints/transformer/pretraining_50m_campaign/best_model.pt`, CC-BY-4.0) and exploratory pilot checkpoint (`checkpoints/transformer/flagship_100m_pilot/best_model.pt`, internal only) are permanently preserved.

---

## License

- **Software Source Code & Custom CUDA Kernels**: Licensed under the [MIT License](LICENSE) (Copyright © 2026 CUDA GPU Acceleration Engine Contributors).
- **NEXORA GPT-86.8M Model Weights**: Governed by [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) — see [MODEL_CARD.md](MODEL_CARD.md) (Intended distribution: Public; Final release authorization: Pending).
- **NEXORA GPT-100M Pilot Checkpoint**: Internal engineering benchmark material (withheld from public distribution).
