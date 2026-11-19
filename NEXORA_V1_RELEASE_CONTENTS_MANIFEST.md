# NEXORA v1.0.0 — Release Contents Manifest

**Candidate Archive:** `dist/cuda-acceleration-engine-v1.0.0-rc4-source.zip`  
**Candidate Checksum File:** `dist/cuda-acceleration-engine-v1.0.0-rc4-checksums.sha256`  
**Archive SHA-256 Digest:** `DBDD670654F15955FEF46F9CEF9EB5E8E5CDCDD7E2ECB407C566C28EDECB5AF0`  
**Archive Size:** 1,558,575 bytes (1.49 MB compressed, 6,786,419 bytes uncompressed)  
**Total File Count:** 388 files  
**Top-Level Archive Root:** `cuda-acceleration-engine/`  
**Date of Certification:** November 17, 2026  

---

## 1. Directory Structure & File Count Overview

```
cuda-acceleration-engine/
├── [Root Files]                   (12 files)   - Licenses, Notices, Configs, Startup Scripts
├── backend/                       (15 files)   - FastAPI backend routers, models, serving
├── checkpoints/                   (4 files)    - Manifests, documentation, seed test fixture
├── config/ & configs/             (5 files)    - Model presets, logging configurations
├── custom_kernels/                (17 files)   - Python CUDA kernel bindings, autograd functions
├── data/                          (5 files)    - Manifests, small validation fixtures
├── engine/                        (52 files)   - Core ML engine, adaptive training, telemetry
├── frontend/                      (47 files)   - React SPA source + pre-built dist bundle
├── include/                       (5 files)    - Native C++/CUDA header files
├── results/                       (118 files)  - Benchmark logs, telemetry records, campaign results
├── scripts/                       (67 files)   - Automation, packaging, verification scripts
├── src/                           (5 files)    - C++/CUDA kernel implementation sources
└── tests/                         (36 files)   - Complete pytest regression & verification suite
────────────────────────────────────────────────────────────────────────────────
Total Packaged Files:              388 files
```

---

## 2. Itemized Release File Inventory

### 2.1 Root Files (12 files)
| File Path | Description |
| :--- | :--- |
| `cuda-acceleration-engine/.gitignore` | Repository gitignore rules |
| `cuda-acceleration-engine/CMakeLists.txt` | CMake build configuration for native C++/CUDA extension |
| `cuda-acceleration-engine/LICENSE` | Project open-source license declaration |
| `cuda-acceleration-engine/MODEL_CARD.md` | Model architecture, training provenance, and licensing card |
| `cuda-acceleration-engine/NOTICE` | Formal attribution and copyright notice |
| `cuda-acceleration-engine/README.md` | Platform documentation, architectural overview, and quickstart |
| `cuda-acceleration-engine/THIRD_PARTY_LICENSES.md` | Inventory of 36 dependencies, datasets, and licenses |
| `cuda-acceleration-engine/environment.yml` | Conda environment specification |
| `cuda-acceleration-engine/pytest.ini` | Pytest configuration and test filters |
| `cuda-acceleration-engine/requirements.txt` | Production Python pip dependencies |
| `cuda-acceleration-engine/start_platform.bat` | Windows batch single-server platform launcher |
| `cuda-acceleration-engine/start_platform.ps1` | Windows PowerShell single-server platform launcher |

### 2.2 Backend (`backend/`, 15 files)
| File Path | Description |
| :--- | :--- |
| `backend/__init__.py` | Backend package initialization |
| `backend/app.py` | FastAPI application entrypoint, CORS, SPA static mounts |
| `backend/database.py` | Platform database and session state management |
| `backend/main.py` | Direct Uvicorn backend runner |
| `backend/models.py` | Pydantic API request/response schemas |
| `backend/serving.py` | Inference serving engine, model caching, and VRAM management |
| `backend/routers/__init__.py` | Router subpackage initialization |
| `backend/routers/adaptive_training.py` | Adaptive training hardware and recommendation endpoints |
| `backend/routers/benchmarks.py` | GPU benchmark results and comparative performance API |
| `backend/routers/dashboard.py` | Overview telemetry and hardware summary endpoints |
| `backend/routers/experiments.py` | Laboratory experiment execution and history API |
| `backend/routers/hardware.py` | Real-time NVML GPU hardware metrics and telemetry API |
| `backend/routers/inference.py` | Vision and Transformer inference execution endpoints |
| `backend/routers/models.py` | Model catalog registry, loading, and unloading endpoints |
| `backend/routers/training.py` | Training job lifecycle, queue management, and metrics API |

### 2.3 Checkpoints & Seed Fixtures (`checkpoints/`, 4 files)
| File Path | Description |
| :--- | :--- |
| `checkpoints/.gitkeep` | Git directory anchor |
| `checkpoints/MODEL_MANIFEST.json` | Registry manifest detailing 5 model architectures & hashes |
| `checkpoints/README.md` | Instructions on downloading heavy pre-trained weights |
| `checkpoints/test_checkpoint.pt` | Verified seed model fixture (`BaselineMLP`, 17,226 params) |

### 2.4 Configuration (`config/` & `configs/`, 5 files)
| File Path | Description |
| :--- | :--- |
| `config/logging.conf` | Platform structured logging configuration |
| `configs/cifar10_mlp.json` | Baseline MLP training and architecture configuration |
| `configs/logging.json` | Telemetry logging output formatting rules |
| `configs/nexora_gpt_25m.json` | 25M Foundation Transformer hyperparameter configuration |
| `configs/resnet18.json` | ResNet18 vision architecture training configuration |

### 2.5 Custom Kernels (`custom_kernels/`, 17 files)
| File Path | Description |
| :--- | :--- |
| `custom_kernels/__init__.py` | Kernel package exports |
| `custom_kernels/activations.py` | Custom activation kernel bindings (GELU, ReLU, Sigmoid) |
| `custom_kernels/autograd.py` | PyTorch autograd function wrappers for custom CUDA kernels |
| `custom_kernels/benchmarks.py` | Custom kernel micro-benchmark harnesses |
| `custom_kernels/fallback.py` | Pure PyTorch fallback implementations for CPU/unsigned environments |
| `custom_kernels/fused_ops.py` | Fused bias-activation kernel wrappers |
| `custom_kernels/loader.py` | Dynamic C++/CUDA `.pyd` native loader with graceful fallback |
| `custom_kernels/matmul.py` | Tiled shared-memory matrix multiplication kernel wrappers |
| `custom_kernels/reductions.py` | Parallel reduction kernel wrappers (Sum, Max, Argmax) |
| `custom_kernels/vector_ops.py` | Elementwise vector addition and scaling kernel wrappers |
| `custom_kernels/wmma.py` | Tensor Core WMMA FP16 matrix multiplication kernel wrappers |
| `custom_kernels/csrc/__init__.py` | C++ source package anchor |
| `custom_kernels/csrc/bindings.cpp` | Pybind11 C++ module declarations and bindings |
| `custom_kernels/csrc/activations.cu` | CUDA kernels for activation functions |
| `custom_kernels/csrc/matmul.cu` | Tiled CUDA kernels for GEMM |
| `custom_kernels/csrc/reductions.cu` | Warp-shuffle CUDA kernels for reductions |
| `custom_kernels/csrc/vector_ops.cu` | CUDA kernels for elementwise vector operations |

### 2.6 Data Fixtures & Manifests (`data/`, 5 files)
| File Path | Description |
| :--- | :--- |
| `data/text/README.md` | Corpus documentation and dataset provenance summary |
| `data/text/expanded_pretraining/dataset_manifest.json` | Cryptographic fingerprint manifest for 200M token corpus |
| `data/text/pretraining_campaign/dataset_manifest.json` | Cryptographic fingerprint manifest for 50M token corpus |
| `data/text/production_10m/dataset_manifest.json` | Cryptographic fingerprint manifest for 10M token corpus |
| `data/text/tiny_shakespeare.txt` | Complete Tiny Shakespeare corpus fixture (1,115,394 bytes) |

### 2.7 Core Engine Subsystem (`engine/`, 52 files)
Includes all core training, model architectures, memory management, and telemetry modules:
- **Telemetry:** `engine/telemetry/nvml_monitor.py`, `engine/telemetry/metrics_collector.py`, `engine/telemetry/logger.py`
- **Adaptive Training:** `engine/training/adaptive_engine.py`, `engine/training/hardware_detector.py`, `engine/training/batch_optimizer.py`, `engine/training/oom_recovery.py`, `engine/training/precision_selector.py`, `engine/training/orchestrator.py`, `engine/training/job_store.py`
- **Model Architectures:** `engine/models/cifar10_mlp.py`, `engine/models/resnet18.py`, `engine/models/transformer.py`, `engine/models/registry.py`
- **Compiler & Autotuning:** `engine/compiler/nvcc_wrapper.py`, `engine/compiler/autotuner.py`, `engine/compiler/kernel_cache.py`
- **Data Loaders & Tokenizers:** `engine/data/memmap_dataset.py`, `engine/data/tokenizer.py`, `engine/data/scalable_loader.py`

### 2.8 Frontend Application & Pre-Built Bundle (`frontend/`, 47 files)
- **Pre-Built Production Bundle (`frontend/dist/`):**
  - `frontend/dist/index.html` (Single-server production SPA entrypoint)
  - `frontend/dist/assets/index-Dr7YA8EJ.css` (Production minified stylesheet, 46 KB)
  - `frontend/dist/assets/index-BhQcNE8G.js` (Production minified JavaScript bundle, 856 KB)
- **Source Code & Components:**
  - `frontend/src/App.tsx`, `frontend/src/main.tsx`, `frontend/src/index.css`
  - `frontend/src/components/Navigation.tsx`, `frontend/src/components/GPUStatusBar.tsx`, `frontend/src/components/ThemeToggle.tsx`
  - `frontend/src/pages/Overview.tsx`, `frontend/src/pages/TrainingStudio.tsx`, `frontend/src/pages/ModelRegistry.tsx`, `frontend/src/pages/CudaLab.tsx`, `frontend/src/pages/GpuMonitoring.tsx`
  - `frontend/src/types/api.ts` (API schemas and hardware status types)
  - `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tailwind.config.js`

### 2.9 Native CUDA Headers (`include/`, 5 files)
| File Path | Description |
| :--- | :--- |
| `include/activations.cuh` | C++ header declarations for CUDA activation kernels |
| `include/cuda_utils.cuh` | CUDA error checking macros and stream management helpers |
| `include/matmul.cuh` | C++ header declarations for GEMM kernels |
| `include/reductions.cuh` | C++ header declarations for reduction kernels |
| `include/vector_ops.cuh` | C++ header declarations for vector operation kernels |

### 2.10 Results, Benchmarks & Validation Artifacts (`results/`, 118 files)
Comprehensive audit trail of historical validation experiments, benchmark JSON records, comparative kernel analysis CSVs, and pretraining telemetry summaries across Milestones 1 through 13.

### 2.11 Release & Operational Scripts (`scripts/`, 67 files)
Operational tooling including:
- Packaging: `scripts/package_release.py`, `scripts/build_frontend.py`
- Clean-Install Verification: `scripts/verify_clean_install_rc4.py`
- Benchmark Runners: `scripts/run_benchmarks.py`, `scripts/benchmark_comparative.py`
- Pretraining & Diagnostics: `scripts/pretrain_flagship.py`, `scripts/evaluate_checkpoints.py`
- Weight Acquisition: `scripts/download_weights.py`

### 2.12 Native C++/CUDA Sources (`src/`, 5 files)
| File Path | Description |
| :--- | :--- |
| `src/activations.cu` | Standalone CUDA activation kernel implementations |
| `src/cuda_utils.cu` | Standalone CUDA memory and device utility functions |
| `src/main.cpp` | Standalone native CUDA benchmark execution binary |
| `src/matmul.cu` | Standalone CUDA matrix multiplication kernel implementations |
| `src/vector_ops.cu` | Standalone CUDA vector addition and scaling implementations |

### 2.13 Automated PyTest Suites (`tests/`, 36 files)
The complete verification and regression test battery:
- `tests/test_adaptive_training.py`
- `tests/test_single_server_deployment.py`
- `tests/test_first_run_model_verification.py`
- `tests/test_ui_acceptance_and_stabilization.py`
- `tests/test_platform_integration_e2e.py`
- `tests/test_milestone9_e2e_acceptance.py`
- `tests/test_milestone10_flagship_transformer.py`
- `tests/test_milestone11_platform_integration.py`
- `tests/test_milestone12_pretraining_campaign.py`
- `tests/test_milestone13_*` (All 8 sub-suites)
- Kernel autograd and numerical correctness suites (`test_custom_kernels.py`, `test_cuda_portability.py`, etc.)

---

## 3. Explicit Inventory of Excluded Items

To ensure that the release archive remains self-contained, secure, and lightweight, the following items are **strictly excluded** by design:

| Excluded Item / Pattern | Reason for Exclusion | Resolution / Alternative |
| :--- | :--- | :--- |
| `.venv/`, `.venv-phase3/` | Host-specific Python virtual environments (non-portable, multi-gigabyte binary bloat) | User recreates environment via `pip install -r requirements.txt` |
| `node_modules/` | Node development dependencies (non-portable, multi-gigabyte bloat) | Production bundle is pre-built in `frontend/dist/`; no Node required for execution |
| `__pycache__/`, `*.pyc`, `*.pyo` | Host-specific precompiled Python bytecode | Python automatically regenerates bytecode on the host |
| `.pytest_cache/` | Local test runner cache artifacts | Clean state upon extraction |
| `build/`, `*.obj`, `*.o`, `*.pyd`, `*.pdb` | Compiled C++ object files and dynamic libraries specific to developer host | Built on target if native toolchain exists; otherwise seamless fallback operates |
| `.git/`, `.vs/`, `.vscode/` | Version control internals and IDE workspace metadata | Clean source release distribution |
| `checkpoints/*.pt` (except test fixture) | Heavy trained weights: ResNet18 (134 MB), 50M (331 MB), 86.8M (1.02 GB) | Downloaded on demand via `scripts/download_weights.py` or trained via Studio |
| `data/cifar-10-python.tar.gz` | Heavy raw training dataset (170 MB compressed) | Acquired on demand during training execution |
| Root `dist/` | Prevents recursive embedding of past candidate zip archives | Separately archived in release distribution portal |

---

## 4. Manifest Integrity Affirmation

This manifest accurately reflects the complete, unaltered contents of `cuda-acceleration-engine-v1.0.0-rc4-source.zip`. Every entry has been verified via archive introspection and cryptographic hashing.
