# DRAFT — NOT EXECUTED
# NEXORA — Sole Contributor and Authorship Declaration
## Project Contributor Governance & Intellectual Property Affirmation

> [!WARNING]
> **GOVERNANCE STATUS: DRAFT DOCUMENT FOR PROJECT OWNER REVIEW — NOT FORMALLY EXECUTED**  
> This draft instrument is prepared strictly for legal review and factual verification by the Project Owner. No binding legal transfer, copyright assignment, or irrevocable licensing grant is effected until formally executed.

---

### 1. Project Identification
- **Project Name:** NEXORA — CUDA GPU Acceleration Engine
- **Target Release:** Production Release `v1.0.0`
- **Primary Repository:** `c:\Projects\CUDA-GPU-Acceleration-Engine`
- **Governing Software License:** MIT License ([`LICENSE`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/LICENSE))
- **Collective Copyright Notice:** `Copyright (c) 2026 CUDA GPU Acceleration Engine Contributors`

---

### 2. Sole Contributor & Factual Authorship Declaration
The undersigned **Project Owner**, **Jai Akash** (operating under Git identity `lethalchrist07 <jaiakash2007@gmail.com>`), hereby factually affirms, declares, and represents as follows:

1. **Sole First-Party Human Authorship:**
   I am the sole human author, creator, and software architect of the original computer software, CUDA C++ GPU acceleration kernels, machine learning model architectures, backend service APIs, frontend user interfaces, evaluation harnesses, and technical documentation comprising the **NEXORA** repository.

2. **Absence of External Individual Contributors:**
   No other individual, collaborator, peer reviewer, employee, or independent contractor contributed creative software code, CUDA kernel implementations, training scripts, or documentation to the repository. The collective notice *"CUDA GPU Acceleration Engine Contributors"* was adopted conventionally to designate the project's development identity and does not represent multiple distinct contributing authors.

---

### 3. Detailed Scope of Personal Contributions
Every primary technical component of the platform was personally developed by me, specifically:

1. **CUDA Kernel Engineering & GPU Optimization:**
   - Custom CUDA C++ fused elementwise kernels (`custom_kernels/fused_elementwise.cu`).
   - Custom fused Bias + GELU autograd kernels (`custom_kernels/fused_bias_gelu.cu` and `custom_kernels/autograd.py`).
   - Tensor Core WMMA mixed-precision matrix multiplication kernels (`custom_kernels/matmul_wmma.cu`).
   - Dynamic MSVC and CUDA Toolkit host discovery loaders (`custom_kernels/loader.py`).
2. **Machine Learning Architecture, Training & Experimentation:**
   - Autoregressive decoder Transformer architecture with KV-cache and SDPA integration (`engine/transformer/model.py`).
   - In-house Byte-Pair Encoding (BPE) tokenizer engine (`engine/transformer/tokenizer.py`).
   - Pretraining execution pipelines across 12,208 optimization steps on local hardware (`scripts/train_production_50m.py`).
   - Three-backend CIFAR-10 training architecture and dynamic checkpointing managers (`engine/training/`).
3. **FastAPI Backend Architecture:**
   - REST service layer and routing (`backend/app.py`, `backend/routes/`).
   - Real-time hardware telemetry querying, memory tracking, and process mutual exclusion (`backend/serving.py`).
   - Single-server unified static asset hosting integration on port 8000 (`backend/main.py`).
4. **React Frontend User Interface:**
   - Interactive Single-Page Application (SPA) dashboard (`frontend/src/App.tsx`).
   - AI Training Studio, Hardware Command Center, CUDA Laboratory, and Inference UI components (`frontend/src/components/`).
   - Vite and Tailwind CSS build configurations (`frontend/package.json`, `frontend/vite.config.ts`).
5. **Dataset Preparation, Provenance & Curation:**
   - Automated ingestion and cleaning pipelines for public-domain historical literature and IETF standards (`scripts/prepare_pretraining_dataset.py`).
   - Verification and generation of the 36-document corpus manifest (`data/text/production_10m/dataset_manifest.json`).
   - In-house technical computing reference text (`data/text/computing_foundations.txt`).
6. **Documentation, Packaging & Testing:**
   - Clean-room release packaging automation (`scripts/package_release.py`).
   - Regression test suites covering all 59 platform unit and integration tests (`tests/`).
   - System documentation, technical specifications, and model card dossiers (`README.md`, `MODEL_CARD.md`).

---

### 4. Clarification of Functional Role Designations
In earlier planning and remediation specifications (specifically Section 6.2 of [`NEXORA_LICENSING_REMEDIATION_PLAN.md`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/NEXORA_LICENSING_REMEDIATION_PLAN.md)), the following three roles were described:
- **Primary Architect**
- **CUDA Kernel Engineer**
- **ML Training Engineer**

**Factual Clarification:**  
These designations represented **functional technical responsibilities and operational perspectives**, not separate legal persons. All three functional roles were executed exclusively by the same individual, **Jai Akash**. No separate individuals ever held or exercised these roles.

---

### 5. Git Authorship & Physical Workstation Evidence
1. **Commit History Parity:**
   The entire recorded commit history of the repository (Commits `1a3b04a`, `b255b60`, `5f01a27`, `fb83162`, `a77071c`, `ded3156`, `a24b85b`, `df3e34e`) reflects:
   - Author: `lethalchrist07 <jaiakash2007@gmail.com>` (100% of commits)
   - Committer: `lethalchrist07 <jaiakash2007@gmail.com>` (100% of commits)
   - Co-authors / trailers: Zero (`0`)
2. **Workstation Environment:**
   All local development, compilation, and training were performed on the Project Owner's local machine (`C:\Users\Jai akash\`).

---

### 6. Model Training Provenance & Local Execution
- **Target Foundation Model Checkpoint:** `checkpoints/transformer/pretraining_50m_campaign/best_model.pt` (`1,042,091,280 bytes`, SHA-256: `ABCD0FE7B305CD1CEF7809359CA7128F0AC94BED592A188F8432329CA1406AAC`).
- **Physical Silicon Execution:** The model was trained entirely on the Project Owner's physical NVIDIA GeForce RTX 3050 Laptop GPU (6GB GDDR6, GA107, SM 8.6).
- **Zero Third-Party Model Weights:** Initialized purely from Gaussian pseudo-random float tensors (`mean=0.0, std=0.02`). Zero foundation weights from external models (e.g. LLaMA, GPT-2, Mistral) were loaded or fine-tuned.
- **Zero Distillation:** Training loss was calculated via standard cross-entropy over raw public-domain token sequences. Zero synthetic outputs from proprietary commercial APIs (OpenAI, Anthropic) were ingested.

---

### 7. Explicit Exclusions of Third-Party Material
This declaration affirms sole authorship strictly over first-party created assets. The following external materials are **expressly excluded** and remain governed by their respective upstream legal licenses:

1. **Third-Party Open-Source Runtime Libraries (38 Python & 15 Frontend Packages):**
   - PyTorch (`torch`, BSD-3-Clause / Caffe2 BSD)
   - NumPy (`numpy`, BSD-3-Clause)
   - Requests (`requests`, Apache-2.0, Kenneth Reitz)
   - OpenTelemetry (`opentelemetry`, Apache-2.0, OpenTelemetry Authors)
   - FastAPI & Pydantic (`fastapi`, `pydantic`, MIT)
   - Uvicorn & Starlette (`uvicorn`, `starlette`, BSD-3-Clause, Encode OSS)
   - Certifi (`certifi`, MPL-2.0, Mozilla NSS / Kenneth Reitz)
   - React & React-DOM (MIT, Meta Platforms, Inc.)
   - All third-party dependencies are obtained out-of-band via package managers (PyPI, npm) and are fully attributed in [`THIRD_PARTY_LICENSES.md`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/THIRD_PARTY_LICENSES.md) and [`NOTICE`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/NOTICE).
2. **NVIDIA CUDA Toolkit Components:**
   - NVIDIA CUDA compiler headers, runtime libraries, and driver interfaces are proprietary property of NVIDIA Corporation, used pursuant to the NVIDIA CUDA Toolkit EULA.
3. **Public Domain Literary & Scientific Texts:**
   - Creative texts comprising the 36-document pretraining corpus are historical works in the worldwide public domain or open IETF standards, as documented in [`data/text/production_10m/dataset_manifest.json`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/data/text/production_10m/dataset_manifest.json).

---

### 8. Clarification of Copyright Ownership & Inbound/Outbound Terms
1. **Software Codebase:**
   As sole author, I confirm that all first-party software code in the repository is licensed to downstream users under the terms of the **MIT License** set forth in root [`LICENSE`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/LICENSE).
2. **No Third-Party Assignment Claims:**
   I affirm that no employer, academic institution, or commercial client holds an unencumbered work-for-hire or intellectual property assignment claim over the software or model weights created in this repository.

---

### 9. Proposed Non-Exclusive Model Licensing Authorization
*(Separately presented for Project Owner consideration and formal legal execution)*

With respect to the trained neural network parameter checkpoint `checkpoints/transformer/pretraining_50m_campaign/best_model.pt` (`NEXORA GPT-86.8M`), I hereby declare:

1. **Legal Standing to License:**  
   As the individual who designed the architecture, curated the public-domain training data, implemented the custom kernels, and executed the optimization steps, I hold undivided standing to determine the licensing terms for `best_model.pt`.
2. **Proposed Licensing Grant:**  
   Subject to formal execution of Section 9.2 in [`MODEL_CARD.md`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/MODEL_CARD.md), I authorize the public distribution of `best_model.pt` under the **Creative Commons Attribution 4.0 International (CC-BY-4.0)** license, covering both statutory copyright and *Sui Generis Database Rights* (EU Directive 96/9/EC).
3. **Attribution Requirement:**  
   Downstream recipients exercising CC-BY-4.0 permissions must provide attribution to *"CUDA GPU Acceleration Engine Contributors"*.

---

### 10. Execution Block (Pending Formal Review)

```text
====================================================================================================
                        PROJECT OWNER SOLE CONTRIBUTOR EXECUTION BLOCK
====================================================================================================
I hereby certify and declare under penalty of perjury and applicable laws of intellectual property 
that the foregoing factual representations regarding sole authorship and training provenance 
are true, complete, and accurate.

Project Owner Name:      Jai Akash
Git Author Identity:     lethalchrist07 <jaiakash2007@gmail.com>

Project Owner Signature: __________________________________
Date of Execution:       __________________________________
Location (City/State):   __________________________________
====================================================================================================
```

*Note: This document remains in DRAFT status until signed and dated above.*
