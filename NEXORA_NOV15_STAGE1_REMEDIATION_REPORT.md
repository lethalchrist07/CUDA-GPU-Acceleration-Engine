# NEXORA — November 15 Remediation Stage 1 Report: BLK-01 & BLK-04 Resolution

**Document Version:** 1.0.0  
**Date:** November 15, 2026  
**Auditor / Role:** Senior Release Engineering Developer  
**Status:** **STAGE 1 REMEDIATION COMPLETE — BLK-01 & BLK-04 RESOLVED (PASS)**  
**Target Release:** v1.0.0-rc3  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, sm_86, Driver 591.44, CUDA 12.9)  

---

## Executive Summary

Pursuant to the **NEXORA November 15 Release Remediation Mandate**, this Stage 1 intervention strictly resolved the two critical licensing and packaging compliance blockers:
1. **BLK-01 (Root LICENSE Omission in Release Archive):** The packaging automation script (`scripts/package_release.py`) was remediated to include the root `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, and `MODEL_CARD.md` files in top-level archive builds, and the clean **v1.0.0-rc3** source archive was assembled and verified.
2. **BLK-04 (Missing Physical Legal Notices):** Both physical notice files—`NOTICE` and `THIRD_PARTY_LICENSES.md`—were authored and committed to the repository root using verified supply-chain data, establishing unambiguous boundaries between first-party source code and out-of-band third-party dependencies.

All strict development constraints were rigorously respected:
- **Zero modification of model checkpoints or weights:** Checkpoints in `checkpoints/` and `runs/` were untouched.
- **Zero training execution:** No pretraining or fine-tuning campaigns were launched.
- **Zero modification of CUDA kernels:** All C++/CUDA kernels in `custom_kernels/` remain untouched.
- **Zero unauthorized remote publishing:** No commits were pushed to GitHub, and no releases were published.
- **Zero signing of unverified declarations:** The Model Redistribution Declaration in `MODEL_CARD.md` Section 9 remains unsigned, and unverified areas remain clearly designated as `UNVERIFIED / PENDING`.

---

## 1. Blocker Resolution Scorecard

| Blocker ID | Severity | Description | Resolution Actions | Empirical Verification | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **BLK-01** | CRITICAL | Root `LICENSE` omitted from release archive (`rc2-source.zip`) | Patched `scripts/package_release.py` `INCLUDE_ROOT_FILES` to bundle `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, and `MODEL_CARD.md`. Defaulted version to `v1.0.0-rc3`. Rebuilt source archive. | Bit-for-bit SHA-256 match between disk and archive. Clean-room extraction verified. | **RESOLVED (PASS)** |
| **BLK-04** | HIGH | Physical legal notice files missing from repository root | Created physical root `NOTICE` (3,359 bytes) and root `THIRD_PARTY_LICENSES.md` (20,294 bytes) with comprehensive 38 Python + 15 Frontend dependency inventories and verbatim license texts. | Presence and content verified by automated test suite (`tests/test_stage1_packaging_compliance.py`). | **RESOLVED (PASS)** |

---

## 2. BLK-01 Forensic Remediation: Release Packaging Automation

### 2.1 Defect Analysis
In `v1.0.0-rc2`, the release archive `cuda-acceleration-engine-v1.0.0-rc2-source.zip` contained all source directories, but omitted `LICENSE`. Root files are packaged solely via explicit array enumeration in `INCLUDE_ROOT_FILES`. `LICENSE` had been omitted from this array, resulting in silent omission and technical breach of MIT License Section 1 condition (*"The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software"*).

### 2.2 Implementation in `scripts/package_release.py`
The packaging script was patched with minimal surgical changes:
1. Updated default target version to `v1.0.0-rc3`.
2. Appended all required legal instruments to `INCLUDE_ROOT_FILES`.
3. Preserved all existing packaging exclusions (`EXCLUDE_PATTERNS`, `should_exclude` directory and extension filters).

#### Exact Diff Applied to [`scripts/package_release.py`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/scripts/package_release.py):
```diff
--- a/scripts/package_release.py
+++ b/scripts/package_release.py
@@ -14,7 +14,7 @@ from pathlib import Path
 
 PROJECT_ROOT = Path(__file__).resolve().parent.parent
 OUTPUT_DIR = PROJECT_ROOT / "dist"
-VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1.0.0-rc2"
+VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1.0.0-rc3"
 ARCHIVE_NAME = f"cuda-acceleration-engine-{VERSION}-source.zip"
 ARCHIVE_PATH = OUTPUT_DIR / ARCHIVE_NAME
 
@@ -30,6 +30,10 @@ INCLUDE_DIRS = [
 ]
 
 INCLUDE_ROOT_FILES = [
+    "LICENSE",
+    "NOTICE",
+    "THIRD_PARTY_LICENSES.md",
+    "MODEL_CARD.md",
     "README.md",
     "requirements.txt",
     "environment.yml",
```

### 2.3 Generated Release Candidate 3 Artifacts
Executing `python scripts/package_release.py v1.0.0-rc3` generated:

- **Archive File:** [`dist/cuda-acceleration-engine-v1.0.0-rc3-source.zip`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/dist/cuda-acceleration-engine-v1.0.0-rc3-source.zip)
  - **Size:** 860,360 bytes (840.2 KB)
  - **Total Files Packaged:** 280 files
  - **SHA-256 Checksum:** `E36683940F0EAF53A183CB6CB6D856AFAD8EEBE2067BC75E71C4FE47985C9E91`
- **Checksum Manifest:** [`dist/cuda-acceleration-engine-v1.0.0-rc3-checksums.sha256`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/dist/cuda-acceleration-engine-v1.0.0-rc3-checksums.sha256)
  - **Size:** 114 bytes
  - **Content:** `E36683940F0EAF53A183CB6CB6D856AFAD8EEBE2067BC75E71C4FE47985C9E91  cuda-acceleration-engine-v1.0.0-rc3-source.zip`

---

## 3. BLK-04 Forensic Remediation: Physical Legal Notice Files

Prior to Stage 1, third-party disclosure details were spread across various audit and preparation reports without physical root presence. In Stage 1, physical notice files were instantiated directly in the repository root.

### 3.1 Root [`NOTICE`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/NOTICE) (3,359 bytes)
- **Primary Product Copyright:**
  ```text
  NEXORA — CUDA GPU Acceleration Engine
  Copyright (c) 2026 CUDA GPU Acceleration Engine Contributors
  This product includes software developed by the NEXORA Project.
  ```
- **NVIDIA CUDA Toolkit Attribution & Trademark Disclaimer:**
  Acknowledges NVIDIA CUDA Toolkit components while strictly adhering to NVIDIA non-endorsement policies (CUDA, GeForce RTX are trademarks of NVIDIA Corporation).
- **Out-of-Band Dependency Clarification:**
  Explicitly declares that third-party runtime libraries are obtained out-of-band by end-users via package managers (`pip`, `npm`) and are **not** bundled in source form inside the release archive.
- **Upstream License Conditions:**
  - **Requests** (Apache-2.0 Section 4(d) notice attribution to Kenneth Reitz)
  - **OpenTelemetry Python** (Apache-2.0 Section 4(d) attribution)
  - **PyTorch / Caffe2** (BSD-3 attribution notices, UC Regents, Kakao Brain, Yangqing Jia)
  - **TypeScript** (Apache-2.0 Section 4(d) attribution to Microsoft Corporation and DefinitelyTyped)

### 3.2 Root [`THIRD_PARTY_LICENSES.md`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/THIRD_PARTY_LICENSES.md) (20,294 bytes)
Contains complete legal documentation structured into five distinct sections:
1. **NVIDIA CUDA Toolkit Acknowledgement:** Details runtime compiler and header usage under NVIDIA EULA.
2. **Master Dependency Inventory:**
   - **Backend / Python (38 Packages):** Complete version, license type, package URL, and license expression inventory (FastAPI, Uvicorn, Starlette, Pydantic, PyTorch, NumPy, Psutil, Pynvml, Requests, etc.).
   - **Frontend / Node.js (15 Packages):** Complete inventory of React, Lucide React, Tailwind CSS, TypeScript, Vite, Recharts, etc.
3. **Deep-Dive Attribution Notices:**
   - Explicit subcomponent copyright breakdowns (NumPy's 14 BSD subcomponents including Dragon4, Cephes, Lapack-lite, PocketFFT; PyTorch Caffe2 contributors; Packaging elected BSD-2-Clause).
4. **Verbatim Upstream License Texts:**
   - MIT License
   - BSD 3-Clause ("Modified BSD") License
   - BSD 2-Clause ("Simplified") License
   - Apache License, Version 2.0
   - Mozilla Public License, Version 2.0 (MPL-2.0)
   - Python Software Foundation License Version 2 (PSF-2.0)
   - ISC License
5. **Formal Designation of Unverified Rights & Disclosures:**
   - **Model Checkpoint Weights Redistribution:** Formally marked as **UNVERIFIED / HOSTING PAUSED**. Model parameter tensors (`*.pt`) do not inherit software MIT terms. Redistribution on external hubs remains paused pending Project Owner execution of `MODEL_CARD.md` Section 9.
   - **Contributor Title Affirmation Execution:** Formally marked as **UNVERIFIED / PENDING FORMAL EXECUTION**. Original authorship is affirmed, while formal signed DCO instruments remain pending.

---

## 4. Empirical Verification & Test Results

A dedicated automated test suite was constructed and executed: [`tests/test_stage1_packaging_compliance.py`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/tests/test_stage1_packaging_compliance.py).

### 4.1 Test Battery Execution Log
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- .venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Projects\CUDA-GPU-Acceleration-Engine
configfile: pytest.ini
plugins: anyio-4.15.1
collecting ... collected 7 items

tests/test_stage1_packaging_compliance.py::TestStage1PackagingCompliance::test_blk04_root_legal_files_exist PASSED [ 14%]
tests/test_stage1_packaging_compliance.py::TestStage1PackagingCompliance::test_blk04_notice_content_integrity PASSED [ 28%]
tests/test_stage1_packaging_compliance.py::TestStage1PackagingCompliance::test_blk04_third_party_licenses_content_integrity PASSED [ 42%]
tests/test_stage1_packaging_compliance.py::TestStage1PackagingCompliance::test_blk01_packaging_script_includes_root_files PASSED [ 57%]
tests/test_stage1_packaging_compliance.py::TestStage1PackagingCompliance::test_blk01_rc3_archive_presence_and_checksum PASSED [ 71%]
tests/test_stage1_packaging_compliance.py::TestStage1PackagingCompliance::test_blk01_rc3_archive_contains_legal_files_bit_for_bit PASSED [ 85%]
tests/test_stage1_packaging_compliance.py::TestStage1PackagingCompliance::test_rc3_archive_zero_unintended_cuda_redistributables PASSED [100%]

============================== 7 passed in 0.07s ==============================
```

### 4.2 Bit-for-Bit Hash Equality Register
The SHA-256 hashes of physical files on disk match the archived files bit-for-bit:

| File Target | Physical File Size | Archive Relative Path | SHA-256 Hash | Verification Result |
| :--- | :---: | :--- | :--- | :---: |
| `LICENSE` | 1,098 bytes | `cuda-acceleration-engine/LICENSE` | `1459A94A7C2D49201FBAF61B280FDA4AD34B8D68C9B4425BEF5EC88005F704C6` | **IDENTICAL (PASS)** |
| `NOTICE` | 3,359 bytes | `cuda-acceleration-engine/NOTICE` | `F86020F64DF69374F4DEE4118A320F48F2FF63D49EE77F93C72C3CED233D26CD` | **IDENTICAL (PASS)** |
| `THIRD_PARTY_LICENSES.md` | 20,294 bytes | `cuda-acceleration-engine/THIRD_PARTY_LICENSES.md` | `C614731A179E5A369AFFA8D4276FCE48610F3AB88118F5087FEEC151F79EB673` | **IDENTICAL (PASS)** |
| `MODEL_CARD.md` | 21,227 bytes | `cuda-acceleration-engine/MODEL_CARD.md` | `F512C044288C88BC266A265977796CD8E7C7FE0C88153B133B5D89D823C8E2B5` | **IDENTICAL (PASS)** |

### 4.3 Clean-Room Extraction Verification
Extraction of `cuda-acceleration-engine-v1.0.0-rc3-source.zip` into an isolated sterile temporary directory (`%TEMP%/nexora_cleanroom_*`) verified:
- Correct top-level prefix folder: `cuda-acceleration-engine/`
- All required project build and legal files extracted cleanly (`LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, `CMakeLists.txt`, `requirements.txt`).
- Zero prohibited binary runtimes (`.dll`, `.so`, `.lib`, `.pyd`, `.dylib`) or virtual environment directories (`.venv`, `node_modules`, `__pycache__`) in the archive.
- Result: **PASS: Clean-room extraction verified successfully.**

---

## 5. Remaining Risks and Outstanding Blockers

With BLK-01 and BLK-04 resolved, the release pipeline transitions to Stage 2. The remaining pre-publication blockers identified in the November 15 distribution-scope audit are:

| Blocker ID | Description | Root Cause / Scope | Planned Stage |
| :--- | :--- | :--- | :---: |
| **BLK-02** | Hardcoded filesystem paths | Absolute paths (`C:/Projects/...`) hardcoded in `engine/hub/loader.py`, `backend/data.py`, and test files impede execution on arbitrary end-user systems. | Stage 2 |
| **BLK-03** | Frontend build asset serving | Backend `app.py` serves API endpoints but does not mount compiled frontend static assets (`frontend/dist/`), requiring two dev servers. | Stage 2 |
| **BLK-05** | Standalone seed checkpoint access | Source archive excludes 1.7GB checkpoints by design; an automated downloader or synthetic seed generation mechanism is required for zero-configuration startup. | Stage 2 |
| **GOV-01** | Model Weights Redistribution | Public weight hosting remains paused until Project Owner reviews and executes `MODEL_CARD.md` Section 9. | Stage 3 |
| **GOV-02** | Contributor Title Affirmation | Project Owner execution of DCO + License Ratification document. | Stage 3 |

---

## 6. Conclusion & Handoff

Stage 1 remediation has successfully met all objective criteria:
- Root `LICENSE` is bundled into the source distribution archive.
- Root `NOTICE` and `THIRD_PARTY_LICENSES.md` provide full legal and supply-chain transparency.
- Archive `cuda-acceleration-engine-v1.0.0-rc3-source.zip` is reproducible, clean, and verified by automated tests.
- Repository is clean, compliant, and ready for Stage 2 remediation (resolving BLK-02 and BLK-03).
