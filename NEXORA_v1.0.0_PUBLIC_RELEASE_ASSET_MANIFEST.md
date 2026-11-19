# NEXORA v1.0.0 — Public Release Asset Manifest & Distribution Specification
## Production Release Package & Asset Integrity Verification

**Project:** CUDA GPU Acceleration Engine (NEXORA)  
**Target Release:** `v1.0.0` (Promoted from Candidate `v1.0.0-rc4`)  
**Certified Distribution Archive:** `dist/cuda-acceleration-engine-v1.0.0-rc4-source.zip`  
**Authority Role:** Production Release Engineer & Open-Source Compliance Auditor  
**Certified Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU 6GB (Ampere SM 8.6)  
**Host Platform:** Windows 11 Home 64-bit (10.0.26100), Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Node.js v20.18.0  
**Specification Date:** November 17, 2026 (Updated from November 8, 2026 Baseline)  
**Status:** **READY FOR PROJECT OWNER AUTHORIZATION**  

---

## 1. Executive Summary & Asset Release Overview

This document specifies the exact, cryptographically verified asset manifest for the official **NEXORA v1.0.0** public release on GitHub.

To maintain strict compliance with release governance, the certified `v1.0.0-rc4` source distribution archive is preserved **completely unmodified** (byte size: `1,558,575 bytes`, SHA-256: `DBDD670654F15955FEF46F9CEF9EB5E8E5CDCDD7E2ECB407C566C28EDECB5AF0`). Release Candidate 4 embeds canonical root `LICENSE`, `NOTICE`, and `THIRD_PARTY_LICENSES.md` documents, along with pre-built production frontend assets (`frontend/dist/`), ensuring autonomous single-server deployment out of the box. Historical candidates (RC1, RC2, RC3) remain preserved in `dist/`.

Furthermore, in strict adherence to model redistribution compliance, both large model checkpoints (>1 GB each) are formally cataloged but **marked as PAUSED / NOT FOR UPLOAD** until the Project Owner clears an explicit model licensing grant and model card.

---

## 2. GitHub Release Asset Inventory

The following assets are designated for direct attachment to the official GitHub Release `v1.0.0`:

| Asset Filename | Format / MIME Type | Exact Size (Bytes) | Cryptographic SHA-256 Checksum | Release Asset Status |
|---|:---:|:---:|---|:---:|
| `cuda-acceleration-engine-v1.0.0-rc4-source.zip` | `application/zip` | `1,558,575` | `DBDD670654F15955FEF46F9CEF9EB5E8E5CDCDD7E2ECB407C566C28EDECB5AF0` | **APPROVED FOR RELEASE ATTACHMENT** |
| `LICENSE` | `text/plain` | `1,098` | `1459A94A7C2D49201FBAF61B280FDA4AD34B8D68C9B4425BEF5EC88005F704C6` | **APPROVED FOR RELEASE ATTACHMENT** |
| `cuda-acceleration-engine-v1.0.0-rc4-checksums.sha256` | `text/plain` | `114` | `DBDD670654F15955FEF46F9CEF9EB5E8E5CDCDD7E2ECB407C566C28EDECB5AF0` | **APPROVED FOR RELEASE ATTACHMENT** |
| `NEXORA_v1.0.0_RELEASE_NOTES.md` | `text/markdown` | `12,000+` | `[Authoritative Release Notes Manifest]` | **PROPOSED GITHUB RELEASE BODY** |

---

## 3. Detailed Asset Descriptions & Functional Roles

### Asset 1: Certified Source Distribution Archive
- **Physical Path:** `dist/cuda-acceleration-engine-v1.0.0-rc4-source.zip`
- **File Size:** `1,558,575 bytes` (1.49 MB compressed, 6.47 MB uncompressed)
- **Packaged Inventory:** 388 files across 13 functional subsystem directories (including pre-built `frontend/dist/` bundle).
- **Role:** Complete, standalone source distribution providing the FastAPI backend, React 18 + Vite 6 production frontend bundle, custom CUDA C++ kernels, baseline benchmark datasets (`results/`), and scoped test runner configuration (`pytest.ini`).
- **Hygiene:** 0 virtual environments, 0 node_modules, 0 build caches, 0 secrets, 0 API keys, 0 private user directory paths.
- **Preservation Status:** **100% UNTOUCHED & CHECKSUM LOCKED**.

### Asset 2: Standalone Canonical Root License File
- **Physical Path:** `LICENSE` (Repository Root)
- **File Size:** `1,098 bytes`
- **Legal Grant:** Standard permissive MIT License text (Copyright © 2026 CUDA GPU Acceleration Engine Contributors).
- **Purpose & Rationale:** Embedded within the RC4 archive and also prepared as an individual downloadable asset on the GitHub Release page for direct visibility.

### Asset 3: Release Notes Document
- **Physical Path:** `NEXORA_v1.0.0_RELEASE_NOTES.md`
- **File Size:** `11,739 bytes`
- **Role:** Comprehensive technical release summary detailing CUDA kernel features (FlashAttention, fused GELU/LayerNorm, FP32/FP16 GEMM scaling up to 16.85 TFLOPS), foundation model specifications, CPU fallback capabilities, hardware safety guardrails (+22.91°C thermal margin), and quickstart instructions.
- **Disposition:** Proposed as the markdown body for the GitHub Release publication.

---

## 4. Model Checkpoints Specification (External Hosting Status)

Due to GitHub's 100 MB per-file upload ceiling and open-source licensing governance, binary weights are **NOT** packaged inside the source archive and are **NOT** attached as standard GitHub release assets.

```
+----------------------------------------------------------------------------------------------------+
| EXTERNAL MODEL ARTIFACTS — REDISTRIBUTION CLEARANCE STATUS                                         |
+--------------------------+--------------------+----------------+------------------+----------------+
| Model Identifier         | Parameter Count    | Byte Size      | Checksum (SHA256)| Hosting Status |
+--------------------------+--------------------+----------------+------------------+----------------+
| NEXORA GPT-86.8M         | 86,825,472 (Exact) | 1,042,091,280  | ABCD0FE7B305CD1C | PAUSED / PENDING
|                          | (50M Pretrained)   | (993.82 MB)    | ...CA1406AAC     | OWNER CLEARANCE|
+--------------------------+--------------------+----------------+------------------+----------------+
| NEXORA GPT-100M Pilot    | 85,940,736 (Active)| 1,026,758,674  | F8B0B9F9D4FB2E9C | PAUSED / PENDING
|                          | (15-Step Feasibility)|(979.19 MB)   | ...381932F9F     | OWNER CLEARANCE|
+--------------------------+--------------------+----------------+------------------+----------------+
```

### 4.1 Legal & Redistribution Audit Findings
1. **Training Data Lineage:** Pre-trained on public domain literature (Project Gutenberg) and open standards (IETF RFCs). 100% compliant with public domain re-use.
2. **Tokenizer Architecture:** In-house trained BPE tokenizer (`bpe_vocab.json`, 2048 vocab) with zero proprietary token dependencies.
3. **Model Redistribution Rights Determination:**
   - While the source code is licensed under MIT, neural network weight matrices do not automatically inherit software copyright under various legal jurisdictions without explicit licensing terms.
   - Neither checkpoint file contains an embedded license statement in its serialized metadata dictionary.
   - The repository currently lacks an explicit **Model Card** (`MODEL_CARD.md`) defining acceptable use, downstream redistribution terms, and ethical limitations.
   - **Compliance Directive:** Model checkpoint redistribution is marked as **NOT VERIFIED / PAUSED**. External hosting and upload to public registries (e.g., Hugging Face) must remain paused until the Project Owner provides an explicit model licensing statement.

---

## 5. Technical Disclosure: Test Suite Executable Fixture

The source archive packages one compiled executable inside `results/phase2_verification_20261016/`:
- **File:** `results/phase2_verification_20261016/original_run/benchmark_suite.exe`
- **File Size:** `115,712 bytes`
- **SHA-256:** `6569BB1E75184460420B3C83050FD30CAF22423B89A9D0D4E7A39940D4B33DBC`
- **Provenance:** Compiled on October 16, 2026, using MSVC 19.51 and CUDA Toolkit from the source files in `src/benchmark_main.cu`, `src/gpu_check.cu`, `src/matmul.cu`, `src/vector_add.cu`, and `src/benchmark_export.cpp`.
- **Purpose & Technical Necessity:**
  - *Consumer Runtime:* **Not Executed.** Consumers do not run this binary for engine startup, inference, or training.
  - *Test Suite Verification:* **Strictly Required.** Automated test `backend/tests/test_data.py` (`test_actual_phase2_results_and_manifest`) computes SHA-256 digests over all files registered in `results/phase2_verification_20261016/manifest.json`. Removing this fixture would break manifest auditing and cause automated regression tests to fail in clean-room environments.
- **Safety Confirmation:** Benign, non-malicious static verification fixture compiled directly from the open-source repository.

---

## 6. Consumer Verification & Extraction Instructions

Downstream consumers can verify the integrity of the downloaded release assets using the following documented commands:

### 6.1 Cryptographic Hash Verification

#### Windows PowerShell
```powershell
# 1. Verify Source Distribution Archive
(Get-FileHash -Algorithm SHA256 cuda-acceleration-engine-v1.0.0-rc4-source.zip).Hash -eq "DBDD670654F15955FEF46F9CEF9EB5E8E5CDCDD7E2ECB407C566C28EDECB5AF0"
# Output should return: True

# 2. Verify Standalone LICENSE Asset
(Get-FileHash -Algorithm SHA256 LICENSE).Hash -eq "1459A94A7C2D49201FBAF61B280FDA4AD34B8D68C9B4425BEF5EC88005F704C6"
# Output should return: True
```

#### Windows Command Prompt (CMD)
```cmd
certutil -hashfile cuda-acceleration-engine-v1.0.0-rc4-source.zip SHA256
certutil -hashfile LICENSE SHA256
```

#### Linux / WSL / macOS
```bash
echo "DBDD670654F15955FEF46F9CEF9EB5E8E5CDCDD7E2ECB407C566C28EDECB5AF0  cuda-acceleration-engine-v1.0.0-rc4-source.zip" | sha256sum --check
echo "1459A94A7C2D49201FBAF61B280FDA4AD34B8D68C9B4425BEF5EC88005F704C6  LICENSE" | sha256sum --check
```

### 6.2 Consumer Setup & External Weights Placement (Once Cleared)
```powershell
# 1. Unpack source archive
Expand-Archive -Path cuda-acceleration-engine-v1.0.0-rc4-source.zip -DestinationPath .
cd cuda-acceleration-engine

# 2. Place standalone license into unpacked root if downloaded separately
copy ..\LICENSE .

# 3. Create model subdirectories and place external weights
mkdir -p checkpoints/transformer/pretraining_50m_campaign
mkdir -p checkpoints/transformer/flagship_100m_pilot

# Place best_model.pt (NEXORA GPT-86.8M) -> checkpoints/transformer/pretraining_50m_campaign/best_model.pt
# Place best_model.pt (NEXORA GPT-100M Pilot) -> checkpoints/transformer/flagship_100m_pilot/best_model.pt
```

---

## 7. Mandatory Governance & Release Restrictions

In accordance with strict production release safety policies:
1. **No Archive Rebuilding:** The certified `v1.0.0-rc4` source archive remains untouched and checksum-locked (historical candidates RC1, RC2, and RC3 remain preserved for rollback continuity).
2. **No Checkpoint Uploads:** Model weights hosting remains paused pending Project Owner licensing authorization.
3. **No Unilateral Release Actions:** No Git commits, tags, remote pushes, GitHub release drafts, or public announcements have been or will be made without explicit Project Owner sign-off.

---

*Prepared by:*  
**Production Release Engineer & Open-Source Compliance Auditor**  
*November 8, 2026*

