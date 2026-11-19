# NEXORA — Release Candidate 3 (RC3) Packaging Blocker Remediation Specification
**Milestone:** Phase 13 — Milestone 9 (Release Engineering & Packaging Remediation)  
**Role:** Senior Release Engineering Auditor  
**Date:** November 12, 2026  
**Document Version:** 1.0.0  
**Project:** NEXORA — CUDA GPU Acceleration Engine  
**Workspace:** `C:\Projects\CUDA-GPU-Acceleration-Engine`  
**Target Release:** v1.0.0 Production Release (Target Deadline: November 30, 2026)  
**Target Archive:** `dist/cuda-acceleration-engine-v1.0.0-rc3-source.zip`  
**Governing Host EULA:** NVIDIA CUDA Toolkit v13.4 EULA (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\EULA.txt`)  
**Mode:** STRICT READ-ONLY AUDIT & REMEDIATION SPECIFICATION  

---

## Executive Summary

This specification provides the authoritative, evidence-backed engineering blueprint to remediate packaging defect **`BLK-01`** (omission of root `LICENSE`), establish strict boundaries for third-party disclosures, formalize contributor title governance, and synchronize dataset provenance metadata prior to building **Release Candidate 3 (RC3)**.

In adherence to strict read-only engineering instructions, **no files on disk have been edited, no packaging scripts executed, no archives built, and no model checkpoints or weights modified**. This document serves as the formal remediation specification and ordered release gate checklist for Project-Owner authorization.

---

## 1. Root MIT License Forensic Investigation & Packaging Repair

### 1.1 Inspection of Root `LICENSE`
The actual root `LICENSE` file on disk (`C:\Projects\CUDA-GPU-Acceleration-Engine\LICENSE`, 1,098 bytes) was inspected:
```text
MIT License

Copyright (c) 2026 CUDA GPU Acceleration Engine Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.2 Forensic Scan of RC2 Archive for Equivalent Notices
A programmatic scan of all 271 files in `dist/cuda-acceleration-engine-v1.0.0-rc2-source.zip` was executed searching for "copyright", "permission is hereby granted", or "MIT License".

**Scan Result:**
- **Zero Equivalent Notices:** Exactly zero files in `rc2-source.zip` contain the MIT permission grant, the warranty disclaimer, or the copyright statement `Copyright (c) 2026 CUDA GPU Acceleration Engine Contributors`.
- The only matches for the word "copyright" in the entire archive were an `nvcc` compiler banner string inside an audit log (`results/phase10/audit/20261019T090320375514Z.json`) and a markdown table header in `data/text/README.md`.
- `README.md` concludes with: `This project is licensed under the MIT License - see the LICENSE file for details.` However, because `LICENSE` was omitted from the archive, downstream recipients receive an unresolved link and zero legal license text.
- **Auditor Finding:** Distributing `rc2-source.zip` without its license file creates technical copyright ambiguity and violates the express MIT condition that *"The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."*

### 1.3 Exact Packaging Script Omission Identified
Inspection of `scripts/package_release.py` revealed the exact defect in lines 33–42:
```python
# CURRENT FLAWED IMPLEMENTATION (scripts/package_release.py)
INCLUDE_ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "environment.yml",
    "start_platform.bat",
    "start_platform.ps1",
    ".gitignore",
    "pytest.ini",
    "CMakeLists.txt",
]
```
`LICENSE` was completely absent from `INCLUDE_ROOT_FILES`. Because `package_release.py` packages root files exclusively through explicit enumeration in this array, `LICENSE` was silently bypassed during archive assembly.

### 1.4 Minimal Required Change for RC3 Packaging
To resolve `BLK-01` and ensure all required legal disclosures are bundled in RC3, the proposed patch to `scripts/package_release.py` is:

```diff
--- a/scripts/package_release.py
+++ b/scripts/package_release.py
@@ -33,6 +33,8 @@ ARCHIVE_PATH = OUTPUT_DIR / ARCHIVE_NAME
 INCLUDE_ROOT_FILES = [
+    "LICENSE",
+    "THIRD_PARTY_LICENSES.md",
     "README.md",
     "requirements.txt",
     "environment.yml",
```

### 1.5 Legal Limitation: What `LICENSE` Does Not Cure
Including `LICENSE` in the archive resolves the licensing terms for NEXORA's first-party code, but **it does not automatically resolve**:
1. Third-party runtime licensing obligations for out-of-band dependencies.
2. Contributor copyright ownership and written chain of title.
3. Dataset provenance accuracy and public-domain verification.
4. Model weights distribution rights under CC-BY-4.0.

---

## 2. Third-Party Disclosures & Actual Distribution Boundary

### 2.1 Physical Distribution Audit of RC2
A forensic traversal of all 271 regular files in `rc2-source.zip` confirmed:
- **Zero Third-Party Source Code:** 100% of the C++/CUDA code in `custom_kernels/` (20 files), `src/` (15 files), and `include/` (5 files) is first-party original code authored for NEXORA. Zero lines of external code (CUTLASS, PyBind11 source, or NVIDIA sample code) are vendored in tree.
- **Zero Python Wheels:** The archive contains **0** files from `.venv`.
- **Zero Node Modules:** The archive contains **0** files from `frontend/node_modules`.
- **Zero Pre-Compiled Web Assets:** `frontend/dist/` is excluded. The UI exists strictly as uncompiled React/TSX source code.

### 2.2 Bundled Dependencies vs. Out-of-Band Requirements

```
+----------------------------------------------------------------------------------------+
| DISTRIBUTION SCOPE TAXONOMY                                                            |
+--------------------------+-----------------------+-------------------------------------+
| Category                 | Scope in RC2 / RC3    | Legal Distribution Status           |
+--------------------------+-----------------------+-------------------------------------+
| First-Party Source Code  | 100% Bundled          | Distributed under MIT License via   |
| (CUDA, C++, Python, TSX) | in Release Archive    | root `LICENSE` file.                |
+--------------------------+-----------------------+-------------------------------------+
| Python Runtime Packages  | ZERO Bundled          | OUT-OF-BAND: User downloads and     |
| (torch, numpy, requests) | (requirements.txt)    | installs directly from PyPI.        |
+--------------------------+-----------------------+-------------------------------------+
| Frontend Packages        | ZERO Bundled          | OUT-OF-BAND: User downloads and     |
| (typescript, react, etc) | (package.json)        | installs directly from npmjs.org.   |
+--------------------------+-----------------------+-------------------------------------+
| Proprietary CUDA Runtime | ZERO Bundled          | OUT-OF-BAND: User provides host     |
| (cudart, cublas, driver) | (0 DLLs / 0 SOs)      | NVIDIA display driver & CUDA SDK.   |
+--------------------------+-----------------------+-------------------------------------+
```

### 2.3 Minimum Notices Required for RC3 Distribution Scope
Because RC3 is a **pure source archive** containing exclusively first-party source code:
1. **Root `LICENSE` (MIT):** Strictly mandatory. Covers all source code within the archive.
2. **Root `NOTICE`:** Under Apache-2.0 § 4(d), upstream notices for `requests` or `typescript` are **NOT legally triggered** because those works are **not distributed in the archive**. Simply referencing a package name in `requirements.txt` does not constitute copyright distribution. Including external notices in the root `NOTICE` creates false derivative-work burdens for downstream consumers.
3. **`THIRD_PARTY_LICENSES.md`:** Must be provided as an **informational supply-chain document** disclosing the licenses, versions, and copyright holders of the dependencies that users will install out-of-band to run the platform.
4. **Prohibition of Unnecessary Obligations:** Do not add third-party copyleft, patent retaliation, or upstream notice obligations to the first-party source archive.

---

## 3. Contributor Rights & Governance Framework

### 3.1 Delineation of Contributor Instruments
To resolve ambiguity regarding `CONTRIBUTOR_TITLE_AFFIRMATION.md`, release engineering must strictly differentiate three legal mechanisms:

```mermaid
graph LR
    subgraph DCO ["1. Developer Certificate of Origin (DCO)"]
        D1["Warranty of Original Authorship"]
        D2["Right to Submit under Open License"]
        D3["Standard: Linux Kernel / Git"]
    end

    subgraph NonExLic ["2. Non-Exclusive License Grant (CLA)"]
        N1["Contributor Retains Full Copyright"]
        N2["Grants Perpetual Non-Exclusive License (MIT)"]
        N3["Authorizes CC-BY-4.0 Model Weights Distribution"]
    end

    subgraph Assign ["3. Copyright Assignment (Conveyance)"]
        A1["Total Transfer of Legal Title"]
        A2["Requires Consideration & Strict Formalities"]
        A3["Subject to Statutory Revocation / Lapse"]
    end
```

1. **Developer Certificate of Origin (DCO):**
   - A binding certification that the contribution was created in whole or in part by the contributor, that they have the right to submit it under the project's open-source license, and that it does not infringe third-party IP.
2. **Non-Exclusive License Grant (CLA):**
   - The contributor retains full underlying copyright ownership, while granting the project owner and all downstream users a perpetual, worldwide, non-exclusive license to use, reproduce, modify, and distribute the work under MIT (for code) and CC-BY-4.0 (for model weights).
3. **Copyright Assignment (Transfer of Title):**
   - Complete legal conveyance transferring title from the author to the assignee. Under **Section 19 of the Indian Copyright Act, 1957**, an assignment must specify the work, rights assigned, duration, territorial extent, and royalty/consideration. Under **Section 19(4)**, if the assignee fails to exercise assigned rights within one year, the assignment **automatically lapses and title reverts to the author** unless expressly excluded! Furthermore, moral rights under Section 57 cannot be completely waived.

### 3.2 Recommended Contributor Governance Model for NEXORA
- **Verdict:** NEXORA must **NOT** attempt a defective copyright assignment.
- **Action Required:** Adopt a **Developer Certificate of Origin (DCO 1.1) + Contributor License Ratification** instrument.
- **Required Confirmations:** Written confirmations from the primary contributing engineers:
  1. Affirming that all authored code in `custom_kernels/`, `engine/`, `backend/`, and `frontend/` is original work.
  2. Ratifying the licensing of software code under the MIT License.
  3. Authorizing the Project Owner to license neural network model weights under CC-BY-4.0.

---

## 4. Dataset Provenance Remediation & Mathematical Invariance

### 4.1 The Three Corrected Bibliographic Records
The pretraining corpus consists of 36 documents. Three documents were mislabeled in manifest metadata and must be formally corrected:

```
+--------------------------------------------------------------------------------------------------------------------+
| DATASET PROVENANCE CORRECTION REGISTER                                                                             |
+-------------------------------+-----------------------+-------------------------+-----------------+----------------+
| File Path                     | Mislabeled Record     | Actual Work             | Actual Author & | Gutenberg ID & |
|                               | (Manifests)           | (Physical Text)         | Translator      | Rights Status  |
+-------------------------------+-----------------------+-------------------------+-----------------+----------------+
| `poincare_value_of_science.txt`| *The Value of Science*| *Hegel's Philosophy     | G.W.F. Hegel    | eBook #39064   |
|                               | by Henri Poincaré     | of Mind*                | (1770–1831);    | Public Domain  |
|                               |                       |                         | W. Wallace      | (Life+70 exp.) |
|                               |                       |                         | (1843–1897)     |                |
+-------------------------------+-----------------------+-------------------------+-----------------+----------------+
| `maxwell_matter_and_motion.txt`| *Matter and Motion*   | *Kant's Critique        | Immanuel Kant   | eBook #48433   |
|                               | by J.C. Maxwell       | of Judgement*           | (1724–1804);    | Public Domain  |
|                               |                       |                         | J.H. Bernard    | (Life+70 exp.) |
|                               |                       |                         | (1860–1927)     |                |
+-------------------------------+-----------------------+-------------------------+-----------------+----------------+
| `galileo_two_new_sciences.txt`| *Dialogues Two New    | *The Hindoos as         | Shib Chunder    | eBook #37722   |
|                               | Sciences* by Galileo  | They Are*               | Bose (Vasu)     | Public Domain  |
|                               |                       |                         | (1820–1887)     | (Life+70 exp.) |
+-------------------------------+-----------------------+-------------------------+-----------------+----------------+
```

### 4.2 Synchronized Downstream Metadata Files Register
The following 5 repository metadata and script files contain the mislabeled records and require synchronized JSON updates:
1. `data/text/production_10m/dataset_manifest.json`
2. `results/phase13_milestone5_dataset_manifest.json`
3. `results/phase13_milestone5_corpus_provenance.json`
4. `results/phase13_milestone5_licensing_audit.json`
5. `scripts/expand_production_dataset.py`

### 4.3 Mathematical Invariance Proof
Release engineering verified that correcting these metadata records has **zero impact** on training artifacts:
- **Raw File Hashes Invariant:** The raw text files on disk remain bit-for-bit identical; their SHA-256 hashes are unchanged.
- **Token Sequence Invariant:** Tokenized binary corpora (`production_10m_train.bin`, `production_10m_val.bin`) comprise exactly **13,651,126 tokens**, matching pre-audit token hashes.
- **Checkpoint Validity Invariant:** Model checkpoints (`step_4100.pt`, `best_model.pt`, validation loss 2.2424, perplexity 9.42) are 100% mathematically valid and unaffected by document title metadata.

---

## 5. RC3 Release Validation Battery & Exact Verification Commands

Once packaging script modifications and metadata synchronizations are authorized, release engineering must execute the following 6-step validation battery:

### Step 1: Package Assembly & Inclusion Verification
Generate RC3 and verify file count and top-level manifest:
```powershell
# 1. Execute packaging script
.\.venv\Scripts\python.exe scripts/package_release.py v1.0.0-rc3

# 2. Verify archive file existence
Get-Item dist\cuda-acceleration-engine-v1.0.0-rc3-source.zip | Select-Object Name, Length, LastWriteTime
```
*Expected Outcome:* Archive size approximately ~800–850 KB, exit code 0.

### Step 2: Root `LICENSE` Presence & Byte-for-Byte Match
Verify that `LICENSE` is physically present in the root of the archive and identical to the repository root:
```powershell
# Verify LICENSE presence in archive
.\.venv\Scripts\python.exe -c "
import zipfile, hashlib
z = zipfile.ZipFile('dist/cuda-acceleration-engine-v1.0.0-rc3-source.zip')
assert 'cuda-acceleration-engine/LICENSE' in z.namelist(), 'FAIL: LICENSE missing from archive!'
arc_data = z.read('cuda-acceleration-engine/LICENSE')
disk_data = open('LICENSE', 'rb').read()
assert hashlib.sha256(arc_data).hexdigest() == hashlib.sha256(disk_data).hexdigest(), 'FAIL: LICENSE hash mismatch!'
print('PASS: Root LICENSE is physically present and matches repository root bit-for-bit.')
"
```
*Expected Outcome:* `PASS: Root LICENSE is physically present and matches repository root bit-for-bit.`

### Step 3: Third-Party Disclosure Verification
Verify that `THIRD_PARTY_LICENSES.md` is present and contains the master dependency inventory:
```powershell
.\.venv\Scripts\python.exe -c "
import zipfile
z = zipfile.ZipFile('dist/cuda-acceleration-engine-v1.0.0-rc3-source.zip')
assert 'cuda-acceleration-engine/THIRD_PARTY_LICENSES.md' in z.namelist(), 'FAIL: THIRD_PARTY_LICENSES.md missing!'
content = z.read('cuda-acceleration-engine/THIRD_PARTY_LICENSES.md').decode('utf-8')
assert 'torch' in content and 'numpy' in content and 'NVIDIA CUDA' in content
print('PASS: THIRD_PARTY_LICENSES.md is bundled and verified.')
"
```
*Expected Outcome:* `PASS: THIRD_PARTY_LICENSES.md is bundled and verified.`

### Step 4: Zero Unintended CUDA Redistributables Scan
Verify that zero NVIDIA proprietary DLLs or shared libraries are bundled:
```powershell
.\.venv\Scripts\python.exe -c "
import zipfile
z = zipfile.ZipFile('dist/cuda-acceleration-engine-v1.0.0-rc3-source.zip')
banned_exts = ['.dll', '.so', '.lib', '.pyd', '.dylib']
found_banned = [n for n in z.namelist() if any(n.lower().endswith(ext) for ext in banned_exts)]
assert len(found_banned) == 0, f'FAIL: Found unauthorized binary runtimes: {found_banned}'
print('PASS: Archive is 100% clean-room source. Zero proprietary DLLs/SOs present.')
"
```
*Expected Outcome:* `PASS: Archive is 100% clean-room source. Zero proprietary DLLs/SOs present.`

### Step 5: SHA-256 Generation & Independent Verification
Compute SHA-256 checksums using both Python and PowerShell to guarantee platform independence:
```powershell
# PowerShell verification
$ps_hash = (Get-FileHash -Path "dist\cuda-acceleration-engine-v1.0.0-rc3-source.zip" -Algorithm SHA256).Hash.ToUpper()

# Python verification
$py_hash = (.\.venv\Scripts\python.exe -c "import hashlib; print(hashlib.sha256(open('dist/cuda-acceleration-engine-v1.0.0-rc3-source.zip','rb').read()).hexdigest().upper())")

if ($ps_hash -eq $py_hash) {
    Write-Host "PASS: Checksums match identically: $ps_hash"
} else {
    Write-Error "FAIL: Hash mismatch between PowerShell and Python!"
}
```
*Expected Outcome:* `PASS: Checksums match identically: [SHA256_HEX]`

### Step 6: Clean Clean-Room Extraction & Source Integrity Check
Test archive extraction in a sterile temporary directory and verify compilation configuration:
```powershell
$tempDir = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "nexora_rc3_test")
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
Expand-Archive -Path "dist\cuda-acceleration-engine-v1.0.0-rc3-source.zip" -DestinationPath $tempDir

# Verify key unpacked files
$rootExtracted = Join-Path $tempDir "cuda-acceleration-engine"
assert (Test-Path (Join-Path $rootExtracted "LICENSE"))
assert (Test-Path (Join-Path $rootExtracted "CMakeLists.txt"))
assert (Test-Path (Join-Path $rootExtracted "requirements.txt"))
Write-Host "PASS: Clean-room extraction verified successfully."
Remove-Item -Recurse -Force $tempDir
```
*Expected Outcome:* `PASS: Clean-room extraction verified successfully.`

---

## 6. Actionable Ordered Blocker Checklist (Pre-RC3 Gating)

```
========================================================================================
   NEXORA v1.0.0 — ORDERED RELEASE CANDIDATE 3 (RC3) REMEDIATION GATES
========================================================================================
[ ] GATE 1: SCRIPT PATCH AUTHORIZATION
    - Target: scripts/package_release.py
    - Action: Add "LICENSE" and "THIRD_PARTY_LICENSES.md" to INCLUDE_ROOT_FILES.
    - Owner: Release Engineer / Project Owner

[ ] GATE 2: DISCLOSURE DOCUMENT DEPLOYMENT
    - Target: Repository Root
    - Action: Write finalized THIRD_PARTY_LICENSES.md to repository root.
    - Owner: Compliance Engineer

[ ] GATE 3: DATASET PROVENANCE SYNCHRONIZATION
    - Target: data/text/production_10m/dataset_manifest.json & 4 downstream results
    - Action: Update records for Hegel (#39064), Kant (#48433), Vasu (#37722).
    - Owner: Data Provenance Lead

[ ] GATE 4: CONTRIBUTOR DCO / CLA RATIFICATION
    - Target: Governance Records
    - Action: Collect signed contributor certifications (DCO + License Ratification).
    - Owner: Project Owner

[ ] GATE 5: RC3 REPRODUCIBLE PACKAGING EXECUTION
    - Target: dist/cuda-acceleration-engine-v1.0.0-rc3-source.zip
    - Action: Execute scripts/package_release.py v1.0.0-rc3.
    - Owner: Release Engineer

[ ] GATE 6: SIX-POINT VALIDATION BATTERY
    - Target: RC3 Archive & Checksums
    - Action: Execute Section 5 validation suite (Steps 1 through 6).
    - Outcome: 6/6 PASS, SHA-256 published.
========================================================================================
STATUS: ALL SPECIFICATIONS READY. STRICT READ-ONLY AUDIT COMPLETE.
========================================================================================
```

