# Model Checkpoints Directory & Distribution Guide

This directory holds neural network model checkpoints for the **CUDA GPU Acceleration Engine**.

Due to GitHub's per-file upload ceiling (100 MB), large binary model weights are not tracked directly in git. Public release model weights (**NEXORA GPT-86.8M**) are published as versioned GitHub Release assets and Hugging Face model repository artifacts under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**, verified against cryptographic SHA-256 hashes.

The software source code remains governed exclusively by the **MIT License**.

---

## 1. Release Models & Placement Classification

| Model | Checkpoint File Path | Size (Bytes / MB) | SHA-256 Checksum | Lineage / Distribution Status |
| :--- | :--- | :---: | :---: | :---: |
| **NEXORA GPT-86.8M** | `checkpoints/transformer/pretraining_50m_campaign/best_model.pt` | 1,042,091,280<br>(993.82 MB) | `ABCD0FE7B305CD1CEF7809359CA7128F0AC94BED592A188F8432329CA1406AAC` | **Certified Production Model (Intended Public Release — CC-BY-4.0; Release Auth Pending)**<br>50M tokens, 12,208 steps<br>`is_partially_trained: false` |
| **NEXORA GPT-100M Pilot** | `checkpoints/transformer/flagship_100m_pilot/best_model.pt` | 1,026,758,674<br>(979.19 MB) | `F8B0B9F9D4FB2E9CE482AEA9C619E2C588404CAA7E5691606E9340F381932F9F` | **Internal Engineering Fixture (Withheld from Public Distribution)**<br>15 optimizer steps (Smoke verification)<br>`is_partially_trained: true` |

> [!NOTE]
> **Distribution & Governance Boundary:** Only **NEXORA GPT-86.8M** is designated for intended public release (under CC-BY-4.0), subject to pending final release authorization by the Project Owner. External download and public upload remain on hold until final release authorization is executed. The **NEXORA GPT-100M Pilot** is an exploratory 15-step feasibility prototype retained exclusively on private engineering infrastructure for architecture verification.

---

## 2. Checkpoint Verification Commands

### Intended Public Release Model (NEXORA GPT-86.8M — Release Auth Pending)

```powershell
# Windows PowerShell
Get-FileHash -Algorithm SHA256 checkpoints\transformer\pretraining_50m_campaign\best_model.pt
# Expected: ABCD0FE7B305CD1CEF7809359CA7128F0AC94BED592A188F8432329CA1406AAC

# Windows Command Prompt (CMD)
certutil -hashfile checkpoints\transformer\pretraining_50m_campaign\best_model.pt SHA256

# Linux / WSL / macOS
sha256sum checkpoints/transformer/pretraining_50m_campaign/best_model.pt
```

### Internal Engineering Fixture (NEXORA GPT-100M Pilot — Internal Developers Only)

```powershell
# Windows PowerShell
Get-FileHash -Algorithm SHA256 checkpoints\transformer\flagship_100m_pilot\best_model.pt
# Expected: F8B0B9F9D4FB2E9CE482AEA9C619E2C588404CAA7E5691606E9340F381932F9F
```

---

## 3. Machine Placement Instructions (Intended Public Release)

When deploying a fresh clone of the engine for public operation (upon final release authorization):

1. Create the production checkpoint directory:
   ```powershell
   mkdir -p checkpoints/transformer/pretraining_50m_campaign
   ```
2. Place the downloaded `best_model.pt` (NEXORA GPT-86.8M, governed by CC-BY-4.0) into `checkpoints/transformer/pretraining_50m_campaign/`.
3. Validate hash against `MODEL_MANIFEST.json` before starting the platform service:
   ```powershell
   Get-FileHash -Algorithm SHA256 checkpoints\transformer\pretraining_50m_campaign\best_model.pt
   ```
4. *Governance & Internal Notes:*
   - Public download and external hosting of `best_model.pt` remain on hold pending final release authorization.
   - The `checkpoints/transformer/flagship_100m_pilot/` directory is an internal test fixture path and is not required for public end-user operation.

