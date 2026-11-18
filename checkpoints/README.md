# Model Checkpoints Directory & Distribution Guide

This directory holds neural network model checkpoints for the **CUDA GPU Acceleration Engine**.

Due to GitHub's per-file upload ceiling (100 MB), large binary model weights are not tracked directly in git. Instead, release checkpoints are published as versioned GitHub Release assets and Hugging Face model repository artifacts, verified against cryptographic SHA-256 hashes.

---

## 1. Required Release Models & Placements

| Model | Checkpoint File Path | Size (Bytes / MB) | SHA-256 Checksum | Lineage / Status |
| :--- | :--- | :---: | :---: | :---: |
| **NEXORA GPT-86.8M** | `checkpoints/transformer/pretraining_50m_campaign/best_model.pt` | 1,042,091,280<br>(993.82 MB) | `ABCD0FE7B305CD1CEF7809359CA7128F0AC94BED592A188F8432329CA1406AAC` | **Certified Production Model**<br>50M tokens, 12,208 steps<br>`is_partially_trained: false` |
| **NEXORA GPT-100M Pilot** | `checkpoints/transformer/flagship_100m_pilot/best_model.pt` | 1,026,758,674<br>(979.19 MB) | `F8B0B9F9D4FB2E9CE482AEA9C619E2C588404CAA7E5691606E9340F381932F9F` | **Exploratory Pilot Prototype**<br>15 optimizer steps<br>`is_partially_trained: true` |

---

## 2. Checkpoint Verification Commands

### Windows PowerShell
```powershell
# Verify NEXORA GPT-86.8M Checkpoint
Get-FileHash -Algorithm SHA256 checkpoints\transformer\pretraining_50m_campaign\best_model.pt

# Verify NEXORA GPT-100M Pilot Checkpoint
Get-FileHash -Algorithm SHA256 checkpoints\transformer\flagship_100m_pilot\best_model.pt
```

### Windows Command Prompt (CMD)
```cmd
certutil -hashfile checkpoints\transformer\pretraining_50m_campaign\best_model.pt SHA256
certutil -hashfile checkpoints\transformer\flagship_100m_pilot\best_model.pt SHA256
```

### Linux / WSL / macOS
```bash
sha256sum checkpoints/transformer/pretraining_50m_campaign/best_model.pt
sha256sum checkpoints/transformer/flagship_100m_pilot/best_model.pt
```

---

## 3. Machine Placement Instructions

When deploying a fresh clone of the engine:

1. Create target directories:
   ```powershell
   mkdir -p checkpoints/transformer/pretraining_50m_campaign
   mkdir -p checkpoints/transformer/flagship_100m_pilot
   ```
2. Place downloaded `best_model.pt` files into their respective subdirectories.
3. Validate hashes against `MODEL_MANIFEST.json` before starting the platform service.

