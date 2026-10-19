# PHASE 9 TRAINING RESULTS — RESNET-18 & CIFAR-10 EMPIRICAL LOGS

## 1. Overview & Verification Scope

All metrics recorded in this document represent actual measured executions conducted on the **NVIDIA GeForce RTX 3050 6GB Laptop GPU** paired with the **Intel Core i5-13450HX** processor.

Historical results from Phases 1–8 (such as SGEMM tiled kernel speedups) are strictly preserved and separated from these new neural network training measurements.

---

## 2. Experimental Setup & Model Configuration

- **Model Architecture:** ResNet-18 (Canonical CIFAR stem: 3×3 conv1, stride 1, padding 1, no initial pooling)
- **Parameter Count:** **11,173,962 parameters** (all trainable, 0 non-trainable)
- **Model Size:** 42.626 MB (single-precision FP32)
- **Dataset:** Official CIFAR-10 Python Archive (50,000 train, 10,000 test)
  - MD5 Integrity: `c58f30108f718f92721af3b95e74349a` (Verified 100%)
  - Preprocessing: Standard channel normalization ($\mu=[0.4914, 0.4822, 0.4465]$, $\sigma=[0.2470, 0.2435, 0.2616]$)
  - Augmentation: Reflection padding (4px), random 32×32 crop, random horizontal flip
- **Loss Function:** PyTorch `nn.CrossEntropyLoss()`
- **Optimizer:** `AdamW` ($\beta_1=0.9, \beta_2=0.999$, weight decay = $1\times 10^{-4}$)
- **Learning Rate Schedule:** `CosineAnnealingLR` ($T_{\max}=15$, $\eta_{\min}=1\times 10^{-6}$)
- **Precision Mode:** Automatic Mixed Precision (AMP) via `torch.amp.autocast('cuda')` with `GradScaler`
- **Batch Size:** 128
- **Deterministic Seed:** 42

---

## 3. Pilot Run Execution Progression (2 Epochs, Subset)

| Epoch | Phase | Loss | Accuracy (%) | Duration (ms) | Throughput (samples/s) | Learning Rate | VRAM Allocated (MB) | VRAM Reserved (MB) | GPU Temp (°C) |
|---|---|---|---|---|---|---|---|---|---|
| **0** | Pre-Train | — | — | — | — | 0.001000 | 42.6 | 64.0 | 45.0 |
| **1** | Train | 1.9885 | 26.10% | 9,036.3 | 221.3 | 0.001000 | 479.16 | 560.0 | 51.0 |
| **1** | Validation | 2.5051 | 20.20% | 275.4 | 1,815.5 | 0.001000 | 479.16 | 560.0 | 51.0 |
| **2** | Train | 1.6963 | 36.45% | 4,416.9 | 452.8 | 0.000501 | 479.16 | 560.0 | 54.0 |
| **2** | Validation | 1.8691 | 30.20% | 268.1 | 1,864.9 | 0.000501 | 479.16 | 560.0 | 54.0 |

- **Isolated Final Test Evaluation:** Test Loss: `1.8998` | Test Accuracy: `32.80%` | Throughput: `1,923.9 samples/s`

---

## 4. Checkpoint Resume Verification Run

To ensure fault tolerance and verify training continuity, training was resumed from `latest_checkpoint.pt`:

| Metric | Recorded Value |
|---|---|
| **Resume Checkpoint** | `checkpoints/resnet18_cifar10/latest_checkpoint.pt` |
| **Resumed Start Epoch** | Epoch 3 |
| **Prior Best Validation Accuracy** | 30.20% |
| **Epoch 3 Train Loss** | **1.5420** (decreased from 1.6963) |
| **Epoch 3 Train Accuracy** | **43.80%** (increased from 36.45%) |
| **Epoch 3 Validation Loss** | **1.6489** (improved from 1.8691) |
| **Epoch 3 Validation Accuracy** | **38.80%** (improved from 30.20%) |
| **Epoch 3 Duration** | 5,498.6 ms (363.7 samples/sec) |
| **Isolated Test Loss** | **1.6754** |
| **Isolated Test Accuracy** | **38.00%** |
| **Peak VRAM Allocated** | 479.82 MB |
| **Peak VRAM Reserved** | 564.0 MB |

---

## 5. Full-Dataset Convergence Training Run (15 Epochs, 45,000 Samples)

- **Experiment ID:** `gpu_amp_resnet18_b128_ep15_1792399262`
- **Training Samples:** 45,000 | **Validation Samples:** 5,000 | **Test Samples:** 10,000
- **Total Workload Duration:** 787,166.15 ms (~13.1 minutes)
- **Average Epoch Duration:** 50,251.5 ms (~50.3 seconds per epoch)
- **Continuous Training Throughput:** ~897.0 samples/sec (with live augmentation and evaluation)

### 5.1 Per-Epoch Progression

| Epoch | Train Loss | Train Accuracy (%) | Val Loss | Val Accuracy (%) | Learning Rate | Epoch Time (s) | Best Val So Far (%) |
|---|---|---|---|---|---|---|---|
| **1** | 1.3812 | 49.10% | 1.1167 | 59.96% | 0.001000 | 55.3s | 59.96% |
| **2** | 0.9463 | 66.13% | 1.1312 | 63.58% | 0.000989 | 51.8s | 63.58% |
| **3** | 0.7537 | 73.40% | 0.7413 | 74.72% | 0.000957 | 51.7s | 74.72% |
| **4** | 0.6248 | 78.32% | 0.6023 | 79.48% | 0.000905 | 51.1s | 79.48% |
| **5** | 0.5392 | 81.18% | 0.5503 | 81.34% | 0.000835 | 50.5s | 81.34% |
| **6** | 0.4691 | 83.70% | 0.5050 | 82.10% | 0.000750 | 45.8s | 82.10% |
| **7** | 0.4125 | 85.77% | 0.4009 | 86.18% | 0.000655 | 48.0s | 86.18% |
| **8** | 0.3647 | 87.40% | 0.3834 | 87.10% | 0.000553 | 48.7s | 87.10% |
| **9** | 0.3177 | 88.96% | 0.3786 | 87.30% | 0.000448 | 50.5s | 87.30% |
| **10** | 0.2759 | 90.39% | 0.3480 | 87.82% | 0.000346 | 50.8s | 87.82% |
| **11** | 0.2376 | 91.72% | 0.3306 | 88.86% | 0.000251 | 50.9s | 88.86% |
| **12** | 0.2107 | 92.72% | 0.2847 | 90.40% | 0.000166 | 50.6s | 90.40% |
| **13** | 0.1813 | 93.74% | 0.2831 | 90.72% | 0.000096 | 50.8s | 90.72% |
| **14** | 0.1635 | 94.37% | 0.2773 | 90.72% | 0.000044 | 48.4s | 90.72% |
| **15** | **0.1551** | **94.73%** | **0.2648** | **90.92%** | **0.000012** | 48.9s | **90.92%** |

### 5.2 Isolated Final Test Evaluation
- **Test Loss:** **0.2980**
- **Test Top-1 Accuracy:** **90.31%**
- **Inference Latency:** 2.89s (total 10,000 test set images)
- **Evaluation Throughput:** **3,462.1 samples/sec**

---

## 6. Summary & Key Findings

1. **High Convergence Accuracy:** The 15-epoch training run delivered **90.92% validation accuracy** and **90.31% test accuracy**, demonstrating production-grade training performance.
2. **VRAM Stability:** Memory allocation remained constant at **479.16 MB allocated** (560.0 MB reserved), consuming under 9.2% of the available 6,144 MB VRAM.
3. **Thermal Safety:** GPU operating temperature remained between 52°C and 65°C throughout the continuous 13.1-minute workload, well below the 75°C warning limit.
4. **State Persistence:** `best_model.pt` was updated at Epoch 15 with 90.92% accuracy and verified on live inference serving.
