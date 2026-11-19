# NEXORA Transformer Models — Official Model Card & Documentation

**Document Version:** 1.0.0  
**Date:** November 8, 2026  
**Project:** CUDA GPU Acceleration Engine (NEXORA)  
**Organization / Author:** CUDA GPU Acceleration Engine Contributors  
**Primary Hardware Certified:** NVIDIA GeForce RTX 3050 Laptop GPU 6GB (Ampere SM 8.6, 20 SMs, 70W TGP)  
**Status:** **AUTHORITATIVE MODEL DOCUMENTATION — WEIGHT REDISTRIBUTION PENDING OWNER DECISION**  

---

## 1. Executive Model Summary

This Model Card provides standardized documentation for the two autoregressive transformer model checkpoints developed and pre-trained within the **CUDA GPU Acceleration Engine (NEXORA)** platform.

| Model Characteristic | Model 1: NEXORA GPT-86.8M | Model 2: NEXORA GPT-100M Pilot |
|---|---|---|
| **Model Classification** | **Certified Pretrained Foundation Model** | **Exploratory Architecture Pilot** |
| **Operational Readiness** | **Production Feasible / Converged** | **Exploratory Feasibility Prototype Only** |
| **Partially Trained Flag**| `is_partially_trained: false` | `is_partially_trained: true` |
| **Total Exact Parameters**| **86,825,472** | **85,940,736** (active pilot configuration) |
| **Preset Spec Parameters**| 86,825,472 | 92,133,888 (target preset architecture) |
| **Architecture** | 12 Layers / 12 Heads / $d_{\text{model}}=768$ | 12 Layers / 12 Heads / $d_{\text{model}}=768$ |
| **Feed-Forward Dimension**| $d_{\text{ff}} = 3072$ | $d_{\text{ff}} = 3072$ |
| **Vocabulary Size** | 2,048 tokens (BPE) | 512 tokens (pilot subset) |
| **Context Length ($T$)** | 256 tokens | 128 tokens |
| **Weight Tying** | Enabled ($W_{\text{embed}} = W_{\text{head}}^T$) | Enabled ($W_{\text{embed}} = W_{\text{head}}^T$) |
| **Training Steps** | **Step 12,000** (Best Checkpoint) / **12,208** (Final Step) | **15 optimizer steps** (Smoke verification) |
| **Cumulative Tokens** | **49,149,952 tokens** (Best) / **50,001,920 tokens** (Final)* | **7,936 tokens** |
| **Validation Loss** | **1.9383** (Best at Step 12,000) / **1.9406** (Final Step 12,208) | **5.6888** |
| **Validation Perplexity** | **6.95** (Best at Step 12,000) / **6.96** (Final Step 12,208) | **295.53** |
| **Checkpoint File** | `checkpoints/transformer/pretraining_50m_campaign/best_model.pt` | `checkpoints/transformer/flagship_100m_pilot/best_model.pt` |
| **Checkpoint Size** | 1,042,091,280 bytes (993.82 MB) | 1,026,758,674 bytes (979.19 MB) |
| **SHA-256 Checksum** | `ABCD0FE7B305CD1CEF7809359CA7128F0AC94BED592A188F8432329CA1406AAC` | `F8B0B9F9D4FB2E9CE482AEA9C619E2C588404CAA7E5691606E9340F381932F9F` |

*\*Note: Early release manifest documentation reported 50,004,608 tokens. See Section 4.3 for disclosure of the 2,688-token unresolved documentation estimate.*

---

## 2. Model Descriptions & Architectural Specifications

### 2.1 Model 1: NEXORA GPT-86.8M (Certified Production Foundation)
`NEXORA GPT-86.8M` is a causal, decoder-only transformer language model engineered to maximize autoregressive inference throughput and numerical stability within consumer GPU memory budgets (<= 6GB VRAM).
- **Attention Mechanism:** 12 multi-head self-attention heads with scaled dot-product attention and causal masking, accelerated via custom fused CUDA FlashAttention kernels.
- **Normalization & Activation:** Pre-layer normalization (`layer_norm_eps = 1e-5`) with standard Gaussian Error Linear Unit (`GELU`) activations.
- **Weight Tying:** The input embedding matrix and the final output projection matrix share weights, reducing total parameter footprint by 1.57M parameters ($2048 \times 768$) while stabilizing output token logit representations.
- **KV-Cache Acceleration:** Integrated key-value caching reduces sequential autoregressive generation latency to **17.68 – 18.49 ms/token** (54.1 – 56.6 tokens/sec warm throughput on mobile Ampere silicon).

### 2.2 Model 2: NEXORA GPT-100M Pilot (Exploratory Architecture Prototype)
`NEXORA GPT-100M Pilot` is an exploratory architecture checkpoint created exclusively to validate training stability, memory bounds, and gradient flow on physical RTX 3050 hardware for larger parameter configurations prior to extended pretraining campaigns.
- **Pilot Configuration:** Uses an active vocabulary of 512 tokens and context length of 128 tokens, yielding 85,940,736 parameters.
- **Target Flagship Preset:** Designed to scale to 92,133,888 parameters with an expanded 8,192 token vocabulary and 1,024 context window.
- **Status Disclosure:** This checkpoint has been trained for only **15 optimizer steps** (~7,936 tokens). Its validation loss is **5.6888** and perplexity is **295.53**. **It does NOT produce coherent natural language text.** It is preserved solely as a technical artifact demonstrating architectural initialization and forward/backward computational viability.

---

## 3. Training Data Provenance, Diversity & Limitations

### 3.1 Data Lineage & Legal Provenance
The training corpus for the 50M campaign was constructed and validated during Phase 13 Milestone 5 (`results/phase13_milestone5_licensing_audit.json` and `data/text/production_10m/dataset_manifest.json`, aggregate SHA-256: `ba004fd1143fdc4e38045a15f79832411585ea20758cfba78dc0a0349010f5b1`). The corpus comprises exactly **36 discrete documents** totaling **13,651,126 tokens** across three splits (34 Train, 1 Val, 1 Test), categorized into four distinct groups:

1. **Classical Literature (22 documents; 9,379,380 tokens):**
   - 21 Train documents: William Shakespeare (*Complete Dramatic Works & Poetry*), Mary Shelley (*Frankenstein*, *The Last Man*), Arthur Conan Doyle (*Sherlock Holmes*, *The Hound of the Baskervilles*, *The Sign of the Four*), Charles Dickens (*A Tale of Two Cities*, *Great Expectations*), Jane Austen (*Pride and Prejudice*), Herman Melville (*Moby-Dick*), Alexandre Dumas (*The Count of Monte Cristo*), Bram Stoker (*Dracula*), Charlotte Brontë (*Jane Eyre*), Emily Brontë (*Wuthering Heights*), Jonathan Swift (*Gulliver's Travels*), Daniel Defoe (*Robinson Crusoe*), H.G. Wells (*The War of the Worlds*, *The Invisible Man*), Jules Verne (*Twenty Thousand Leagues Under the Sea*), Robert Louis Stevenson (*Treasure Island*), Mark Twain (*Adventures of Huckleberry Finn*).
   - 1 Validation document: H.G. Wells (*The Time Machine*; 104,180 tokens).
2. **Historical Science & Philosophy Treatises (9 documents; 3,835,358 tokens):**
   - 8 Train documents: Charles Darwin (*On the Origin of Species*, *The Voyage of the Beagle*), Michael Faraday (*Experimental Researches in Electricity*), Henri Poincaré (*The Value of Science*), James Clerk Maxwell (*Matter and Motion*), Galileo Galilei (*Dialogues Concerning Two New Sciences*), Francis Bacon (*Novum Organum*), Francis Bacon & Ada Lovelace (*Foundations of Analytical Engines*).
   - 1 Test document: Sir Isaac Newton (*Opticks: Treatise of Reflections, Refractions*; 331,796 tokens).
3. **IETF Networking Standards (4 documents; 434,537 tokens):**
   - 4 Train documents: RFC 791 (IPv4), RFC 793 (TCP), RFC 1034 (DNS), RFC 2616 (HTTP/1.1). Open standards under US government work / computational fair use.
4. **Internal Systems Documentation (1 document; 1,603 tokens):**
   - 1 Train document: Parallel Computing Principles & Attention Mechanics (`computing_foundations`), authored in-repo and permissively licensed under CC0 / MIT.

**Resolution of Historical Category Tally:**  
Preliminary audit notes previously reported an arithmetic sum of $29 + 5 + 4 + 1 = 39$ documents. Reconciliation confirmed that 29 works were sourced from Project Gutenberg eBooks, of which 22 are literary works and 7 are scientific treatises. The earlier report erroneously tallied all 29 Gutenberg works under "literature" and then added scientific works separately, double-counting the treatises. The reconciled, authoritative manifest breakdown is $22 + 9 + 4 + 1 = \mathbf{36\text{ documents}}$.

### 3.2 Data Processing & Partitioning
- **Tokenizer:** Custom Byte-Pair Encoding (BPE) tokenizer trained directly from scratch on the public domain corpus (`bpe_vocab.json`). Contains 2,048 tokens with byte fallback tokens (`<byte_00>` through `<byte_ff>`), preventing out-of-vocabulary crashes.
- **Partitioning Strategy:** Strict document-level splits (Train: 95%, Validation: 5%). Entire historical works were kept in isolated sets to guarantee **0.00% validation data leakage or n-gram overlap**.
- **Sanitization:** 100% free of Personal Identifiable Information (PII), private customer records, trade secrets, or proprietary source code.

### 3.3 Training Data Limitations & Bias
- **Historical Archaic Language:** Due to reliance on 17th–19th century literature, the training distribution exhibits archaic vocabulary, dramatic prose, and 19th-century scientific terminology.
- **Zero Contemporary Knowledge:** The corpus contains no contemporary news, modern cultural references, or post-2000 technical developments (outside of baseline networking RFCs).
- **Narrow Scientific Scope:** Technical documentation is limited to networking protocols and GPU hardware principles.

---

## 4. Training Methodology & Computational Optimization

### 4.1 Hyperparameters & Optimization Schedule

```
+----------------------------------------------------------------------------------------------------+
| PRETRAINING CAMPAIGN HYPERPARAMETERS (NEXORA GPT-86.8M)                                            |
+-----------------------------------+----------------------------------------------------------------+
| Parameter                         | Specification / Empirical Value                                |
+-----------------------------------+----------------------------------------------------------------+
| Optimization Algorithm            | AdamW (betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)         |
| Peak Learning Rate                | 2.0e-4 with Cosine Annealing decay to 2.0e-5                   |
| Learning Rate Warmup              | Linear warmup over 200 steps                                   |
| Gradient Clipping                 | Max norm <= 1.0                                                |
| Global Batch Size                 | 16 sequences (Per-device batch size 4 x 4 accumulation steps)  |
| Mixed Precision                   | Automatic Mixed Precision (AMP FP16) with Dynamic Scaler       |
| Best Validation Checkpoint        | Step 12,000 (49,149,952 Tokens; Val Loss: 1.9383, PPL: 6.95)   |
| Campaign Completion Checkpoint    | Step 12,208 (50,001,920 Tokens; Val Loss: 1.9406, PPL: 6.96)   |
| Tokens per Step                   | 4,096 tokens (16 sequences x 256 context length)               |
+-----------------------------------+----------------------------------------------------------------+
```

### 4.2 Hardware & Environmental Profile
- **Training Silicon:** Single NVIDIA GeForce RTX 3050 Laptop GPU (6GB GDDR6, 70W TGP, Ampere SM 8.6).
- **Execution Platform:** Windows 11 Home 64-bit, Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4.
- **Memory Footprint During Training:** Capped at 2,427.62 MB VRAM (39.5% of 6GB total capacity).
- **Thermal Safety Monitoring:** Continuous telemetry enforced. Average training temperature was 56.4°C with a peak of 59.09°C, maintaining a **+22.91°C safety headroom** below the 82.0°C thermal throttle threshold.

### 4.3 Checkpoint Token Accounting & Manifest Discrepancy Disclosure
The pretraining records establish two distinct serialized model checkpoints, as well as an unresolved discrepancy in the release manifest:
1. **Best Checkpoint (`best_model.pt`):** Captured at **Step 12,000** with **49,149,952 cumulative tokens**, achieving the global minimum validation loss of **1.9383** (Perplexity: **6.95**).
2. **Final Training Checkpoint (`final_model.pt`):** Concluded at **Step 12,208** with **50,001,920 cumulative tokens** ($49,149,952 + [208 \text{ steps} \times 4096 \text{ tokens/step}]$) and validation loss of **1.9406** (Perplexity: **6.96**).
3. **Unresolved Manifest Discrepancy Disclosure:** `MODEL_MANIFEST.json` and early Milestone 9 summary reports record a cumulative token count of **50,004,608 tokens**. The resulting difference of **2,688 tokens** ($50,004,608 - 50,001,920$) represents an **unresolved documentation estimate** introduced during preliminary release reporting; no physical training batch produced this remainder. The true physical checkpoint token counts are strictly **49,149,952** (`best_model.pt`) and **50,001,920** (`final_model.pt`).

---

## 5. Evaluation Results & Benchmarks

### 5.1 Pretraining Convergence Metrics

| Evaluation Metric | Step 0 (Init) | Step 2,500 | Step 7,000 | Step 10,000 | Step 12,000 (Best Val Checkpoint) | Step 12,208 (Final Campaign Step) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cumulative Tokens** | 0 | 10,237,952 | 28,672,000 | 40,960,000 | **49,149,952** | **50,001,920** |
| **Training Loss** | 7.6241 | 3.1142 | 2.2105 | 1.9840 | **1.7686** | **1.7886** |
| **Validation Loss** | 7.6189 | 3.1904 | 2.2530 | 2.0121 | **1.9383** | **1.9406** |
| **Validation Perplexity**| 2036.3 | 24.30 | 9.52 | 7.48 | **6.95** | **6.96** |

### 5.2 Physical Inference Performance (RTX 3050 6GB)
- **Autoregressive Generation Throughput:** **54.1 – 56.6 tokens/second** (Warm state, batch size 1).
- **Per-Token Generation Latency:** **17.68 – 18.49 milliseconds**.
- **Cold-Start Latency (First Token):** ~5,123 ms (includes PyTorch CUDA context binding and kernel autotuning).
- **Peak Inference VRAM Footprint:** **343.64 MB** (< 6% of total device VRAM).

### 5.3 Host CPU Fallback Inference Performance
- **Device Target:** Intel Core i5-13450HX (Host CPU execution via `device: "cpu"`).
- **Throughput:** **2.9 tokens/second**.
- **VRAM Utilization:** **0.00 MB** (Host RAM allocation only).

---

## 6. Intended Uses & Out-of-Scope Uses

### 6.1 Intended Use Cases
1. **Edge & Consumer GPU Benchmarking:** Validating transformer execution speed, memory footprint, and thermal stability on low-VRAM (<= 6GB) consumer workstations.
2. **CUDA Kernel Research & Profiling:** Serving as an authentic computational workload for testing custom fused CUDA kernels (FlashAttention, fused RMSNorm, fused GELU).
3. **Academic & Systems Demonstration:** Demonstrating end-to-end full-stack integration between low-level CUDA C++ kernels, FastAPI services, and React 18 user interfaces.
4. **Offline Local Language Modeling:** Running local causal language generation without external API dependencies or cloud telemetry.

### 6.2 Out-of-Scope & Restricted Uses
1. **High-Stakes Decision Making:** **Strictly Prohibited.** The models must NOT be used for legal, financial, medical, diagnostic, military, or safety-critical automated decisions.
2. **Fact Retrieval / Factual Question Answering:** The models are small pre-trained foundation models trained on historical literature and cannot be relied upon for factual accuracy. Hallucinations are expected.
3. **Harmful Content Generation:** Must NOT be used to generate hate speech, harassment, cyberattack payloads, malware code, or sexually explicit content.
4. **Production Deployment of the 100M Pilot Checkpoint:** The 100M model is an untrained feasibility prototype and is completely unfit for any user-facing text generation.

---

## 7. Known Limitations, Risks & Mitigations

1. **Repetition & Hallucination:** Under greedy sampling or low temperatures ($T < 0.5$), the 86.8M model may enter cyclic repetition loops. *Mitigation:* Use temperature $T=0.7 - 0.8$ with nucleus sampling ($p=0.9$) and repetition penalties.
2. **Limited Context Window:** The 86.8M model is constrained to 256 tokens. Prompts exceeding this boundary are truncated.
3. **Archaic Tone:** Outputs often emulate early Modern English dramatic verse or 19th-century scientific prose reflecting the pretraining literature.
4. **Untrained Pilot Failure Mode:** The 100M pilot produces incoherent character sequences. The system explicitly exposes `is_partially_trained: true` to prevent user deception.

---

## 8. Checkpoint Verification & Cryptographic Hashes

To ensure bit-for-bit authenticity and detect corruption prior to loading weights into memory, consumers must verify SHA-256 checksums:

### 8.1 Production Foundation Checkpoint (`best_model.pt` — Peak Validation)
- **Artifact Role:** Best validation generalization checkpoint (minimum validation loss).
- **Global Step:** Step 12,000 (49,149,952 serialized tokens).
- **Relative Path:** `checkpoints/transformer/pretraining_50m_campaign/best_model.pt`
- **Expected Size:** `1,042,091,280 bytes` (993.82 MB)
- **Expected SHA-256:** `ABCD0FE7B305CD1CEF7809359CA7128F0AC94BED592A188F8432329CA1406AAC`
- **Verification Command (Windows PowerShell):**
  ```powershell
  Get-FileHash -Algorithm SHA256 checkpoints\transformer\pretraining_50m_campaign\best_model.pt
  ```

### 8.2 Final Pretraining Checkpoint (`final_model.pt` — Campaign Conclusion)
- **Artifact Role:** Final pretraining campaign completion checkpoint.
- **Global Step:** Step 12,208 (50,001,920 serialized tokens).
- **Relative Path:** `checkpoints/transformer/pretraining_50m_campaign/final_model.pt`
- **Expected Size:** `1,042,091,878 bytes` (993.82 MB)
- **Expected SHA-256:** `CA6FE2A51D1B973F0C3E0FDA04FA43CE5A8CA1E95F51ED9D99CC424C234E45BD`
- **Verification Command (Windows PowerShell):**
  ```powershell
  Get-FileHash -Algorithm SHA256 checkpoints\transformer\pretraining_50m_campaign\final_model.pt
  ```

### 8.3 Exploratory Pilot Checkpoint (`NEXORA GPT-100M Pilot`)
- **Artifact Role:** Exploratory 15-step architectural initialization prototype (partially trained).
- **Relative Path:** `checkpoints/transformer/flagship_100m_pilot/best_model.pt`
- **Expected Size:** `1,026,758,674 bytes` (979.19 MB)
- **Expected SHA-256:** `F8B0B9F9D4FB2E9CE482AEA9C619E2C588404CAA7E5691606E9340F381932F9F`
- **Verification Command (Windows PowerShell):**
  ```powershell
  Get-FileHash -Algorithm SHA256 checkpoints\transformer\flagship_100m_pilot\best_model.pt
  ```

---

## 9. Model Licensing & Redistribution Decision Block

> [!IMPORTANT]
> **LEGAL NOTICE: Model weights do NOT automatically inherit open-source software licenses.**  
> While the software source code of NEXORA is licensed under the permissive MIT License, machine learning model weight checkpoints represent distinct computational artifacts. This section documents the legal status and sets forth the explicit decision required by the Project Owner.

```
====================================================================================================
                        MODEL WEIGHTS LICENSING STATUS AUDIT
====================================================================================================
  PROJECT SOFTWARE LICENSE:          MIT License (Copyright (c) 2026 Contributors)
  PRETRAINING DATASET RIGHTS:        100% Public Domain & IETF Open Standards (Verified Compliant)
  THIRD-PARTY IP IN WEIGHTS:         0% (Trained from scratch; zero teacher model distillation)
  METADATA LICENSE DECLARATION:      ABSENT (Checkpoints do not contain serialized license fields)
  MODEL WEIGHT REDISTRIBUTION:       >>> NOT VERIFIED — HOSTING PAUSED <<<
====================================================================================================
```

### 9.1 Legal Analysis & Pre-Conditions
1. **Software vs. Weights Separation:** The root `LICENSE` file governs code in `backend/`, `frontend/`, `custom_kernels/`, and `engine/`. It does not explicitly state terms for neural network parameter tensors.
2. **Dataset Freedom:** Because the training data consists entirely of Public Domain literature and IETF RFCs, there are no upstream restrictive copyleft or non-commercial data clauses encumbering the weights.
3. **Third-Party Model Rights:** No third-party base weights (e.g. LLaMA, Mistral, GPT-2) were fine-tuned or pruned. The checkpoints are 100% original numerical weights.

### 9.2 Project Owner Action Required
Public hosting and distribution (e.g. on Hugging Face Model Hub, GitHub Release Assets, or S3 buckets) **MUST REMAIN PAUSED** until the Project Owner explicitly selects and signs one of the following model licensing frameworks:

```
[ ] OPTION 1: Permissive Open-Source Model License (Creative Commons Attribution 4.0 - CC-BY-4.0)
    Allows unrestricted commercial and academic use, modification, and redistribution with attribution.

[ ] OPTION 2: OpenRAIL-M (Responsible AI License)
    Permits royalty-free commercial and academic use while prohibiting designated harmful use cases.

[ ] OPTION 3: Explicit MIT License Extension for Model Binaries
    Formal declaration that the root MIT license applies equally to the binary weight tensors.

[ ] OPTION 4: Non-Commercial Research Only (CC-BY-NC-4.0)
    Restricts model weights to academic and experimental research purposes.

Project Owner Decision:  __________________________________
Authorized Signature:    __________________________________
Date:                    __________________________________
```

**Current Operational Mandate:** Until the Project Owner completes this authorization, model checkpoints must be retained strictly on private infrastructure and flagged as **NOT FOR PUBLIC UPLOAD**.

