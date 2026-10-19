# PHASE 9 GPU PERFORMANCE REPORT — RESNET-18 TRAINING & INFERENCE BENCHMARKS

## 1. Executive Summary & Measurement Integrity

This report documents the empirical performance benchmarks executed under **Stage C** of Phase 9. All metrics are derived from actual system executions using explicit CUDA event synchronizations (`torch.cuda.synchronize()`), verified warmup iterations, and defined measurement boundaries.

All speedups are calculated strictly relative to measured single-threaded CPU reference executions conducted on the **Intel Core i5-13450HX**. Historical kernel micro-benchmarks from Phases 1–8 (e.g. 89.9x end-to-end SGEMM) are distinct from these end-to-end neural network workloads.

---

## 2. Experimental Benchmark Matrix

- **Workload:** ResNet-18 Deep Residual Network (11,173,962 parameters, 42.6 MB FP32 footprint)
- **Input Tensor Dimensions:** `(BatchSize, 3, 32, 32)`
- **GPU Target:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, 2,560 CUDA Cores, Compute Capability `sm_86`)
- **CPU Reference:** Intel Core i5-13450HX (16 threads, x86_64)
- **Deep Learning Runtime:** PyTorch `2.6.0+cu124` on Windows 11

---

## 3. Training Performance Benchmark Results

The training step benchmark measures a complete training iteration encompassing the forward pass, cross-entropy loss computation, backward gradient backpropagation, and AdamW optimizer parameter updates.

### 3.1 Empirical Training Performance Table

| Configuration | Precision Mode | Batch Size | Step Time (ms) | Throughput (samples/s) | Peak VRAM Allocated (MB) | Measured Speedup vs CPU Reference |
|---|---|---|---|---|---|---|
| **CPU Reference** | FP32 | 64 | 3,749.99 | **17.1** | 0.0 | **1.00×** (Baseline) |
| **GPU FP32** | FP32 | 64 | 61.59 | **1,039.2** | 471.9 | **60.77×** |
| **GPU AMP** | FP16 / AMP | 64 | 38.07 | **1,681.2** | 514.0 | **98.32×** |
| **GPU AMP** | FP16 / AMP | 32 | 24.86 | **1,287.0** | 613.5 | **75.26×** |
| **GPU AMP** | FP16 / AMP | 128 | 64.93 | **1,971.4** | 829.9 | **115.29×** |
| **GPU AMP** | FP16 / AMP | 256 | 121.09 | **2,114.2** | 1,123.7 | **123.64×** |

### 3.2 Performance Insights & Scaling Analysis

1. **GPU Acceleration vs CPU Baseline:**
   - Single-precision GPU training achieved a **60.77×** speedup over the CPU baseline at identical batch sizes ($64$).
   - Step latency dropped from 3,749.99 ms on CPU down to 61.59 ms on CUDA FP32.
2. **Automatic Mixed Precision (AMP) Benefit:**
   - Enabling CUDA AMP (`torch.amp.autocast('cuda')`) accelerated training throughput from 1,039.2 to **1,681.2 samples/sec** at batch size 64 (**1.62× faster** than GPU FP32 alone).
   - Speedup reached **98.32×** vs the CPU baseline.
3. **Batch Size Scaling:**
   - As batch size scaled from 32 to 256, training throughput scaled from 1,287.0 to **2,114.2 samples/sec** (**123.64× speedup** vs CPU).
   - Larger batch sizes maximized GPU Streaming Multiprocessor (SM) utilization, saturating the 20 SMs on the GA107 core.
4. **VRAM Headroom:**
   - Even at batch size 256, peak allocated VRAM was 1,123.66 MB (under 18.3% of total available VRAM), proving the 6GB RTX 3050 has ample capacity for deep vision training.

---

## 4. Inference Latency & Throughput Benchmark Results

The inference benchmark evaluates evaluation-mode forward passes (`torch.no_grad()`) across batch sizes 1 (single-sample interactive serving), 32, and 128 (high-throughput batch processing).

### 4.1 Latency Distribution & Throughput Table

| Configuration | Batch Size | Mean Latency (ms) | p50 Median (ms) | p95 Latency (ms) | p99 Latency (ms) | Throughput (inferences/s) | Inference Speedup vs CPU (b=1) |
|---|---|---|---|---|---|---|---|
| **CPU FP32** | 1 | 8.66 | 8.66 | 9.06 | 9.08 | **115.4** | **1.00×** (Baseline) |
| **GPU FP32** | 1 | 2.26 | 2.18 | 2.64 | 3.51 | **442.2** | **3.83×** |
| **GPU AMP** | 1 | 2.70 | 2.66 | 2.85 | 3.41 | **370.9** | **3.21×** |
| **GPU AMP** | 32 | 6.16 | 6.14 | 6.22 | 6.35 | **5,199.2** | **45.05×** |
| **GPU AMP** | 128 | 19.21 | 19.19 | 19.34 | 19.41 | **6,663.8** | **57.75×** |

### 4.2 Latency Distribution Analysis

1. **Single-Sample Real-Time Latency:**
   - Single-sample inference latency on GPU FP32 is **2.26 ms** (median **2.18 ms**), providing sub-3ms real-time responsiveness for interactive model serving.
   - For batch size 1, FP32 outperforms AMP (2.26 ms vs 2.70 ms) because kernel launch overhead and casting dominate over arithmetic intensity at single-item tensors.
2. **High-Throughput Batch Serving:**
   - At batch size 32, mean latency is 6.16 ms (**0.19 ms per image**), delivering **5,199.2 inferences/sec**.
   - At batch size 128, mean latency is 19.21 ms (**0.15 ms per image**), delivering **6,663.8 inferences/sec** (**57.75×** total throughput acceleration over CPU).
3. **Latency Predictability:**
   - Tail latency is tight: for batch size 128, p50 is 19.19 ms, p95 is 19.34 ms, and p99 is 19.41 ms (under 0.22 ms variance).

---

## 5. Thermal & Power Telemetry During Workloads

Continuous hardware telemetry was recorded across the full benchmark battery:

| Telemetry Attribute | Idle Baseline | Peak Sustained Workload | Safety Limit | Margin to Limit |
|---|---|---|---|---|
| **GPU Core Temperature** | 44.0°C | **63.0°C** | 83.0°C | +20.0°C safe headroom |
| **GPU Utilization** | 0.0% | **99.0%** | 100.0% | Fully saturated |
| **Board Power Draw** | 11.78 W | **74.08 W** | Manufacturer Spec | Safely within power envelope |
| **VRAM Allocated** | 0.0 MB | **1,123.7 MB** | 6,143.5 MB | +5,019.8 MB free |

Zero thermal throttling events, driver resets, or out-of-memory errors were observed throughout testing.
