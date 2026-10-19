# Dashboard Telemetry & Benchmark Screenshots

This directory is designated for authentic, full-resolution screenshots of the **CUDA GPU Acceleration Engine** interactive dashboard and live telemetry monitor.

> [!IMPORTANT]
> **Zero Fabrication Policy**: All screenshots in this directory must be captured directly from the live, running dashboard at `http://127.0.0.1:5173/` or `http://127.0.0.1:4173/`. Never fabricate, modify, or mock telemetry values.

---

## Recommended Screenshot Catalog

| Filename | Dashboard Section | Content & Focus Areas |
|---|---|---|
| `01_overview_dashboard.png` | **1. Overview** | Hero card, verified KPI badges (89.9× E2E speedup, 116.8× kernel speedup, 760.5 GFLOPS), system hardware spec card (SM 8.6, 20 SMs, 6GB VRAM, CUDA 13.4). |
| `02_cuda_benchmarks.png` | **2. CUDA Benchmarks** | Recharts execution time comparisons (log scale), speedup multipliers vs single-threaded CPU, interactive category filter tabs, and 12-row verified measurements table. |
| `03_pytorch_training.png` | **3. PyTorch Training** | BaselineMLP training comparisons (CPU, GPU FP32, GPU AMP FP16), total training duration & speedup charts, 10-epoch validation accuracy, and amber historical provenance banner. |
| `04_llm_inference.png` | **4. LLM Inference** | Quantized local SLM/LLM inference benchmarks (Qwen, DeepSeek, TinyLlama), tokens/sec throughput chart, TTFT latency metrics, and historical provenance banner. |
| `05_live_telemetry.png` | **5. Live Telemetry** | Real-time NVIDIA-SMI live telemetry: GPU utilization %, dedicated VRAM progress meter (MiB / GB), junction temperature sensor (°C), power draw (W), and 3-second auto-refresh polling badge. |
| `06_verification_manifest.png` | **6. Verification & Manifest** | 17/17 SHA-256 cryptographic manifest parity audit table (100% matched), downloadable raw artifact archive list, and read-only confinement badge. |

---

## Step-by-Step Capture Instructions

### 1. Start the Backend and Frontend Servers
Open a terminal in the project root:

```powershell
# Terminal 1: Start FastAPI Backend
.\.venv-phase3\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Start Vite Dev Server
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

### 2. Configure Browser Viewport
1. Open Google Chrome, Microsoft Edge, or Mozilla Firefox.
2. Navigate to `http://127.0.0.1:5173/`.
3. Set your browser zoom level to `100%`.
4. Recommended display resolution: **1920 × 1080 (1080p)** or **2560 × 1440 (1440p)** for optimal glassmorphism contrast and clean typography.
5. Press `F11` (Full Screen) or use browser DevTools Device Mode (`Ctrl + Shift + M`) set to `1920 × 1080` to remove browser chrome and address bars.

### 3. Capture Each Tab
1. **Overview**: Click the **Overview** tab. Press `Win + Shift + S` (Windows Snipping Tool) or `Ctrl + Shift + P` -> *Capture full size screenshot* in DevTools. Save as `screenshots/01_overview_dashboard.png`.
2. **CUDA Benchmarks**: Click **CUDA Benchmarks**. Ensure charts have rendered and table is visible. Save as `screenshots/02_cuda_benchmarks.png`.
3. **PyTorch Training**: Click **PyTorch Training**. Ensure the warning banner and Recharts bar charts are captured. Save as `screenshots/03_pytorch_training.png`.
4. **LLM Inference**: Click **LLM Inference**. Capture token throughput and TTFT latency graphs. Save as `screenshots/04_llm_inference.png`.
5. **Live Telemetry**: Click **Live Telemetry**. Wait for an active polling interval so live temperature, power, and VRAM are visible. Save as `screenshots/05_live_telemetry.png`.
6. **Verification & Manifest**: Click **Verification & Manifest**. Ensure the "17 / 17 Manifest Files Matched" green badge and hash audit table are visible. Save as `screenshots/06_verification_manifest.png`.

---

## File Format & Optimization Guidelines
- Format: PNG (`.png`) for crisp font rendering and lossless color reproduction.
- Compression: Optionally optimize PNG images using `pngquant` or `oxipng` prior to final commit (keep file sizes under 1.5 MB per image).
