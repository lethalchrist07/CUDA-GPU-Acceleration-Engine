"""
scripts/run_milestone8_baseline_reproduction.py
Phase 13 Milestone 8: Stage 2 Scientific Baseline Reproduction.
Reproduces the 3-way training comparison (Config A, Config B, Config C) under strictly matched
conditions on the NVIDIA GeForce RTX 3050 6GB Laptop GPU.

Measures:
- Median, Mean, StdDev, p95 step latency
- Forward, Backward, Optimizer latencies
- End-to-end throughput (tokens/sec)
- Peak allocated & reserved VRAM
- GPU temperature and utilization
- Step loss trajectory and validation loss

Outputs: results/phase13_milestone8_baseline_benchmarks.json
"""

import copy
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.transformer.config import TransformerConfig
from engine.transformer.model import GPTTransformer
from engine.transformer.scalable_dataset import ScalableMemmapDataset
from engine.transformer.trainer import safe_perplexity
import custom_kernels.autograd as cka

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone8BaselineRepro")


class CustomFusedTransformerMLP(nn.Module):
    """Drop-in MLP replacement using Custom CUDA Fused Bias+GELU."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.d_ff = config.d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        gemm_out = torch.matmul(x, self.c_fc.weight.T)
        gemm_2d = gemm_out.view(-1, self.d_ff)
        h_2d = cka.fused_bias_activation(gemm_2d, self.c_fc.bias, "gelu")
        h = h_2d.view(B, T, self.d_ff)
        h = self.c_proj(h)
        return self.dropout(h)


def inject_custom_fused_mlp(model: GPTTransformer) -> GPTTransformer:
    for block in model.blocks:
        fused_mlp = CustomFusedTransformerMLP(model.config).to(block.mlp.c_fc.weight.device)
        with torch.no_grad():
            fused_mlp.c_fc.weight.copy_(block.mlp.c_fc.weight)
            fused_mlp.c_fc.bias.copy_(block.mlp.c_fc.bias)
            fused_mlp.c_proj.weight.copy_(block.mlp.c_proj.weight)
            fused_mlp.c_proj.bias.copy_(block.mlp.c_proj.bias)
        block.mlp = fused_mlp
    return model


def get_gpu_telemetry() -> Tuple[float, float]:
    """Query live GPU temperature and utilization via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0


def evaluate_dataset_loss(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    use_sdpa: bool,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in val_batches:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
                _, loss = model(x, targets=y, use_sdpa=use_sdpa)
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / max(1, len(val_batches))
    return round(avg_loss, 4), round(safe_perplexity(avg_loss), 2)


def benchmark_configuration(
    config_name: str,
    base_model: GPTTransformer,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    use_sdpa: bool,
    use_amp: bool,
    use_custom_fused: bool,
    steps: int = 30,
    device: torch.device = torch.device("cuda:0"),
) -> Dict[str, Any]:
    logger.info(f"Running Baseline Reproduction: {config_name} ({steps} steps)...")

    # Cloned initial state for bitwise-identical starting point
    model = copy.deepcopy(base_model).to(device)
    if use_custom_fused:
        model = inject_custom_fused_mlp(model)

    initial_val_loss, initial_val_ppl = evaluate_dataset_loss(
        model, val_batches, use_sdpa, use_amp
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Warmup
    logger.info("  Executing 2 warmup steps...")
    for i in range(2):
        x, y = train_batches[i]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
            _, loss = model(x, targets=y, use_sdpa=use_sdpa)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    step_latencies: List[float] = []
    forward_latencies: List[float] = []
    backward_latencies: List[float] = []
    optimizer_latencies: List[float] = []
    loss_curve: List[float] = []

    start_event = torch.cuda.Event(enable_timing=True)
    fwd_end_event = torch.cuda.Event(enable_timing=True)
    bwd_end_event = torch.cuda.Event(enable_timing=True)
    step_end_event = torch.cuda.Event(enable_timing=True)

    t_start_gpu, u_start_gpu = get_gpu_telemetry()

    for i in range(steps):
        x, y = train_batches[i % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)

        start_event.record()

        # Forward
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
            _, loss = model(x, targets=y, use_sdpa=use_sdpa)
        fwd_end_event.record()

        # Backward & Step
        if use_amp:
            scaler.scale(loss).backward()
            bwd_end_event.record()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            bwd_end_event.record()
            optimizer.step()
        step_end_event.record()

        torch.cuda.synchronize()

        fwd_ms = start_event.elapsed_time(fwd_end_event)
        bwd_ms = fwd_end_event.elapsed_time(bwd_end_event)
        opt_ms = bwd_end_event.elapsed_time(step_end_event)
        total_ms = start_event.elapsed_time(step_end_event)

        forward_latencies.append(fwd_ms)
        backward_latencies.append(bwd_ms)
        optimizer_latencies.append(opt_ms)
        step_latencies.append(total_ms)
        loss_curve.append(round(float(loss.item()), 4))

    t_end_gpu, u_end_gpu = get_gpu_telemetry()
    final_val_loss, final_val_ppl = evaluate_dataset_loss(
        model, val_batches, use_sdpa, use_amp
    )

    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    tokens_per_step = train_batches[0][0].size(0) * train_batches[0][0].size(1)
    median_latency_ms = float(np.median(step_latencies))
    mean_latency_ms = float(np.mean(step_latencies))
    std_latency_ms = float(np.std(step_latencies))
    p95_latency_ms = float(np.percentile(step_latencies, 95))
    jitter_pct = (std_latency_ms / mean_latency_ms) * 100.0
    throughput = (tokens_per_step / (median_latency_ms / 1000.0))

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    logger.info(f"  Result: {median_latency_ms:.2f} ms median ({p95_latency_ms:.2f} ms p95), {throughput:,.1f} tok/s, "
                f"Peak VRAM: {peak_alloc_mb:.1f} MB, Loss: {loss_curve[0]} -> {loss_curve[-1]}")

    return {
        "configuration_name": config_name,
        "use_sdpa": use_sdpa,
        "use_amp": use_amp,
        "use_custom_fused": use_custom_fused,
        "steps": steps,
        "tokens_per_step": tokens_per_step,
        "median_step_latency_ms": round(median_latency_ms, 2),
        "mean_step_latency_ms": round(mean_latency_ms, 2),
        "std_step_latency_ms": round(std_latency_ms, 2),
        "p95_step_latency_ms": round(p95_latency_ms, 2),
        "latency_jitter_pct": round(jitter_pct, 2),
        "forward_latency_ms": round(float(np.mean(forward_latencies)), 2),
        "backward_latency_ms": round(float(np.mean(backward_latencies)), 2),
        "optimizer_latency_ms": round(float(np.mean(optimizer_latencies)), 2),
        "throughput_tokens_per_sec": round(throughput, 1),
        "peak_vram_allocated_mb": round(peak_alloc_mb, 2),
        "peak_vram_reserved_mb": round(peak_res_mb, 2),
        "temperature_celsius_start": t_start_gpu,
        "temperature_celsius_end": t_end_gpu,
        "gpu_utilization_pct": u_end_gpu,
        "initial_train_loss": loss_curve[0],
        "final_train_loss": loss_curve[-1],
        "train_loss_reduction": round(loss_curve[0] - loss_curve[-1], 4),
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "val_loss_reduction": round(initial_val_loss - final_val_loss, 4),
        "initial_val_ppl": initial_val_ppl,
        "final_val_ppl": final_val_ppl,
        "loss_curve": loss_curve,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for baseline reproduction.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)
    total_vram_mb = props.total_memory / (1024 ** 2)

    logger.info("=" * 80)
    logger.info("PHASE 13 MILESTONE 8: STAGE 2 BASELINE REPRODUCTION")
    logger.info(f"Target GPU: {gpu_name} (Ampere SM 8.6, {props.multi_processor_count} SMs, {total_vram_mb:.0f} MB VRAM)")
    logger.info("=" * 80)

    # Load production dataset batches
    data_dir = PROJECT_ROOT / "data" / "text" / "production_10m"
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"

    train_ds = ScalableMemmapDataset(train_bin, seq_len=256, stride=256, vocab_size=2048)
    val_ds = ScalableMemmapDataset(val_bin, seq_len=256, stride=256, vocab_size=2048)

    # Prepare matched deterministic batches (B=16, T=256 -> 4096 tokens per step)
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i in range(35):
        x_list, y_list = [], []
        for j in range(16):
            idx = (i * 16 + j) % len(train_ds)
            x, y = train_ds[idx]
            x_list.append(x)
            y_list.append(y)
        train_batches.append((torch.stack(x_list).to(device), torch.stack(y_list).to(device)))

    for i in range(8):
        x_list, y_list = [], []
        for j in range(16):
            idx = (i * 16 + j) % len(val_ds)
            x, y = val_ds[idx]
            x_list.append(x)
            y_list.append(y)
        val_batches.append((torch.stack(x_list).to(device), torch.stack(y_list).to(device)))

    # Instantiate base model
    cfg = TransformerConfig(
        vocab_size=2048,
        max_seq_len=256,
        d_model=768,
        n_layer=12,
        n_head=12,
        d_ff=3072,
        tie_weights=True,
    )
    torch.manual_seed(42)
    base_model = GPTTransformer(cfg)

    # 1. Config A: Standard PyTorch FP32
    res_a = benchmark_configuration(
        config_name="Configuration A: Standard PyTorch Baseline (Manual Causal Attention + FP32)",
        base_model=base_model,
        train_batches=train_batches,
        val_batches=val_batches,
        use_sdpa=False,
        use_amp=False,
        use_custom_fused=False,
        steps=30,
        device=device,
    )

    # 2. Config B: Optimized PyTorch (SDPA + AMP FP16)
    res_b = benchmark_configuration(
        config_name="Configuration B: Optimized PyTorch (Native SDPA + AMP FP16)",
        base_model=base_model,
        train_batches=train_batches,
        val_batches=val_batches,
        use_sdpa=True,
        use_amp=True,
        use_custom_fused=False,
        steps=30,
        device=device,
    )

    # 3. Config C: Optimized PyTorch + Custom CUDA (SDPA + AMP + Fused Bias-GELU)
    res_c = benchmark_configuration(
        config_name="Configuration C: Optimized PyTorch + Custom CUDA (SDPA + AMP + Fused Bias-GELU)",
        base_model=base_model,
        train_batches=train_batches,
        val_batches=val_batches,
        use_sdpa=True,
        use_amp=True,
        use_custom_fused=True,
        steps=30,
        device=device,
    )

    # Speedups & VRAM savings
    base_ms = res_a["median_step_latency_ms"]
    res_a["speedup_factor"] = 1.0
    res_b["speedup_factor"] = round(base_ms / res_b["median_step_latency_ms"], 2)
    res_c["speedup_factor"] = round(base_ms / res_c["median_step_latency_ms"], 2)

    vram_a = res_a["peak_vram_allocated_mb"]
    res_a["vram_savings_pct"] = 0.0
    res_b["vram_savings_pct"] = round((1.0 - (res_b["peak_vram_allocated_mb"] / vram_a)) * 100, 1)
    res_c["vram_savings_pct"] = round((1.0 - (res_c["peak_vram_allocated_mb"] / vram_a)) * 100, 1)

    delta_c_vs_b_pct = round(((res_c["median_step_latency_ms"] - res_b["median_step_latency_ms"]) / res_b["median_step_latency_ms"]) * 100, 2)
    delta_c_vs_b_ms = round(res_c["median_step_latency_ms"] - res_b["median_step_latency_ms"], 2)

    logger.info("=" * 80)
    logger.info("BASELINE REPRODUCTION SUMMARY:")
    logger.info(f"  Config A: {res_a['median_step_latency_ms']:6.2f} ms ({res_a['p95_step_latency_ms']:6.2f} ms p95) | {res_a['throughput_tokens_per_sec']:7.1f} tok/s | {res_a['peak_vram_allocated_mb']} MB VRAM | 1.00x")
    logger.info(f"  Config B: {res_b['median_step_latency_ms']:6.2f} ms ({res_b['p95_step_latency_ms']:6.2f} ms p95) | {res_b['throughput_tokens_per_sec']:7.1f} tok/s | {res_b['peak_vram_allocated_mb']} MB VRAM | {res_b['speedup_factor']}x ({res_b['vram_savings_pct']}% saved)")
    logger.info(f"  Config C: {res_c['median_step_latency_ms']:6.2f} ms ({res_c['p95_step_latency_ms']:6.2f} ms p95) | {res_c['throughput_tokens_per_sec']:7.1f} tok/s | {res_c['peak_vram_allocated_mb']} MB VRAM | {res_c['speedup_factor']}x ({res_c['vram_savings_pct']}% saved)")
    logger.info(f"  Config C vs Config B delta: {delta_c_vs_b_ms:+.2f} ms ({delta_c_vs_b_pct:+.2f}%)")
    logger.info("=" * 80)

    summary_record = {
        "milestone": "Phase 13 Milestone 8",
        "stage": "Stage 2 - Scientific Baseline Reproduction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": gpu_name,
            "architecture": "Ampere SM 8.6",
            "sm_count": props.multi_processor_count,
            "total_vram_mb": round(total_vram_mb, 2),
        },
        "model_parameters": 86825472,
        "effective_batch_size_tokens": 4096,
        "configurations": [res_a, res_b, res_c],
        "comparative_analysis": {
            "config_b_speedup_vs_baseline": res_b["speedup_factor"],
            "config_c_speedup_vs_baseline": res_c["speedup_factor"],
            "config_c_vs_config_b_delta_ms": delta_c_vs_b_ms,
            "config_c_vs_config_b_delta_pct": delta_c_vs_b_pct,
            "vram_savings_b_vs_a_pct": res_b["vram_savings_pct"],
            "vram_savings_c_vs_a_pct": res_c["vram_savings_pct"],
        }
    }

    out_file = PROJECT_ROOT / "results" / "phase13_milestone8_baseline_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_record, f, indent=2)

    logger.info(f"Saved baseline reproduction benchmarks to: {out_file}")
    return summary_record


if __name__ == "__main__":
    main()

