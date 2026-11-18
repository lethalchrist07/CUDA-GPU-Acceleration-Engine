"""
Forensic DataLoader Performance and Memory Benchmark.
Phase 13 Milestone 1: Measures batch loading throughput, token streaming rates,
host memory footprint, and causal sequence integrity.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import ctypes
import numpy as np
import torch

# Ensure repository root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.transformer.scalable_dataset import ScalableMemmapDataset, create_scalable_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataLoaderBenchmark")


def get_process_memory_mb() -> float:
    """Retrieves current process working set RAM in MiB via Windows psapi."""
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_ulong),
                ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t)
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        func = ctypes.windll.psapi.GetProcessMemoryInfo
        func.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
        func.restype = ctypes.c_int
        if func(handle, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def run_dataloader_benchmarks():
    data_dir = PROJECT_ROOT / "data" / "text" / "pretraining_campaign"
    train_bin = data_dir / "train_tokens.bin"

    if not train_bin.is_file():
        raise FileNotFoundError(f"Binary token file not found at: {train_bin}")

    base_ram_mb = get_process_memory_mb()
    logger.info(f"Baseline Host RSS Memory: {base_ram_mb:.2f} MiB")

    # Configurations to profile
    configs_to_test = [
        {"batch_size": 2, "seq_len": 256, "num_workers": 0},
        {"batch_size": 4, "seq_len": 256, "num_workers": 0},
        {"batch_size": 8, "seq_len": 256, "num_workers": 0},
        {"batch_size": 16, "seq_len": 256, "num_workers": 0},
        {"batch_size": 4, "seq_len": 128, "num_workers": 0},
        {"batch_size": 4, "seq_len": 512, "num_workers": 0},
    ]

    benchmark_records = []
    num_batches_to_profile = 200

    logger.info(f"Executing DataLoader benchmarks ({num_batches_to_profile} batches per configuration)...")

    for cfg in configs_to_test:
        bs = cfg["batch_size"]
        sl = cfg["seq_len"]
        nw = cfg["num_workers"]

        dataset = ScalableMemmapDataset(
            bin_path=train_bin,
            seq_len=sl,
            dtype=np.uint16,
            vocab_size=2048,
        )

        loaders = create_scalable_dataloaders(
            train_dataset=dataset,
            batch_size=bs,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )
        loader = loaders["train"]

        # Warmup
        warmup_batches = 5
        loader_iter = iter(loader)
        for _ in range(warmup_batches):
            try:
                _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                _ = next(loader_iter)

        # Timed streaming
        total_tokens = 0
        total_sequences = 0
        alignment_verified = True

        t0 = time.perf_counter()
        for batch_idx in range(num_batches_to_profile):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            B, T = x.shape
            total_tokens += B * T
            total_sequences += B

            # Spot-check causal target alignment on first batch
            if batch_idx == 0:
                # y must be shifted by 1 relative to x: x[0, 1:] == y[0, :-1]
                if not torch.equal(x[0, 1:], y[0, :-1]):
                    alignment_verified = False

        duration = time.perf_counter() - t0
        current_ram_mb = get_process_memory_mb()
        ram_delta_mb = max(0.0, current_ram_mb - base_ram_mb)

        batches_per_sec = num_batches_to_profile / max(duration, 1e-6)
        sequences_per_sec = total_sequences / max(duration, 1e-6)
        tokens_per_sec = total_tokens / max(duration, 1e-6)

        record = {
            "batch_size": bs,
            "seq_len": sl,
            "num_workers": nw,
            "batches_profiled": num_batches_to_profile,
            "duration_sec": round(duration, 4),
            "batches_per_second": round(batches_per_sec, 1),
            "sequences_per_second": round(sequences_per_sec, 1),
            "tokens_per_second": round(tokens_per_sec, 1),
            "mb_per_second": round((tokens_per_sec * 2) / (1024**2), 2),  # uint16
            "peak_rss_mb": round(current_ram_mb, 2),
            "ram_delta_mb": round(ram_delta_mb, 2),
            "causal_target_alignment_verified": alignment_verified,
        }
        benchmark_records.append(record)

        logger.info(
            f"  B={bs}, T={sl}, NW={nw}: {batches_per_sec:.1f} batches/s | "
            f"{tokens_per_sec:,.0f} tok/s | RAM: {current_ram_mb:.1f} MiB (Delta: +{ram_delta_mb:.1f} MiB)"
        )

    out_file = PROJECT_ROOT / "results" / "dataloader_benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "benchmark": "ScalableMemmapDataset DataLoader Performance",
        "dataset_file": str(train_bin.relative_to(PROJECT_ROOT).as_posix()),
        "total_dataset_tokens": dataset.total_tokens,
        "dataset_file_size_bytes": train_bin.stat().st_size,
        "baseline_rss_mb": round(base_ram_mb, 2),
        "configurations": benchmark_records,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"DataLoader benchmarks successfully recorded in: {out_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_dataloader_benchmarks()
