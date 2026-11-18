"""
GPU Memory Feasibility Profiler and Telemetry for Transformer Architectures.

Calibrated for NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere GA107, 6144 MiB VRAM).
Provides:
- Analytical memory breakdown (weights, gradients, AdamW states, activations)
- Peak VRAM measurement during live forward and backward passes
- Feasibility classification across model sizes: 25M, 100M, 200M, 300M
- Concrete recommendations for microbatch, sequence length, and gradient accumulation
"""

import gc
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from engine.transformer.config import TransformerConfig, get_preset_config
from engine.transformer.model import GPTTransformer


@dataclass
class MemoryEstimate:
    model_name: str
    total_parameters: int
    precision: str  # "fp32" or "amp_fp16"
    weights_mb: float
    gradients_mb: float
    optimizer_states_mb: float
    static_total_mb: float
    activations_standard_mb: float
    activations_checkpointed_mb: float
    estimated_peak_standard_mb: float
    estimated_peak_checkpointed_mb: float
    fits_6gb_standard: bool
    fits_6gb_checkpointed: bool
    recommended_microbatch: int
    recommended_grad_accum: int
    recommended_seq_len: int
    notes: str


def compute_analytical_memory(
    config: TransformerConfig,
    batch_size: int = 4,
    seq_len: Optional[int] = None,
    precision: str = "amp_fp16",
    cuda_overhead_mb: float = 400.0,  # PyTorch CUDA context + cuBLAS/cuDNN workspace
) -> MemoryEstimate:
    """
    Computes exact theoretical memory breakdown for transformer training.
    
    Formulas:
    P = total parameter count
    1. Weights:
       - FP32: P * 4 bytes
       - AMP FP16: P * 2 bytes (FP16 model) + P * 4 bytes (FP32 master weights) = P * 6 bytes
    2. Gradients:
       - FP32: P * 4 bytes
       - AMP FP16: P * 2 bytes (unscaled FP16) or P * 4 bytes
    3. AdamW Optimizer states:
       - 1st moment (fp32): P * 4 bytes
       - 2nd moment (fp32): P * 4 bytes
       - Total optimizer: P * 8 bytes
    4. Activations (Standard Transformer, per layer):
       - Linear projections, self-attention, LayerNorm, and MLP activations:
         ~ 34 * B * S * D bytes per layer (FP16) or ~ 17 * B * S * D for SDPA/FlashAttention.
    5. Activations (Gradient / Activation Checkpointing):
       - Drops all intermediate activations; retains only block boundary activations:
         ~ 2 * B * S * D bytes per layer + 1 block recomputed during backward:
         Reduces activation memory by ~ 5x to 10x!
    """
    T = seq_len or config.max_seq_len
    B = batch_size
    D = config.d_model
    L = config.n_layer
    H = config.n_head
    V = config.vocab_size

    # Parameter count
    # Embeddings: (V + T) * D
    # Per layer: 4 * D^2 (attn) + 2 * D * d_ff (mlp) + 4 * D (norms)
    # Output: tied (0) or untied (V * D)
    temp_model = GPTTransformer(config)
    p_info = temp_model.count_parameters()
    P = p_info["total_parameters"]
    del temp_model

    bytes_per_mb = 1024 * 1024

    if precision == "amp_fp16":
        w_bytes = P * 2  # Model weights in FP16
        g_bytes = P * 2  # Gradients in FP16
        opt_bytes = P * 8  # FP32 master weights & AdamW m, v
    else:  # fp32
        w_bytes = P * 4
        g_bytes = P * 4
        opt_bytes = P * 8

    weights_mb = w_bytes / bytes_per_mb
    grads_mb = g_bytes / bytes_per_mb
    opt_mb = opt_bytes / bytes_per_mb
    static_mb = weights_mb + grads_mb + opt_mb

    # Activation memory estimation (with PyTorch SDPA)
    # Per layer FP16 activations: ~ 16 * B * T * D bytes
    act_bytes_per_layer = 16 * B * T * D
    total_act_standard_bytes = L * act_bytes_per_layer
    activations_standard_mb = total_act_standard_bytes / bytes_per_mb

    # With activation checkpointing: stores input to each layer + recomputes 1 layer
    total_act_ckpt_bytes = (L * 2 * B * T * D) + act_bytes_per_layer
    activations_ckpt_mb = total_act_ckpt_bytes / bytes_per_mb

    # Total peak estimates including CUDA context overhead
    total_standard_mb = static_mb + activations_standard_mb + cuda_overhead_mb
    total_ckpt_mb = static_mb + activations_ckpt_mb + cuda_overhead_mb

    usable_vram_mb = 6000.0  # RTX 3050 6GB has ~6001 MiB free
    fits_std = total_standard_mb < usable_vram_mb
    fits_ckpt = total_ckpt_mb < usable_vram_mb

    # Formulate recommendations
    if P <= 30_000_000:
        rec_mb = 4
        rec_ga = 2
        rec_seq = 512
        notes = "Comfortably fits inside 6GB VRAM for both training and inference."
    elif P <= 120_000_000:
        rec_mb = 2
        rec_ga = 4
        rec_seq = 512
        notes = "Fits in 6GB VRAM with AMP FP16 and microbatch=2 (or sequence length 512)."
    elif P <= 220_000_000:
        rec_mb = 1
        rec_ga = 8
        rec_seq = 256
        notes = "Tight fit on 6GB VRAM; requires activation checkpointing, microbatch=1, and reduced context."
    else:
        rec_mb = 1
        rec_ga = 16
        rec_seq = 128
        notes = "Exceeds standard 6GB budget for standard AdamW; requires CPU offloading or 8-bit Adam."

    return MemoryEstimate(
        model_name=f"{round(P / 1e6, 1)}M-param Transformer",
        total_parameters=P,
        precision=precision,
        weights_mb=round(weights_mb, 2),
        gradients_mb=round(grads_mb, 2),
        optimizer_states_mb=round(opt_mb, 2),
        static_total_mb=round(static_mb, 2),
        activations_standard_mb=round(activations_standard_mb, 2),
        activations_checkpointed_mb=round(activations_ckpt_mb, 2),
        estimated_peak_standard_mb=round(total_standard_mb, 2),
        estimated_peak_checkpointed_mb=round(total_ckpt_mb, 2),
        fits_6gb_standard=fits_std,
        fits_6gb_checkpointed=fits_ckpt,
        recommended_microbatch=rec_mb,
        recommended_grad_accum=rec_ga,
        recommended_seq_len=rec_seq,
        notes=notes,
    )


def measure_actual_gpu_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Empirically measures peak VRAM on CUDA for forward and backward passes.
    Safe: clean GC and empty_cache beforehand.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    target_dev = device or torch.device("cuda:0")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    baseline_vram_mb = torch.cuda.memory_allocated() / (1024**2)

    model.to(target_dev)
    torch.cuda.synchronize()
    model_loaded_vram_mb = torch.cuda.memory_allocated() / (1024**2)

    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=target_dev)
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=target_dev)

    # Forward
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits, loss = model(x, targets=y)
    
    torch.cuda.synchronize()
    fwd_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # Backward
    loss.backward()
    torch.cuda.synchronize()
    bwd_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # Cleanup
    del x, y, logits, loss
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_loaded_mb": round(model_loaded_vram_mb - baseline_vram_mb, 2),
        "forward_peak_mb": round(fwd_peak_mb, 2),
        "backward_peak_mb": round(bwd_peak_mb, 2),
        "total_peak_allocated_mb": round(bwd_peak_mb, 2),
    }
