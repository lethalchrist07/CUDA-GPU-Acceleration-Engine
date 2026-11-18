"""
Scientific Evaluation Framework for Transformer Language Models.
Phase 13 Milestone 1: Metrics for held-out validation loss, perplexity, train-val divergence,
lexical diversity (Distinct-N), repetition behavior, and GPU memory telemetry.
"""

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from engine.transformer.model import GPTTransformer
from engine.transformer.tokenizer import BaseTokenizer, BPETokenizer
from engine.transformer.trainer import safe_perplexity


def compute_distinct_ngrams(tokens: List[int], n: int) -> float:
    """Computes Distinct-N ratio: unique n-grams / total n-grams."""
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_index(tokens: List[int], n: int = 3) -> float:
    """
    Computes repetition index: fraction of n-grams that occur more than once.
    High repetition index indicates degenerate text looping.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated_instances = sum(c for gram, c in counts.items() if c > 1)
    return repeated_instances / len(ngrams)


class ModelEvaluator:
    """
    Comprehensive scientific evaluation harness for decoder-only Transformers.
    """

    STANDARD_PROMPTS = [
        "The time traveler",
        "To be or not to be",
        "The mind of man",
        "In the beginning",
        "Parallel computing on GPU architecture",
    ]

    def __init__(
        self,
        model: GPTTransformer,
        tokenizer: BaseTokenizer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.is_cuda = self.device.type == "cuda"
        self.model.to(self.device)

    def evaluate_loss_and_perplexity(
        self,
        loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates cross-entropy loss and safe perplexity across a DataLoader.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        total_tokens = 0

        t0 = time.perf_counter()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break
                x = x.to(self.device, non_blocking=self.is_cuda)
                y = y.to(self.device, non_blocking=self.is_cuda)

                total_tokens += x.numel()

                with torch.amp.autocast("cuda", enabled=self.is_cuda):
                    _, loss = self.model(x, targets=y)

                if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    total_batches += 1

        eval_duration = time.perf_counter() - t0
        avg_loss = total_loss / max(1, total_batches)
        ppl = safe_perplexity(avg_loss)
        tok_per_sec = total_tokens / max(eval_duration, 1e-6)

        return {
            "loss": round(avg_loss, 4),
            "perplexity": round(ppl, 2),
            "batches_evaluated": total_batches,
            "tokens_evaluated": total_tokens,
            "eval_duration_sec": round(eval_duration, 3),
            "eval_tokens_per_sec": round(tok_per_sec, 1),
        }

    def evaluate_divergence(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        sample_batches: int = 50,
    ) -> Dict[str, Any]:
        """
        Computes training versus validation divergence to diagnose overfitting:
        Delta = L_val - L_train
        """
        train_eval = self.evaluate_loss_and_perplexity(train_loader, max_batches=sample_batches)
        val_eval = self.evaluate_loss_and_perplexity(val_loader, max_batches=sample_batches)

        delta = val_eval["loss"] - train_eval["loss"]
        gap_ratio = val_eval["loss"] / max(train_eval["loss"], 1e-6)

        status = "healthy_generalization"
        if delta > 1.5 or gap_ratio > 1.5:
            status = "severe_overfitting"
        elif delta > 0.5:
            status = "moderate_divergence"
        elif delta < -0.2:
            status = "validation_underestimation_or_distribution_shift"

        return {
            "train_loss": train_eval["loss"],
            "val_loss": val_eval["loss"],
            "train_perplexity": train_eval["perplexity"],
            "val_perplexity": val_eval["perplexity"],
            "divergence_delta": round(delta, 4),
            "generalization_gap_ratio": round(gap_ratio, 3),
            "status": status,
        }

    def evaluate_generations(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 40,
        temperature: float = 0.7,
        top_k: int = 15,
        top_p: float = 0.9,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Generates completions and computes quantitative text quality and diversity metrics.
        """
        self.model.eval()
        prompt_list = prompts or self.STANDARD_PROMPTS
        results = []

        all_distinct_1 = []
        all_distinct_2 = []
        all_distinct_3 = []
        all_repetition = []

        for p in prompt_list:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            prompt_ids = self.tokenizer.encode(p)
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                out_ids = self.model.generate(
                    idx=idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            full_tokens = out_ids[0].tolist()
            new_tokens = full_tokens[len(prompt_ids):]

            gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            full_text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)

            d1 = compute_distinct_ngrams(new_tokens, 1)
            d2 = compute_distinct_ngrams(new_tokens, 2)
            d3 = compute_distinct_ngrams(new_tokens, 3)
            rep = compute_repetition_index(new_tokens, 3)

            all_distinct_1.append(d1)
            all_distinct_2.append(d2)
            all_distinct_3.append(d3)
            all_repetition.append(rep)

            results.append({
                "prompt": p,
                "continuation": gen_text.strip(),
                "full_text": full_text.strip(),
                "generated_tokens_count": len(new_tokens),
                "distinct_1": round(d1, 3),
                "distinct_2": round(d2, 3),
                "distinct_3": round(d3, 3),
                "repetition_index_3gram": round(rep, 3),
                "is_degenerate_loop": rep > 0.50,
            })

        avg_d1 = sum(all_distinct_1) / max(1, len(all_distinct_1))
        avg_d2 = sum(all_distinct_2) / max(1, len(all_distinct_2))
        avg_d3 = sum(all_distinct_3) / max(1, len(all_distinct_3))
        avg_rep = sum(all_repetition) / max(1, len(all_repetition))

        return {
            "generation_parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
            },
            "aggregate_metrics": {
                "mean_distinct_1": round(avg_d1, 3),
                "mean_distinct_2": round(avg_d2, 3),
                "mean_distinct_3": round(avg_d3, 3),
                "mean_repetition_index": round(avg_rep, 3),
                "degenerate_completions_detected": sum(1 for r in results if r["is_degenerate_loop"]),
            },
            "samples": results,
        }

    def measure_inference_telemetry(
        self,
        batch_size: int = 1,
        seq_len: int = 256,
        warmup: int = 3,
        runs: int = 10,
    ) -> Dict[str, Any]:
        """Measures single-batch and batched inference latency, throughput, and peak VRAM."""
        dummy_input = torch.randint(0, self.tokenizer.vocab_size, (batch_size, seq_len), device=self.device)

        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_input)
            if self.is_cuda:
                torch.cuda.synchronize(self.device)
            latencies.append(time.perf_counter() - t0)

        avg_latency_ms = (sum(latencies) / len(latencies)) * 1000.0
        total_tokens_per_pass = batch_size * seq_len
        tokens_per_sec = total_tokens_per_pass / (avg_latency_ms / 1000.0)

        peak_vram_mb = 0.0
        if self.is_cuda:
            peak_vram_mb = round(torch.cuda.max_memory_allocated(self.device) / (1024**2), 2)

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "mean_forward_latency_ms": round(avg_latency_ms, 2),
            "inference_tokens_per_second": round(tokens_per_sec, 1),
            "peak_vram_mb": peak_vram_mb,
        }
