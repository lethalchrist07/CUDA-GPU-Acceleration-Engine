"""
BPE Tokenizer Validation and Forensic Evaluation Suite.
Phase 13 Milestone 1: Rigorous verification of vocabulary coverage, round-trip correctness,
compression ratio, unknown-token handling, throughput, and checkpoint compatibility.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from engine.transformer.tokenizer import BaseTokenizer, BPETokenizer

logger = logging.getLogger(__name__)


class TokenizerValidator:
    """
    Forensic evaluation suite for validating production Transformer tokenizers.
    """

    BENCHMARK_PROMPTS = {
        "classical_literature": (
            "The Time Traveller was expounding a recondite matter to us. His grey eyes shone and "
            "twinkled, and his usually pale face was flushed and animated. The fire burnt brightly, "
            "and the soft radiance of the incandescent lights in the lilies of silver caught the "
            "bubbles that flashed and passed in our glasses."
        ),
        "scientific_technical": (
            "The Analytical Engine has no pretensions whatever to originate anything. It can do "
            "whatever we know how to order it to perform. It can follow analysis; but it has no "
            "power of anticipating any analytical revelations or truths."
        ),
        "cuda_programming": (
            "__global__ void matrixMulKernel(float* C, const float* A, const float* B, int wA, int wB) {\n"
            "    int tx = threadIdx.x;\n"
            "    int ty = threadIdx.y;\n"
            "    __shared__ float sA[16][16];\n"
            "    __shared__ float sB[16][16];\n"
            "    // Perform warp-level tensor core reduction\n"
            "}\n"
        ),
        "complex_unicode_and_symbols": (
            "Mathematical theorem: ∀ x ∈ ℝ, e^(iπ) + 1 = 0. "
            "Greek glyphs: α, β, γ, δ, ε, ζ, η, θ. "
            "Foreign scripts & currency: Здравствуйте! £50, €100, ¥1,000, $25.50. "
            "Quotation typography: “Double curly quotes” and ‘single curly quotes’."
        ),
    }

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer

    def validate_roundtrip(self, test_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validates exact roundtrip decode(encode(x)) == x across benchmark texts."""
        texts = test_texts or list(self.BENCHMARK_PROMPTS.values())
        failures = []
        tested_count = len(texts)

        for i, text in enumerate(texts):
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            if decoded != text:
                failures.append({
                    "index": i,
                    "expected_length": len(text),
                    "decoded_length": len(decoded),
                    "mismatch_sample": repr(decoded[:100]),
                })

        return {
            "tested_texts_count": tested_count,
            "passed": len(failures) == 0,
            "failure_count": len(failures),
            "failures": failures,
        }

    def measure_compression(self, test_texts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Measures compression ratio (bytes per token, chars per token) across domains."""
        texts = test_texts or self.BENCHMARK_PROMPTS
        domain_metrics = {}
        total_raw_bytes = 0
        total_tokens = 0
        total_chars = 0

        for domain, text in texts.items():
            raw_bytes = len(text.encode("utf-8"))
            char_count = len(text)
            tokens = self.tokenizer.encode(text)
            n_tokens = len(tokens)

            bytes_per_tok = raw_bytes / max(1, n_tokens)
            chars_per_tok = char_count / max(1, n_tokens)

            domain_metrics[domain] = {
                "raw_bytes": raw_bytes,
                "character_count": char_count,
                "token_count": n_tokens,
                "bytes_per_token": round(bytes_per_tok, 3),
                "chars_per_token": round(chars_per_tok, 3),
                "compression_ratio": round(raw_bytes / max(1, n_tokens * 2), 3),  # relative to uint16
            }

            total_raw_bytes += raw_bytes
            total_tokens += n_tokens
            total_chars += char_count

        overall_bytes_per_tok = total_raw_bytes / max(1, total_tokens)
        overall_chars_per_tok = total_chars / max(1, total_tokens)

        return {
            "overall_bytes_per_token": round(overall_bytes_per_tok, 3),
            "overall_chars_per_token": round(overall_chars_per_tok, 3),
            "total_bytes_analyzed": total_raw_bytes,
            "total_tokens_generated": total_tokens,
            "domains": domain_metrics,
        }

    def validate_unknown_token_behavior(self) -> Dict[str, Any]:
        """Tests handling of exotic characters, emojis, and unlearned multi-byte sequences."""
        stress_samples = [
            "Emoji string: 🚀 ⚡ 🤖 🔬 🧠 🌌",
            "Zero-width and non-breaking: Zero\u200bWidth\u00a0Space",
            "CJK Ideographs: 深度学习 自然语言处理 计算机视觉",
            "Arabic / Hebrew: اللغة العربية / עברית",
            "Combining diacritics: e\u0301 a\u0300 u\u0308",
        ]

        results = []
        all_passed = True
        for s in stress_samples:
            encoded = self.tokenizer.encode(s)
            decoded = self.tokenizer.decode(encoded)
            is_lossless = (decoded == s)
            if not is_lossless:
                all_passed = False
            # Count how many tokens are base byte tokens (ID 4..259)
            base_byte_count = sum(1 for t in encoded if 4 <= t < 260)
            results.append({
                "sample": s,
                "token_count": len(encoded),
                "base_byte_tokens": base_byte_count,
                "subword_tokens": len(encoded) - base_byte_count,
                "lossless": is_lossless,
            })

        return {
            "all_lossless": all_passed,
            "oov_rate_percent": 0.0,  # UTF-8 byte fallback guarantees 0% true OOV
            "samples_tested": results,
        }

    def measure_throughput(self, target_chars: int = 200_000, warmup_runs: int = 3) -> Dict[str, Any]:
        """Measures encoding and decoding throughput in tokens/sec and characters/sec."""
        seed_text = (
            "The quick brown fox jumps over the lazy dog. Causal language modeling on NVIDIA GPU "
            "architecture with PyTorch SDPA, flash attention, and optimized BPE tokenization. "
        )
        repeats = max(1, target_chars // len(seed_text))
        bench_text = seed_text * repeats
        actual_chars = len(bench_text)
        actual_bytes = len(bench_text.encode("utf-8"))

        # Warmup
        for _ in range(warmup_runs):
            _ = self.tokenizer.encode(seed_text)

        # Benchmark encode
        t0 = time.perf_counter()
        token_ids = self.tokenizer.encode(bench_text)
        t_enc = time.perf_counter() - t0
        enc_tokens_per_sec = len(token_ids) / max(t_enc, 1e-6)
        enc_chars_per_sec = actual_chars / max(t_enc, 1e-6)
        enc_mb_per_sec = (actual_bytes / (1024**2)) / max(t_enc, 1e-6)

        # Benchmark decode
        t0 = time.perf_counter()
        _ = self.tokenizer.decode(token_ids)
        t_dec = time.perf_counter() - t0
        dec_tokens_per_sec = len(token_ids) / max(t_dec, 1e-6)
        dec_chars_per_sec = actual_chars / max(t_dec, 1e-6)
        dec_mb_per_sec = (actual_bytes / (1024**2)) / max(t_dec, 1e-6)

        return {
            "benchmark_chars": actual_chars,
            "benchmark_tokens": len(token_ids),
            "encode_time_sec": round(t_enc, 4),
            "encode_tokens_per_sec": round(enc_tokens_per_sec, 1),
            "encode_chars_per_sec": round(enc_chars_per_sec, 1),
            "encode_mb_per_sec": round(enc_mb_per_sec, 2),
            "decode_time_sec": round(t_dec, 4),
            "decode_tokens_per_sec": round(dec_tokens_per_sec, 1),
            "decode_chars_per_sec": round(dec_chars_per_sec, 1),
            "decode_mb_per_sec": round(dec_mb_per_sec, 2),
        }

    def validate_checkpoint_compatibility(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Verifies checkpoint weight compatibility with this tokenizer's vocabulary size."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            return {
                "compatible": False,
                "error": f"Checkpoint file not found: {ckpt_path}",
            }

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            meta = ckpt.get("metadata", {})
            model_cfg = meta.get("model_config", {})
            state_dict = ckpt.get("model_state_dict", {})

            cfg_vocab = model_cfg.get("vocab_size")
            emb_weight = state_dict.get("token_embeddings.weight")
            lm_head_weight = state_dict.get("lm_head.weight")

            emb_vocab = emb_weight.shape[0] if emb_weight is not None else None
            head_vocab = lm_head_weight.shape[0] if lm_head_weight is not None else None

            is_compat = (
                self.tokenizer.vocab_size == cfg_vocab
                and self.tokenizer.vocab_size == emb_vocab
                and (head_vocab is None or self.tokenizer.vocab_size == head_vocab)
            )

            return {
                "compatible": is_compat,
                "checkpoint_path": str(ckpt_path),
                "tokenizer_vocab_size": self.tokenizer.vocab_size,
                "model_config_vocab_size": cfg_vocab,
                "embedding_matrix_shape": list(emb_weight.shape) if emb_weight is not None else None,
                "lm_head_shape": list(lm_head_weight.shape) if lm_head_weight is not None else None,
                "step": ckpt.get("metrics", {}).get("step"),
                "val_loss": ckpt.get("metrics", {}).get("val_loss"),
                "val_perplexity": ckpt.get("metrics", {}).get("val_perplexity"),
            }
        except Exception as exc:
            return {
                "compatible": False,
                "error": f"Failed to load checkpoint: {exc}",
            }

    def run_full_audit(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Runs the complete end-to-end tokenizer audit and returns a structured report."""
        roundtrip_res = self.validate_roundtrip()
        compression_res = self.measure_compression()
        oov_res = self.validate_unknown_token_behavior()
        throughput_res = self.measure_throughput()

        compat_res = None
        if checkpoint_path:
            compat_res = self.validate_checkpoint_compatibility(checkpoint_path)

        report = {
            "tokenizer_type": self.tokenizer.__class__.__name__,
            "vocab_size": self.tokenizer.vocab_size,
            "merges_count": len(self.tokenizer.merges) if hasattr(self.tokenizer, "merges") else 0,
            "roundtrip_validation": roundtrip_res,
            "compression_efficiency": compression_res,
            "unknown_token_behavior": oov_res,
            "throughput_benchmark": throughput_res,
            "checkpoint_compatibility": compat_res,
        }
        return report
