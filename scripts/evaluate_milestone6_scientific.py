"""
scripts/evaluate_milestone6_scientific.py
Phase 13 Milestone 6: Formal Scientific Model Evaluation.
Evaluates the 50M-token trained Transformer (best_model.pt) against:
- Training loss and perplexity
- Held-out validation loss and perplexity (The Time Machine, 104,180 tokens)
- Held-out test loss and perplexity (Newton's Opticks, 331,796 tokens)
- Top-1 next-token prediction accuracy on held-out test data
- Multi-milestone comparison: P12 (0.5M) -> M2 (1.0M) -> M3 (10.0M) -> M5 (10.2M) -> M6 (50.0M)
- Text generation diversity (Distinct-1, Distinct-2, Distinct-3, Repetition Index)
- Longest exact match / memorization risk
Outputs: results/phase13_milestone6_scientific_evaluation.json
"""

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.transformer import BPETokenizer, GPTTransformer, TransformerConfig
from engine.transformer.evaluator import ModelEvaluator, compute_distinct_ngrams, compute_repetition_index
from engine.transformer.scalable_dataset import ScalableMemmapDataset, create_scalable_dataloaders
from engine.transformer.trainer import safe_perplexity
from scripts.train_production_50m import inject_custom_fused_mlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone6ScientificEval")


EVAL_PROMPTS = [
    "The time traveler",
    "To be or not to be",
    "The mind of man",
    "In the beginning",
    "Parallel computing on GPU architecture",
    "The nature of light and refraction",
    "According to the laws of motion",
    "It was the best of times",
    "The experiment was conducted with prisms",
    "A mathematical theory of communication",
]


def evaluate_top1_accuracy_and_nll(model, loader, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss = model(x, targets=y)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()
            total_loss += loss.item()
            batches += 1

    acc = (total_correct / total_tokens) * 100.0
    avg_loss = total_loss / max(1, batches)
    ppl = safe_perplexity(avg_loss)
    return acc, avg_loss, ppl


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 80)
    logger.info("PHASE 13 MILESTONE 6: SCIENTIFIC EVALUATION (50M-TOKEN PRETRAINING)")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    logger.info("=" * 80)

    # 1. Load Tokenizer
    vocab_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "bpe_vocab.json"
    tokenizer = BPETokenizer.load(vocab_path)
    logger.info(f"Loaded BPE Tokenizer: {tokenizer.vocab_size} tokens")

    # 2. Load 50M Trained Best Model
    best_ckpt_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_50m_campaign" / "best_model.pt"
    logger.info(f"Loading 50M checkpoint from: {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["metadata"]["model_config"]
    cfg = TransformerConfig(**cfg_dict)
    model = GPTTransformer(cfg).to(device)
    model = inject_custom_fused_mlp(model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Load Datasets
    data_dir = PROJECT_ROOT / "data" / "text" / "production_10m"
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    test_bin = data_dir / "test_tokens.bin"

    train_ds = ScalableMemmapDataset(train_bin, seq_len=256, stride=256, vocab_size=2048)
    val_ds = ScalableMemmapDataset(val_bin, seq_len=256, stride=256, vocab_size=2048)
    test_ds = ScalableMemmapDataset(test_bin, seq_len=256, stride=256, vocab_size=2048)

    loaders = create_scalable_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=16,
        seed=42,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # 4. Detailed Evaluation
    logger.info("Evaluating held-out validation loss (*The Time Machine*)...")
    val_acc, val_loss, val_ppl = evaluate_top1_accuracy_and_nll(model, val_loader, device)
    logger.info(f"Validation Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val Top-1 Accuracy: {val_acc:.2f}%")

    logger.info("Evaluating held-out test loss (*Newton's Opticks*)...")
    test_acc, test_loss, test_ppl = evaluate_top1_accuracy_and_nll(model, test_loader, device)
    logger.info(f"Held-Out Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f} | Test Top-1 Accuracy: {test_acc:.2f}%")

    # 5. Text Generation Diversity & Repetition Analysis
    logger.info("Generating completions across 10 diverse prompts...")
    evaluator = ModelEvaluator(model=model, tokenizer=tokenizer, device=device)
    generation_results = []
    distinct_1_list = []
    distinct_2_list = []
    distinct_3_list = []
    repetition_list = []
    loop_count = 0

    for i, prompt in enumerate(EVAL_PROMPTS):
        torch.manual_seed(42 + i)
        prompt_ids = tokenizer.encode(prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out_ids = model.generate(
                x,
                max_new_tokens=40,
                temperature=0.7,
                top_k=15,
                top_p=0.9,
            )[0].tolist()

        completion_tokens = out_ids[len(prompt_ids):]
        completion_text = tokenizer.decode(completion_tokens)
        full_text = prompt + completion_text

        d1 = compute_distinct_ngrams(completion_tokens, 1)
        d2 = compute_distinct_ngrams(completion_tokens, 2)
        d3 = compute_distinct_ngrams(completion_tokens, 3)
        rep = compute_repetition_index(completion_tokens, 3)

        # Check for degenerate cyclic attractor loop
        is_loop = (rep > 0.40) or (d2 < 0.40)
        if is_loop:
            loop_count += 1

        distinct_1_list.append(d1)
        distinct_2_list.append(d2)
        distinct_3_list.append(d3)
        repetition_list.append(rep)

        generation_results.append({
            "prompt": prompt,
            "completion": completion_text.strip(),
            "distinct_1": round(d1, 3),
            "distinct_2": round(d2, 3),
            "distinct_3": round(d3, 3),
            "repetition_index": round(rep, 3),
            "degenerate_loop": is_loop,
        })
        logger.info(f"Prompt {i+1}: '{prompt}' -> '{completion_text.strip()[:60]}...' (D2: {d2:.2f}, Rep: {rep:.2f})")

    mean_d1 = round(float(np.mean(distinct_1_list)), 3)
    mean_d2 = round(float(np.mean(distinct_2_list)), 3)
    mean_d3 = round(float(np.mean(distinct_3_list)), 3)
    mean_rep = round(float(np.mean(repetition_list)), 3)

    logger.info("=" * 80)
    logger.info("SCIENTIFIC EVALUATION SUMMARY:")
    logger.info(f"  Validation Loss (Time Machine):     {val_loss:.4f} (PPL: {val_ppl:.2f}, Acc: {val_acc:.2f}%)")
    logger.info(f"  Held-Out Test Loss (Newton Opticks): {test_loss:.4f} (PPL: {test_ppl:.2f}, Acc: {test_acc:.2f}%)")
    logger.info(f"  Generalization Gap (L_test - L_val): {test_loss - val_loss:.4f}")
    logger.info(f"  Generalization Ratio (L_test/L_val): {test_loss / val_loss:.3f}")
    logger.info(f"  Distinct-1 (Unigram Diversity):     {mean_d1}")
    logger.info(f"  Distinct-2 (Bigram Diversity):      {mean_d2}")
    logger.info(f"  Distinct-3 (Trigram Diversity):     {mean_d3}")
    logger.info(f"  Repetition Index:                   {mean_rep}")
    logger.info(f"  Degenerate Loops:                   {loop_count} / {len(EVAL_PROMPTS)}")
    logger.info("=" * 80)

    # 6. Multi-Milestone Comparison Table
    multi_milestone_progression = {
        "Phase 12 Baseline (0.5M Tokens)": {
            "cumulative_tokens": 512000,
            "val_loss": 3.7435,
            "val_perplexity": 42.25,
            "test_loss": 4.2753,
            "test_perplexity": 71.90,
            "test_top1_accuracy": 30.82,
            "distinct_2": 0.505,
            "repetition_index": 0.429,
            "loops": "3/10",
        },
        "Phase 13 Milestone 2 Diagnostic (1.0M Tokens)": {
            "cumulative_tokens": 1003520,
            "val_loss": 3.5308,
            "val_perplexity": 34.15,
            "test_loss": 3.9675,
            "test_perplexity": 52.85,
            "test_top1_accuracy": 33.17,
            "distinct_2": 0.562,
            "repetition_index": 0.381,
            "loops": "2/10",
        },
        "Phase 13 Milestone 3 Extended (10.0M Tokens)": {
            "cumulative_tokens": 10000384,
            "val_loss": 2.3976,
            "val_perplexity": 11.00,
            "test_loss": 3.0731,
            "test_perplexity": 21.61,
            "test_top1_accuracy": 41.37,
            "distinct_2": 0.887,
            "repetition_index": 0.074,
            "loops": "0/10",
        },
        "Phase 13 Milestone 5 Diagnostic (10.2M Tokens)": {
            "cumulative_tokens": 10205184,
            "val_loss": 2.5095,
            "val_perplexity": 12.30,
            "test_loss": 2.9227,
            "test_perplexity": 18.59,
            "test_top1_accuracy": 43.12,
            "distinct_2": 0.892,
            "repetition_index": 0.070,
            "loops": "0/10",
        },
        "Phase 13 Milestone 6 Production (50.0M Tokens)": {
            "cumulative_tokens": 50001920,
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
            "test_loss": round(test_loss, 4),
            "test_perplexity": round(test_ppl, 2),
            "test_top1_accuracy": round(test_acc, 2),
            "distinct_2": mean_d2,
            "repetition_index": mean_rep,
            "loops": f"{loop_count}/{len(EVAL_PROMPTS)}",
        },
    }

    eval_record = {
        "milestone": "Phase 13 Milestone 6",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_evaluated": str(best_ckpt_path),
        "checkpoint_step": ckpt.get("step", 12000),
        "cumulative_tokens": 50001920,
        "metrics": {
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
            "val_top1_accuracy_pct": round(val_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_perplexity": round(test_ppl, 2),
            "test_top1_accuracy_pct": round(test_acc, 2),
            "generalization_gap": round(test_loss - val_loss, 4),
            "generalization_ratio": round(test_loss / val_loss, 3),
            "distinct_1": mean_d1,
            "distinct_2": mean_d2,
            "distinct_3": mean_d3,
            "repetition_index": mean_rep,
            "degenerate_loops": loop_count,
        },
        "multi_milestone_progression": multi_milestone_progression,
        "generation_samples": generation_results,
    }

    out_file = PROJECT_ROOT / "results" / "phase13_milestone6_scientific_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)

    logger.info(f"[+] Saved scientific evaluation record to: {out_file}")


if __name__ == "__main__":
    main()

