"""
Scientific Evaluation CLI Script for Flagship Transformer Checkpoints.
Phase 13 Milestone 1: Evaluates validation loss, perplexity, train-val divergence,
lexical diversity (Distinct-1/2/3), repetition index, and GPU telemetry.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.transformer import GPTTransformer, TransformerConfig, BPETokenizer
from engine.transformer.evaluator import ModelEvaluator
from engine.transformer.scalable_dataset import ScalableMemmapDataset, create_scalable_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScientificEvaluator")


def run_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Evaluation Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'Host CPU'})")

    # 1. Load active tokenizer
    tok_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "tokenizer" / "bpe_vocab.json"
    tokenizer = BPETokenizer.load(tok_path)
    logger.info(f"Loaded Tokenizer: {tokenizer.vocab_size} tokens")

    # 2. Load active checkpoint
    ckpt_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "best_model.pt"
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    meta = checkpoint.get("metadata", {})
    cfg_dict = meta.get("model_config", {})
    config = TransformerConfig.from_dict(cfg_dict)
    model = GPTTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Instantiated model: {config.n_layer}L, {config.n_head}H, d_model={config.d_model}, {model.count_parameters()['total_parameters']:,} parameters")

    # 3. Load datasets
    data_dir = PROJECT_ROOT / "data" / "text" / "pretraining_campaign"
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"

    train_ds = ScalableMemmapDataset(train_bin, seq_len=config.max_seq_len, vocab_size=config.vocab_size)
    val_ds = ScalableMemmapDataset(val_bin, seq_len=config.max_seq_len, vocab_size=config.vocab_size)

    loaders = create_scalable_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=4,
        seed=42,
    )

    # 4. Execute scientific evaluation
    evaluator = ModelEvaluator(model=model, tokenizer=tokenizer, device=device)

    logger.info("Evaluating held-out validation loss and perplexity...")
    val_metrics = evaluator.evaluate_loss_and_perplexity(loaders["val"])
    logger.info(f"  Val Loss: {val_metrics['loss']:.4f} | Val Perplexity: {val_metrics['perplexity']:.2f}")

    logger.info("Evaluating training versus validation divergence...")
    div_metrics = evaluator.evaluate_divergence(loaders["train"], loaders["val"], sample_batches=30)
    logger.info(f"  Train Loss: {div_metrics['train_loss']:.4f} | Val Loss: {div_metrics['val_loss']:.4f} | Delta: {div_metrics['divergence_delta']:+.4f} ({div_metrics['status']})")

    logger.info("Evaluating generation quality and repetition behavior...")
    gen_metrics = evaluator.evaluate_generations()
    logger.info(f"  Mean Distinct-1: {gen_metrics['aggregate_metrics']['mean_distinct_1']:.3f}")
    logger.info(f"  Mean Distinct-2: {gen_metrics['aggregate_metrics']['mean_distinct_2']:.3f}")
    logger.info(f"  Mean Repetition Index: {gen_metrics['aggregate_metrics']['mean_repetition_index']:.3f}")

    logger.info("Measuring inference telemetry...")
    telemetry = evaluator.measure_inference_telemetry(batch_size=1, seq_len=256)
    logger.info(f"  Forward Latency: {telemetry['mean_forward_latency_ms']:.2f} ms | Throughput: {telemetry['inference_tokens_per_second']:.1f} tok/s | Peak VRAM: {telemetry['peak_vram_mb']:.2f} MiB")

    report = {
        "evaluation_name": "Phase 13 Milestone 1 Scientific Evaluation",
        "checkpoint_evaluated": str(ckpt_path.relative_to(PROJECT_ROOT).as_posix()),
        "checkpoint_step": checkpoint.get("metrics", {}).get("step"),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "Host CPU",
        "validation_metrics": val_metrics,
        "divergence_metrics": div_metrics,
        "generation_metrics": gen_metrics,
        "inference_telemetry": telemetry,
    }

    out_file = PROJECT_ROOT / "results" / "scientific_evaluation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Full scientific evaluation report saved to: {out_file}")


if __name__ == "__main__":
    run_evaluation()
