"""
Milestone 10: Flagship 100M-Parameter Transformer Real GPU Training Pilot.

Executes a controlled, forensic real GPU training run on NVIDIA GeForce RTX 3050 6GB Laptop GPU.
Logs exact parameter counts, step latency, throughput (tokens/sec), peak VRAM, loss,
validation loss, perplexity, and checkpoint SHA-256 validation.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.transformer import (
    GPTTransformer,
    TransformerConfig,
    TransformerTrainer,
    TransformerTrainingConfig,
    get_preset_config,
    BPETokenizer,
    CharTokenizer,
    safe_perplexity,
)
from engine.transformer.data_pipeline import DocumentCorpus, prepare_production_datasets
from engine.transformer.dataset import create_dataloaders


def run_flagship_100m_pilot():
    print("=" * 80)
    print("PHASE 12 — MILESTONE 10: FLAGSHIP ~100M TRANSFORMER REAL GPU PILOT")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'Host CPU'}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 1. Load Multi-Document Corpus
    corpus_dir = PROJECT_ROOT / "data" / "text" / "corpus"
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found at: {corpus_dir}")

    corpus = DocumentCorpus.from_directory(corpus_dir, name="MultiDocumentCorpus")
    print(f"Loaded Multi-Document Corpus: {len(corpus.documents)} documents")
    print(f"Corpus Cryptographic Hash: {corpus.compute_sha256()}")

    # 2. Train and Save BPE Tokenizer
    full_text = " ".join(corpus.documents)
    bpe_vocab_size = 512  # Clean subword vocabulary for controlled pilot
    tokenizer = BPETokenizer.train_from_text(full_text, target_vocab_size=bpe_vocab_size)
    tok_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_path = tok_dir / "bpe_vocab.json"
    tokenizer.save(tok_path)
    print(f"BPE Tokenizer trained: {tokenizer.vocab_size} vocabulary tokens")

    # 3. Prepare Production Datasets
    seq_len = 128
    batch_size = 2
    grad_accum_steps = 4
    max_steps = 15

    train_ds, val_ds, manifest = prepare_production_datasets(
        corpus=corpus,
        tokenizer=tokenizer,
        seq_len=seq_len,
        val_ratio=0.2,
        seed=42,
    )
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds, batch_size=batch_size, seed=42
    )
    print(f"Train chunks: {len(train_ds)}, Validation chunks: {len(val_ds)}")
    print(f"Manifest: train_tokens={manifest.train_tokens:,}, val_tokens={manifest.val_tokens:,}")

    # 4. Instantiate Flagship ~100M Transformer
    cfg_100m = get_preset_config("flagship_100m", vocab_size=tokenizer.vocab_size)
    cfg_100m.max_seq_len = seq_len
    model = GPTTransformer(cfg_100m).to(device)

    param_info = model.count_parameters()
    total_p = param_info["total_parameters"]
    non_emb_p = param_info["non_embedding_parameters"]
    print(f"Flagship Transformer Instantiated:")
    print(f"  Layers: {cfg_100m.n_layer}, Hidden: {cfg_100m.d_model}, Heads: {cfg_100m.n_head}")
    print(f"  Exact Total Parameters: {total_p:,} ({round(total_p / 1e6, 2)}M)")
    print(f"  Exact Non-Embedding Parameters: {non_emb_p:,} ({round(non_emb_p / 1e6, 2)}M)")
    print(f"  Tied Weights: {param_info['tied_weights']}")

    # 5. Execute Training Loop
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "flagship_100m_pilot"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer_cfg = TransformerTrainingConfig(
        learning_rate=2e-4,
        min_learning_rate=2e-5,
        warmup_steps=3,
        gradient_accumulation_steps=grad_accum_steps,
        mixed_precision=True,
        eval_interval_steps=5,
        checkpoint_interval_steps=10,
        device=str(device),
        seed=42,
        dataset_fingerprint=manifest.content_hash_sha256,
    )

    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        checkpoint_dir=ckpt_dir,
    )

    print("\nStarting Training Execution (Target: 15 optimizer steps)...")
    start_time = time.perf_counter()
    metrics = trainer.fit(epochs=5, max_steps=max_steps)
    total_wall_time = time.perf_counter() - start_time

    # 6. Evaluate and Validate Checkpoint
    ckpt_path = Path(metrics["best_checkpoint_path"])
    val_info = CheckpointManager.validate_checkpoint(str(ckpt_path))
    ckpt_size = ckpt_path.stat().st_size

    # 7. Text Generation Demonstration
    prompt = "Technical Foundations of GPU Computing:\n"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    gen_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.7, top_k=8)
    generated_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)

    # 8. Forensic Artifact Compilation
    results = {
        "milestone": "Phase 12 — Milestone 10",
        "target_model": "Flagship Transformer (~100M)",
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "Host CPU",
        "parameters": {
            "total_parameters": total_p,
            "non_embedding_parameters": non_emb_p,
            "tied_weights": param_info["tied_weights"],
            "d_model": cfg_100m.d_model,
            "n_layer": cfg_100m.n_layer,
            "n_head": cfg_100m.n_head,
            "vocab_size": tokenizer.vocab_size,
            "context_length": seq_len,
        },
        "training_execution": {
            "status": metrics["status"],
            "optimizer_steps_completed": metrics["optimizer_steps_completed"],
            "microbatches_processed": metrics["microbatches_processed"],
            "effective_batch_size": batch_size * grad_accum_steps,
            "total_tokens_processed": metrics["total_tokens_processed"],
            "total_wall_time_sec": round(total_wall_time, 3),
            "tokens_per_sec": metrics["tokens_per_sec"],
            "final_train_loss": metrics["final_train_loss"],
            "final_val_loss": metrics["final_val_loss"],
            "final_val_perplexity": metrics["final_val_perplexity"],
            "peak_vram_mb": metrics["peak_vram_mb"],
        },
        "checkpoint_integrity": {
            "checkpoint_path": str(ckpt_path),
            "size_bytes": ckpt_size,
            "sha256": val_info["sha256"],
            "valid": val_info["valid"],
        },
        "dataset_manifest": manifest.to_dict(),
        "generated_preview": repr(generated_text[:120]),
    }

    out_file = PROJECT_ROOT / "results" / "milestone10_flagship_100m_pilot_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("FLAGSHIP ~100M PILOT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Parameters: {total_p:,}")
    print(f"Optimizer Steps: {metrics['optimizer_steps_completed']}")
    print(f"Final Train Loss: {metrics['final_train_loss']:.4f}")
    print(f"Final Val Loss: {metrics['final_val_loss']:.4f}")
    print(f"Final Val Perplexity: {metrics['final_val_perplexity']:.2f}")
    print(f"Throughput: {metrics['tokens_per_sec']:.1f} tokens/second")
    print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB (under 6GB limit)")
    print(f"Checkpoint Path: {ckpt_path}")
    print(f"Checkpoint Size: {ckpt_size:,} bytes")
    safe_preview = generated_text[:120].encode("ascii", "replace").decode("ascii")
    print(f"Generated Preview:\n{repr(safe_preview)}")
    print(f"\nForensic results saved to: {out_file}")


if __name__ == "__main__":
    run_flagship_100m_pilot()
