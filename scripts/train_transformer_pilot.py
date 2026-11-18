"""
Milestone 9: Real Transformer Training Pilot Execution Script.

Executes real training pilots on NVIDIA GeForce RTX 3050 6GB Laptop GPU:
1. Small 1M model short training run
2. Foundation ~25M model training pilot
Records exact parameter counts, loss, validation loss, perplexity, throughput, peak VRAM, and checkpoint sizes.
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
    CharTokenizer,
    create_train_val_datasets,
    create_dataloaders,
)


def run_pilots():
    print("=" * 80)
    print("PHASE 12 — MILESTONE 9: REAL TRANSFORMER TRAINING PILOTS")
    print("=" * 80)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    print(f"Target Hardware: {device_name}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 1. Load Text Corpus
    data_path = PROJECT_ROOT / "data" / "text" / "tiny_shakespeare.txt"
    if not data_path.exists():
        raise FileNotFoundError(f"Training corpus not found at {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    
    print(f"Loaded corpus: {len(corpus):,} characters")

    # 2. Train and save CharTokenizer
    tokenizer = CharTokenizer.train_from_text(corpus)
    tok_dir = PROJECT_ROOT / "checkpoints" / "transformer" / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tok_dir / "vocab.json")
    print(f"Tokenizer trained: {tokenizer.vocab_size} vocabulary tokens")

    # Encode entire corpus
    encoded_tokens = tokenizer.encode(corpus)
    print(f"Total encoded tokens: {len(encoded_tokens):,}")

    results = {}

    # =========================================================================
    # PILOT 1: Small Smoke Model (1M Parameters) Real Training Run
    # =========================================================================
    print("\n" + "-" * 80)
    print("Starting Pilot 1: Small Smoke Model (1M params) on GPU")
    print("-" * 80)

    seq_len_1m = 64
    batch_size_1m = 8
    epochs_1m = 3

    train_ds_1m, val_ds_1m = create_train_val_datasets(
        encoded_tokens,
        seq_len=seq_len_1m,
        val_ratio=0.1,
        vocab_size=tokenizer.vocab_size,
    )
    train_loader_1m, val_loader_1m = create_dataloaders(
        train_ds_1m, val_ds_1m, batch_size=batch_size_1m, seed=42
    )

    cfg_1m = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len_1m,
        d_model=128,
        n_layer=4,
        n_head=4,
        dropout=0.1,
        tie_weights=True,
    )
    model_1m = GPTTransformer(cfg_1m)
    p_info_1m = model_1m.count_parameters()
    print(f"Model 1M parameters: {p_info_1m['total_parameters']:,} (non-embedding: {p_info_1m['non_embedding_parameters']:,})")

    ckpt_dir_1m = PROJECT_ROOT / "checkpoints" / "transformer" / "pilot_1m"
    ckpt_dir_1m.mkdir(parents=True, exist_ok=True)

    trainer_1m = TransformerTrainer(
        model=model_1m,
        train_loader=train_loader_1m,
        val_loader=val_loader_1m,
        config=TransformerTrainingConfig(
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=10,
            epochs=epochs_1m,
            mixed_precision=True,
            eval_interval_steps=20,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=42,
        ),
        checkpoint_dir=ckpt_dir_1m,
    )

    p1_metrics = trainer_1m.fit(epochs=epochs_1m)
    ckpt_1m_path = Path(p1_metrics["best_checkpoint_path"])
    ckpt_1m_size = ckpt_1m_path.stat().st_size if ckpt_1m_path.exists() else 0
    val_1m_info = CheckpointManager.validate_checkpoint(str(ckpt_1m_path))

    # Test generation on 1M model
    prompt_text = "First Citizen:\n"
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=trainer_1m.device)
    gen_ids = model_1m.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=10)
    generated_sample_1m = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)

    results["pilot_1m"] = {
        "model_name": "Small Smoke Transformer (1M)",
        "total_parameters": p_info_1m["total_parameters"],
        "non_embedding_parameters": p_info_1m["non_embedding_parameters"],
        "status": p1_metrics["status"],
        "epochs": epochs_1m,
        "global_steps": p1_metrics["global_steps"],
        "total_tokens_processed": p1_metrics["total_tokens_processed"],
        "wall_time_sec": p1_metrics["total_wall_time_sec"],
        "tokens_per_sec": p1_metrics["tokens_per_sec"],
        "final_train_loss": p1_metrics["final_train_loss"],
        "final_val_loss": p1_metrics["final_val_loss"],
        "final_val_perplexity": p1_metrics["final_val_perplexity"],
        "peak_vram_mb": p1_metrics["peak_vram_mb"],
        "checkpoint_path": str(ckpt_1m_path),
        "checkpoint_size_bytes": ckpt_1m_size,
        "checkpoint_sha256": val_1m_info["sha256"],
        "generated_sample": generated_sample_1m,
    }

    print("\nPilot 1 Complete:")
    print(f"  Train Loss: {p1_metrics['final_train_loss']:.4f}")
    print(f"  Val Loss: {p1_metrics['final_val_loss']:.4f}")
    print(f"  Val Perplexity: {p1_metrics['final_val_perplexity']:.2f}")
    print(f"  Throughput: {p1_metrics['tokens_per_sec']:.1f} tokens/sec")
    print(f"  Peak VRAM: {p1_metrics['peak_vram_mb']:.2f} MB")
    print(f"  Checkpoint Size: {ckpt_1m_size:,} bytes")
    print(f"  Generated Preview:\n{repr(generated_sample_1m[:100])}")

    # =========================================================================
    # PILOT 2: Foundation Model (~25M Parameters) Short Real Training Pilot
    # =========================================================================
    print("\n" + "-" * 80)
    print("Starting Pilot 2: Foundation Model (~25M params) on GPU")
    print("-" * 80)

    seq_len_25m = 128
    batch_size_25m = 4
    max_steps_25m = 25  # Controlled short pilot

    train_ds_25m, val_ds_25m = create_train_val_datasets(
        encoded_tokens,
        seq_len=seq_len_25m,
        val_ratio=0.1,
        vocab_size=tokenizer.vocab_size,
    )
    train_loader_25m, val_loader_25m = create_dataloaders(
        train_ds_25m, val_ds_25m, batch_size=batch_size_25m, seed=42
    )

    cfg_25m = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len_25m,
        d_model=384,
        n_layer=12,
        n_head=6,
        dropout=0.1,
        tie_weights=True,
    )
    model_25m = GPTTransformer(cfg_25m)
    p_info_25m = model_25m.count_parameters()
    print(f"Model 25M parameters: {p_info_25m['total_parameters']:,} (non-embedding: {p_info_25m['non_embedding_parameters']:,})")

    ckpt_dir_25m = PROJECT_ROOT / "checkpoints" / "transformer" / "pilot_25m"
    ckpt_dir_25m.mkdir(parents=True, exist_ok=True)

    trainer_25m = TransformerTrainer(
        model=model_25m,
        train_loader=train_loader_25m,
        val_loader=val_loader_25m,
        config=TransformerTrainingConfig(
            learning_rate=3e-4,
            min_learning_rate=3e-5,
            warmup_steps=5,
            gradient_accumulation_steps=2,
            mixed_precision=True,
            eval_interval_steps=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=42,
        ),
        checkpoint_dir=ckpt_dir_25m,
    )

    p2_metrics = trainer_25m.fit(max_steps=max_steps_25m)
    ckpt_25m_path = Path(p2_metrics["best_checkpoint_path"])
    ckpt_25m_size = ckpt_25m_path.stat().st_size if ckpt_25m_path.exists() else 0
    val_25m_info = CheckpointManager.validate_checkpoint(str(ckpt_25m_path))

    # Test generation on 25M model
    gen_ids_25m = model_25m.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=10)
    generated_sample_25m = tokenizer.decode(gen_ids_25m[0].tolist(), skip_special_tokens=True)

    results["pilot_25m"] = {
        "model_name": "Foundation Transformer (~25M)",
        "total_parameters": p_info_25m["total_parameters"],
        "non_embedding_parameters": p_info_25m["non_embedding_parameters"],
        "status": p2_metrics["status"],
        "global_steps": p2_metrics["global_steps"],
        "total_tokens_processed": p2_metrics["total_tokens_processed"],
        "wall_time_sec": p2_metrics["total_wall_time_sec"],
        "tokens_per_sec": p2_metrics["tokens_per_sec"],
        "final_train_loss": p2_metrics["final_train_loss"],
        "final_val_loss": p2_metrics["final_val_loss"],
        "final_val_perplexity": p2_metrics["final_val_perplexity"],
        "peak_vram_mb": p2_metrics["peak_vram_mb"],
        "checkpoint_path": str(ckpt_25m_path),
        "checkpoint_size_bytes": ckpt_25m_size,
        "checkpoint_sha256": val_25m_info["sha256"],
        "generated_sample": generated_sample_25m,
    }

    print("\nPilot 2 Complete:")
    print(f"  Train Loss: {p2_metrics['final_train_loss']:.4f}")
    print(f"  Val Loss: {p2_metrics['final_val_loss']:.4f}")
    print(f"  Val Perplexity: {p2_metrics['final_val_perplexity']:.2f}")
    print(f"  Throughput: {p2_metrics['tokens_per_sec']:.1f} tokens/sec")
    print(f"  Peak VRAM: {p2_metrics['peak_vram_mb']:.2f} MB")
    print(f"  Checkpoint Size: {ckpt_25m_size:,} bytes")
    print(f"  Generated Preview:\n{repr(generated_sample_25m[:100])}")

    # 3. Save Final Results Artifact
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "milestone9_transformer_pilot_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll pilot results written to: {out_file}")


if __name__ == "__main__":
    run_pilots()
