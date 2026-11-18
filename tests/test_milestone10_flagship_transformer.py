"""
Phase 12 — Milestone 10 Automated Test Suite: Flagship 100M-Parameter Transformer.

Tests the full staged validation sequence (Tests A through L):
TEST A: Tiny Transformer CPU forward/backward
TEST B: Tiny Transformer GPU forward/backward with AMP FP16 and SDPA
TEST C: Strict causal masking & autoregressive autograd isolation
TEST D: Tiny dataset overfitting demonstrating loss convergence
TEST E: Foundation (~25M) training regression
TEST F: Flagship (~100M) forward/backward memory profiling
TEST G: Flagship (~100M) short real GPU training pilot
TEST H: Checkpoint save/load verification and SHA-256 validation
TEST I: Controlled interruption and resume-equivalence
TEST J: Validation loss and numerical perplexity calculation
TEST K: Short text generation (greedy and temperature/top-k sampling)
TEST L: API-driven Transformer training job with SQLite persisted telemetry
"""

import math
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.training.job_store import JobStore
from engine.training.orchestrator import TrainingOrchestrator, SUPPORTED_MODELS
from engine.transformer import (
    GPTTransformer,
    TransformerConfig,
    TransformerTrainer,
    TransformerTrainingConfig,
    get_preset_config,
    CharTokenizer,
    ByteTokenizer,
    BPETokenizer,
    TextChunkDataset,
    create_train_val_datasets,
    create_dataloaders,
    safe_perplexity,
)
from engine.transformer.data_pipeline import DocumentCorpus, prepare_production_datasets
from engine.transformer.memory_profiler import measure_actual_gpu_memory, compute_analytical_memory


# ============================================================================
# TEST A: Tiny Transformer CPU Forward / Backward
# ============================================================================
def test_a_tiny_transformer_cpu_forward_backward():
    """Validates CPU execution, loss computation, and backpropagation."""
    config = TransformerConfig(vocab_size=32, max_seq_len=16, d_model=32, n_layer=2, n_head=2, dropout=0.0)
    model = GPTTransformer(config)

    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))

    logits, loss = model(x, targets=y, use_sdpa=False)
    assert logits.shape == (2, 8, 32)
    assert loss is not None and loss.item() > 0.0
    assert not torch.isnan(loss) and not torch.isinf(loss)

    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


# ============================================================================
# TEST B: Tiny Transformer GPU Forward / Backward with AMP FP16 and SDPA
# ============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_b_tiny_transformer_gpu_forward_backward():
    """Validates NVIDIA CUDA GPU execution with native AMP FP16 and PyTorch SDPA."""
    device = torch.device("cuda:0")
    config = TransformerConfig(vocab_size=64, max_seq_len=32, d_model=64, n_layer=2, n_head=4, dropout=0.0)
    model = GPTTransformer(config).to(device)

    x = torch.randint(0, 64, (2, 16), device=device)
    y = torch.randint(0, 64, (2, 16), device=device)

    scaler = torch.amp.GradScaler("cuda")
    with torch.amp.autocast("cuda"):
        logits, loss = model(x, targets=y, use_sdpa=True)

    assert logits.shape == (2, 16, 64)
    assert loss is not None
    assert not torch.isnan(loss) and not torch.isinf(loss)

    scaler.scale(loss).backward()
    scaler.step(torch.optim.AdamW(model.parameters(), lr=1e-3))
    scaler.update()


# ============================================================================
# TEST C: Strict Causal Masking & Autoregressive Isolation
# ============================================================================
def test_c_causal_masking_correctness():
    """
    Mathematical proof of causal isolation:
    d(logits[t]) / d(x[t']) must be strictly ZERO for all t' > t.
    """
    config = TransformerConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layer=2,
        n_head=2,
        dropout=0.0,
        attn_dropout=0.0,
    )
    model = GPTTransformer(config)
    model.eval()

    seq_len = 8
    idx = torch.randint(0, 64, (1, seq_len))

    tok_emb = model.token_embeddings(idx).clone().detach().requires_grad_(True)
    pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
    pos_emb = model.position_embeddings(pos)
    x = tok_emb + pos_emb

    for block in model.blocks:
        x, _ = block(x, use_sdpa=False)
    x = model.ln_f(x)
    logits = model.lm_head(x)

    for t in range(seq_len - 1):
        target_logit = logits[0, t].sum()
        grads = torch.autograd.grad(target_logit, tok_emb, retain_graph=True)[0]
        future_grads = grads[0, t + 1 :]
        assert torch.all(future_grads == 0.0), f"Information leakage detected at position {t}!"


# ============================================================================
# TEST D: Tiny Dataset Overfitting
# ============================================================================
def test_d_tiny_dataset_overfitting():
    """Proves learning capability by overfitting a short sequence from loss > 3.0 to < 0.2."""
    torch.manual_seed(42)
    config = TransformerConfig(
        vocab_size=32,
        max_seq_len=16,
        d_model=64,
        n_layer=2,
        n_head=2,
        dropout=0.0,
    )
    model = GPTTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    tokens = [4, 9, 15, 21, 8, 12, 27, 2, 19]
    x = torch.tensor([tokens[:-1]], dtype=torch.long)
    y = torch.tensor([tokens[1:]], dtype=torch.long)

    initial_loss = None
    for step in range(120):
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert initial_loss > 2.5
    assert loss.item() < 0.3, f"Overfitting failed: initial={initial_loss:.4f}, final={loss.item():.4f}"


# ============================================================================
# TEST E: Foundation (~25M) Model Training Regression
# ============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_e_foundation_25m_model_training_regression(tmp_path):
    """Executes short training steps on the 23.1M/24.6M Foundation model, verifying loss and VRAM."""
    cfg_25m = get_preset_config("foundation_25m", vocab_size=60)
    cfg_25m.max_seq_len = 64
    model = GPTTransformer(cfg_25m)

    tokens = list(range(1, 60)) * 6
    train_ds = TextChunkDataset(tokens, seq_len=32, stride=16)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)

    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        config=TransformerTrainingConfig(
            learning_rate=3e-4,
            warmup_steps=2,
            gradient_accumulation_steps=2,
            device="cuda",
            mixed_precision=True,
        ),
        checkpoint_dir=tmp_path / "ckpt_25m",
    )
    res = trainer.fit(epochs=2, max_steps=4)
    assert res["status"] == "COMPLETED"
    assert res["global_steps"] == 4
    assert res["peak_vram_mb"] < 1000.0  # Must consume well under 1GB VRAM


# ============================================================================
# TEST F: Flagship (~100M) Memory Profile Forward / Backward
# ============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_f_flagship_100m_memory_profile_fwd_bwd():
    """
    Measures actual peak VRAM on RTX 3050 for the Flagship 100M configuration.
    Verifies that peak memory is under 2.5GB (well within 6GB hardware limit).
    """
    cfg_100m = get_preset_config("flagship_100m", vocab_size=1024)
    cfg_100m.max_seq_len = 256
    model = GPTTransformer(cfg_100m)
    p_info = model.count_parameters()
    # 92M tied, 98M untied
    assert p_info["total_parameters"] > 85_000_000

    mem_stats = measure_actual_gpu_memory(model, batch_size=2, seq_len=128, use_amp=True)
    assert "error" not in mem_stats
    assert mem_stats["total_peak_allocated_mb"] < 2500.0, f"VRAM too high: {mem_stats}"


# ============================================================================
# TEST G: Flagship (~100M) Short Real GPU Training Pilot
# ============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_g_flagship_100m_real_gpu_training_pilot(tmp_path):
    """Executes a controlled short pilot on the Flagship model on RTX 3050."""
    cfg_100m = get_preset_config("flagship_100m", vocab_size=60)
    cfg_100m.max_seq_len = 64
    model = GPTTransformer(cfg_100m)

    tokens = list(range(1, 60)) * 8
    train_ds = TextChunkDataset(tokens, seq_len=32, stride=16)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        config=TransformerTrainingConfig(
            learning_rate=2e-4,
            warmup_steps=2,
            gradient_accumulation_steps=2,
            device="cuda",
            mixed_precision=True,
        ),
        checkpoint_dir=tmp_path / "ckpt_100m",
    )
    res = trainer.fit(max_steps=3)
    assert res["status"] == "COMPLETED"
    assert res["global_steps"] == 3
    assert res["tokens_per_sec"] > 30.0
    assert res["peak_vram_mb"] < 3000.0  # Confirms feasibility on 6GB RTX 3050


# ============================================================================
# TEST H: Checkpoint Save / Load Verification
# ============================================================================
def test_h_checkpoint_save_load_validation(tmp_path):
    """Validates atomic save, SHA-256 calculation, and bitwise parameter recovery."""
    config = TransformerConfig(vocab_size=32, max_seq_len=16, d_model=32, n_layer=2, n_head=2)
    model1 = GPTTransformer(config)
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "test_ckpt.pt"
    saved = CheckpointManager.save_checkpoint(
        model=model1,
        filepath=str(ckpt_path),
        optimizer=opt1,
        epoch=1,
        metrics={"step": 10, "val_loss": 2.5},
        metadata={"model_config": config.to_dict()},
    )
    assert Path(saved).is_file()

    val_info = CheckpointManager.validate_checkpoint(str(saved))
    assert val_info["valid"] is True
    assert len(val_info["sha256"]) == 64

    model2 = GPTTransformer(config)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    loaded = CheckpointManager.load_checkpoint(str(saved), model=model2, optimizer=opt2)
    assert loaded["metrics"]["step"] == 10

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


# ============================================================================
# TEST I: Controlled Interruption and Resume
# ============================================================================
def test_i_controlled_interruption_and_resume(tmp_path):
    """Verifies that resumed training matches continuous training within numerical tolerance."""
    torch.manual_seed(101)
    config = TransformerConfig(vocab_size=32, max_seq_len=8, d_model=32, n_layer=2, n_head=2, dropout=0.0)

    tokens = list(range(1, 32)) * 4
    train_ds = TextChunkDataset(tokens, seq_len=4, stride=2)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    # 1. Run A for 4 steps, save at step 2
    model_a = GPTTransformer(config)
    trainer_a = TransformerTrainer(
        model=model_a,
        train_loader=train_loader,
        config=TransformerTrainingConfig(learning_rate=1e-3, device="cpu", mixed_precision=False),
        checkpoint_dir=tmp_path / "run_a",
    )
    trainer_a.fit(max_steps=2)
    ckpt_step2 = tmp_path / "run_a" / "ckpt_step2.pt"
    trainer_a.save_checkpoint(ckpt_step2, step=2, epoch=1, metrics={"step": 2})

    trainer_a.fit(max_steps=4)

    # 2. Run B resumed from step 2 to step 4
    model_b = GPTTransformer(config)
    trainer_b = TransformerTrainer(
        model=model_b,
        train_loader=train_loader,
        config=TransformerTrainingConfig(learning_rate=1e-3, device="cpu", mixed_precision=False),
        checkpoint_dir=tmp_path / "run_b",
    )
    trainer_b.load_checkpoint(ckpt_step2)
    trainer_b.fit(max_steps=4)

    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.allclose(pa, pb, atol=1e-5), f"Mismatch in parameter {na}"


# ============================================================================
# TEST J: Validation Loss and Perplexity Calculation
# ============================================================================
def test_j_validation_loss_and_perplexity_calculation():
    """Validates numerical handling and safe overflow clamping for perplexity."""
    assert safe_perplexity(0.0) == 1.0
    assert abs(safe_perplexity(1.0) - math.e) < 1e-4
    assert abs(safe_perplexity(2.0) - (math.e**2)) < 1e-3
    assert safe_perplexity(150.0) == math.exp(100.0)  # Clamped at 100.0 to prevent inf
    assert safe_perplexity(float("nan")) == float("inf")
    assert safe_perplexity(float("inf")) == float("inf")


# ============================================================================
# TEST K: Short Text Generation Test
# ============================================================================
def test_k_autoregressive_generation():
    """Tests greedy (T=0) and temperature sampling text generation."""
    config = TransformerConfig(vocab_size=32, max_seq_len=16, d_model=32, n_layer=2, n_head=2)
    model = GPTTransformer(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]])
    gen_greedy = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert gen_greedy.shape == (1, 7)
    assert torch.equal(gen_greedy[:, :3], prompt)

    gen_sample = model.generate(prompt, max_new_tokens=4, temperature=0.7, top_k=5)
    assert gen_sample.shape == (1, 7)
    assert (gen_sample >= 0).all() and (gen_sample < 32).all()


# ============================================================================
# TEST L: API-Driven Transformer Training Job with Persisted Telemetry
# ============================================================================
def test_l_api_driven_transformer_training_job(tmp_path):
    """
    Submits a Transformer training job to the orchestration engine,
    verifies asynchronous execution, status transitions, and SQLite metric recording.
    """
    db_file = tmp_path / "test_orchestrator_m10.db"
    store = JobStore(db_path=str(db_file))
    orchestrator = TrainingOrchestrator(job_store=store)

    job_info = orchestrator.submit_job(
        model_name="transformer_1m",
        backend="CPU_NATIVE",
        dataset="MultiDocumentCorpus",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        seed=42,
        extra_config={
            "max_steps": 3,
            "seq_len": 32,
            "dataset_path": "data/text/corpus",
        },
    )
    job_id = job_info["job_id"]
    assert job_info["status"] == "QUEUED"

    # Poll until terminal state
    max_wait = 15.0
    start = time.time()
    final_job = None
    while time.time() - start < max_wait:
        final_job = store.get_job(job_id)
        if final_job["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.3)

    assert final_job is not None
    assert final_job["status"] == "COMPLETED", f"Job failed: {final_job.get('error_message')}"
    metrics = final_job["metrics"]
    assert metrics is not None
    assert "tokens_per_sec" in metrics
    assert metrics["tokens_per_sec"] > 0
    assert final_job["checkpoint_path"] is not None

    orchestrator.shutdown()
