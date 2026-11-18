"""
Phase 10 Stage 2 Milestone 5: Training Orchestrator & Job Scheduler
Coordinates training jobs across CPU_NATIVE, GPU_NATIVE, and GPU_CUSTOM backends:
- Strict GPU mutual exclusion (at most 1 active GPU job on RTX 3050 6GB)
- Persistent SQLite state tracking (QUEUED -> RUNNING -> COMPLETED / CANCELLED / FAILED)
- Cooperative cancellation tokens and safe resource recovery
- Live progress callbacks and atomic checkpoint isolation
"""
from __future__ import annotations

import hashlib
import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from engine.checkpointing.checkpoint_manager import CheckpointManager
from engine.data.cifar10 import create_cifar10_dataloaders
from engine.models.cifar10_mlp import create_cifar10_model, get_canonical_initial_weights
from .cifar10_three_backend_trainer import ThreeBackendTrainer, TrainingCancelledException
from .job_store import JobStore, utc_now

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "cifar10_mlp": {
        "model_id": "cifar10_mlp",
        "name": "CIFAR10MLP",
        "dataset": "CIFAR-10",
        "supported_backends": ["CPU_NATIVE", "GPU_NATIVE", "GPU_CUSTOM"],
        "parameters": 3426250,
        "input_shape": [3, 32, 32],
        "classes": 10,
        "custom_cuda_supported": True,
        "description": "4-Layer Deep MLP (3072 -> 1024 -> 256 -> 64 -> 10) with custom CUDA autograd GEMM and fused bias-activations.",
    },
    "resnet18": {
        "model_id": "resnet18",
        "name": "ResNet18",
        "dataset": "CIFAR-10",
        "supported_backends": ["CPU_NATIVE", "GPU_NATIVE"],
        "parameters": 11173962,
        "input_shape": [3, 32, 32],
        "classes": 10,
        "custom_cuda_supported": False,
        "description": "Deep Residual Network (ResNet-18) using PyTorch native CPU or cuBLAS CUDA backends.",
    },
    "baseline_mlp": {
        "model_id": "baseline_mlp",
        "name": "BaselineMLP",
        "dataset": "SyntheticClassification",
        "supported_backends": ["CPU_NATIVE", "GPU_NATIVE"],
        "parameters": 465172,
        "input_shape": [128],
        "classes": 20,
        "custom_cuda_supported": False,
        "description": "3-Layer MLP for rapid synthetic benchmark verification.",
    },
    "transformer_50m": {
        "model_id": "transformer_50m",
        "name": "NEXORA GPT-86.8M",
        "dataset": "MultiDocumentCorpus",
        "task": "CausalLanguageModeling",
        "supported_backends": ["CPU_NATIVE", "GPU_NATIVE"],
        "parameters": 86825472,
        "parameters_pilot": 86825472,
        "input_shape": [256],
        "classes": 2048,
        "custom_cuda_supported": False,
        "description": "12-Layer Production Pretrained Decoder-Only Transformer (86,825,472 params, 12 layers, 12 heads, d=768) with Causal Self-Attention, AMP FP16, and SDPA.",
    },
    "transformer_100m": {
        "model_id": "transformer_100m",
        "name": "NEXORA GPT-100M",
        "dataset": "MultiDocumentCorpus",
        "task": "CausalLanguageModeling",
        "supported_backends": ["CPU_NATIVE", "GPU_NATIVE"],
        "parameters": 92133888,
        "parameters_pilot": 85940736,
        "input_shape": [1024],
        "classes": 8192,
        "custom_cuda_supported": False,
        "description": "12-Layer Flagship Decoder-Only Transformer (92.13M params tied / 98.42M untied, 85.94M pilot) with Causal Self-Attention, AMP FP16, and SDPA.",
    },
    "transformer_25m": {
        "model_id": "transformer_25m",
        "name": "FoundationTransformer25M",
        "dataset": "MultiDocumentCorpus",
        "task": "CausalLanguageModeling",
        "supported_backends": ["CPU_NATIVE", "GPU_NATIVE"],
        "parameters": 23063808,
        "input_shape": [512],
        "classes": 4096,
        "custom_cuda_supported": False,
        "description": "12-Layer Foundation Decoder-Only Transformer (23.1M params tied / 24.6M untied) with Causal Self-Attention and AMP FP16.",
    },
    "transformer_1m": {
        "model_id": "transformer_1m",
        "name": "SmokeTransformer1M",
        "dataset": "MultiDocumentCorpus",
        "task": "CausalLanguageModeling",
        "supported_backends": ["CPU_NATIVE", "GPU_NATIVE"],
        "parameters": 957184,
        "input_shape": [256],
        "classes": 1024,
        "custom_cuda_supported": False,
        "description": "4-Layer Smoke-Test Decoder-Only Transformer (~1.0M params) for rapid verification.",
    },
}


class TrainingOrchestrator:
    """
    Central orchestration engine for AI model training.
    Enforces GPU resource safety, schedules queued workloads, tracks live progress,
    and manages atomic checkpoints.
    """

    _instance: Optional[TrainingOrchestrator] = None
    _init_lock = threading.Lock()

    def __init__(
        self,
        job_store: Optional[JobStore] = None,
        checkpoints_base_dir: Optional[Path] = None,
    ):
        self.job_store = job_store or JobStore()
        if checkpoints_base_dir is None:
            self.checkpoints_base_dir = REPO_ROOT / "checkpoints/training_jobs"
        else:
            p = Path(checkpoints_base_dir)
            self.checkpoints_base_dir = p if p.is_absolute() else (REPO_ROOT / p)
        self.checkpoints_base_dir.mkdir(parents=True, exist_ok=True)

        self.queue: queue.Queue[str] = queue.Queue()
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.active_jobs_lock = threading.Lock()

        # Strict GPU mutual exclusion lock
        self.gpu_lock = threading.Lock()
        self.active_gpu_job_id: Optional[str] = None

        # Background scheduler thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="TrainingOrchestratorScheduler",
        )
        self.scheduler_thread.start()

        # Audit and recover any interrupted jobs from previous server runs
        self.job_store.recover_interrupted_jobs()

    @classmethod
    def get_instance(cls, job_store: Optional[JobStore] = None) -> TrainingOrchestrator:
        with cls._init_lock:
            if cls._instance is None or not getattr(cls._instance, "is_running", False):
                cls._instance = TrainingOrchestrator(job_store=job_store)
            return cls._instance

    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        return list(SUPPORTED_MODELS.values())

    def submit_job(
        self,
        model_name: str,
        backend: str,
        dataset: str = "CIFAR-10",
        epochs: int = 3,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        seed: int = 42,
        max_train_samples: Optional[int] = 10000,
        max_val_samples: Optional[int] = 2000,
        max_test_samples: Optional[int] = 2000,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate, register, and queue a new training job."""
        norm_model = model_name.lower().replace("-", "_")
        if norm_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. Must be one of: {list(SUPPORTED_MODELS.keys())}"
            )

        model_spec = SUPPORTED_MODELS[norm_model]
        norm_backend = backend.upper()
        if norm_backend not in model_spec["supported_backends"]:
            if norm_backend == "GPU_CUSTOM" and not model_spec["custom_cuda_supported"]:
                raise ValueError(
                    f"Model '{model_name}' does not support GPU_CUSTOM (custom CUDA autograd). "
                    f"Custom CUDA kernels are currently implemented for MLP only. "
                    f"ResNet-18 supports CPU_NATIVE and GPU_NATIVE."
                )
            raise ValueError(
                f"Unsupported backend '{backend}' for model '{model_name}'. "
                f"Supported: {model_spec['supported_backends']}"
            )

        if epochs < 1 or epochs > 50:
            raise ValueError(f"Epochs must be between 1 and 50 (got {epochs})")
        if batch_size < 1 or batch_size > 512:
            raise ValueError(f"Batch size must be between 1 and 512 (got {batch_size})")
        if learning_rate <= 0 or learning_rate > 1.0:
            raise ValueError(f"Learning rate must be between 1e-6 and 1.0 (got {learning_rate})")

        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Apply adaptive optimizations if requested
        if extra_config and (extra_config.get("adaptive_mode") or extra_config.get("auto_batch_size")):
            from engine.training.adaptive_engine import AdaptiveTrainingEngine
            adaptive_engine = AdaptiveTrainingEngine()
            rec = adaptive_engine.recommend_training_config(
                model_name=norm_model,
                backend=norm_backend,
                requested_batch_size=batch_size if not extra_config.get("auto_batch_size") else None,
                requested_precision=extra_config.get("precision"),
                seq_len=extra_config.get("seq_len"),
                gradient_accumulation_steps=extra_config.get("gradient_accumulation_steps", 1),
                target_vram_ratio=extra_config.get("target_vram_ratio", 0.85),
            )
            batch_size = rec.applied_batch_size
            extra_config["precision"] = rec.applied_precision
            extra_config["gradient_accumulation_steps"] = rec.gradient_accumulation_steps
            extra_config["adaptive_decision"] = rec.to_dict()

        config = {
            "model_id": norm_model,
            "backend": norm_backend,
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "max_test_samples": max_test_samples,
            **(extra_config or {}),
        }

        job = self.job_store.create_job(
            job_id=job_id,
            model_name=norm_model,
            backend=norm_backend,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            config=config,
        )

        with self.active_jobs_lock:
            self.active_jobs[job_id] = {
                "cancel_event": threading.Event(),
                "thread": None,
                "backend": norm_backend,
            }

        self.queue.put(job_id)
        logger.info(f"Training job {job_id} [{norm_backend} - {norm_model}] registered and queued.")
        return job

    def cancel_job(self, job_id: str) -> bool:
        """Cooperatively request safe cancellation of a queued or running job."""
        job = self.job_store.get_job(job_id)
        if not job:
            return False

        current_status = job["status"]
        if current_status in ("COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"):
            return False

        with self.active_jobs_lock:
            active = self.active_jobs.get(job_id)
            if active and "cancel_event" in active:
                active["cancel_event"].set()

        if current_status == "QUEUED":
            self.job_store.cancel_job(job_id)
            logger.info(f"Queued job {job_id} cancelled directly before execution.")
            return True

        self.job_store.request_cancellation(job_id)
        logger.info(f"Cancellation requested for running job {job_id}.")
        return True

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.job_store.get_job(job_id)

    def list_jobs(
        self,
        status: Optional[str] = None,
        backend: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return self.job_store.list_jobs(status=status, backend=backend, limit=limit)

    def get_job_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.job_store.get_job(job_id)
        if not job:
            return None
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "metrics": job["metrics"],
            "error_message": job["error_message"],
        }

    def get_job_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        return self.job_store.get_job_checkpoints(job_id)

    def _scheduler_loop(self) -> None:
        """Main dispatcher loop continuously servicing the bounded job queue."""
        while self.is_running:
            try:
                job_id = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            job = self.job_store.get_job(job_id)
            if not job or job["status"] in ("CANCELLED", "FAILED", "COMPLETED", "CANCEL_REQUESTED"):
                if job and job["status"] == "CANCEL_REQUESTED":
                    self.job_store.cancel_job(job_id)
                self.queue.task_done()
                continue

            backend = job["backend"]
            is_gpu_job = backend in ("GPU_NATIVE", "GPU_CUSTOM")

            if is_gpu_job:
                # Acquire GPU mutual exclusion lock (blocks other GPU jobs safely)
                self.gpu_lock.acquire()
                self.active_gpu_job_id = job_id

            # Launch job worker in dedicated thread
            t = threading.Thread(
                target=self._run_job_worker,
                args=(job_id, is_gpu_job),
                name=f"Worker-{job_id}",
            )
            with self.active_jobs_lock:
                if job_id in self.active_jobs:
                    self.active_jobs[job_id]["thread"] = t

            t.start()
            self.queue.task_done()

    def _run_job_worker(self, job_id: str, is_gpu_job: bool) -> None:
        """Worker thread executing the full training lifecycle with guards and cleanup."""
        cancel_event = threading.Event()
        with self.active_jobs_lock:
            if job_id in self.active_jobs:
                cancel_event = self.active_jobs[job_id]["cancel_event"]

        job = self.job_store.get_job(job_id)
        if (
            not job
            or job["status"] in ("CANCELLED", "CANCEL_REQUESTED", "FAILED", "COMPLETED")
            or cancel_event.is_set()
        ):
            if job and job["status"] in ("QUEUED", "CANCEL_REQUESTED"):
                self.job_store.cancel_job(job_id)
            if is_gpu_job:
                self.active_gpu_job_id = None
                try:
                    self.gpu_lock.release()
                except RuntimeError:
                    pass
            with self.active_jobs_lock:
                self.active_jobs.pop(job_id, None)
            return

        cfg = job["config"]
        checkpoint_dir = self.checkpoints_base_dir / job_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "best_model.pt"

        self.job_store.update_job_status(
            job_id,
            "RUNNING",
            started_at=utc_now(),
            worker_pid=os.getpid(),
        )

        def on_progress(p: Dict[str, Any]) -> None:
            if cancel_event.is_set():
                raise TrainingCancelledException("Job was cancelled by client request.")
            self.job_store.update_job_progress(job_id, p)

        try:
            if cancel_event.is_set():
                raise TrainingCancelledException("Job was cancelled before dataloader setup.")

            norm_model = cfg.get("model_id", "cifar10_mlp")
            norm_backend = cfg.get("backend", "GPU_CUSTOM")

            # Apply adaptive training optimizations if requested
            if cfg.get("adaptive_mode") or cfg.get("auto_batch_size"):
                from engine.training.adaptive_engine import AdaptiveTrainingEngine
                adaptive_engine = AdaptiveTrainingEngine()
                rec = adaptive_engine.recommend_training_config(
                    model_name=norm_model,
                    backend=norm_backend,
                    requested_batch_size=cfg.get("batch_size") if not cfg.get("auto_batch_size") else None,
                    requested_precision=cfg.get("precision"),
                    seq_len=cfg.get("seq_len"),
                    gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
                    target_vram_ratio=cfg.get("target_vram_ratio", 0.85),
                )
                cfg["batch_size"] = rec.applied_batch_size
                cfg["precision"] = rec.applied_precision
                cfg["gradient_accumulation_steps"] = rec.gradient_accumulation_steps
                cfg["adaptive_decision"] = rec.to_dict()
                logger.info(
                    f"Adaptive training applied for job {job_id}: batch_size={rec.applied_batch_size}, "
                    f"precision={rec.applied_precision}, accum={rec.gradient_accumulation_steps}"
                )

            # Check if this is a language model job
            is_transformer = norm_model in ("transformer_100m", "transformer_25m", "transformer_1m") or (
                norm_model in SUPPORTED_MODELS and SUPPORTED_MODELS[norm_model].get("task") == "CausalLanguageModeling"
            )

            if is_transformer:
                if norm_backend == "GPU_CUSTOM":
                    raise ValueError(
                        f"Model '{norm_model}' does not support GPU_CUSTOM (custom CUDA autograd). "
                        f"Transformer models execute via PyTorch native GPU_NATIVE (Ampere SM 8.6 SDPA/AMP) or CPU_NATIVE."
                    )

                from engine.transformer import (
                    GPTTransformer,
                    TransformerConfig,
                    TransformerTrainer,
                    TransformerTrainingConfig,
                    get_preset_config,
                    CharTokenizer,
                    ByteTokenizer,
                    BPETokenizer,
                )
                from engine.transformer.data_pipeline import DocumentCorpus, prepare_production_datasets
                from engine.transformer.dataset import create_dataloaders

                device = torch.device("cuda:0" if norm_backend == "GPU_NATIVE" and torch.cuda.is_available() else "cpu")

                # Load text corpus
                raw_path = cfg.get("dataset_path") or "data/text/corpus"
                p = Path(raw_path)
                if not p.is_absolute():
                    p = REPO_ROOT / p
                if p.is_dir():
                    corpus = DocumentCorpus.from_directory(p)
                elif p.is_file():
                    corpus = DocumentCorpus.from_file(p)
                else:
                    fallback = REPO_ROOT / "data/text/tiny_shakespeare.txt"
                    if fallback.is_file():
                        corpus = DocumentCorpus.from_file(fallback)
                    else:
                        corpus = DocumentCorpus.from_file("data/text/tiny_shakespeare.txt")

                full_text = " ".join(corpus.documents)
                tok_type = str(cfg.get("tokenizer_type", "char")).lower()
                if tok_type == "bpe":
                    tok_vocab = int(cfg.get("vocab_size", 1024))
                    tokenizer = BPETokenizer.train_from_text(full_text, target_vocab_size=tok_vocab)
                elif tok_type == "byte":
                    tokenizer = ByteTokenizer()
                else:
                    tokenizer = CharTokenizer.train_from_text(full_text)

                seq_len = int(cfg.get("seq_len", 128 if norm_model == "transformer_100m" else 64))
                train_ds, val_ds, manifest = prepare_production_datasets(
                    corpus=corpus,
                    tokenizer=tokenizer,
                    seq_len=seq_len,
                    val_ratio=0.1,
                    seed=cfg.get("seed", 42),
                )
                train_loader, val_loader = create_dataloaders(
                    train_ds,
                    val_ds,
                    batch_size=max(1, int(cfg.get("batch_size", 4))),
                    seed=cfg.get("seed", 42),
                )

                if norm_model == "transformer_50m":
                    model_cfg = get_preset_config("production_86m", vocab_size=tokenizer.vocab_size)
                    model_cfg.max_seq_len = max(seq_len, model_cfg.max_seq_len)
                elif norm_model == "transformer_100m":
                    model_cfg = get_preset_config("flagship_100m", vocab_size=tokenizer.vocab_size)
                    model_cfg.max_seq_len = max(seq_len, model_cfg.max_seq_len)
                elif norm_model == "transformer_25m":
                    model_cfg = get_preset_config("foundation_25m", vocab_size=tokenizer.vocab_size)
                    model_cfg.max_seq_len = max(seq_len, model_cfg.max_seq_len)
                else:
                    model_cfg = get_preset_config("smoke_1m", vocab_size=tokenizer.vocab_size)
                    model_cfg.max_seq_len = max(seq_len, model_cfg.max_seq_len)

                if cancel_event.is_set():
                    raise TrainingCancelledException("Job was cancelled before transformer setup.")

                model = GPTTransformer(model_cfg).to(device)

                grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 2)))
                precision = str(cfg.get("precision", "amp")).lower()
                use_amp = (precision == "amp") and (device.type == "cuda")

                max_steps_cfg = cfg.get("max_steps")
                eval_interval = int(cfg.get("eval_interval_steps", 50))
                if max_steps_cfg and max_steps_cfg < eval_interval:
                    eval_interval = max(1, max_steps_cfg // 2)

                trainer_cfg = TransformerTrainingConfig(
                    learning_rate=float(cfg.get("learning_rate", 3e-4)),
                    min_learning_rate=float(cfg.get("min_learning_rate", 3e-5)),
                    gradient_accumulation_steps=grad_accum,
                    mixed_precision=use_amp,
                    device=str(device),
                    seed=int(cfg.get("seed", 42)),
                    eval_interval_steps=eval_interval,
                    dataset_fingerprint=manifest.content_hash_sha256,
                )

                trainer = TransformerTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=trainer_cfg,
                    checkpoint_dir=checkpoint_dir,
                    progress_callback=on_progress,
                    is_cancelled=cancel_event.is_set,
                )

                epochs = int(cfg.get("epochs", 1))
                max_steps = cfg.get("max_steps")
                resume_path = cfg.get("resume_checkpoint_path")
                results = trainer.fit(epochs=epochs, max_steps=max_steps, resume_from=resume_path)

            else:
                # 1. Prepare Vision DataLoaders
                train_loader, val_loader, test_loader, _ = create_cifar10_dataloaders(
                    data_dir=str(REPO_ROOT / "data"),
                    batch_size=cfg.get("batch_size", 128),
                    val_split=0.1,
                    seed=cfg.get("seed", 42),
                    augment=True,
                    max_train_samples=cfg.get("max_train_samples", 10000),
                    max_val_samples=cfg.get("max_val_samples", 2000),
                    max_test_samples=cfg.get("max_test_samples", 2000),
                    pin_memory=(is_gpu_job and torch.cuda.is_available()),
                )

                if norm_model == "resnet18":
                    if norm_backend == "GPU_CUSTOM":
                        raise ValueError(
                            "Model 'resnet18' does not support GPU_CUSTOM (custom CUDA autograd). "
                            "Custom CUDA kernels are currently implemented for MLP only."
                        )

                    from engine.models.resnet import create_resnet18
                    from engine.training.resnet_trainer import ResNetTrainer

                    device = torch.device("cuda:0" if norm_backend == "GPU_NATIVE" and torch.cuda.is_available() else "cpu")
                    torch.manual_seed(cfg.get("seed", 42))
                    model = create_resnet18(num_classes=10, cifar_stem=True).to(device)

                    if cancel_event.is_set():
                        raise TrainingCancelledException("Job was cancelled before optimizer setup.")

                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=cfg.get("learning_rate", 0.01),
                        momentum=0.9,
                        weight_decay=1e-4,
                    )

                    lr_sched_type = str(cfg.get("lr_scheduler", "none")).lower()
                    epochs = int(cfg.get("epochs", 3))
                    scheduler = None
                    if lr_sched_type == "cosine":
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                    elif lr_sched_type == "step":
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)

                    precision = str(cfg.get("precision", "float32")).lower()
                    use_amp = (precision == "amp") and (device.type == "cuda")
                    grad_accum_steps = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
                    resume_path = cfg.get("resume_checkpoint_path")

                    trainer = ResNetTrainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        criterion=torch.nn.CrossEntropyLoss(),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device=device,
                        backend=norm_backend,
                        use_amp=use_amp,
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_path=checkpoint_file,
                        gradient_accumulation_steps=grad_accum_steps,
                        seed=cfg.get("seed", 42),
                        cancel_event=cancel_event,
                        progress_callback=on_progress,
                        config=cfg,
                    )

                    results = trainer.fit(
                        epochs=epochs,
                        resume_from=resume_path,
                        save_checkpoints=True,
                    )
                else:
                    if cancel_event.is_set():
                        raise TrainingCancelledException("Job was cancelled before model setup.")

                    # 2. Prepare Canonical Initialized Model
                    canonical_weights = get_canonical_initial_weights(seed=cfg.get("seed", 42))
                    model = create_cifar10_model(
                        backend=cfg.get("backend", "GPU_CUSTOM"),
                        canonical_state_dict=canonical_weights,
                    )

                    if cancel_event.is_set():
                        raise TrainingCancelledException("Job was cancelled before optimizer setup.")

                    # 3. Prepare Optimizer
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=cfg.get("learning_rate", 0.01),
                        momentum=0.9,
                        weight_decay=1e-4,
                    )

                    # 4. Instantiate ThreeBackendTrainer
                    trainer = ThreeBackendTrainer(
                        backend=cfg.get("backend", "GPU_CUSTOM"),
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        optimizer=optimizer,
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_path=checkpoint_file,
                        seed=cfg.get("seed", 42),
                        cancel_event=cancel_event,
                        progress_callback=on_progress,
                    )

                    # 5. Execute Training Loop
                    results = trainer.fit(epochs=cfg.get("epochs", 3))

            if cancel_event.is_set() or (isinstance(results, dict) and results.get("status") == "CANCELLED"):
                raise TrainingCancelledException("Job was cancelled cooperatively.")

            # 6. Record Checkpoint Metadata if saved
            target_ckpt = checkpoint_dir / "best_model.pt"
            if not target_ckpt.is_file():
                target_ckpt = checkpoint_dir / "final_model.pt"
            if not target_ckpt.is_file() and checkpoint_file.is_file():
                target_ckpt = checkpoint_file

            if target_ckpt.is_file():
                try:
                    ckpt_val = CheckpointManager.validate_checkpoint(str(target_ckpt.resolve()))
                    ckpt_bytes = ckpt_val["size_bytes"]
                    ckpt_hash = ckpt_val["sha256"]
                except Exception as val_err:
                    logger.warning(f"Checkpoint validation warning for {target_ckpt}: {val_err}")
                    ckpt_bytes = target_ckpt.stat().st_size
                    ckpt_hash = hashlib.sha256(target_ckpt.read_bytes()).hexdigest()

                self.job_store.add_checkpoint(
                    job_id=job_id,
                    epoch=results.get("epochs") or results.get("total_epochs", 1),
                    accuracy=results.get("best_val_acc") or results.get("final_val_perplexity", 0.0),
                    checkpoint_path=str(target_ckpt.resolve()),
                    size_bytes=ckpt_bytes,
                    sha256=ckpt_hash,
                )

            # 7. Complete Job
            self.job_store.complete_job(
                job_id=job_id,
                metrics=results,
                checkpoint_path=str(target_ckpt.resolve()) if target_ckpt.is_file() else None,
            )
            logger.info(f"Training job {job_id} successfully completed.")

        except TrainingCancelledException:
            logger.warning(f"Training job {job_id} cancelled cooperatively.")
            self.job_store.cancel_job(job_id)

        except torch.cuda.OutOfMemoryError as oom:
            logger.error(f"CUDA OutOfMemory in training job {job_id}: {oom}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            import gc
            gc.collect()
            self.job_store.fail_job(
                job_id,
                "CUDA Out of Memory: Workload memory requirements exceeded available 6GB VRAM on RTX 3050 Laptop GPU. "
                "Mitigation recommendations: Enable adaptive mode (auto_batch_size=True), reduce batch size, reduce sequence length, "
                "enable gradient accumulation, or switch to CPU_NATIVE backend. Existing checkpoints remain uncorrupted.",
            )

        except Exception as exc:
            logger.error(f"Error executing training job {job_id}: {exc}", exc_info=True)
            self.job_store.fail_job(
                job_id,
                f"Training failed ({type(exc).__name__}): {str(exc)}",
            )

        finally:
            if is_gpu_job and torch.cuda.is_available():
                torch.cuda.empty_cache()

            with self.active_jobs_lock:
                self.active_jobs.pop(job_id, None)

            if is_gpu_job:
                self.active_gpu_job_id = None
                try:
                    self.gpu_lock.release()
                except RuntimeError:
                    pass

    def shutdown(self) -> None:
        self.is_running = False
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=2.0)
        with self._init_lock:
            TrainingOrchestrator._instance = None
