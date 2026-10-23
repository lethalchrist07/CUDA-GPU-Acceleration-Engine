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
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from engine.data.cifar10 import create_cifar10_dataloaders
from engine.models.cifar10_mlp import create_cifar10_model, get_canonical_initial_weights
from .cifar10_three_backend_trainer import ThreeBackendTrainer, TrainingCancelledException
from .job_store import JobStore, utc_now

logger = logging.getLogger(__name__)

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
        checkpoints_base_dir: Path = Path("checkpoints/training_jobs"),
    ):
        self.job_store = job_store or JobStore()
        self.checkpoints_base_dir = Path(checkpoints_base_dir)
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
            if cls._instance is None:
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
                    f"Custom CUDA is available for: {[m for m, s in SUPPORTED_MODELS.items() if s['custom_cuda_supported']]}"
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

        self.job_store.update_job_status(job_id, "RUNNING", started_at=utc_now())

        def on_progress(p: Dict[str, Any]) -> None:
            if cancel_event.is_set():
                raise TrainingCancelledException("Job was cancelled by client request.")
            self.job_store.update_job_progress(job_id, p)

        try:
            if cancel_event.is_set():
                raise TrainingCancelledException("Job was cancelled before dataloader setup.")

            # 1. Prepare DataLoaders
            train_loader, val_loader, test_loader, _ = create_cifar10_dataloaders(
                data_dir="data",
                batch_size=cfg.get("batch_size", 128),
                val_split=0.1,
                seed=cfg.get("seed", 42),
                augment=True,
                max_train_samples=cfg.get("max_train_samples", 10000),
                max_val_samples=cfg.get("max_val_samples", 2000),
                max_test_samples=cfg.get("max_test_samples", 2000),
                pin_memory=(is_gpu_job and torch.cuda.is_available()),
            )

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

            # 6. Record Checkpoint Metadata if saved
            if checkpoint_file.is_file():
                ckpt_bytes = checkpoint_file.stat().st_size
                ckpt_hash = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
                self.job_store.add_checkpoint(
                    job_id=job_id,
                    epoch=results["epochs"],
                    accuracy=results.get("best_val_acc", 0.0),
                    checkpoint_path=str(checkpoint_file.resolve()),
                    size_bytes=ckpt_bytes,
                    sha256=ckpt_hash,
                )

            # 7. Complete Job
            self.job_store.complete_job(
                job_id=job_id,
                metrics=results,
                checkpoint_path=str(checkpoint_file.resolve()) if checkpoint_file.is_file() else None,
            )
            logger.info(f"Training job {job_id} successfully completed.")

        except TrainingCancelledException:
            logger.warning(f"Training job {job_id} cancelled cooperatively.")
            self.job_store.cancel_job(job_id)

        except torch.cuda.OutOfMemoryError as oom:
            logger.error(f"CUDA OutOfMemory in training job {job_id}: {oom}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.job_store.fail_job(
                job_id,
                "CUDA Out of Memory: Batch memory requirements exceeded 6GB VRAM on RTX 3050 Laptop GPU. "
                "Reduce batch size or image sample count.",
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
