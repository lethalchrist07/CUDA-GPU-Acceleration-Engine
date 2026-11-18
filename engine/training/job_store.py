"""
Phase 12 Milestone 8: Hardened SQLite Training Job Store
Provides ACID-compliant storage with an explicit lifecycle state machine:
  QUEUED -> RUNNING -> COMPLETED / CANCELLED / FAILED / INTERRUPTED
Enforces:
- Legal state transitions and terminal state immutability
- Process ownership tracking (worker_pid, server_instance_id, last_heartbeat)
- Safe restart recovery that validates process liveness before marking INTERRUPTED
- Dynamic database path isolation via CUDA_ENGINE_DB_PATH
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/training_jobs.db")

VALID_STATUSES: Set[str] = {
    "QUEUED",
    "RUNNING",
    "CANCEL_REQUESTED",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
}

TERMINAL_STATUSES: Set[str] = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
}

ALLOWED_TRANSITIONS: Dict[str, Set[str]] = {
    "QUEUED": {"RUNNING", "CANCELLED", "INTERRUPTED"},
    "RUNNING": {"CANCEL_REQUESTED", "COMPLETED", "FAILED", "INTERRUPTED"},
    "CANCEL_REQUESTED": {"CANCELLED", "FAILED", "INTERRUPTED"},
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
    "INTERRUPTED": set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_configured_db_path() -> Path:
    env_path = os.getenv("CUDA_ENGINE_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def is_pid_alive(pid: Optional[int]) -> bool:
    """
    Check if an operating system process ID is actively executing.
    Uses Win32 API on Windows and kill-0 on POSIX.
    """
    if pid is None or pid <= 0:
        return False

    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h_proc:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(h_proc, ctypes.byref(exit_code)):
                    STILL_ACTIVE = 259
                    return exit_code.value == STILL_ACTIVE
                return False
            finally:
                kernel32.CloseHandle(h_proc)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


class JobStore:
    """
    SQLite-backed persistent store for training jobs with state-machine guards.
    Survives API restarts, prevents duplicate worker races, and guarantees
    atomic, legal status transitions.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else get_configured_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.server_instance_id = uuid.uuid4().hex[:12]
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                backend TEXT NOT NULL,
                dataset TEXT NOT NULL,
                status TEXT NOT NULL,
                epochs INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                learning_rate REAL NOT NULL,
                config_json TEXT NOT NULL,
                progress_json TEXT NOT NULL,
                metrics_json TEXT,
                error_message TEXT,
                checkpoint_path TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS job_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                checkpoint_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES training_jobs(job_id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status ON training_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON training_jobs(created_at);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_job ON job_checkpoints(job_id);
            """)

            # Automatic schema migration for process ownership and liveness
            existing_cols = {col["name"] for col in conn.execute("PRAGMA table_info(training_jobs);").fetchall()}
            if "worker_pid" not in existing_cols:
                conn.execute("ALTER TABLE training_jobs ADD COLUMN worker_pid INTEGER;")
            if "server_instance_id" not in existing_cols:
                conn.execute("ALTER TABLE training_jobs ADD COLUMN server_instance_id TEXT;")
            if "last_heartbeat" not in existing_cols:
                conn.execute("ALTER TABLE training_jobs ADD COLUMN last_heartbeat TEXT;")
            if "final_state_reason" not in existing_cols:
                conn.execute("ALTER TABLE training_jobs ADD COLUMN final_state_reason TEXT;")

            conn.commit()

    def create_job(
        self,
        job_id: str,
        model_name: str,
        backend: str,
        dataset: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        config: Dict[str, Any],
        worker_pid: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Insert a newly queued job with initial state."""
        now = utc_now()
        initial_progress = {
            "current_epoch": 0,
            "total_epochs": epochs,
            "current_step": 0,
            "total_steps": 0,
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "val_loss": 0.0,
            "val_accuracy": 0.0,
            "elapsed_time_sec": 0.0,
            "samples_processed": 0,
            "samples_per_sec": 0.0,
        }

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO training_jobs (
                    job_id, model_name, backend, dataset, status,
                    epochs, batch_size, learning_rate, config_json,
                    progress_json, metrics_json, error_message, checkpoint_path,
                    created_at, started_at, completed_at, worker_pid,
                    server_instance_id, last_heartbeat, final_state_reason
                ) VALUES (?, ?, ?, ?, 'QUEUED', ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL, NULL, ?, ?, ?, NULL)
                """,
                (
                    job_id,
                    model_name,
                    backend,
                    dataset,
                    epochs,
                    batch_size,
                    learning_rate,
                    json.dumps(config),
                    json.dumps(initial_progress),
                    now,
                    worker_pid,
                    self.server_instance_id,
                    now,
                ),
            )
            conn.commit()

        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Failed to retrieve newly created job {job_id}")
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def list_jobs(
        self,
        status: Optional[str] = None,
        backend: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM training_jobs"
        params: List[Any] = []
        conditions: List[str] = []

        if status:
            conditions.append("status = ?")
            params.append(status.upper())
        if backend:
            conditions.append("backend = ?")
            params.append(backend.upper())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            cur = conn.execute(query, params)
            return [self._row_to_dict(row) for row in cur.fetchall()]

    def update_job_status(
        self,
        job_id: str,
        status: str,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        error_message: Optional[str] = None,
        final_state_reason: Optional[str] = None,
        worker_pid: Optional[int] = None,
    ) -> bool:
        """
        Transition job status with legal state-machine enforcement.
        Terminal states (COMPLETED, FAILED, CANCELLED, INTERRUPTED) cannot be overwritten.
        """
        new_status = status.upper()
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of: {VALID_STATUSES}")

        now = utc_now()
        with self._connection() as conn:
            cur = conn.execute("SELECT status FROM training_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return False

            current_status = row["status"]
            if current_status in TERMINAL_STATUSES:
                logger.warning(
                    f"Refusing illegal transition for job {job_id}: "
                    f"current state '{current_status}' is terminal, cannot transition to '{new_status}'."
                )
                return False

            allowed = ALLOWED_TRANSITIONS.get(current_status, set())
            if new_status not in allowed:
                logger.warning(
                    f"Illegal state transition for job {job_id}: {current_status} -> {new_status}."
                )
                return False

            updates = ["status = ?", "last_heartbeat = ?"]
            params: List[Any] = [new_status, now]

            if started_at:
                updates.append("started_at = ?")
                params.append(started_at)
            if completed_at:
                updates.append("completed_at = ?")
                params.append(completed_at)
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)
            if final_state_reason is not None:
                updates.append("final_state_reason = ?")
                params.append(final_state_reason)
            if worker_pid is not None:
                updates.append("worker_pid = ?")
                params.append(worker_pid)

            params.append(job_id)
            sql = f"UPDATE training_jobs SET {', '.join(updates)} WHERE job_id = ? AND status NOT IN ('COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED')"
            res = conn.execute(sql, params)
            conn.commit()
            return res.rowcount > 0

    def update_job_progress(self, job_id: str, progress: Dict[str, Any]) -> bool:
        """Update live progress payload and timestamp heartbeat."""
        now = utc_now()
        with self._connection() as conn:
            res = conn.execute(
                """
                UPDATE training_jobs
                SET progress_json = ?, last_heartbeat = ?
                WHERE job_id = ?
                """,
                (json.dumps(progress), now, job_id),
            )
            conn.commit()
            return res.rowcount > 0

    def update_heartbeat(self, job_id: str) -> bool:
        now = utc_now()
        with self._connection() as conn:
            res = conn.execute(
                "UPDATE training_jobs SET last_heartbeat = ? WHERE job_id = ?",
                (now, job_id),
            )
            conn.commit()
            return res.rowcount > 0

    def complete_job(
        self,
        job_id: str,
        metrics: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
    ) -> bool:
        """
        Record COMPLETED only after training and validation successfully finish.
        Refuses to overwrite terminal states.
        """
        now = utc_now()
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT status, epochs, progress_json FROM training_jobs WHERE job_id = ?",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return False

            current_status = row["status"]
            if current_status in TERMINAL_STATUSES:
                logger.warning(
                    f"complete_job called on job {job_id} which is already in terminal state '{current_status}'. Overwrite refused."
                )
                return False

            total_epochs = row["epochs"] or 1
            try:
                p = json.loads(row["progress_json"]) if row["progress_json"] else {}
            except Exception:
                p = {}

            p["epoch"] = total_epochs
            p["current_epoch"] = total_epochs
            p["total_epochs"] = total_epochs

            # Distinguish step-based transformer models from epoch-based vision models
            opt_steps = (
                metrics.get("optimizer_steps_completed")
                or metrics.get("global_steps")
                or (len(metrics.get("step_records", [])) if metrics.get("step_records") else None)
            )
            if opt_steps is not None:
                # Step-based optimizer progress
                p["step"] = opt_steps
                p["current_step"] = opt_steps
                p["total_steps"] = opt_steps
                p["optimizer_steps_completed"] = opt_steps
                microbatches = (
                    metrics.get("microbatches_processed")
                    or p.get("microbatches_processed")
                    or p.get("batch")
                )
                if microbatches:
                    p["batch"] = microbatches
                    p["microbatches_processed"] = microbatches
                    p["total_batches"] = microbatches
            else:
                # Epoch-based vision progress (CIFAR10 MLP / ResNet)
                total_batches = p.get("total_batches") or p.get("total_steps") or 1
                p["batch"] = total_batches
                p["current_step"] = total_batches
                p["total_batches"] = total_batches
                p["total_steps"] = total_batches

            if "final_train_loss" in metrics:
                p["train_loss"] = metrics["final_train_loss"]
            if "best_val_acc" in metrics:
                p["val_accuracy"] = metrics["best_val_acc"]
                p["val_acc"] = metrics["best_val_acc"]
            if "samples_per_sec" in metrics:
                p["samples_per_sec"] = metrics["samples_per_sec"]
            if "tokens_per_sec" in metrics:
                p["tokens_per_sec"] = metrics["tokens_per_sec"]
            if "final_val_loss" in metrics:
                p["val_loss"] = metrics["final_val_loss"]
            if "final_val_perplexity" in metrics:
                p["val_perplexity"] = metrics["final_val_perplexity"]
            if "physical_vram_mb" in metrics:
                p["physical_vram_mb"] = metrics["physical_vram_mb"]
            if "gpu_temperature_c" in metrics:
                p["gpu_temperature_c"] = metrics["gpu_temperature_c"]
            if "gpu_utilization_pct" in metrics:
                p["gpu_utilization_pct"] = metrics["gpu_utilization_pct"]

            updated_progress = json.dumps(p)

            res = conn.execute(
                """
                UPDATE training_jobs
                SET status = 'COMPLETED',
                    metrics_json = ?,
                    checkpoint_path = ?,
                    progress_json = ?,
                    final_state_reason = 'Training and validation completed successfully.',
                    last_heartbeat = ?,
                    completed_at = ?
                WHERE job_id = ? AND status = 'RUNNING'
                """,
                (json.dumps(metrics), checkpoint_path, updated_progress, now, now, job_id),
            )
            conn.commit()
            return res.rowcount > 0

    def fail_job(
        self,
        job_id: str,
        error_message: str,
        structured_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record FAILED state with structured diagnostic context.
        Refuses to overwrite terminal states.
        """
        now = utc_now()
        with self._connection() as conn:
            cur = conn.execute("SELECT status FROM training_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row or row["status"] in TERMINAL_STATUSES:
                return False

            diag_str = json.dumps(structured_diagnostics) if structured_diagnostics else error_message
            res = conn.execute(
                """
                UPDATE training_jobs
                SET status = 'FAILED',
                    error_message = ?,
                    final_state_reason = ?,
                    last_heartbeat = ?,
                    completed_at = ?
                WHERE job_id = ? AND status NOT IN ('COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED')
                """,
                (error_message, diag_str, now, now, job_id),
            )
            conn.commit()
            return res.rowcount > 0

    def cancel_job(self, job_id: str, reason: str = "Client requested cancellation.") -> bool:
        """
        Record CANCELLED state.
        Refuses to overwrite terminal states.
        """
        now = utc_now()
        with self._connection() as conn:
            cur = conn.execute("SELECT status FROM training_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row or row["status"] in TERMINAL_STATUSES:
                return False

            res = conn.execute(
                """
                UPDATE training_jobs
                SET status = 'CANCELLED',
                    final_state_reason = ?,
                    last_heartbeat = ?,
                    completed_at = ?
                WHERE job_id = ? AND status IN ('QUEUED', 'RUNNING', 'CANCEL_REQUESTED')
                """,
                (reason, now, now, job_id),
            )
            conn.commit()
            return res.rowcount > 0

    def request_cancellation(self, job_id: str) -> bool:
        """Transition active job to intermediate CANCEL_REQUESTED state."""
        now = utc_now()
        with self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE training_jobs
                SET status = 'CANCEL_REQUESTED', last_heartbeat = ?
                WHERE job_id = ? AND status IN ('QUEUED', 'RUNNING')
                """,
                (now, job_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def add_checkpoint(
        self,
        job_id: str,
        epoch: int,
        accuracy: float,
        checkpoint_path: str,
        size_bytes: int,
        sha256: str,
    ) -> None:
        now = utc_now()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO job_checkpoints (
                    job_id, epoch, accuracy, checkpoint_path, size_bytes, sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, epoch, accuracy, checkpoint_path, size_bytes, sha256, now),
            )
            conn.commit()

    def get_job_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT * FROM job_checkpoints WHERE job_id = ? ORDER BY epoch ASC",
                (job_id,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "job_id": r["job_id"],
                    "epoch": r["epoch"],
                    "accuracy": r["accuracy"],
                    "checkpoint_path": r["checkpoint_path"],
                    "size_bytes": r["size_bytes"],
                    "sha256": r["sha256"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def recover_interrupted_jobs(self) -> int:
        """
        Audit and recover orphaned jobs after an actual server termination.
        CRITICAL SAFETY RULE:
        Only marks a job as INTERRUPTED if its recorded worker PID is confirmed DEAD
        in the operating system. If the worker PID is alive, leaves the job intact.
        """
        now = utc_now()
        recovered_count = 0

        with self._connection() as conn:
            cur = conn.execute(
                "SELECT job_id, status, worker_pid, server_instance_id, last_heartbeat FROM training_jobs WHERE status IN ('RUNNING', 'CANCEL_REQUESTED')"
            )
            candidates = cur.fetchall()

            current_pid = os.getpid()
            for cand in candidates:
                job_id = cand["job_id"]
                pid = cand["worker_pid"]
                cand_instance_id = cand["server_instance_id"]

                # Liveness check:
                # 1. If PID belongs to another process and is alive, DO NOT interrupt (e.g. running production server)
                if pid and pid != current_pid and is_pid_alive(pid):
                    logger.info(
                        f"Job {job_id} is actively executing in alive worker PID {pid}. Skipping recovery."
                    )
                    continue

                # 2. If PID is our current process and belongs to THIS server instance, DO NOT interrupt
                if pid and pid == current_pid and cand_instance_id == self.server_instance_id:
                    logger.info(
                        f"Job {job_id} is owned by current server instance {self.server_instance_id}. Skipping recovery."
                    )
                    continue

                # Process is confirmed dead, unattached, or from a prior server instance
                conn.execute(
                    """
                    UPDATE training_jobs
                    SET status = 'INTERRUPTED',
                        error_message = ?,
                        final_state_reason = 'Worker process terminated before job completion.',
                        last_heartbeat = ?,
                        completed_at = ?
                    WHERE job_id = ? AND status IN ('RUNNING', 'CANCEL_REQUESTED')
                    """,
                    (
                        f"Job was interrupted by server restart or worker process terminated before completion (worker PID: {pid or 'unknown'}).",
                        now,
                        now,
                        job_id,
                    ),
                )
                recovered_count += 1
                logger.warning(
                    f"Recovered orphaned job {job_id} (PID {pid}, instance {cand_instance_id}). Status set to INTERRUPTED."
                )

            conn.commit()

        return recovered_count

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        keys = row.keys()
        return {
            "job_id": row["job_id"],
            "model_name": row["model_name"],
            "backend": row["backend"],
            "dataset": row["dataset"],
            "status": row["status"],
            "epochs": row["epochs"],
            "batch_size": row["batch_size"],
            "learning_rate": row["learning_rate"],
            "config": json.loads(row["config_json"]) if row["config_json"] else {},
            "progress": json.loads(row["progress_json"]) if row["progress_json"] else {},
            "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else None,
            "error_message": row["error_message"],
            "checkpoint_path": row["checkpoint_path"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "worker_pid": row["worker_pid"] if "worker_pid" in keys else None,
            "server_instance_id": row["server_instance_id"] if "server_instance_id" in keys else None,
            "last_heartbeat": row["last_heartbeat"] if "last_heartbeat" in keys else None,
            "final_state_reason": row["final_state_reason"] if "final_state_reason" in keys else None,
        }
