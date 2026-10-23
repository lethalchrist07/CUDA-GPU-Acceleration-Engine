"""
Phase 10 Stage 2 Milestone 5: SQLite Training Job Store
Provides persistent, ACID-compliant storage for training job lifecycles,
live batch/epoch progress, metrics, checkpoints, and server restart recovery.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/training_jobs.db")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """
    SQLite-backed persistent store for training jobs.
    Survives API restarts and enables atomic status transitions.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
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
    ) -> Dict[str, Any]:
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
                    created_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, 'QUEUED', ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL, NULL)
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
                ),
            )
            conn.commit()

        return self.get_job(job_id)  # type: ignore

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
    ) -> None:
        updates = ["status = ?"]
        params: List[Any] = [status.upper()]

        if started_at:
            updates.append("started_at = ?")
            params.append(started_at)
        if completed_at:
            updates.append("completed_at = ?")
            params.append(completed_at)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(job_id)
        sql = f"UPDATE training_jobs SET {', '.join(updates)} WHERE job_id = ?"

        with self._connection() as conn:
            conn.execute(sql, params)
            conn.commit()

    def update_job_progress(self, job_id: str, progress: Dict[str, Any]) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE training_jobs SET progress_json = ? WHERE job_id = ?",
                (json.dumps(progress), job_id),
            )
            conn.commit()

    def complete_job(
        self,
        job_id: str,
        metrics: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
    ) -> None:
        now = utc_now()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE training_jobs
                SET status = 'COMPLETED', metrics_json = ?, checkpoint_path = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(metrics), checkpoint_path, now, job_id),
            )
            conn.commit()

    def fail_job(self, job_id: str, error_message: str) -> None:
        now = utc_now()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE training_jobs
                SET status = 'FAILED', error_message = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (error_message, now, job_id),
            )
            conn.commit()

    def cancel_job(self, job_id: str) -> None:
        now = utc_now()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE training_jobs
                SET status = 'CANCELLED', completed_at = ?
                WHERE job_id = ?
                """,
                (now, job_id),
            )
            conn.commit()

    def request_cancellation(self, job_id: str) -> bool:
        with self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE training_jobs
                SET status = 'CANCEL_REQUESTED'
                WHERE job_id = ? AND status IN ('QUEUED', 'RUNNING')
                """,
                (job_id,),
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
        Audit and recover jobs left in non-terminal states during an abrupt server shutdown.
        Transitions them to INTERRUPTED and records explanation.
        """
        now = utc_now()
        with self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE training_jobs
                SET status = 'INTERRUPTED',
                    error_message = 'Job was interrupted by server restart or process termination before completion.',
                    completed_at = ?
                WHERE status IN ('RUNNING', 'CANCEL_REQUESTED')
                """,
                (now,),
            )
            conn.commit()
            count = cur.rowcount
            if count > 0:
                logger.warning(f"Recovered {count} interrupted training jobs from previous session.")
            return count

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
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
        }
