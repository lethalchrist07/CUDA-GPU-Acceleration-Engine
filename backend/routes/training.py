"""
Phase 10 Stage 2 Milestone 5: Training Orchestration API Routes
Exposes validated endpoints for AI training job creation, status polling,
live metrics, safe cancellation, checkpoint discovery, and model registry.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from backend.schemas.training import (
    CancelJobResponse,
    ModelCapabilityInfo,
    TrainingJobCheckpointsResponse,
    TrainingJobCreateRequest,
    TrainingJobListResponse,
    TrainingJobMetricsResponse,
    TrainingJobResponse,
    TrainingModelsResponse,
)
from engine.training.orchestrator import TrainingOrchestrator

router = APIRouter(prefix="/api/training", tags=["training"])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_job_id(job_id: str) -> str:
    """Guard against path traversal or malicious input strings."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid job identifier format: '{job_id}'.",
        )
    return job_id


@router.get("/models", response_model=TrainingModelsResponse)
def list_training_models():
    """List supported model architectures, tasks, datasets, and backend capabilities."""
    models = TrainingOrchestrator.list_models()
    return TrainingModelsResponse(
        status="available",
        fetched_at=utc_now(),
        models=[ModelCapabilityInfo(**m) for m in models],
    )


@router.post("/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
def create_training_job(req: TrainingJobCreateRequest):
    """
    Register and queue a new training job across CPU_NATIVE, GPU_NATIVE, or GPU_CUSTOM.
    Enforces GPU mutual exclusion and safety bounds.
    """
    orchestrator = TrainingOrchestrator.get_instance()
    try:
        job = orchestrator.submit_job(
            model_name=req.model_name,
            backend=req.backend,
            dataset=req.dataset,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            seed=req.seed,
            max_train_samples=req.max_train_samples,
            max_val_samples=req.max_val_samples,
            max_test_samples=req.max_test_samples,
            extra_config={"precision": req.precision},
        )
        return TrainingJobResponse(**job)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit training job: {type(exc).__name__}",
        )


@router.get("/jobs", response_model=TrainingJobListResponse)
def list_training_jobs(
    job_status: Optional[str] = Query(None, alias="status", description="Filter by status (QUEUED, RUNNING, COMPLETED, etc.)"),
    backend: Optional[str] = Query(None, description="Filter by backend (CPU_NATIVE, GPU_NATIVE, GPU_CUSTOM)"),
    limit: int = Query(50, ge=1, le=200, description="Max jobs to return"),
):
    """List recent training jobs with optional filtering by status or backend."""
    orchestrator = TrainingOrchestrator.get_instance()
    jobs = orchestrator.list_jobs(status=job_status, backend=backend, limit=limit)
    return TrainingJobListResponse(
        status="available",
        fetched_at=utc_now(),
        total_jobs=len(jobs),
        jobs=[TrainingJobResponse(**j) for j in jobs],
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str):
    """Retrieve full status, live progress, and completion summary of a specific job."""
    valid_id = validate_job_id(job_id)
    orchestrator = TrainingOrchestrator.get_instance()
    job = orchestrator.get_job(valid_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{job_id}' not found.",
        )
    return TrainingJobResponse(**job)


@router.get("/jobs/{job_id}/metrics", response_model=TrainingJobMetricsResponse)
def get_training_job_metrics(job_id: str):
    """Retrieve fine-grained epoch and batch metrics for an active or completed job."""
    valid_id = validate_job_id(job_id)
    orchestrator = TrainingOrchestrator.get_instance()
    metrics_data = orchestrator.get_job_metrics(valid_id)
    if not metrics_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{job_id}' not found.",
        )
    return TrainingJobMetricsResponse(
        status="available",
        fetched_at=utc_now(),
        job_id=metrics_data["job_id"],
        job_status=metrics_data["status"],
        progress=metrics_data["progress"],
        metrics=metrics_data["metrics"],
        error_message=metrics_data["error_message"],
    )


@router.post("/jobs/{job_id}/cancel", response_model=CancelJobResponse)
def cancel_training_job(job_id: str):
    """
    Request cooperative cancellation of a queued or running training job.
    Releases GPU resources and transitions status safely to CANCELLED.
    """
    valid_id = validate_job_id(job_id)
    orchestrator = TrainingOrchestrator.get_instance()
    job = orchestrator.get_job(valid_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{job_id}' not found.",
        )

    current_status = job["status"]
    if current_status in ("COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"):
        return CancelJobResponse(
            status="terminal",
            job_id=valid_id,
            cancelled=False,
            message=f"Job is already in terminal state '{current_status}'.",
        )

    success = orchestrator.cancel_job(valid_id)
    return CancelJobResponse(
        status="cancel_requested",
        job_id=valid_id,
        cancelled=success,
        message="Cancellation signal transmitted. Job will terminate at next step boundary." if success else "Could not signal cancellation.",
    )


@router.get("/jobs/{job_id}/checkpoints", response_model=TrainingJobCheckpointsResponse)
def get_training_job_checkpoints(job_id: str):
    """Retrieve saved model checkpoint metadata for a specific job."""
    valid_id = validate_job_id(job_id)
    orchestrator = TrainingOrchestrator.get_instance()
    job = orchestrator.get_job(valid_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{job_id}' not found.",
        )

    checkpoints = orchestrator.get_job_checkpoints(valid_id)
    return TrainingJobCheckpointsResponse(
        status="available",
        fetched_at=utc_now(),
        job_id=valid_id,
        checkpoints=checkpoints,
    )
