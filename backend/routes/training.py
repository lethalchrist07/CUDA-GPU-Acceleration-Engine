"""
Phase 10 Stage 2 Milestone 5: Training Orchestration API Routes
Exposes validated endpoints for AI training job creation, status polling,
live metrics, safe cancellation, checkpoint discovery, and model registry.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request, status

from backend.schemas.training import (
    AdaptiveHardwareResponse,
    AdaptiveRecommendationRequest,
    AdaptiveRecommendationResponse,
    CancelJobResponse,
    ModelCapabilityInfo,
    TrainingJobCheckpointsResponse,
    TrainingJobCreateRequest,
    TrainingJobListResponse,
    TrainingJobMetricsResponse,
    TrainingJobResponse,
    TrainingModelsResponse,
)
from engine.training.adaptive_engine import AdaptiveTrainingEngine
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


@router.get("/adaptive/hardware", response_model=AdaptiveHardwareResponse)
def get_adaptive_hardware():
    """Inspect and return runtime hardware capability profile, VRAM headroom, and precision flags."""
    engine = AdaptiveTrainingEngine()
    hw = engine.get_hardware_profile()
    return AdaptiveHardwareResponse(
        status="available",
        fetched_at=utc_now(),
        device_name=hw.device_name,
        device_type=hw.device_type,
        compute_capability=[hw.major, hw.minor],
        sm_count=hw.sm_count,
        total_vram_mb=hw.total_vram_mb,
        free_vram_mb=hw.free_vram_mb,
        supports_ampere_tensor_cores=hw.supports_ampere_tensor_cores,
        supports_fp16=hw.supports_fp16,
        supports_bf16=hw.supports_bf16,
        supports_custom_cuda=hw.supports_custom_cuda,
        supports_cuda_graphs=hw.supports_cuda_graphs,
        temperature_c=hw.temperature_c,
        utilization_gpu_pct=hw.utilization_gpu_pct,
    )


@router.post("/adaptive/recommend", response_model=AdaptiveRecommendationResponse)
def get_adaptive_recommendation(req: AdaptiveRecommendationRequest, request: Request):
    """
    Evaluate training configuration and recommend optimal batch size, precision,
    and gradient accumulation steps based on live GPU VRAM headroom.
    """
    from backend.config import get_settings
    from backend.auth import is_owner_authorized
    if get_settings().demo_mode and not is_owner_authorized(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Adaptive training recommendation is disabled in public demo mode.",
        )

    engine = AdaptiveTrainingEngine()
    try:
        decision = engine.recommend_training_config(
            model_name=req.model_name,
            backend=req.backend,
            requested_batch_size=req.batch_size,
            requested_precision=req.precision,
            seq_len=req.seq_len,
            gradient_accumulation_steps=req.gradient_accumulation_steps,
            target_vram_ratio=req.target_vram_ratio,
        )
        hw = engine.get_hardware_profile(force_cpu=(req.backend == "CPU_NATIVE"))
        return AdaptiveRecommendationResponse(
            status="available",
            fetched_at=utc_now(),
            model_name=decision.model_name,
            device=decision.device,
            applied_batch_size=decision.applied_batch_size,
            applied_precision=decision.applied_precision,
            gradient_accumulation_steps=decision.gradient_accumulation_steps,
            effective_batch_size=decision.effective_batch_size,
            execution_path=decision.execution_path,
            estimated_peak_vram_mb=decision.estimated_peak_vram_mb,
            available_vram_mb=decision.available_vram_mb,
            safety_margin_mb=decision.safety_margin_mb,
            vram_safety_ratio=decision.vram_safety_ratio,
            is_safe=decision.is_safe,
            rationale=decision.rationale,
            adjustments_made=decision.adjustments_made,
            hardware=hw.to_dict(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to generate adaptive recommendation: {str(exc)}",
        )


@router.post("/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
def create_training_job(req: TrainingJobCreateRequest, request: Request):
    """
    Register and queue a new training job across CPU_NATIVE, GPU_NATIVE, or GPU_CUSTOM.
    Enforces GPU mutual exclusion and safety bounds. Applies adaptive recommendations if enabled.
    """
    from backend.config import get_settings
    from backend.auth import is_owner_authorized
    if get_settings().demo_mode and not is_owner_authorized(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AI Training Studio job creation and execution is disabled in public demo mode.",
        )

    orchestrator = TrainingOrchestrator.get_instance()
    try:
        final_batch_size = req.batch_size
        final_precision = req.precision
        final_grad_accum = req.gradient_accumulation_steps
        adaptive_metadata = None

        if req.adaptive_mode or req.auto_batch_size:
            engine = AdaptiveTrainingEngine()
            rec = engine.recommend_training_config(
                model_name=req.model_name,
                backend=req.backend,
                requested_batch_size=req.batch_size if not req.auto_batch_size else None,
                requested_precision=req.precision,
                seq_len=req.seq_len,
                gradient_accumulation_steps=req.gradient_accumulation_steps,
                target_vram_ratio=req.target_vram_ratio,
            )
            final_batch_size = rec.applied_batch_size
            final_precision = rec.applied_precision
            final_grad_accum = rec.gradient_accumulation_steps
            adaptive_metadata = rec.to_dict()

        job = orchestrator.submit_job(
            model_name=req.model_name,
            backend=req.backend,
            dataset=req.dataset,
            epochs=req.epochs,
            batch_size=final_batch_size,
            learning_rate=req.learning_rate,
            seed=req.seed,
            max_train_samples=req.max_train_samples,
            max_val_samples=req.max_val_samples,
            max_test_samples=req.max_test_samples,
            extra_config={
                "precision": final_precision,
                "lr_scheduler": req.lr_scheduler,
                "gradient_accumulation_steps": final_grad_accum,
                "resume_checkpoint_path": req.resume_checkpoint_path,
                "max_steps": req.max_steps,
                "seq_len": req.seq_len,
                "tokenizer_type": req.tokenizer_type,
                "vocab_size": req.vocab_size,
                "warmup_steps": req.warmup_steps,
                "weight_decay": req.weight_decay,
                "grad_clip": req.grad_clip,
                "dataset_path": req.dataset_path,
                "adaptive_mode": req.adaptive_mode,
                "auto_batch_size": req.auto_batch_size,
                "target_vram_ratio": req.target_vram_ratio,
                "adaptive_decision": adaptive_metadata,
            },
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
    request: Request,
    job_status: Optional[str] = Query(None, alias="status", description="Filter by status (QUEUED, RUNNING, COMPLETED, etc.)"),
    backend: Optional[str] = Query(None, description="Filter by backend (CPU_NATIVE, GPU_NATIVE, GPU_CUSTOM)"),
    limit: int = Query(50, ge=1, le=200, description="Max jobs to return"),
):
    from backend.config import get_settings
    from backend.auth import is_owner_authorized
    if get_settings().demo_mode and not is_owner_authorized(request):
        return TrainingJobListResponse(
            status="available",
            fetched_at=utc_now(),
            total_jobs=0,
            jobs=[],
        )

    orchestrator = TrainingOrchestrator.get_instance()
    norm_status = None if (job_status and job_status.upper() == "ALL") else job_status
    norm_backend = None if (backend and backend.upper() == "ALL") else backend
    jobs = orchestrator.list_jobs(status=norm_status, backend=norm_backend, limit=limit)
    return TrainingJobListResponse(
        status="available",
        fetched_at=utc_now(),
        total_jobs=len(jobs),
        jobs=[TrainingJobResponse(**j) for j in jobs],
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str, request: Request):
    """Retrieve full status, live progress, and completion summary of a specific job."""
    from backend.config import get_settings
    from backend.auth import is_owner_authorized
    if get_settings().demo_mode and not is_owner_authorized(request):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{job_id}' not found.",
        )

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
def get_training_job_metrics(job_id: str, request: Request):
    """Retrieve fine-grained epoch and batch metrics for an active or completed job."""
    from backend.config import get_settings
    from backend.auth import is_owner_authorized
    if get_settings().demo_mode and not is_owner_authorized(request):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{job_id}' not found.",
        )

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
def cancel_training_job(job_id: str, request: Request):
    """
    Request cooperative cancellation of a queued or running training job.
    Releases GPU resources and transitions status safely to CANCELLED.
    """
    from backend.config import get_settings
    from backend.auth import is_owner_authorized
    if get_settings().demo_mode and not is_owner_authorized(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Training job cancellation is disabled in public demo mode.",
        )

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
def get_training_job_checkpoints(job_id: str, request: Request):
    """Retrieve saved model checkpoint metadata for a specific job."""
    from backend.config import get_settings
    from backend.auth import is_owner_authorized
    if get_settings().demo_mode and not is_owner_authorized(request):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{job_id}' not found.",
        )

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
