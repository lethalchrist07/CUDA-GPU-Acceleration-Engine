"""
Phase 10 Stage 2 Milestone 5: Pydantic Validation Schemas for Training Orchestration API
Provides robust validation for job creation, progress streaming, metrics, and models.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TrainingJobCreateRequest(BaseModel):
    model_name: str = Field(
        default="cifar10_mlp",
        description="Target model architecture name (e.g. cifar10_mlp, resnet18, baseline_mlp)",
    )
    backend: str = Field(
        default="GPU_CUSTOM",
        description="Execution backend: CPU_NATIVE, GPU_NATIVE, or GPU_CUSTOM",
    )
    dataset: str = Field(
        default="CIFAR-10",
        description="Dataset name (e.g. CIFAR-10, SyntheticClassification)",
    )
    epochs: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Total training epochs (1 to 50)",
    )
    batch_size: int = Field(
        default=128,
        ge=16,
        le=512,
        description="Training batch size (16 to 512)",
    )
    learning_rate: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="SGD learning rate (1e-6 to 1.0)",
    )
    seed: int = Field(
        default=42,
        ge=0,
        le=2147483647,
        description="Deterministic random seed",
    )
    max_train_samples: Optional[int] = Field(
        default=10000,
        ge=10,
        le=50000,
        description="Subset limit for training samples (10 to 50,000)",
    )
    max_val_samples: Optional[int] = Field(
        default=2000,
        ge=10,
        le=10000,
        description="Subset limit for validation samples (10 to 10,000)",
    )
    max_test_samples: Optional[int] = Field(
        default=2000,
        ge=10,
        le=10000,
        description="Subset limit for test samples (10 to 10,000)",
    )
    precision: str = Field(
        default="float32",
        description="Floating point precision: float32",
    )

    @field_validator("model_name")
    def validate_model_name(cls, v: str) -> str:
        clean = v.strip().lower().replace("-", "_")
        valid = {"cifar10_mlp", "resnet18", "baseline_mlp"}
        if clean not in valid:
            raise ValueError(f"Invalid model_name '{v}'. Supported models: {sorted(valid)}")
        return clean

    @field_validator("backend")
    def validate_backend(cls, v: str) -> str:
        clean = v.strip().upper()
        valid = {"CPU_NATIVE", "GPU_NATIVE", "GPU_CUSTOM"}
        if clean not in valid:
            raise ValueError(f"Invalid backend '{v}'. Must be one of: {sorted(valid)}")
        return clean

    @field_validator("precision")
    def validate_precision(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean != "float32":
            raise ValueError(f"Precision '{v}' is currently unsupported for Milestone 5. Only 'float32' is supported.")
        return clean


class TrainingJobResponse(BaseModel):
    job_id: str
    model_name: str
    backend: str
    dataset: str
    status: str
    epochs: int
    batch_size: int
    learning_rate: float
    config: Dict[str, Any]
    progress: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TrainingJobListResponse(BaseModel):
    status: str = "available"
    fetched_at: str
    total_jobs: int
    jobs: List[TrainingJobResponse]


class TrainingJobMetricsResponse(BaseModel):
    status: str = "available"
    fetched_at: str
    job_id: str
    job_status: str
    progress: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ModelCapabilityInfo(BaseModel):
    model_id: str
    name: str
    dataset: str
    supported_backends: List[str]
    parameters: int
    input_shape: List[int]
    classes: int
    custom_cuda_supported: bool
    description: str


class TrainingModelsResponse(BaseModel):
    status: str = "available"
    fetched_at: str
    models: List[ModelCapabilityInfo]


class CheckpointMetadata(BaseModel):
    id: int
    job_id: str
    epoch: int
    accuracy: float
    checkpoint_path: str
    size_bytes: int
    sha256: str
    created_at: str


class TrainingJobCheckpointsResponse(BaseModel):
    status: str = "available"
    fetched_at: str
    job_id: str
    checkpoints: List[CheckpointMetadata]


class CancelJobResponse(BaseModel):
    status: str
    job_id: str
    cancelled: bool
    message: str
