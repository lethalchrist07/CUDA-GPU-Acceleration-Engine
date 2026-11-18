"""
Phase 10 Stage 2 Milestone 5: Pydantic Validation Schemas for Training Orchestration API
Provides robust validation for job creation, progress streaming, metrics, and models.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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
        ge=1,
        le=512,
        description="Training batch size (1 to 512)",
    )

    @model_validator(mode="after")
    def validate_batch_size_for_model(self):
        is_lm = "transformer" in self.model_name.lower() or "gpt" in self.model_name.lower()
        if not is_lm and self.batch_size < 16:
            raise ValueError("Batch size must be between 16 and 512 for vision models")
        return self
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
        description="Floating point precision: float32, amp, float16",
    )
    lr_scheduler: Optional[str] = Field(
        default="none",
        description="Learning rate scheduler: none, cosine, step",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Gradient accumulation steps (1 to 16)",
    )
    resume_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional filesystem path to checkpoint for full training resume",
    )
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000,
        description="Target optimizer steps for language model training",
    )
    seq_len: Optional[int] = Field(
        default=None,
        ge=16,
        le=2048,
        description="Sequence length in tokens",
    )
    tokenizer_type: Optional[str] = Field(
        default=None,
        description="Tokenizer type: bpe, byte, char",
    )
    vocab_size: Optional[int] = Field(
        default=None,
        ge=64,
        le=32000,
        description="Vocabulary size for BPE tokenizer",
    )
    warmup_steps: Optional[int] = Field(
        default=None,
        ge=0,
        le=500,
        description="Optimizer warmup steps",
    )
    weight_decay: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="AdamW weight decay",
    )
    grad_clip: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Gradient clipping max norm",
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to text corpus file or directory",
    )
    adaptive_mode: bool = Field(
        default=False,
        description="Enable adaptive hardware-informed hyperparameter optimization",
    )
    auto_batch_size: bool = Field(
        default=False,
        description="Enable automatic batch-size calculation based on available VRAM",
    )
    target_vram_ratio: float = Field(
        default=0.85,
        ge=0.5,
        le=0.98,
        description="Target VRAM utilization ratio for safe batch sizing (0.5 to 0.98)",
    )

    @field_validator("model_name")
    def validate_model_name(cls, v: str) -> str:
        clean = v.strip().lower().replace("-", "_")
        valid = {
            "cifar10_mlp",
            "resnet18",
            "baseline_mlp",
            "transformer_100m",
            "transformer_50m",
            "transformer_25m",
            "transformer_1m",
            "transformer",
            "nexora_gpt_100m",
            "nexora_gpt_86m",
            "nexora_gpt_86.8m",
            "nexora_gpt",
        }
        if clean not in valid:
            raise ValueError(f"Invalid model_name '{v}'. Supported models: {sorted(valid)}")
        if clean in ("transformer", "nexora_gpt_100m"):
            return "transformer_100m"
        if clean in ("transformer_50m", "nexora_gpt_86m", "nexora_gpt_86.8m", "nexora_gpt"):
            return "transformer_50m"
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
        if clean == "bf16":
            return "bfloat16"
        valid = {"float32", "amp", "float16", "bfloat16"}
        if clean not in valid:
            raise ValueError(f"Precision '{v}' is unsupported. Must be one of: {sorted(valid)}")
        return clean

    @field_validator("lr_scheduler")
    def validate_lr_scheduler(cls, v: Optional[str]) -> str:
        if v is None:
            return "none"
        clean = v.strip().lower()
        valid = {"none", "cosine", "step"}
        if clean not in valid:
            raise ValueError(f"Invalid lr_scheduler '{v}'. Must be one of: {sorted(valid)}")
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


class AdaptiveHardwareResponse(BaseModel):
    status: str = "available"
    fetched_at: str
    device_name: str
    device_type: str
    compute_capability: List[int]
    sm_count: int
    total_vram_mb: float
    free_vram_mb: float
    supports_ampere_tensor_cores: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_custom_cuda: bool
    supports_cuda_graphs: bool
    temperature_c: Optional[float] = None
    utilization_gpu_pct: Optional[float] = None


class AdaptiveRecommendationRequest(BaseModel):
    model_name: str = Field(default="cifar10_mlp", description="Target model name")
    backend: str = Field(default="GPU_NATIVE", description="Target execution backend")
    batch_size: Optional[int] = Field(default=None, ge=1, le=512, description="Optional user requested batch size")
    precision: Optional[str] = Field(default=None, description="Optional user requested precision (float32, amp, bfloat16)")
    seq_len: Optional[int] = Field(default=None, ge=16, le=2048, description="Optional sequence length for transformers")
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=16, description="Starting gradient accumulation steps")
    target_vram_ratio: float = Field(default=0.85, ge=0.5, le=0.98, description="Target VRAM ratio (default 0.85)")


class AdaptiveRecommendationResponse(BaseModel):
    status: str = "available"
    fetched_at: str
    model_name: str
    device: str
    applied_batch_size: int
    applied_precision: str
    gradient_accumulation_steps: int
    effective_batch_size: int
    execution_path: str
    estimated_peak_vram_mb: float
    available_vram_mb: float
    safety_margin_mb: float
    vram_safety_ratio: float
    is_safe: bool
    rationale: str
    adjustments_made: List[str]
    hardware: Dict[str, Any]

