"""
backend/schemas/laboratory.py
Pydantic Schemas for Phase 13 Milestone 7 Comparative Training Laboratory.
Provides structured request/response validation for:
- Model configurations
- Comparative experiments
- Custom CUDA benchmarks
- GPU monitoring & telemetry
- Checkpoint metadata
- Model capabilities & evaluation
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelConfigurationInfo(BaseModel):
    id: str
    name: str
    description: str
    attention_type: str
    precision: str
    custom_kernels_enabled: bool
    target_hardware: str
    parameters: int = 86825472
    vram_requirement_mb: float


class LaboratoryConfigurationsResponse(BaseModel):
    status: str = "available"
    total_configurations: int
    configurations: List[ModelConfigurationInfo]


class CreateExperimentRequest(BaseModel):
    name: str = Field(default="Comparative Benchmark", min_length=1, max_length=100)
    configurations: List[str] = Field(
        default=["Config A", "Config B", "Config C", "Config D"],
        description="Configurations to compare: Config A (FP32), Config B (AMP SDPA), Config C (Custom CUDA), Config D (Adaptive)",
    )
    steps: int = Field(default=25, ge=5, le=100, description="Steps per configuration")
    batch_size: int = Field(default=16, ge=1, le=32)
    seq_len: int = Field(default=256, ge=64, le=512)
    seed: int = Field(default=42, ge=0)


class ExperimentRecord(BaseModel):
    experiment_id: str
    timestamp: str
    status: str
    effective_batch_size: int
    results: List[Dict[str, Any]]
    speedup_summary: Dict[str, Any]


class CUDABenchmarkMetrics(BaseModel):
    operation: str
    pytorch_native_ms: float
    custom_cuda_ms: float
    speedup_or_delta: str
    memory_bandwidth_gb_s: Optional[float] = None
    bus_saturation_pct: Optional[float] = None


class GPUMonitoringState(BaseModel):
    device_name: str
    compute_capability: str
    sm_count: int
    temperature_celsius: float
    gpu_utilization_pct: float
    vram_total_mb: float
    vram_allocated_mb: float
    vram_reserved_mb: float
    vram_free_mb: float
    thermal_throttling_active: bool
    active_gpu_job_id: Optional[str] = None
    can_launch_experiment: bool


class CheckpointMetadataRecord(BaseModel):
    filename: str
    step: Optional[int]
    cumulative_tokens: Optional[int]
    val_loss: Optional[float]
    val_perplexity: Optional[float]
    sha256: str
    size_bytes: int
    verified: bool


class LaboratoryCheckpointsResponse(BaseModel):
    status: str = "available"
    total_checkpoints: int
    checkpoints: List[CheckpointMetadataRecord]


class LaboratoryEvaluationResponse(BaseModel):
    status: str = "available"
    checkpoint_evaluated: str
    overall_throughput_tokens_per_sec: float
    overall_latency_per_token_ms: float
    regime_summaries: Dict[str, Any]
    category_summaries: Dict[str, Any]
    scientific_insights: Dict[str, str]
    sample_count: int


class AdaptiveSelectRequest(BaseModel):
    workload: str = Field(default="training", description="'training' or 'inference'")
    precision: str = Field(default="amp_fp16", description="'amp_fp16', 'fp32', or 'bf16'")
    batch_size: int = Field(default=16, ge=1, le=64)
    sequence_length: int = Field(default=256, ge=16, le=1024)
    force_path: Optional[str] = Field(default=None, description="Optional override path name")


class AdaptiveDecisionInfo(BaseModel):
    """Structured and validated adaptive execution decision payload."""
    selected_path: str = Field(..., description="Machine-readable path key for backend orchestration")
    path_name: str = Field(..., description="Human-readable display name for the execution path")
    workload_type: str = Field(..., description="'training', 'inference', or 'benchmark'")
    precision: str = Field(..., description="Precision string, e.g. 'amp_fp16', 'fp32'")
    execution_mode: str = Field(..., description="'GPU_NATIVE', 'GPU_CUSTOM', 'COMPILER_ASSISTED', or 'CPU_NATIVE'")
    kernel_dispatch: str = Field(..., description="Detailed kernel dispatch description")
    use_sdpa: bool = Field(..., description="Whether Scaled Dot-Product Attention is enabled")
    use_amp: bool = Field(..., description="Whether Automatic Mixed Precision is active")
    amp_enabled: bool = Field(default=True, description="Alias for use_amp matching frontend contract")
    use_custom_fused: bool = Field(..., description="Whether custom C++/CUDA fused bias+GELU is active")
    compiler_backend: Optional[str] = Field(default=None, description="Optional compiler backend, e.g. 'cudagraphs'")
    benchmark_expected_latency_ms: Optional[float] = Field(default=None, description="Empirically validated latency in ms/step")
    empirical_throughput_tokens_per_sec: Optional[float] = Field(default=None, description="Derived or measured throughput in tokens/second")
    throughput_is_estimated: bool = Field(default=True, description="True if throughput is derived from expected latency")
    empirical_vram_mb: Optional[float] = Field(default=None, description="Empirical peak VRAM footprint in MB")
    numerical_parity_verified: bool = Field(default=True, description="Whether numerical parity is certified")
    rationale: str = Field(..., description="Architectural rationale for the decision")
    fallback_reason: Optional[str] = Field(default=None, description="Reason if fallback occurred")
    confidence_score: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    hardware_target: str = Field(default="NVIDIA Ampere SM 8.6", description="Hardware architecture target")
    cache_hit: bool = Field(default=False, description="Whether the decision was served from cache")
    cache_key: Optional[str] = Field(default=None, description="Internal cache key")


class AdaptiveSelectResponse(BaseModel):
    """Validated response envelope for adaptive execution selection."""
    status: str = "success"
    decision: AdaptiveDecisionInfo


class AdaptiveStatusResponse(BaseModel):
    status: str = "available"
    hardware: Dict[str, Any]
    available_paths: List[str]
    default_training_path: str
    default_inference_path: str
    empirical_benchmarks_ms: Dict[str, float]
    numerical_certification_status: str
    cache_stats: Dict[str, Any]


