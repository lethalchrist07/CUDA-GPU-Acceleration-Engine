export interface SourceInfo {
  path: string;
  modified_at: string;
  sha256: string;
  provenance: 'phase2_hash_verified' | 'historical_unverified' | 'hash_mismatch' | 'recorded_execution';
}

export interface BenchmarkRecord {
  id: number;
  category: string;
  variant: string;
  workload_size: string;
  dimension: number;
  memory_mb: number;
  cpu_time_ms: number;
  gpu_h2d_ms: number;
  gpu_kernel_ms: number;
  gpu_d2h_ms: number;
  gpu_end_to_end_ms: number;
  kernel_speedup: number;
  end_to_end_speedup: number;
  gflops: number;
  memory_bandwidth_gb_s: number;
  max_abs_error: number;
  rmse: number;
  tolerance: number;
  mismatched_elements: number;
  verified: boolean;
  warmup_runs: number;
  timed_runs: number;
  cpu_timed_runs: number;
  samples_ms?: {
    cpu: number[];
    h2d: number[];
    kernel: number[];
    d2h: number[];
    end_to_end: number[];
  };
}

export interface BenchmarkResponse {
  status: 'available' | 'unavailable' | 'degraded';
  fetched_at: string;
  source: SourceInfo | null;
  warnings: string[];
  rows: BenchmarkRecord[];
  system_info?: Record<string, any>;
  summary?: Record<string, any>;
  measurement_notes?: Record<string, any>;
}

export interface TrainingRecord {
  experiment_id: string;
  model_name: string;
  parameters: number;
  device: string;
  device_name: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  seed: number;
  use_amp: boolean;
  total_train_time_ms: number;
  avg_epoch_time_ms: number;
  final_train_loss: number;
  final_val_loss: number;
  best_val_accuracy: number;
  peak_gpu_memory_mb: number;
  speedup_vs_cpu: number;
  timestamp: string;
}

export interface TrainingResponse {
  status: 'available' | 'unavailable' | 'degraded';
  fetched_at: string;
  source: SourceInfo | null;
  warnings: string[];
  rows: TrainingRecord[];
}

export interface InferenceRecord {
  experiment_id: string;
  timestamp: string;
  model_name: string;
  quantization: string;
  parameter_size: string;
  device: string;
  device_name: string;
  prompt_tokens: number;
  output_tokens: number;
  ttft_ms: number;
  generation_duration_ms: number;
  total_duration_ms: number;
  tokens_per_second: number;
  backend_vram_mb: number;
  pytorch_vram_mb: number;
  global_vram_used_mb: number;
}

export interface InferenceResponse {
  status: 'available' | 'unavailable' | 'degraded';
  fetched_at: string;
  source: SourceInfo | null;
  warnings: string[];
  rows: InferenceRecord[];
}

export interface GpuTelemetry {
  index: number;
  name: string;
  uuid: string;
  driver_version: string;
  memory_total_mib: number | null;
  memory_used_mib: number | null;
  memory_free_mib: number | null;
  utilization_percent: number | null;
  temperature_c: number | null;
  power_w: number | null;
  power_limit_w: number | null;
}

export interface HardwareResponse {
  status: 'available' | 'unavailable' | 'degraded' | 'cpu_only';
  fetched_at: string;
  gpus: GpuTelemetry[];
  host: {
    platform: string;
    processor: string;
  };
  warnings: string[];
  source: string;
}

export interface ManifestCheck {
  path: string;
  status: 'matched' | 'mismatch';
  expected_sha256: string;
  actual_sha256: string | null;
}

export interface ArtifactFile {
  id: string;
  name: string;
  path: string;
  size_bytes: number;
  download_url: string;
}

export interface VerificationResponse {
  status: 'available' | 'unavailable' | 'degraded';
  fetched_at: string;
  manifest_status: 'matched' | 'mismatch' | 'unavailable';
  checks: ManifestCheck[];
  artifacts: ArtifactFile[];
  warnings: string[];
}

export interface OverviewResponse {
  status: 'available' | 'unavailable' | 'degraded';
  fetched_at: string;
  benchmark_count: number;
  benchmark_passed: number;
  training_count: number;
  inference_count: number;
  benchmark_source: SourceInfo | null;
  system_info: Record<string, any>;
  benchmark_rows: BenchmarkRecord[];
  warnings: string[];
}

export interface HealthResponse {
  status: string;
  version: string;
  time: string;
  mode: string;
}

export interface GenerateTextRequest {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  seed?: number;
  checkpoint?: string;
  device?: string;
}

export interface GenerateTextResponse {
  status: string;
  model_name: string;
  device: string;
  checkpoint: string;
  prompt: string;
  prompt_tokens: number;
  generated_tokens: number;
  total_tokens: number;
  generation_duration_ms: number;
  tokens_per_second: number;
  peak_vram_mb: number;
  is_partially_trained: boolean;
  status_label: string;
  generated_text: string;
  full_text: string;
}

// -------------------------------------------------------------
// Model Serving & Registry Interfaces
// -------------------------------------------------------------
export interface AvailableModelItem {
  model_id: string;
  name: string;
  task: string;
  dataset: string;
  classes?: number;
  parameters: number;
  parameters_pilot?: number;
  layers?: number;
  hidden_dim?: number;
  heads?: number;
  context_length?: number;
  vocab_size?: number;
  tokenizer?: string;
  tokenizer_display?: string;
  supported_backends?: string[];
  training_capability?: boolean;
  inference_capability?: boolean;
  training_status?: string;
  status: 'available' | 'loaded';
  device: string | null;
  active_checkpoint: string | null;
  checkpoints: {
    id: string;
    path: string;
    size_bytes: number;
    filename: string;
    modified_at?: string;
  }[];
}

export interface ModelsListResponse {
  status: string;
  fetched_at: string;
  models: AvailableModelItem[];
}

export interface LoadModelRequest {
  model_name?: string;
  checkpoint?: string | null;
  device?: string;
}

export interface LoadModelResponse {
  status: string;
  model: string;
  device: string;
  parameters: number;
  checkpoint: string | null;
  classes?: number;
  message?: string;
}

export interface PredictVisionRequest {
  sample_index?: number;
  tensor?: any[];
}

export interface PredictVisionResponse {
  status: string;
  model: string;
  device: string;
  latency_ms: number;
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  correct: boolean;
  top_predictions: {
    class_name: string;
    class_index: number;
    confidence: number;
    confidence_percent: number;
  }[];
  metadata: {
    source: string;
    sample_index: number;
    ground_truth_class: string;
    ground_truth_index: number;
  };
}

export interface CheckpointCatalogItem {
  id: string;
  filename: string;
  path: string;
  size_bytes: number;
  sha256: string;
  modified_at: string;
}

export interface CheckpointsCatalogResponse {
  status: string;
  total_checkpoints: number;
  checkpoints: CheckpointCatalogItem[];
}

// -------------------------------------------------------------
// Comparative Laboratory & Adaptive Engine Interfaces
// -------------------------------------------------------------
export interface ModelConfigurationInfo {
  id: string;
  name: string;
  description: string;
  attention_type: string;
  precision: string;
  custom_kernels_enabled: boolean;
  target_hardware: string;
  parameters: number;
  vram_requirement_mb: number;
}

export interface LaboratoryConfigurationsResponse {
  status: string;
  total_configurations: number;
  configurations: ModelConfigurationInfo[];
}

export interface CreateExperimentRequest {
  name?: string;
  configurations?: string[];
  steps?: number;
  batch_size?: number;
  seq_len?: number;
  seed?: number;
}

export interface CreateExperimentResponse {
  status: string;
  experiment_id: string;
  created_at: string;
  request: CreateExperimentRequest;
  latest_verified_run?: Record<string, any>;
  message: string;
}

export interface LaboratoryExperimentsResponse {
  status: string;
  fetched_at: string;
  source_artifact: string;
  experiment_data: Record<string, any>;
}

export interface CudaBenchmarksResponse {
  status: string;
  fetched_at: string;
  source_artifact: string;
  benchmarks: Record<string, any>;
}

export interface GPUMonitoringState {
  device_name: string;
  compute_capability: string;
  sm_count: number;
  temperature_celsius: number;
  gpu_utilization_pct: number;
  vram_total_mb: number;
  vram_allocated_mb: number;
  vram_reserved_mb: number;
  vram_free_mb: number;
  thermal_throttling_active: boolean;
  active_gpu_job_id: string | null;
  can_launch_experiment: boolean;
}

export interface LaboratoryCheckpointRecord {
  filename: string;
  step: number | null;
  cumulative_tokens: number | null;
  val_loss: number | null;
  val_perplexity: number | null;
  sha256: string;
  size_bytes: number;
  verified: boolean;
}

export interface LaboratoryCheckpointsResponse {
  status: string;
  total_checkpoints: number;
  checkpoints: LaboratoryCheckpointRecord[];
}

export interface LaboratoryEvaluationResponse {
  status: string;
  checkpoint_evaluated: string;
  overall_throughput_tokens_per_sec: number;
  overall_latency_per_token_ms: number;
  regime_summaries: Record<string, any>;
  category_summaries: Record<string, any>;
  scientific_insights: Record<string, string>;
  sample_count: number;
}

export interface AdaptiveStatusResponse {
  status: string;
  hardware: Record<string, any>;
  available_paths: string[];
  default_training_path: string;
  default_inference_path: string;
  empirical_benchmarks_ms: Record<string, number>;
  numerical_certification_status: string;
  cache_stats: Record<string, any>;
}

export interface AdaptiveSelectRequest {
  workload: string;
  precision: string;
  batch_size: number;
  sequence_length: number;
  force_path?: string | null;
}

export interface AdaptiveDecision {
  selected_path?: string;
  path_name: string;
  kernel_dispatch: string;
  execution_mode: string;
  amp_enabled: boolean;
  precision: string;
  workload_type?: string;
  use_sdpa?: boolean;
  use_amp?: boolean;
  use_custom_fused?: boolean;
  compiler_backend?: string | null;
  benchmark_expected_latency_ms?: number | null;
  empirical_throughput_tokens_per_sec?: number | null;
  throughput_is_estimated?: boolean;
  empirical_vram_mb?: number | null;
  rationale: string;
  fallback_reason?: string | null;
  confidence_score?: number;
  hardware_target?: string;
  numerical_parity_verified: boolean;
  cache_hit?: boolean;
  cache_key?: string | null;
}

export interface AdaptiveSelectResponse {
  status: string;
  decision: AdaptiveDecision;
}
