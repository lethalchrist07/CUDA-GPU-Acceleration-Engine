/**
 * Phase 10 Stage 2 Milestone 6: AI Training Studio TypeScript Types
 * Directly mirrors backend Pydantic models from backend/schemas/training.py
 */

export type JobStatus =
  | 'QUEUED'
  | 'RUNNING'
  | 'COMPLETED'
  | 'CANCEL_REQUESTED'
  | 'CANCELLED'
  | 'FAILED'
  | 'INTERRUPTED';

export type BackendType = 'CPU_NATIVE' | 'GPU_NATIVE' | 'GPU_CUSTOM';

export interface ModelCapabilityInfo {
  model_id: string;
  name: string;
  dataset: string;
  supported_backends: string[];
  parameters: number;
  input_shape: number[];
  classes: number;
  custom_cuda_supported: boolean;
  description: string;
  task?: string;
  context_length?: number;
}

export interface TrainingModelsResponse {
  status: string;
  fetched_at: string;
  models: ModelCapabilityInfo[];
}

export interface TrainingJobCreateRequest {
  model_name: string;
  backend: BackendType;
  dataset?: string;
  dataset_path?: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  seed?: number;
  max_train_samples?: number;
  max_val_samples?: number;
  max_test_samples?: number;
  precision?: string;
  lr_scheduler?: 'none' | 'cosine' | 'step';
  gradient_accumulation_steps?: number;
  resume_checkpoint_path?: string;
  seq_len?: number;
  max_steps?: number;
  max_tokens?: number;
  tokenizer_type?: string;
  vocab_size?: number;
  warmup_steps?: number;
  weight_decay?: number;
  grad_clip?: number;
  eval_interval_steps?: number;
  checkpoint_interval_steps?: number;
  activation_checkpointing?: boolean;
  adaptive_mode?: boolean;
  auto_batch_size?: boolean;
  target_vram_ratio?: number;
}

export interface EpochMetricItem {
  epoch: number;
  train_loss: number;
  train_acc?: number;
  val_loss?: number;
  val_acc: number;
  duration_sec?: number;
  samples_per_sec?: number;
  step?: number;
  tokens_per_sec?: number;
  val_perplexity?: number;
}

export interface JobProgress {
  epoch?: number;
  current_epoch?: number;
  total_epochs?: number;
  batch?: number;
  current_step?: number;
  total_batches?: number;
  total_steps?: number;
  train_loss?: number;
  train_acc?: number;
  train_accuracy?: number;
  val_loss?: number;
  val_acc?: number;
  val_accuracy?: number;
  val_perplexity?: number;
  train_perplexity?: number;
  step_latency_ms?: number;
  samples_per_sec?: number;
  tokens_per_sec?: number;
  tokens_processed?: number;
  total_tokens?: number;
  total_tokens_budget?: number;
  microbatches_processed?: number;
  optimizer_steps_completed?: number;
  physical_vram_mb?: number;
  gpu_temperature_c?: number;
  gpu_utilization_pct?: number;
  step_duration_sec?: number;
  peak_vram_mb?: number;
  peak_allocated_vram_mb?: number;
  reserved_vram_mb?: number;
  checkpoint_status?: string;
  elapsed_sec?: number;
  elapsed_time_sec?: number;
  stage?: string;
  message?: string;
  [key: string]: any;
}

export interface JobMetrics {
  model_name?: string;
  backend?: string;
  device?: string;
  epochs?: number;
  total_wall_time_sec?: number;
  avg_epoch_duration_sec?: number;
  mean_step_latency_ms?: number;
  median_step_latency_ms?: number;
  throughput_samples_per_sec?: number;
  samples_per_sec?: number;
  tokens_per_sec?: number;
  total_tokens_processed?: number;
  optimizer_steps_completed?: number;
  microbatches_processed?: number;
  final_train_loss?: number;
  final_train_acc?: number;
  best_val_acc?: number;
  final_val_loss?: number;
  final_val_perplexity?: number;
  test_acc?: number;
  test_loss?: number;
  peak_allocated_vram_mb?: number;
  physical_vram_mb?: number;
  gpu_temperature_c?: number;
  gpu_utilization_pct?: number;
  custom_cuda_dispatches?: Record<string, number>;
  dispatch_summary?: Record<string, any>;
  epoch_metrics?: EpochMetricItem[];
  epoch_records?: any[];
  step_records?: any[];
}

export interface TrainingJobResponse {
  job_id: string;
  model_name: string;
  backend: BackendType;
  dataset: string;
  status: JobStatus;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  config: Record<string, any>;
  progress: JobProgress;
  metrics?: JobMetrics | null;
  error_message?: string | null;
  checkpoint_path?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface TrainingJobListResponse {
  status: string;
  fetched_at: string;
  total_jobs: number;
  jobs: TrainingJobResponse[];
}

export interface TrainingJobMetricsResponse {
  status: string;
  fetched_at: string;
  job_id: string;
  job_status: JobStatus;
  progress: JobProgress;
  metrics?: JobMetrics | null;
  error_message?: string | null;
}

export interface CheckpointMetadata {
  id: number;
  job_id: string;
  epoch: number;
  accuracy: number;
  val_loss?: number;
  perplexity?: number;
  tokens_processed?: number;
  step?: number;
  model_architecture?: string;
  checkpoint_path: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
}

export interface TrainingJobCheckpointsResponse {
  status: string;
  fetched_at: string;
  job_id: string;
  checkpoints: CheckpointMetadata[];
}

export interface CancelJobResponse {
  status: string;
  job_id: string;
  cancelled: boolean;
  message: string;
}

export interface AdaptiveHardwareInfo {
  status: string;
  fetched_at: string;
  device_name: string;
  device_type: string;
  compute_capability: number[];
  sm_count: number;
  total_vram_mb: number;
  free_vram_mb: number;
  supports_ampere_tensor_cores: boolean;
  supports_fp16: boolean;
  supports_bf16: boolean;
  supports_custom_cuda: boolean;
  supports_cuda_graphs: boolean;
  temperature_c?: number | null;
  utilization_gpu_pct?: number | null;
}

export interface AdaptiveRecommendationRequest {
  model_name: string;
  backend: string;
  batch_size?: number;
  precision?: string;
  seq_len?: number;
  gradient_accumulation_steps?: number;
  target_vram_ratio?: number;
}

export interface AdaptiveRecommendationResponse {
  status: string;
  fetched_at: string;
  model_name: string;
  device: string;
  applied_batch_size: number;
  applied_precision: string;
  gradient_accumulation_steps: number;
  effective_batch_size: number;
  execution_path: string;
  estimated_peak_vram_mb: number;
  available_vram_mb: number;
  safety_margin_mb: number;
  vram_safety_ratio: number;
  is_safe: boolean;
  rationale: string;
  adjustments_made: string[];
  hardware: Record<string, any>;
}
