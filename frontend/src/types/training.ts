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
  epochs: number;
  batch_size: number;
  learning_rate: number;
  seed?: number;
  max_train_samples?: number;
  max_val_samples?: number;
  max_test_samples?: number;
  precision?: string;
}

export interface EpochMetricItem {
  epoch: number;
  train_loss: number;
  train_acc?: number;
  val_loss?: number;
  val_acc: number;
  duration_sec?: number;
  samples_per_sec?: number;
}

export interface JobProgress {
  epoch?: number;
  total_epochs?: number;
  batch?: number;
  total_batches?: number;
  train_loss?: number;
  train_acc?: number;
  train_accuracy?: number;
  val_loss?: number;
  val_acc?: number;
  val_accuracy?: number;
  step_latency_ms?: number;
  samples_per_sec?: number;
  peak_vram_mb?: number;
  peak_allocated_vram_mb?: number;
  elapsed_sec?: number;
  elapsed_time_sec?: number;
  stage?: string;
  message?: string;
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
  final_train_loss?: number;
  final_train_acc?: number;
  best_val_acc?: number;
  final_val_loss?: number;
  test_acc?: number;
  test_loss?: number;
  peak_allocated_vram_mb?: number;
  custom_cuda_dispatches?: Record<string, number>;
  dispatch_summary?: Record<string, any>;
  epoch_metrics?: EpochMetricItem[];
  epoch_records?: any[];
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
