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
  status: 'available' | 'unavailable' | 'degraded';
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
