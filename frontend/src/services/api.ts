import {
  OverviewResponse,
  BenchmarkResponse,
  TrainingResponse,
  InferenceResponse,
  HardwareResponse,
  VerificationResponse,
  HealthResponse,
  GenerateTextRequest,
  GenerateTextResponse,
  ModelsListResponse,
  LoadModelRequest,
  LoadModelResponse,
  PredictVisionRequest,
  PredictVisionResponse,
  CheckpointsCatalogResponse,
  CheckpointCatalogItem,
  LaboratoryConfigurationsResponse,
  LaboratoryExperimentsResponse,
  CreateExperimentRequest,
  CreateExperimentResponse,
  CudaBenchmarksResponse,
  GPUMonitoringState,
  LaboratoryCheckpointsResponse,
  LaboratoryEvaluationResponse,
  AdaptiveStatusResponse,
  AdaptiveSelectRequest,
  AdaptiveSelectResponse,
} from '../types/api';

const API_BASE = '/api';

async function fetchJson<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `API Error ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

async function postJson<T, B = Record<string, any>>(endpoint: string, body?: B): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `API Error ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

export const api = {
  // Core System & Telemetry
  getHealth: () => fetchJson<HealthResponse>('/health'),
  getOverview: () => fetchJson<OverviewResponse>('/overview'),
  getBenchmarks: () => fetchJson<BenchmarkResponse>('/benchmarks'),
  getTraining: () => fetchJson<TrainingResponse>('/training'),
  getInference: () => fetchJson<InferenceResponse>('/inference'),
  getHardware: () => fetchJson<HardwareResponse>('/hardware'),
  getVerification: () => fetchJson<VerificationResponse>('/verification'),
  getArtifactUrl: (artifactId: string) => `${API_BASE}/artifacts/${artifactId}`,

  // Phase 9 Model Serving & Checkpoints
  getModels: () => fetchJson<ModelsListResponse>('/models'),
  loadModel: (req: LoadModelRequest) => postJson<LoadModelResponse, LoadModelRequest>('/models/load', req),
  unloadModel: () => postJson<{ status: string; unloaded_models: string[] }>('/models/unload', {}),
  predictVision: (req: PredictVisionRequest) => postJson<PredictVisionResponse, PredictVisionRequest>('/models/predict', req),
  getCheckpoints: (forceRehash: boolean = false) => fetchJson<CheckpointsCatalogResponse>(`/checkpoints${forceRehash ? '?force_rehash=true' : ''}`),
  getCheckpointDetail: (checkpointId: string) => fetchJson<{ status: string; checkpoint: CheckpointCatalogItem }>(`/checkpoints/${checkpointId}`),
  getModelEvaluation: () => fetchJson<Record<string, any>>('/models/evaluation'),

  // Transformer Text Generation
  generateText: (req: GenerateTextRequest): Promise<GenerateTextResponse> =>
    postJson<GenerateTextResponse, GenerateTextRequest>('/models/generate', req),

  // Phase 13 Milestone 7/8 Comparative Laboratory & Adaptive Engine
  getLaboratoryConfigurations: () => fetchJson<LaboratoryConfigurationsResponse>('/laboratory/configurations'),
  getLaboratoryExperiments: () => fetchJson<LaboratoryExperimentsResponse>('/laboratory/experiments'),
  createLaboratoryExperiment: (req: CreateExperimentRequest) =>
    postJson<CreateExperimentResponse, CreateExperimentRequest>('/laboratory/experiments', req),
  getCudaBenchmarks: () => fetchJson<CudaBenchmarksResponse>('/laboratory/cuda-benchmarks'),
  getLaboratoryGpuMonitoring: () => fetchJson<GPUMonitoringState>('/laboratory/gpu-monitoring'),
  getLaboratoryCheckpoints: () => fetchJson<LaboratoryCheckpointsResponse>('/laboratory/checkpoints'),
  getLaboratoryEvaluations: () => fetchJson<LaboratoryEvaluationResponse>('/laboratory/evaluations'),
  getAdaptiveStatus: () => fetchJson<AdaptiveStatusResponse>('/laboratory/adaptive-status'),
  selectAdaptivePath: (req: AdaptiveSelectRequest) =>
    postJson<AdaptiveSelectResponse, AdaptiveSelectRequest>('/laboratory/adaptive-select', req),
};
