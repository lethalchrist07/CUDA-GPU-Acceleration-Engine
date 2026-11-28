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

export function getApiBase(): string {
  if (typeof window !== 'undefined') {
    if ((window as any).__NEXORA_API_BASE__) {
      return (window as any).__NEXORA_API_BASE__.replace(/\/+$/, '');
    }
    if ((window as any).__NEXORA_BACKEND_URL__) {
      return `${(window as any).__NEXORA_BACKEND_URL__.replace(/\/+$/, '')}/api`;
    }
    if ((import.meta as any)?.env?.VITE_API_BASE_URL) {
      return ((import.meta as any).env.VITE_API_BASE_URL as string).replace(/\/+$/, '');
    }
  }
  return '/api';
}

export function getOwnerKey(): string | null {
  if (typeof window !== 'undefined') {
    // Purge any legacy localStorage entry
    if (localStorage.getItem('nexora_owner_key')) {
      localStorage.removeItem('nexora_owner_key');
    }
    return sessionStorage.getItem('nexora_owner_key') || null;
  }
  return null;
}

export function setOwnerKey(key: string | null): void {
  if (typeof window !== 'undefined') {
    // Purge any legacy localStorage entry
    localStorage.removeItem('nexora_owner_key');
    if (key && key.trim()) {
      sessionStorage.setItem('nexora_owner_key', key.trim());
    } else {
      sessionStorage.removeItem('nexora_owner_key');
    }
  }
}

export interface ApiError extends Error {
  isNetworkError?: boolean;
  isAuthRedirect?: boolean;
  status?: number;
}

async function fetchJson<T>(endpoint: string): Promise<T> {
  let response: Response;
  const baseUrl = getApiBase();
  const headers: Record<string, string> = {
    Accept: 'application/json',
    'ngrok-skip-browser-warning': 'true',
  };
  const ownerKey = getOwnerKey();
  if (ownerKey) {
    headers['X-Owner-Key'] = ownerKey;
  }
  try {
    response = await fetch(`${baseUrl}${endpoint}`, {
      credentials: 'include',
      headers,
    });
  } catch (netErr: any) {
    const error: ApiError = new Error(`Network failure connecting to ${endpoint}: ${netErr?.message || 'Failed to fetch'}`);
    error.isNetworkError = true;
    throw error;
  }

  const contentType = response.headers.get('content-type') || '';
  if (response.redirected || contentType.includes('text/html')) {
    const error: ApiError = new Error(`Authentication required or HTML returned from ${endpoint}`);
    error.isAuthRedirect = true;
    error.status = response.status;
    throw error;
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errJson = await response.json();
      detail = errJson?.detail || detail;
    } catch {
      // Non-JSON response body fallback
    }
    const error: ApiError = new Error(detail || `API Error ${response.status}: ${response.statusText}`);
    error.status = response.status;
    throw error;
  }

  try {
    return await response.json();
  } catch {
    const error: ApiError = new Error(`Invalid JSON received from ${endpoint}`);
    error.status = response.status;
    throw error;
  }
}

async function postJson<T, B = Record<string, any>>(endpoint: string, body?: B): Promise<T> {
  let response: Response;
  const baseUrl = getApiBase();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    'ngrok-skip-browser-warning': 'true',
  };
  const ownerKey = getOwnerKey();
  if (ownerKey) {
    headers['X-Owner-Key'] = ownerKey;
  }
  try {
    response = await fetch(`${baseUrl}${endpoint}`, {
      method: 'POST',
      credentials: 'include',
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (netErr: any) {
    const error: ApiError = new Error(`Network failure posting to ${endpoint}: ${netErr?.message || 'Failed to fetch'}`);
    error.isNetworkError = true;
    throw error;
  }

  const contentType = response.headers.get('content-type') || '';
  if (response.redirected || contentType.includes('text/html')) {
    const error: ApiError = new Error(`Authentication required or HTML returned from ${endpoint}`);
    error.isAuthRedirect = true;
    error.status = response.status;
    throw error;
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errJson = await response.json();
      detail = errJson?.detail || detail;
    } catch {
      // Non-JSON response body fallback
    }
    const error: ApiError = new Error(detail || `API Error ${response.status}: ${response.statusText}`);
    error.status = response.status;
    throw error;
  }

  try {
    return await response.json();
  } catch {
    const error: ApiError = new Error(`Invalid JSON received from ${endpoint}`);
    error.status = response.status;
    throw error;
  }
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
  getArtifactUrl: (artifactId: string) => `${getApiBase()}/artifacts/${artifactId}`,

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
