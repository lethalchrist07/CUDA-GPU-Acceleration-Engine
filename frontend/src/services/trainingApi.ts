/**
 * Phase 10 Stage 2 Milestone 6: AI Training Studio Typed API Client
 * Connects frontend directly to FastAPI /api/training endpoints
 */

import {
  TrainingModelsResponse,
  TrainingJobCreateRequest,
  TrainingJobResponse,
  TrainingJobListResponse,
  TrainingJobMetricsResponse,
  TrainingJobCheckpointsResponse,
  CancelJobResponse,
  AdaptiveHardwareInfo,
  AdaptiveRecommendationRequest,
  AdaptiveRecommendationResponse,
} from '../types/training';

export function getTrainingApiBase(): string {
  if (typeof window !== 'undefined') {
    if ((window as any).__NEXORA_API_BASE__) {
      return `${(window as any).__NEXORA_API_BASE__.replace(/\/+$/, '')}/training`;
    }
    if ((window as any).__NEXORA_BACKEND_URL__) {
      return `${(window as any).__NEXORA_BACKEND_URL__.replace(/\/+$/, '')}/api/training`;
    }
    if ((import.meta as any)?.env?.VITE_API_BASE_URL) {
      return `${((import.meta as any).env.VITE_API_BASE_URL as string).replace(/\/+$/, '')}/training`;
    }
  }
  return '/api/training';
}

export interface TrainingApiError extends Error {
  isNetworkError?: boolean;
  isAuthRedirect?: boolean;
  status?: number;
}

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${getTrainingApiBase()}${endpoint}`;
  const method = (options?.method || 'GET').toUpperCase();
  const ownerKey = typeof window !== 'undefined'
    ? sessionStorage.getItem('nexora_owner_key')
    : null;
  const headers: Record<string, string> = {
    Accept: 'application/json',
    'ngrok-skip-browser-warning': 'true',
    ...(ownerKey ? { 'X-Owner-Key': ownerKey } : {}),
    ...(options?.headers as Record<string, string> || {}),
  };
  // Only inject Content-Type: application/json for methods with a request body
  if (method !== 'GET' && method !== 'HEAD') {
    if (!headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      credentials: 'include',
      headers,
    });
  } catch (netErr: any) {
    const error: TrainingApiError = new Error(`Network failure connecting to ${endpoint}: ${netErr?.message || 'Failed to fetch'}`);
    error.isNetworkError = true;
    throw error;
  }

  const contentType = response.headers.get('content-type') || '';
  if (response.redirected || contentType.includes('text/html')) {
    const error: TrainingApiError = new Error(`Authentication required or HTML returned from ${endpoint}`);
    error.isAuthRedirect = true;
    error.status = response.status;
    throw error;
  }

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson && errorJson.detail) {
        if (typeof errorJson.detail === 'string') {
          errorDetail = errorJson.detail;
        } else if (Array.isArray(errorJson.detail)) {
          errorDetail = errorJson.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ');
        }
      }
    } catch {
      // Non-JSON response fallback
    }
    const err: TrainingApiError = new Error(errorDetail);
    err.status = response.status;
    throw err;
  }

  try {
    return await response.json();
  } catch {
    const error: TrainingApiError = new Error(`Invalid JSON received from ${endpoint}`);
    error.status = response.status;
    throw error;
  }
}

export const trainingApi = {
  /**
   * List available model architectures and backend execution support
   */
  getModels: (): Promise<TrainingModelsResponse> => {
    return request<TrainingModelsResponse>('/models');
  },

  /**
   * Register and queue a new training job across CPU_NATIVE, GPU_NATIVE, or GPU_CUSTOM
   */
  createJob: (req: TrainingJobCreateRequest): Promise<TrainingJobResponse> => {
    return request<TrainingJobResponse>('/jobs', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /**
   * List recent training jobs with optional filtering by status or backend
   */
  listJobs: (params?: {
    status?: string;
    backend?: string;
    limit?: number;
  }): Promise<TrainingJobListResponse> => {
    const query = new URLSearchParams();
    if (params?.status && params.status !== 'ALL') {
      query.set('status', params.status);
    }
    if (params?.backend && params.backend !== 'ALL') {
      query.set('backend', params.backend);
    }
    if (params?.limit) {
      query.set('limit', params.limit.toString());
    }
    const queryString = query.toString();
    return request<TrainingJobListResponse>(`/jobs${queryString ? `?${queryString}` : ''}`);
  },

  /**
   * Retrieve full status, live progress, and completion summary of a specific job
   */
  getJob: (jobId: string): Promise<TrainingJobResponse> => {
    return request<TrainingJobResponse>(`/jobs/${encodeURIComponent(jobId)}`);
  },

  /**
   * Retrieve fine-grained epoch and batch metrics for an active or completed job
   */
  getJobMetrics: (jobId: string): Promise<TrainingJobMetricsResponse> => {
    return request<TrainingJobMetricsResponse>(`/jobs/${encodeURIComponent(jobId)}/metrics`);
  },

  /**
   * Request cooperative cancellation of a queued or running training job
   */
  cancelJob: (jobId: string): Promise<CancelJobResponse> => {
    return request<CancelJobResponse>(`/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    });
  },

  /**
   * Retrieve saved model checkpoint metadata for a specific job
   */
  getJobCheckpoints: (jobId: string): Promise<TrainingJobCheckpointsResponse> => {
    return request<TrainingJobCheckpointsResponse>(`/jobs/${encodeURIComponent(jobId)}/checkpoints`);
  },

  /**
   * Retrieve live adaptive hardware profile and VRAM headroom
   */
  getAdaptiveHardware: (): Promise<AdaptiveHardwareInfo> => {
    return request<AdaptiveHardwareInfo>('/adaptive/hardware');
  },

  /**
   * Request adaptive hyperparameter recommendation (batch size, precision, accum)
   */
  getAdaptiveRecommendation: (req: AdaptiveRecommendationRequest): Promise<AdaptiveRecommendationResponse> => {
    return request<AdaptiveRecommendationResponse>('/adaptive/recommend', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },
};
