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
} from '../types/training';

const BASE_URL = '/api/training';

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...(options?.headers || {}),
    },
  });

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
    throw new Error(errorDetail);
  }

  return response.json();
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
};
