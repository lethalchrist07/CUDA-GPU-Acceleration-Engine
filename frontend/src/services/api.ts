import {
  OverviewResponse,
  BenchmarkResponse,
  TrainingResponse,
  InferenceResponse,
  HardwareResponse,
  VerificationResponse,
  HealthResponse,
} from '../types/api';

const API_BASE = '/api';

async function fetchJson<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`API Error ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

export const api = {
  getHealth: () => fetchJson<HealthResponse>('/health'),
  getOverview: () => fetchJson<OverviewResponse>('/overview'),
  getBenchmarks: () => fetchJson<BenchmarkResponse>('/benchmarks'),
  getTraining: () => fetchJson<TrainingResponse>('/training'),
  getInference: () => fetchJson<InferenceResponse>('/inference'),
  getHardware: () => fetchJson<HardwareResponse>('/hardware'),
  getVerification: () => fetchJson<VerificationResponse>('/verification'),
  getArtifactUrl: (artifactId: string) => `${API_BASE}/artifacts/${artifactId}`,
};
