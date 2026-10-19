import React, { useState, useEffect, useCallback } from 'react';
import { Navbar, SectionTab } from './components/Navbar';
import { OverviewSection } from './components/OverviewSection';
import { BenchmarksSection } from './components/BenchmarksSection';
import { TrainingSection } from './components/TrainingSection';
import { InferenceSection } from './components/InferenceSection';
import { HardwareSection } from './components/HardwareSection';
import { VerificationSection } from './components/VerificationSection';
import { api } from './services/api';
import {
  OverviewResponse,
  BenchmarkResponse,
  TrainingResponse,
  InferenceResponse,
  HardwareResponse,
  VerificationResponse,
  HealthResponse,
} from './types/api';
import { Shield, Server } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<SectionTab>('overview');

  // Health
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<boolean>(false);

  // Section Data
  const [overviewData, setOverviewData] = useState<OverviewResponse | null>(null);
  const [overviewLoading, setOverviewLoading] = useState<boolean>(true);

  const [benchmarksData, setBenchmarksData] = useState<BenchmarkResponse | null>(null);
  const [benchmarksLoading, setBenchmarksLoading] = useState<boolean>(true);

  const [trainingData, setTrainingData] = useState<TrainingResponse | null>(null);
  const [trainingLoading, setTrainingLoading] = useState<boolean>(true);

  const [inferenceData, setInferenceData] = useState<InferenceResponse | null>(null);
  const [inferenceLoading, setInferenceLoading] = useState<boolean>(true);

  const [hardwareData, setHardwareData] = useState<HardwareResponse | null>(null);
  const [hardwareLoading, setHardwareLoading] = useState<boolean>(true);
  const [autoRefreshHardware, setAutoRefreshHardware] = useState<boolean>(true);

  const [verificationData, setVerificationData] = useState<VerificationResponse | null>(null);
  const [verificationLoading, setVerificationLoading] = useState<boolean>(true);

  // Load health
  const checkHealth = useCallback(async () => {
    try {
      const h = await api.getHealth();
      setHealth(h);
      setHealthError(false);
    } catch {
      setHealthError(true);
    }
  }, []);

  // Fetch Hardware
  const fetchHardware = useCallback(async () => {
    try {
      const data = await api.getHardware();
      setHardwareData(data);
    } catch (err) {
      console.error('Failed to fetch hardware telemetry:', err);
    } finally {
      setHardwareLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    checkHealth();

    // Fetch overview
    api.getOverview()
      .then(setOverviewData)
      .catch(console.error)
      .finally(() => setOverviewLoading(false));

    // Fetch benchmarks
    api.getBenchmarks()
      .then(setBenchmarksData)
      .catch(console.error)
      .finally(() => setBenchmarksLoading(false));

    // Fetch training
    api.getTraining()
      .then(setTrainingData)
      .catch(console.error)
      .finally(() => setTrainingLoading(false));

    // Fetch inference
    api.getInference()
      .then(setInferenceData)
      .catch(console.error)
      .finally(() => setInferenceLoading(false));

    // Fetch hardware
    fetchHardware();

    // Fetch verification
    api.getVerification()
      .then(setVerificationData)
      .catch(console.error)
      .finally(() => setVerificationLoading(false));
  }, [checkHealth, fetchHardware]);

  // Periodic hardware telemetry polling
  useEffect(() => {
    if (!autoRefreshHardware) return;

    const interval = setInterval(() => {
      fetchHardware();
      checkHealth();
    }, 3000);

    return () => clearInterval(interval);
  }, [autoRefreshHardware, fetchHardware, checkHealth]);

  return (
    <div className="min-h-screen bg-obsidian-950 text-slate-100 flex flex-col font-sans selection:bg-[#76b900] selection:text-slate-950">
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        health={health}
        healthError={healthError}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {activeTab === 'overview' && (
          <OverviewSection
            data={overviewData}
            loading={overviewLoading}
            onNavigateTab={(tab) => setActiveTab(tab)}
          />
        )}

        {activeTab === 'benchmarks' && (
          <BenchmarksSection
            data={benchmarksData}
            loading={benchmarksLoading}
          />
        )}

        {activeTab === 'training' && (
          <TrainingSection
            data={trainingData}
            loading={trainingLoading}
          />
        )}

        {activeTab === 'inference' && (
          <InferenceSection
            data={inferenceData}
            loading={inferenceLoading}
          />
        )}

        {activeTab === 'hardware' && (
          <HardwareSection
            data={hardwareData}
            loading={hardwareLoading}
            onRefresh={fetchHardware}
            autoRefresh={autoRefreshHardware}
            onToggleAutoRefresh={() => setAutoRefreshHardware(prev => !prev)}
          />
        )}

        {activeTab === 'verification' && (
          <VerificationSection
            data={verificationData}
            loading={verificationLoading}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="glass-panel border-t border-white/[0.07] py-6 mt-12 bg-obsidian-950/90 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex flex-wrap items-center justify-center md:justify-start gap-2 text-center md:text-left">
            <Shield className="w-4 h-4 text-[#76b900]" />
            <span className="text-slate-200 font-semibold">CUDA GPU Acceleration Engine</span>
            <span className="text-slate-600">•</span>
            <span className="text-slate-400">Independent Systems Engineering Portfolio (2028 Target)</span>
            <span className="text-slate-600 hidden sm:inline">•</span>
            <span className="text-slate-400 text-[11px] block sm:inline">Not affiliated with NVIDIA Corporation</span>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3 text-slate-400">
            <span className="flex items-center gap-1.5 font-mono text-[11px] text-slate-300">
              <Server className="w-3.5 h-3.5 text-[#76b900]" />
              API: 127.0.0.1:8000
            </span>
            <span className="text-slate-600">•</span>
            <span className="text-[#76b900] font-mono text-[11px] font-semibold uppercase tracking-wider">
              Read-Only Confined
            </span>
            <span className="text-slate-600">•</span>
            <span className="font-mono text-[11px] text-slate-400">
              SHA-256 Verified
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
