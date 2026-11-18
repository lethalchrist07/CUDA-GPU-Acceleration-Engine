import React, { useState, useEffect, useCallback } from 'react';
import { SectionTab } from './types/navigation';
import { Sidebar } from './components/Sidebar';
import { CommandBar } from './components/CommandBar';
import { OverviewSection } from './components/OverviewSection';
import { GpuMonitoringSection } from './components/GpuMonitoringSection';
import { TrainingStudio } from './components/training/TrainingStudio';
import { ModelRegistrySection } from './components/ModelRegistrySection';
import { CudaLabSection } from './components/CudaLabSection';
import { PerformanceAnalyticsSection } from './components/PerformanceAnalyticsSection';
import { InferencePlaygroundSection } from './components/InferencePlaygroundSection';
import { VerificationSection } from './components/VerificationSection';
import { ShowcaseExperience } from './components/showcase/ShowcaseExperience';
import { api } from './services/api';
import { trainingApi } from './services/trainingApi';
import {
  OverviewResponse,
  BenchmarkResponse,
  TrainingResponse,
  InferenceResponse,
  HardwareResponse,
  VerificationResponse,
  HealthResponse,
} from './types/api';
import { TrainingJobResponse } from './types/training';
import { Shield, Server } from 'lucide-react';

const VALID_TABS: SectionTab[] = [
  'showcase',
  'overview',
  'gpu_monitoring',
  'training_studio',
  'model_registry',
  'cuda_lab',
  'analytics',
  'inference',
  'verification',
];

const getInitialTab = (): SectionTab => {
  if (typeof window === 'undefined') return 'showcase';
  const path = window.location.pathname.replace(/^\/+/, '').split('/')[0];
  const hash = window.location.hash.replace(/^#/, '');
  if (VALID_TABS.includes(path as SectionTab)) return path as SectionTab;
  if (VALID_TABS.includes(hash as SectionTab)) return hash as SectionTab;
  return 'showcase';
};

export const App: React.FC = () => {
  const [activeTab, setActiveTabState] = useState<SectionTab>(getInitialTab);
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    return window.innerWidth < 1200;
  });

  const setActiveTab = useCallback((tab: SectionTab) => {
    setActiveTabState(tab);
    if (typeof window !== 'undefined') {
      const targetPath = tab === 'showcase' ? '/' : `/${tab}`;
      if (window.location.pathname !== targetPath) {
        window.history.pushState({ tab }, '', targetPath);
      }
    }
  }, []);

  // Sync state on browser back/forward buttons
  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname.replace(/^\/+/, '').split('/')[0];
      if (path && VALID_TABS.includes(path as SectionTab)) {
        setActiveTabState(path as SectionTab);
      } else if (!path) {
        setActiveTabState('showcase');
      }
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

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
  const [isRefreshingHardware, setIsRefreshingHardware] = useState<boolean>(false);

  const [verificationData, setVerificationData] = useState<VerificationResponse | null>(null);
  const [verificationLoading, setVerificationLoading] = useState<boolean>(true);

  const [recentJob, setRecentJob] = useState<TrainingJobResponse | null>(null);

  // Health check
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
    setIsRefreshingHardware(true);
    try {
      const data = await api.getHardware();
      setHardwareData(data);
    } catch (err) {
      console.error('Failed to fetch hardware telemetry:', err);
    } finally {
      setHardwareLoading(false);
      setIsRefreshingHardware(false);
    }
  }, []);

  // Fetch Latest Job for Overview
  const fetchLatestJob = useCallback(async () => {
    try {
      const res = await trainingApi.listJobs({ limit: 1 });
      if (res.jobs && res.jobs.length > 0) {
        setRecentJob(res.jobs[0]);
      }
    } catch (err) {
      console.error('Failed to fetch latest job:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    checkHealth();

    api.getOverview()
      .then(setOverviewData)
      .catch(console.error)
      .finally(() => setOverviewLoading(false));

    api.getBenchmarks()
      .then(setBenchmarksData)
      .catch(console.error)
      .finally(() => setBenchmarksLoading(false));

    api.getTraining()
      .then(setTrainingData)
      .catch(console.error)
      .finally(() => setTrainingLoading(false));

    api.getInference()
      .then(setInferenceData)
      .catch(console.error)
      .finally(() => setInferenceLoading(false));

    fetchHardware();
    fetchLatestJob();

    api.getVerification()
      .then(setVerificationData)
      .catch(console.error)
      .finally(() => setVerificationLoading(false));
  }, [checkHealth, fetchHardware, fetchLatestJob]);

  // Periodic telemetry polling (3 seconds)
  useEffect(() => {
    if (!autoRefreshHardware) return;

    const interval = setInterval(() => {
      fetchHardware();
      checkHealth();
      fetchLatestJob();
    }, 3000);

    return () => clearInterval(interval);
  }, [autoRefreshHardware, fetchHardware, checkHealth, fetchLatestJob]);

  // Keyboard navigation shortcuts [1 - 8]
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is currently typing inside an input, textarea, or select
      const target = e.target as HTMLElement;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;

      const keyMap: Record<string, SectionTab> = {
        '0': 'showcase',
        's': 'showcase',
        'S': 'showcase',
        '1': 'overview',
        '2': 'gpu_monitoring',
        '3': 'training_studio',
        '4': 'model_registry',
        '5': 'cuda_lab',
        '6': 'analytics',
        '7': 'inference',
        '8': 'verification',
      };

      if (keyMap[e.key]) {
        setActiveTab(keyMap[e.key]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const gpuName = hardwareData?.gpus?.[0]?.name || 'NVIDIA RTX 3050 Laptop GPU (6GB)';

  return (
    <div className="min-h-screen bg-[#08090B] text-[#F5F5F4] flex font-sans selection:bg-[#76B900] selection:text-[#08090B]">
      {/* Collapsible Left Navigation Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        health={health}
        healthError={healthError}
        gpuName={gpuName}
      />

      {/* Main Content Area (Offset by Sidebar Width) */}
      <div 
        className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ease-in-out ${
          sidebarCollapsed ? 'pl-20' : 'pl-72'
        }`}
      >
        {/* Top Command Bar */}
        <CommandBar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed((prev) => !prev)}
          health={health}
          healthError={healthError}
          hardwareData={hardwareData}
          onRefreshHardware={fetchHardware}
          isRefreshing={isRefreshingHardware}
        />

        {/* Section Router */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {activeTab === 'showcase' && (
            <ShowcaseExperience
              overviewData={overviewData}
              benchmarksData={benchmarksData}
              hardwareData={hardwareData}
              onNavigateTab={(tab) => setActiveTab(tab)}
            />
          )}

          {activeTab === 'overview' && (
            <OverviewSection
              data={overviewData}
              loading={overviewLoading}
              hardwareData={hardwareData}
              recentJob={recentJob}
              onNavigateTab={(tab) => setActiveTab(tab)}
            />
          )}

          {activeTab === 'gpu_monitoring' && (
            <GpuMonitoringSection
              data={hardwareData}
              loading={hardwareLoading}
              onRefresh={fetchHardware}
              autoRefresh={autoRefreshHardware}
              onToggleAutoRefresh={() => setAutoRefreshHardware((prev) => !prev)}
            />
          )}

          {activeTab === 'training_studio' && (
            <TrainingStudio />
          )}

          {activeTab === 'model_registry' && (
            <ModelRegistrySection
              onNavigateTab={(tab) => setActiveTab(tab)}
            />
          )}

          {activeTab === 'cuda_lab' && (
            <CudaLabSection
              data={benchmarksData}
              loading={benchmarksLoading}
            />
          )}

          {activeTab === 'analytics' && (
            <PerformanceAnalyticsSection
              data={trainingData}
              loading={trainingLoading}
            />
          )}

          {activeTab === 'inference' && (
            <InferencePlaygroundSection
              data={inferenceData}
              loading={inferenceLoading}
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
        <footer className="border-t border-[#30353D] py-5 mt-12 bg-[#0D0F12] text-xs text-[#A6ADB7]">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex flex-wrap items-center justify-center md:justify-start gap-2 text-center md:text-left">
              <Shield className="w-4 h-4 text-[#76B900]" />
              <span className="text-[#F5F5F4] font-medium">CUDA GPU Acceleration Engine</span>
              <span className="text-[#30353D]">•</span>
              <span className="text-[#A6ADB7]">Ampere SM 8.6 Architecture Platform</span>
              <span className="text-[#30353D] hidden sm:inline">•</span>
              <span className="text-[#7A828E] text-[11px] block sm:inline">Not affiliated with NVIDIA Corporation</span>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-3 text-[#A6ADB7] font-mono text-[11px]">
              <span className="flex items-center gap-1.5 text-[#F5F5F4]">
                <Server className="w-3.5 h-3.5 text-[#70D6C2]" />
                API: 127.0.0.1:8000
              </span>
              <span className="text-[#30353D]">•</span>
              <span className="text-[#76B900] font-semibold uppercase tracking-wider">
                Read-Only Confined
              </span>
              <span className="text-[#30353D]">•</span>
              <span className="text-[#A6ADB7]">
                SHA-256 Verified
              </span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
