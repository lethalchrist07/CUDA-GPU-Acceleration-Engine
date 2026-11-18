import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  ModelCapabilityInfo,
  TrainingJobResponse,
  TrainingJobCreateRequest,
  BackendType,
} from '../../types/training';
import { trainingApi } from '../../services/trainingApi';
import { ModelSelector } from './ModelSelector';
import { TrainingConfigForm } from './TrainingConfigForm';
import { ActiveJobMonitor } from './ActiveJobMonitor';
import { TrainingCharts } from './TrainingCharts';
import { TrainingHistoryTable } from './TrainingHistoryTable';
import { CheckpointInspector } from './CheckpointInspector';
import { ComparisonView } from './ComparisonView';
import { 
  Flame, 
  Play, 
  History, 
  GitCompare, 
  AlertCircle
} from 'lucide-react';

export type StudioTab = 'studio' | 'history' | 'comparison';

export const TrainingStudio: React.FC = () => {
  const [activeStudioTab, setActiveStudioTab] = useState<StudioTab>('studio');

  // Models
  const [models, setModels] = useState<ModelCapabilityInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('cifar10_mlp');
  const [selectedBackend, setSelectedBackend] = useState<BackendType>('GPU_CUSTOM');

  // Jobs & History
  const [jobs, setJobs] = useState<TrainingJobResponse[]>([]);
  const [jobsLoading, setJobsLoading] = useState<boolean>(true);
  const [activeJob, setActiveJob] = useState<TrainingJobResponse | null>(null);

  // Actions
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [inspectingCheckpointJobId, setInspectingCheckpointJobId] = useState<string | null>(null);

  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch models
  const fetchModels = useCallback(async () => {
    try {
      const res = await trainingApi.getModels();
      setModels(res.models);
      if (res.models.length > 0) {
        setSelectedModelId(res.models[0].model_id);
        if (res.models[0].supported_backends.includes('GPU_CUSTOM')) {
          setSelectedBackend('GPU_CUSTOM');
        } else {
          setSelectedBackend(res.models[0].supported_backends[0] as BackendType);
        }
      }
    } catch (err: any) {
      console.error('Failed to load training models:', err);
    }
  }, []);

  // Fetch jobs
  const fetchJobs = useCallback(async () => {
    try {
      setJobsLoading(true);
      const res = await trainingApi.listJobs({ limit: 50 });
      setJobs(res.jobs);

      // If no active job is selected yet, default to the most recent running or completed job
      setActiveJob((prev) => {
        if (prev) {
          // Keep current active job up to date with fresh record
          const updated = res.jobs.find((j) => j.job_id === prev.job_id);
          return updated || prev;
        }
        return res.jobs[0] || null;
      });
    } catch (err: any) {
      console.error('Failed to load training jobs:', err);
    } finally {
      setJobsLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchModels();
    fetchJobs();
  }, [fetchModels, fetchJobs]);

  // Live polling for non-terminal active job
  useEffect(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    if (
      activeJob &&
      ['QUEUED', 'RUNNING', 'CANCEL_REQUESTED'].includes(activeJob.status)
    ) {
      const activeJobId = activeJob.job_id;
      pollingRef.current = setInterval(async () => {
        try {
          const freshJob = await trainingApi.getJob(activeJobId);
          setActiveJob(freshJob);

          // Update in jobs list as well
          setJobs((prevJobs) =>
            prevJobs.map((j) => (j.job_id === freshJob.job_id ? freshJob : j))
          );

          // If reached terminal state, stop polling and refresh full list
          if (['COMPLETED', 'CANCELLED', 'FAILED', 'INTERRUPTED'].includes(freshJob.status)) {
            if (pollingRef.current) {
              clearInterval(pollingRef.current);
              pollingRef.current = null;
            }
            fetchJobs();
          }
        } catch (err) {
          console.error('Error polling active job:', err);
        }
      }, 1000);
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [activeJob?.status, activeJob?.job_id, fetchJobs]);

  // Handle Model Change
  const handleSelectModel = (modelId: string) => {
    setSelectedModelId(modelId);
    const m = models.find((mod) => mod.model_id === modelId);
    if (m && !m.supported_backends.includes(selectedBackend)) {
      setSelectedBackend(m.supported_backends[0] as BackendType);
    }
  };

  // Launch Training Job
  const handleLaunchJob = async (request: TrainingJobCreateRequest) => {
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      const newJob = await trainingApi.createJob(request);
      setActiveJob(newJob);
      setJobs((prev) => [newJob, ...prev]);
      setActiveStudioTab('studio');
    } catch (err: any) {
      setSubmitError(err.message || 'Failed to submit training job');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Cancel Job
  const handleCancelJob = async (jobId: string) => {
    setIsCancelling(true);
    try {
      await trainingApi.cancelJob(jobId);
      // Immediately reflect CANCEL_REQUESTED locally for snappy UX
      setActiveJob((prev) =>
        prev && prev.job_id === jobId ? { ...prev, status: 'CANCEL_REQUESTED' } : prev
      );
    } catch (err: any) {
      console.error('Failed to cancel job:', err);
    } finally {
      setIsCancelling(false);
    }
  };

  return (
    <div id="section-training-studio" className="space-y-6 animate-fadeIn">
      {/* Studio Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#171A1F] rounded-2xl p-6 border border-[#30353D] shadow-sm">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-[#76B900]/10 text-[#76B900] border border-[#76B900]/30">
              <Flame className="w-6 h-6 text-[#76B900]" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-[#F5F5F4] tracking-tight font-sans">
                AI Training Studio
              </h2>
              <p className="text-xs sm:text-sm text-[#A6ADB7] mt-0.5 font-sans">
                Real PyTorch &amp; Custom CUDA Training Orchestration • Laptop RTX 3050 6GB GPU
              </p>
            </div>
          </div>
        </div>

        {/* Studio Sub-Navigation Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-[#121519] border border-[#30353D] self-start sm:self-auto font-sans text-xs">
          <button
            onClick={() => setActiveStudioTab('studio')}
            id="tab-studio-workspace"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
              activeStudioTab === 'studio'
                ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A]'
            }`}
          >
            <Play className="w-3.5 h-3.5 fill-current text-[#76B900]" />
            <span>Launch &amp; Monitor</span>
          </button>

          <button
            onClick={() => setActiveStudioTab('history')}
            id="tab-studio-history"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
              activeStudioTab === 'history'
                ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A]'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            <span>Job History ({jobs.length})</span>
          </button>

          <button
            onClick={() => setActiveStudioTab('comparison')}
            id="tab-studio-comparison"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
              activeStudioTab === 'comparison'
                ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A]'
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" />
            <span>3-Backend Comparison</span>
          </button>
        </div>
      </div>

      {/* Error Alert Banner */}
      {submitError && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-[#171A1F] border border-[#FB7185]/30 text-xs text-[#F5F5F4] animate-fadeIn">
          <AlertCircle className="w-5 h-5 text-[#FB7185] flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold text-[#FB7185] font-sans">Failed to Launch Training Job</span>
            <p className="font-mono text-[11px] text-[#FB7185]">{submitError}</p>
          </div>
        </div>
      )}

      {/* TAB 1: STUDIO WORKSPACE (Launch & Active Monitor) */}
      {activeStudioTab === 'studio' && (
        <div className="space-y-6">
          {/* 6-Step Workflow Stepper Banner */}
          <div className="p-3.5 rounded-xl bg-[#121519] border border-[#30353D] overflow-x-auto scrollbar-none">
            <div className="flex items-center min-w-[700px] justify-between text-xs font-sans">
              {(() => {
                const isJobRunning = activeJob && ['QUEUED', 'RUNNING', 'CANCEL_REQUESTED'].includes(activeJob.status);
                const isJobTerminal = activeJob && ['COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(activeJob.status);
                return [
                  { step: 1, label: 'Choose Model', active: !activeJob, completed: Boolean(activeJob) },
                  { step: 2, label: 'Configure Training', active: !activeJob && Boolean(selectedModelId), completed: Boolean(activeJob) },
                  { step: 3, label: 'Review Config', active: false, completed: Boolean(activeJob) },
                  { step: 4, label: 'Launch', active: isSubmitting, completed: Boolean(activeJob) && !isSubmitting },
                  { step: 5, label: 'Monitor', active: isJobRunning, completed: isJobTerminal },
                  { step: 6, label: 'Inspect Results', active: isJobTerminal, completed: isJobTerminal && activeJob.status === 'COMPLETED' },
                ];
              })().map((item, idx, arr) => {
                const isHighlighted = item.active || (item.completed && item.step === 6);
                const isDone = item.completed && !item.active;
                return (
                  <React.Fragment key={item.step}>
                    <div className={`flex items-center gap-2 px-2.5 py-1 rounded-lg transition-all ${
                      isHighlighted
                        ? 'bg-[#76B900]/10 text-[#F5F5F4] font-semibold border border-[#76B900]/30'
                        : isDone
                        ? 'bg-[#4ADE80]/5 text-[#A6ADB7] border border-[#4ADE80]/20'
                        : 'text-[#6B7280] opacity-60'
                    }`}>
                      <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono font-bold ${
                        isHighlighted 
                          ? 'bg-[#76B900] text-[#08090B]' 
                          : isDone 
                          ? 'bg-[#4ADE80]/20 text-[#4ADE80] border border-[#4ADE80]/40' 
                          : 'bg-[#171A1F] text-[#A6ADB7] border border-[#30353D]'
                      }`}>
                        {isDone ? '✓' : item.step}
                      </span>
                      <span>{item.label}</span>
                    </div>
                    {idx < arr.length - 1 && (
                      <span className="text-[#30353D] px-1 font-mono">→</span>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>

          {/* Active Job Live Monitor (If any active job) */}
          {activeJob && (
            <div className="space-y-4">
              <ActiveJobMonitor
                job={activeJob}
                onCancel={handleCancelJob}
                isCancelling={isCancelling}
                onViewCheckpoints={(jobId) => setInspectingCheckpointJobId(jobId)}
              />

              {/* Real-time Scientific Charts */}
              <TrainingCharts job={activeJob} />
            </div>
          )}

          {/* Model Selection & Configuration Forms */}
          <div className="bg-[#171A1F] p-5 sm:p-6 rounded-2xl border border-[#30353D] space-y-6 interactive-card">
            <ModelSelector
              models={models}
              selectedModelId={selectedModelId}
              onSelectModel={handleSelectModel}
              selectedBackend={selectedBackend}
              onSelectBackend={setSelectedBackend}
              disabled={isSubmitting || (activeJob?.status === 'RUNNING' && selectedBackend !== 'CPU_NATIVE')}
            />

            <TrainingConfigForm
              modelId={selectedModelId}
              backend={selectedBackend}
              onSubmit={handleLaunchJob}
              isSubmitting={isSubmitting}
              disabled={isSubmitting}
            />
          </div>
        </div>
      )}

      {/* TAB 2: TRAINING HISTORY */}
      {activeStudioTab === 'history' && (
        <TrainingHistoryTable
          jobs={jobs}
          selectedJobId={activeJob?.job_id}
          onSelectJob={(j) => {
            setActiveJob(j);
            setActiveStudioTab('studio');
          }}
          onRefresh={fetchJobs}
          isLoading={jobsLoading}
          onViewCheckpoints={(jobId) => setInspectingCheckpointJobId(jobId)}
        />
      )}

      {/* TAB 3: 3-BACKEND COMPARISON */}
      {activeStudioTab === 'comparison' && (
        <ComparisonView jobs={jobs} />
      )}

      {/* Checkpoint Inspector Modal */}
      {inspectingCheckpointJobId && (
        <CheckpointInspector
          jobId={inspectingCheckpointJobId}
          onClose={() => setInspectingCheckpointJobId(null)}
        />
      )}
    </div>
  );
};
