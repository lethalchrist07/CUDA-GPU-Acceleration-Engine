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
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.08]">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-[#76b900] to-emerald-800 text-slate-950 font-bold border border-[#76b900]/40 shadow-md shadow-[#76b900]/20">
              <Flame className="w-6 h-6 fill-current text-slate-950" />
            </div>
            <div>
              <h2 className="text-2xl font-extrabold text-white tracking-tight">
                AI Training Studio
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
                Real PyTorch & Custom CUDA Training Orchestration • Laptop RTX 3050 6GB GPU
              </p>
            </div>
          </div>
        </div>

        {/* Studio Sub-Navigation Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-obsidian-900 border border-white/[0.08] self-start sm:self-auto font-mono text-xs">
          <button
            onClick={() => setActiveStudioTab('studio')}
            id="tab-studio-workspace"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeStudioTab === 'studio'
                ? 'bg-[#76b900]/20 text-[#76b900] border border-[#76b900]/40 shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
            }`}
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Launch & Monitor</span>
          </button>

          <button
            onClick={() => setActiveStudioTab('history')}
            id="tab-studio-history"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeStudioTab === 'history'
                ? 'bg-[#76b900]/20 text-[#76b900] border border-[#76b900]/40 shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            <span>Job History ({jobs.length})</span>
          </button>

          <button
            onClick={() => setActiveStudioTab('comparison')}
            id="tab-studio-comparison"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeStudioTab === 'comparison'
                ? 'bg-[#76b900]/20 text-[#76b900] border border-[#76b900]/40 shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" />
            <span>3-Backend Comparison</span>
          </button>
        </div>
      </div>

      {/* Error Alert Banner */}
      {submitError && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/25 text-xs text-rose-300 animate-fadeIn">
          <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold">Failed to Launch Training Job</span>
            <p className="font-mono text-[11px] text-rose-200">{submitError}</p>
          </div>
        </div>
      )}

      {/* TAB 1: STUDIO WORKSPACE (Launch & Active Monitor) */}
      {activeStudioTab === 'studio' && (
        <div className="space-y-6">
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
          <div className="glass-panel p-5 sm:p-6 rounded-2xl border border-white/[0.07] space-y-6">
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
