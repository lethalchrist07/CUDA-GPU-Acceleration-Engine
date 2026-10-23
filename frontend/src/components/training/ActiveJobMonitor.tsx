import React from 'react';
import { TrainingJobResponse } from '../../types/training';
import { 
  Activity, 
  Clock, 
  Flame, 
  Zap, 
  StopCircle, 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  Cpu, 
  HardDrive,
  RefreshCw,
  Award
} from 'lucide-react';

interface ActiveJobMonitorProps {
  job: TrainingJobResponse;
  onCancel: (jobId: string) => Promise<void>;
  isCancelling: boolean;
  onViewCheckpoints?: (jobId: string) => void;
}

export const ActiveJobMonitor: React.FC<ActiveJobMonitorProps> = ({
  job,
  onCancel,
  isCancelling,
  onViewCheckpoints,
}) => {
  const isTerminal = ['COMPLETED', 'CANCELLED', 'FAILED', 'INTERRUPTED'].includes(job.status);
  const isCancelRequested = job.status === 'CANCEL_REQUESTED';

  const progress = job.progress || {};
  const metrics = job.metrics || {};

  // Epoch progress
  const currentEpoch = progress.epoch ?? 0;
  const totalEpochs = progress.total_epochs ?? job.epochs ?? 1;
  const currentBatch = progress.batch ?? 0;
  const totalBatches = progress.total_batches ?? 1;

  // Calculate overall percentage
  const epochWeight = 100 / Math.max(1, totalEpochs);
  const currentEpochProgress = totalBatches > 0 ? (currentBatch / totalBatches) * epochWeight : 0;
  const completedEpochsProgress = Math.max(0, currentEpoch - 1) * epochWeight;
  const percentComplete = Math.min(
    100,
    job.status === 'COMPLETED'
      ? 100
      : Math.round(completedEpochsProgress + currentEpochProgress)
  );

  const getStatusBadge = () => {
    switch (job.status) {
      case 'QUEUED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            QUEUED (Waiting for GPU Lock)
          </span>
        );
      case 'RUNNING':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold bg-[#76b900]/15 text-[#76b900] border border-[#76b900]/40 shadow-[0_0_12px_rgba(118,185,0,0.2)]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#76b900] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#76b900]"></span>
            </span>
            RUNNING
          </span>
        );
      case 'CANCEL_REQUESTED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold bg-rose-500/15 text-rose-400 border border-rose-500/30 animate-pulse">
            <RefreshCw className="w-3 h-3 animate-spin text-rose-400" />
            CANCELLING (At Step Boundary)
          </span>
        );
      case 'CANCELLED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold bg-slate-800 text-slate-300 border border-white/[0.1]">
            <StopCircle className="w-3.5 h-3.5 text-slate-400" />
            CANCELLED
          </span>
        );
      case 'COMPLETED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shadow-[0_0_12px_rgba(16,185,129,0.2)]">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            COMPLETED
          </span>
        );
      case 'FAILED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold bg-rose-500/15 text-rose-400 border border-rose-500/30">
            <XCircle className="w-3.5 h-3.5 text-rose-400" />
            FAILED
          </span>
        );
      case 'INTERRUPTED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30">
            <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
            INTERRUPTED (Restart Recovery)
          </span>
        );
      default:
        return null;
    }
  };

  const lossValue = progress.train_loss ?? metrics.final_train_loss;
  const valAccValue = progress.val_acc ?? progress.val_accuracy ?? metrics.best_val_acc;
  const elapsedSec = progress.elapsed_sec ?? progress.elapsed_time_sec ?? metrics.total_wall_time_sec ?? 0;
  const throughput = progress.samples_per_sec ?? metrics.samples_per_sec ?? metrics.throughput_samples_per_sec ?? 0;
  const vramUsage = progress.peak_vram_mb ?? progress.peak_allocated_vram_mb ?? metrics.peak_allocated_vram_mb ?? 0;
  const customDispatches = metrics.custom_cuda_dispatches
    ? Object.values(metrics.custom_cuda_dispatches).reduce((a: any, b: any) => Number(a) + Number(b), 0)
    : (metrics.dispatch_summary?.total_custom_dispatches ?? 0);

  return (
    <div className="glass-panel p-5 sm:p-6 rounded-2xl border border-white/[0.08] space-y-5">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-white/[0.06]">
        <div>
          <div className="flex items-center gap-2.5">
            <h3 className="text-base font-extrabold text-white tracking-tight font-mono">
              {job.job_id}
            </h3>
            {getStatusBadge()}
          </div>
          <div className="flex flex-wrap items-center gap-2 mt-1 text-xs text-slate-400 font-mono">
            <span>Model: <strong className="text-white">{job.model_name}</strong></span>
            <span>•</span>
            <span>Backend: <strong className="text-[#76b900]">{job.backend}</strong></span>
            <span>•</span>
            <span>Dataset: <strong className="text-slate-300">{job.dataset}</strong></span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {!isTerminal && (
            <button
              onClick={() => onCancel(job.job_id)}
              disabled={isCancelling || isCancelRequested}
              id="btn-cancel-job"
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-sm cursor-pointer ${
                isCancelling || isCancelRequested
                  ? 'bg-obsidian-900 text-slate-500 border border-white/[0.05] cursor-not-allowed'
                  : 'bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30'
              }`}
            >
              <StopCircle className="w-3.5 h-3.5" />
              <span>{isCancelRequested ? 'Cancelling...' : 'Cancel Job'}</span>
            </button>
          )}

          {job.checkpoint_path && onViewCheckpoints && (
            <button
              onClick={() => onViewCheckpoints(job.job_id)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-[#76b900]/15 hover:bg-[#76b900]/25 text-[#76b900] border border-[#76b900]/30 transition-all cursor-pointer"
            >
              <HardDrive className="w-3.5 h-3.5" />
              <span>Inspect Checkpoint</span>
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-slate-300 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-[#76b900]" />
            Epoch {currentEpoch} / {totalEpochs}
            {currentBatch > 0 && (
              <span className="text-slate-500">
                (Batch {currentBatch} / {totalBatches})
              </span>
            )}
          </span>
          <span className="text-[#76b900] font-bold">
            {percentComplete}% Complete
          </span>
        </div>
        <div className="w-full h-2.5 rounded-full bg-obsidian-950 border border-white/[0.05] overflow-hidden p-[1px]">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              job.status === 'FAILED'
                ? 'bg-rose-500'
                : job.status === 'CANCELLED'
                ? 'bg-slate-600'
                : 'bg-gradient-to-r from-[#76b900] to-emerald-400 shadow-[0_0_12px_rgba(118,185,0,0.5)]'
            }`}
            style={{ width: `${percentComplete}%` }}
          />
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Loss */}
        <div className="p-3 rounded-xl bg-obsidian-900/80 border border-white/[0.05] space-y-1">
          <div className="flex items-center gap-1 text-[11px] text-slate-400">
            <Flame className="w-3 h-3 text-amber-400" />
            <span>Train Loss</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-white">
            {lossValue !== undefined ? lossValue.toFixed(4) : '--'}
          </div>
        </div>

        {/* Validation Accuracy */}
        <div className="p-3 rounded-xl bg-obsidian-900/80 border border-white/[0.05] space-y-1">
          <div className="flex items-center gap-1 text-[11px] text-slate-400">
            <Award className="w-3 h-3 text-emerald-400" />
            <span>Val Accuracy</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-emerald-400">
            {valAccValue !== undefined ? `${valAccValue.toFixed(2)}%` : '--'}
          </div>
        </div>

        {/* Throughput */}
        <div className="p-3 rounded-xl bg-obsidian-900/80 border border-white/[0.05] space-y-1">
          <div className="flex items-center gap-1 text-[11px] text-slate-400">
            <Zap className="w-3 h-3 text-[#76b900]" />
            <span>Throughput</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-white">
            {throughput > 0 ? `${Math.round(throughput)} smp/s` : '--'}
          </div>
        </div>

        {/* Wall Time */}
        <div className="p-3 rounded-xl bg-obsidian-900/80 border border-white/[0.05] space-y-1">
          <div className="flex items-center gap-1 text-[11px] text-slate-400">
            <Clock className="w-3 h-3 text-cyan-400" />
            <span>Elapsed Time</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-white">
            {elapsedSec.toFixed(2)}s
          </div>
        </div>

        {/* Peak VRAM */}
        <div className="p-3 rounded-xl bg-obsidian-900/80 border border-white/[0.05] space-y-1">
          <div className="flex items-center gap-1 text-[11px] text-slate-400">
            <Cpu className="w-3 h-3 text-purple-400" />
            <span>VRAM Allocated</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-white">
            {job.backend === 'CPU_NATIVE' ? '0.00 MB' : `${vramUsage.toFixed(1)} MB`}
          </div>
        </div>

        {/* Custom CUDA Dispatches */}
        <div className="p-3 rounded-xl bg-obsidian-900/80 border border-white/[0.05] space-y-1">
          <div className="flex items-center gap-1 text-[11px] text-slate-400">
            <Zap className="w-3 h-3 text-[#76b900]" />
            <span>CUDA Dispatches</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-[#76b900]">
            {job.backend === 'GPU_CUSTOM' ? `${customDispatches}` : '0'}
          </div>
        </div>
      </div>

      {/* Error Message Alert */}
      {job.error_message && (
        <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/25 text-xs text-rose-300">
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <span className="font-bold">Execution Error:</span>
            <p className="font-mono text-[11px] text-rose-200">{job.error_message}</p>
          </div>
        </div>
      )}
    </div>
  );
};
