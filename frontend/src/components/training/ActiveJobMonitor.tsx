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

  const isCompleted = job.status === 'COMPLETED';

  // Detect whether this is a language model job
  const isTransformer =
    job.model_name?.toLowerCase().includes('transformer') ||
    job.model_name?.toLowerCase().includes('gpt') ||
    job.config?.model_id?.toLowerCase().includes('transformer');

  // Transformer step & token metrics
  const actualOptimizerSteps = metrics.optimizer_steps_completed 
    ?? (metrics.step_records && metrics.step_records.length > 0 ? metrics.step_records.length : undefined);
  const totalSteps = isCompleted && actualOptimizerSteps !== undefined
    ? actualOptimizerSteps
    : (progress.total_steps ?? job.config?.max_steps ?? actualOptimizerSteps ?? 1);
  const currentStep = isCompleted 
    ? (actualOptimizerSteps ?? totalSteps)
    : (progress.current_step ?? progress.step ?? 0);
  const totalTokens = progress.tokens_processed ?? progress.total_tokens ?? metrics.total_tokens_processed ?? 0;
  const tokenBudget = progress.total_tokens_budget ?? (totalSteps * (job.config?.seq_len || 128) * (job.config?.batch_size || 4) * (job.config?.gradient_accumulation_steps || 2));

  // Epoch progress (for vision / classical models)
  const totalEpochs = progress.total_epochs ?? job.epochs ?? 1;
  const rawEpoch = progress.epoch ?? progress.current_epoch;
  const currentEpoch = isCompleted ? totalEpochs : (rawEpoch ?? (job.status === 'QUEUED' ? 0 : 1));
  const totalBatches = progress.total_batches ?? progress.total_steps ?? 1;
  const rawBatch = progress.batch ?? progress.current_step;
  const currentBatch = isCompleted ? totalBatches : (rawBatch ?? 0);

  // Calculate overall percentage
  let percentComplete = 0;
  if (isCompleted) {
    percentComplete = 100;
  } else if (isTransformer) {
    percentComplete = totalSteps > 0 ? Math.min(99, Math.max(0, Math.round((currentStep / totalSteps) * 100))) : 0;
  } else {
    const epochWeight = 100 / Math.max(1, totalEpochs);
    const currentEpochProgress = totalBatches > 0 ? (currentBatch / totalBatches) * epochWeight : 0;
    const completedEpochsProgress = Math.max(0, currentEpoch - 1) * epochWeight;
    percentComplete = Math.min(
      99,
      Math.max(0, Math.round(completedEpochsProgress + currentEpochProgress))
    );
  }

  const getStatusBadge = () => {
    switch (job.status) {
      case 'QUEUED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30">
            <span className="w-2 h-2 rounded-full bg-[#FBBF24] animate-pulse" />
            QUEUED (Waiting for GPU Lock)
          </span>
        );
      case 'RUNNING':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-[#76B900]/10 text-[#76B900] border border-[#76B900]/30">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#76B900] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#76B900]"></span>
            </span>
            RUNNING
          </span>
        );
      case 'CANCEL_REQUESTED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-[#FB7185]/10 text-[#FB7185] border border-[#FB7185]/30 animate-pulse">
            <RefreshCw className="w-3 h-3 animate-spin text-[#FB7185]" />
            CANCELLING (At Step Boundary)
          </span>
        );
      case 'CANCELLED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-[#121519] text-[#A6ADB7] border border-[#30353D]">
            <StopCircle className="w-3.5 h-3.5 text-[#A6ADB7]" />
            CANCELLED
          </span>
        );
      case 'COMPLETED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-[#4ADE80]/10 text-[#4ADE80] border border-[#4ADE80]/30">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#4ADE80]" />
            COMPLETED
          </span>
        );
      case 'FAILED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-[#FB7185]/10 text-[#FB7185] border border-[#FB7185]/30">
            <XCircle className="w-3.5 h-3.5 text-[#FB7185]" />
            FAILED
          </span>
        );
      case 'INTERRUPTED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30">
            <AlertCircle className="w-3.5 h-3.5 text-[#FBBF24]" />
            INTERRUPTED (Restart Recovery)
          </span>
        );
      default:
        return null;
    }
  };

  const lossValue = metrics.final_train_loss ?? progress.train_loss;
  const valAccValue = metrics.best_val_acc ?? progress.best_val_accuracy ?? progress.val_accuracy ?? progress.val_acc;
  const valPerplexity = progress.val_perplexity ?? metrics.final_val_perplexity;
  const elapsedSec = progress.elapsed_sec ?? progress.elapsed_time_sec ?? metrics.total_wall_time_sec ?? 0;
  const throughput = isTransformer
    ? (progress.tokens_per_sec ?? metrics.tokens_per_sec ?? 0)
    : (progress.samples_per_sec ?? metrics.samples_per_sec ?? metrics.throughput_samples_per_sec ?? 0);
  const vramUsage = progress.peak_vram_mb ?? progress.peak_allocated_vram_mb ?? metrics.peak_allocated_vram_mb ?? 0;
  const reservedVram = progress.reserved_vram_mb ?? progress.peak_reserved_vram_mb;
  const gpuTemp = progress.gpu_temperature_c ?? metrics.gpu_temperature_c;
  const gpuUtil = progress.gpu_utilization_pct ?? metrics.gpu_utilization_pct;
  const physicalVram = progress.physical_vram_mb ?? metrics.physical_vram_mb;
  const customDispatches = metrics.custom_cuda_dispatches
    ? Object.values(metrics.custom_cuda_dispatches).reduce((a: any, b: any) => Number(a) + Number(b), 0)
    : (metrics.dispatch_summary?.total_custom_dispatches ?? 0);

  return (
    <div className="bg-[#171A1F] p-5 sm:p-6 rounded-2xl border border-[#30353D] space-y-5 interactive-card">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#30353D]">
        <div>
          <div className="flex items-center gap-2.5">
            <h3 className="text-base font-bold text-[#F5F5F4] tracking-tight font-mono">
              {job.job_id}
            </h3>
            {getStatusBadge()}
            {(job.config?.adaptive_mode || job.config?.adaptive_decision) && (
              <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#70D6C2]/15 text-[#70D6C2] border border-[#70D6C2]/30">
                <Zap className="w-3 h-3 text-[#70D6C2]" />
                ADAPTIVE MODE
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 mt-1 text-xs text-[#A6ADB7] font-sans">
            <span>Model: <strong className="text-[#F5F5F4] font-mono">{job.model_name}</strong></span>
            <span className="text-[#30353D]">•</span>
            <span>Backend: <strong className="text-[#76B900] font-mono">{job.backend}</strong></span>
            <span className="text-[#30353D]">•</span>
            <span>Dataset: <strong className="text-[#F5F5F4] font-mono">{job.dataset}</strong></span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {!isTerminal && (
            <button
              onClick={() => onCancel(job.job_id)}
              disabled={isCancelling || isCancelRequested}
              id="btn-cancel-job"
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold font-sans transition-all cursor-pointer ${
                isCancelling || isCancelRequested
                  ? 'bg-[#121519] text-[#A6ADB7]/50 border border-[#30353D] cursor-not-allowed'
                  : 'bg-[#FB7185]/15 hover:bg-[#FB7185]/25 text-[#FB7185] border border-[#FB7185]/30'
              }`}
            >
              <StopCircle className="w-3.5 h-3.5" />
              <span>{isCancelRequested ? 'Cancelling...' : 'Cancel Job'}</span>
            </button>
          )}

          {job.checkpoint_path && onViewCheckpoints && (
            <button
              onClick={() => onViewCheckpoints(job.job_id)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold font-sans bg-[#76B900]/15 hover:bg-[#76B900]/25 text-[#76B900] border border-[#76B900]/40 transition-all cursor-pointer btn-primary-lift"
            >
              <HardDrive className="w-3.5 h-3.5" />
              <span>Inspect Checkpoint</span>
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          {isTransformer ? (
            <span className="text-[#A6ADB7] font-sans flex items-center gap-1.5 flex-wrap">
              <Activity className="w-3.5 h-3.5 text-[#76B900]" />
              <span>Step</span>
              <span className="font-mono text-[#F5F5F4] font-semibold">{currentStep}</span>
              <span>/</span>
              <span className="font-mono text-[#F5F5F4] font-semibold">{totalSteps}</span>
              <span className="text-[#30353D]">•</span>
              <span>Tokens:</span>
              <span className="font-mono text-[#70D6C2] font-semibold">{totalTokens.toLocaleString()}</span>
              {tokenBudget > 0 && (
                <>
                  <span>/</span>
                  <span className="font-mono text-[#A6ADB7]">{tokenBudget.toLocaleString()}</span>
                </>
              )}
              {progress.microbatches_processed !== undefined && (
                <span className="text-[#A6ADB7] font-mono text-[11px]">
                  ({progress.microbatches_processed} μBatches)
                </span>
              )}
            </span>
          ) : (
            <span className="text-[#A6ADB7] font-sans flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-[#76B900]" />
              <span>Epoch</span>
              <span className="font-mono text-[#F5F5F4] font-semibold">{currentEpoch}</span>
              <span>/</span>
              <span className="font-mono text-[#F5F5F4] font-semibold">{totalEpochs}</span>
              {!isTerminal && currentBatch > 0 && (
                <span className="text-[#A6ADB7] font-mono">
                  (Batch {currentBatch} / {totalBatches})
                </span>
              )}
            </span>
          )}
          <span className="text-[#76B900] font-mono font-bold">
            {percentComplete}% Complete
          </span>
        </div>
        <div className="w-full h-2 rounded-full bg-[#121519] border border-[#30353D] overflow-hidden p-[1px]">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              job.status === 'FAILED'
                ? 'bg-[#FB7185]'
                : job.status === 'CANCELLED'
                ? 'bg-[#6B7280]'
                : 'bg-[#76B900]'
            }`}
            style={{ width: `${percentComplete}%` }}
          />
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Loss */}
        <div className="p-3 rounded-xl bg-[#121519] border border-[#30353D] space-y-1 hover:border-[#76B900]/30 transition-colors">
          <div className="flex items-center gap-1 text-[11px] text-[#A6ADB7] font-sans">
            <Flame className="w-3 h-3 text-[#76B900]" />
            <span>Train Loss</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-[#F5F5F4]">
            {lossValue !== undefined ? Number(lossValue).toFixed(4) : '--'}
          </div>
        </div>

        {/* Validation Metric (Perplexity for Transformer / Accuracy for Vision) */}
        <div className="p-3 rounded-xl bg-[#121519] border border-[#30353D] space-y-1 hover:border-[#70D6C2]/30 transition-colors">
          <div className="flex items-center gap-1 text-[11px] text-[#A6ADB7] font-sans">
            <Award className="w-3 h-3 text-[#70D6C2]" />
            <span>{isTransformer ? 'Val Perplexity' : 'Val Accuracy'}</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-[#70D6C2]">
            {isTransformer
              ? (valPerplexity !== undefined ? Number(valPerplexity).toFixed(2) : (progress.val_loss !== undefined ? Math.exp(Math.min(20, progress.val_loss)).toFixed(2) : '--'))
              : (valAccValue !== undefined ? `${Number(valAccValue).toFixed(2)}%` : '--')}
          </div>
        </div>

        {/* Throughput */}
        <div className="p-3 rounded-xl bg-[#121519] border border-[#30353D] space-y-1 hover:border-[#70D6C2]/30 transition-colors">
          <div className="flex items-center gap-1 text-[11px] text-[#A6ADB7] font-sans">
            <Zap className="w-3 h-3 text-[#70D6C2]" />
            <span>Throughput</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-[#F5F5F4]">
            {throughput > 0
              ? (isTransformer ? `${Math.round(throughput)} tok/s` : `${Math.round(throughput)} smp/s`)
              : '--'}
          </div>
        </div>

        {/* Wall Time */}
        <div className="p-3 rounded-xl bg-[#121519] border border-[#30353D] space-y-1 hover:border-[#70D6C2]/30 transition-colors">
          <div className="flex items-center gap-1 text-[11px] text-[#A6ADB7] font-sans">
            <Clock className="w-3 h-3 text-[#70D6C2]" />
            <span>Elapsed Time</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-[#F5F5F4]">
            {elapsedSec.toFixed(2)}s
          </div>
        </div>

        {/* Peak VRAM */}
        <div className="p-3 rounded-xl bg-[#121519] border border-[#30353D] space-y-1 hover:border-[#70D6C2]/30 transition-colors">
          <div className="flex items-center justify-between text-[11px] text-[#A6ADB7] font-sans">
            <div className="flex items-center gap-1">
              <Cpu className="w-3 h-3 text-[#70D6C2]" />
              <span>VRAM Allocated</span>
            </div>
            {gpuTemp !== undefined && gpuTemp !== null && (
              <span className="text-[#76B900] font-mono font-medium">{gpuTemp}°C</span>
            )}
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-[#F5F5F4]">
            {job.backend === 'CPU_NATIVE'
              ? '0.00 MB'
              : `${vramUsage.toFixed(1)} MB${reservedVram ? ` (${reservedVram.toFixed(0)} MB res)` : ''}`}
          </div>
          {physicalVram && physicalVram > 0 && job.backend !== 'CPU_NATIVE' ? (
            <div className="text-[10px] font-mono text-[#A6ADB7] truncate">
              Phys: {physicalVram.toFixed(0)}MB {gpuUtil !== null && gpuUtil !== undefined ? `(${gpuUtil}% util)` : ''}
            </div>
          ) : null}
        </div>

        {/* Sixth metric: Learning Rate for Transformer, CUDA Dispatches for Vision */}
        <div className="p-3 rounded-xl bg-[#121519] border border-[#30353D] space-y-1 hover:border-[#76B900]/30 transition-colors">
          <div className="flex items-center gap-1 text-[11px] text-[#A6ADB7] font-sans">
            <Zap className="w-3 h-3 text-[#76B900]" />
            <span>{isTransformer ? 'Learning Rate' : 'CUDA Dispatches'}</span>
          </div>
          <div className="text-sm sm:text-base font-mono font-bold text-[#76B900]">
            {isTransformer
              ? (progress.learning_rate !== undefined
                  ? Number(progress.learning_rate).toExponential(2)
                  : (job.config?.learning_rate ? Number(job.config.learning_rate).toExponential(2) : '--'))
              : (job.backend === 'GPU_CUSTOM' ? `${customDispatches}` : '0')}
          </div>
        </div>
      </div>

      {/* Error Message Alert */}
      {job.error_message && (
        <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-[#171A1F] border border-[#FB7185]/30 text-xs text-[#F5F5F4]">
          <AlertCircle className="w-4 h-4 text-[#FB7185] flex-shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <span className="font-bold text-[#FB7185] font-sans">Execution Error:</span>
            <p className="font-mono text-[11px] text-[#FB7185]">{job.error_message}</p>
          </div>
        </div>
      )}
    </div>
  );
};
