import React, { useState } from 'react';
import { TrainingJobResponse, JobStatus } from '../../types/training';
import { 
  History, 
  RefreshCw, 
  Eye, 
  HardDrive, 
  CheckCircle2, 
  XCircle, 
  StopCircle, 
  AlertCircle
} from 'lucide-react';

interface TrainingHistoryTableProps {
  jobs: TrainingJobResponse[];
  selectedJobId?: string;
  onSelectJob: (job: TrainingJobResponse) => void;
  onRefresh: () => void;
  isLoading: boolean;
  onViewCheckpoints: (jobId: string) => void;
}

export const TrainingHistoryTable: React.FC<TrainingHistoryTableProps> = ({
  jobs,
  selectedJobId,
  onSelectJob,
  onRefresh,
  isLoading,
  onViewCheckpoints,
}) => {
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [backendFilter, setBackendFilter] = useState<string>('ALL');

  const filteredJobs = jobs.filter((j) => {
    if (statusFilter !== 'ALL' && j.status !== statusFilter) return false;
    if (backendFilter !== 'ALL' && j.backend !== backendFilter) return false;
    return true;
  });

  const renderStatus = (status: JobStatus) => {
    switch (status) {
      case 'COMPLETED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#4ADE80]/10 text-[#4ADE80] border border-[#4ADE80]/30">
            <CheckCircle2 className="w-3 h-3" />
            COMPLETED
          </span>
        );
      case 'RUNNING':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#76B900]/10 text-[#76B900] border border-[#76B900]/30 animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-[#76B900]" />
            RUNNING
          </span>
        );
      case 'QUEUED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30">
            QUEUED
          </span>
        );
      case 'CANCEL_REQUESTED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#FB7185]/10 text-[#FB7185] border border-[#FB7185]/30">
            CANCELLING
          </span>
        );
      case 'CANCELLED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#121519] text-[#A6ADB7] border border-[#30353D]">
            <StopCircle className="w-3 h-3" />
            CANCELLED
          </span>
        );
      case 'FAILED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#FB7185]/10 text-[#FB7185] border border-[#FB7185]/30">
            <XCircle className="w-3 h-3" />
            FAILED
          </span>
        );
      case 'INTERRUPTED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30">
            <AlertCircle className="w-3 h-3" />
            INTERRUPTED
          </span>
        );
      default:
        return <span className="text-[10px] font-mono text-[#A6ADB7]">{status}</span>;
    }
  };

  return (
    <div className="bg-[#171A1F] p-5 sm:p-6 rounded-2xl border border-[#30353D] space-y-4 interactive-card">
      {/* Header and Filter Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#30353D]">
        <div className="flex items-center gap-2">
          <History className="w-5 h-5 text-[#76B900]" />
          <h3 className="text-base font-bold text-[#F5F5F4] tracking-tight font-sans">
            Training Experiment History
          </h3>
          <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#121519] text-[#A6ADB7] border border-[#30353D]">
            {filteredJobs.length} {filteredJobs.length === 1 ? 'Job' : 'Jobs'}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-2.5 py-1.5 rounded-lg bg-[#121519] border border-[#30353D] text-xs font-mono text-[#F5F5F4] focus:outline-none cursor-pointer"
          >
            <option value="ALL">Status: All</option>
            <option value="COMPLETED">Completed</option>
            <option value="RUNNING">Running</option>
            <option value="QUEUED">Queued</option>
            <option value="CANCELLED">Cancelled</option>
            <option value="FAILED">Failed</option>
            <option value="INTERRUPTED">Interrupted</option>
          </select>

          {/* Backend Filter */}
          <select
            value={backendFilter}
            onChange={(e) => setBackendFilter(e.target.value)}
            className="px-2.5 py-1.5 rounded-lg bg-[#121519] border border-[#30353D] text-xs font-mono text-[#F5F5F4] focus:outline-none cursor-pointer"
          >
            <option value="ALL">Backend: All</option>
            <option value="CPU_NATIVE">CPU_NATIVE</option>
            <option value="GPU_NATIVE">GPU_NATIVE</option>
            <option value="GPU_CUSTOM">GPU_CUSTOM</option>
          </select>

          {/* Refresh Button */}
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="p-1.5 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D] transition-all cursor-pointer"
            title="Refresh Jobs"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-[#76B900]' : ''}`} />
          </button>
        </div>
      </div>

      {/* Table */}
      {filteredJobs.length === 0 ? (
        <div className="py-12 text-center space-y-2">
          <History className="w-8 h-8 text-[#6B7280] mx-auto" />
          <p className="text-sm font-semibold text-[#F5F5F4] font-sans">No training jobs recorded</p>
          <p className="text-xs text-[#A6ADB7] font-sans">
            Launch a training job above to start tracking real model runs in SQLite.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-[#30353D] text-[#A6ADB7] font-sans">
                <th className="py-2.5 px-3">Job ID</th>
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Backend</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Epochs</th>
                <th className="py-2.5 px-3">Train Loss</th>
                <th className="py-2.5 px-3">Val Acc</th>
                <th className="py-2.5 px-3">Duration</th>
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#30353D]">
              {filteredJobs.map((j) => {
                const isSelected = j.job_id === selectedJobId;
                const m = j.metrics || {};
                const p = j.progress || {};
                const loss = m.final_train_loss ?? p.train_loss;
                const acc = m.best_val_acc ?? p.best_val_accuracy ?? p.val_accuracy ?? p.val_acc;
                const wallTime = m.total_wall_time_sec ?? p.elapsed_time_sec ?? p.elapsed_sec;

                return (
                  <tr
                    key={j.job_id}
                    className={`transition-colors ${
                      isSelected
                        ? 'bg-[#76B900]/10 border-l-2 border-l-[#76B900]'
                        : 'hover:bg-[#121519]'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-bold text-[#F5F5F4] whitespace-nowrap">
                      {j.job_id}
                    </td>
                    <td className="py-2.5 px-3 text-[#A6ADB7] whitespace-nowrap">
                      {j.model_name}
                    </td>
                    <td className="py-2.5 px-3 whitespace-nowrap">
                      <span
                        className={`font-semibold ${
                          j.backend === 'GPU_CUSTOM'
                            ? 'text-[#76B900]'
                            : j.backend === 'GPU_NATIVE'
                            ? 'text-[#70D6C2]'
                            : 'text-[#A6ADB7]'
                        }`}
                      >
                        {j.backend}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 whitespace-nowrap">
                      {renderStatus(j.status)}
                    </td>
                    <td className="py-2.5 px-3 text-[#A6ADB7] whitespace-nowrap">
                      {j.epochs} ep ({j.batch_size} bs)
                    </td>
                    <td className="py-2.5 px-3 text-[#F5F5F4] whitespace-nowrap">
                      {loss !== undefined ? loss.toFixed(4) : '--'}
                    </td>
                    <td className="py-2.5 px-3 text-[#70D6C2] font-semibold whitespace-nowrap">
                      {acc !== undefined ? `${acc.toFixed(2)}%` : '--'}
                    </td>
                    <td className="py-2.5 px-3 text-[#A6ADB7] whitespace-nowrap">
                      {wallTime !== undefined ? `${wallTime.toFixed(2)}s` : '--'}
                    </td>
                    <td className="py-2.5 px-3 text-right whitespace-nowrap">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => onSelectJob(j)}
                          className="inline-flex items-center gap-1 px-2 py-1 rounded bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] border border-[#30353D] transition-all cursor-pointer text-[11px] font-sans"
                          title="Inspect Job"
                        >
                          <Eye className="w-3 h-3 text-[#76B900]" />
                          <span>View</span>
                        </button>

                        {j.checkpoint_path && (
                          <button
                            onClick={() => onViewCheckpoints(j.job_id)}
                            className="inline-flex items-center gap-1 px-2 py-1 rounded bg-[#76B900]/10 hover:bg-[#76B900]/20 text-[#76B900] border border-[#76B900]/30 transition-all cursor-pointer text-[11px] font-sans"
                            title="Inspect Saved Checkpoint"
                          >
                            <HardDrive className="w-3 h-3" />
                            <span>Ckpt</span>
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
