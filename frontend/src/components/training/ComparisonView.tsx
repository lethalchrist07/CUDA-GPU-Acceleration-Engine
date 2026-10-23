import React, { useState, useMemo } from 'react';
import { TrainingJobResponse } from '../../types/training';
import { 
  GitCompare, 
  AlertTriangle, 
  Cpu, 
  Zap, 
  Clock 
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface ComparisonViewProps {
  jobs: TrainingJobResponse[];
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({ jobs }) => {
  const completedJobs = useMemo(
    () => jobs.filter((j) => j.status === 'COMPLETED' && j.metrics),
    [jobs]
  );

  // Find latest completed runs of each backend as defaults
  const latestCpu = completedJobs.find((j) => j.backend === 'CPU_NATIVE');
  const latestGpuNative = completedJobs.find((j) => j.backend === 'GPU_NATIVE');
  const latestGpuCustom = completedJobs.find((j) => j.backend === 'GPU_CUSTOM');

  const [selectedCpuId, setSelectedCpuId] = useState<string>(latestCpu?.job_id || '');
  const [selectedGpuNativeId, setSelectedGpuNativeId] = useState<string>(latestGpuNative?.job_id || '');
  const [selectedGpuCustomId, setSelectedGpuCustomId] = useState<string>(latestGpuCustom?.job_id || '');

  const jobCpu = completedJobs.find((j) => j.job_id === selectedCpuId);
  const jobGpuNative = completedJobs.find((j) => j.job_id === selectedGpuNativeId);
  const jobGpuCustom = completedJobs.find((j) => j.job_id === selectedGpuCustomId);

  const selectedRuns = [
    { label: 'CPU Native', job: jobCpu, color: '#94a3b8' },
    { label: 'GPU Native (cuBLAS)', job: jobGpuNative, color: '#10b981' },
    { label: 'GPU Custom CUDA', job: jobGpuCustom, color: '#76b900' },
  ].filter((r) => r.job !== undefined) as {
    label: string;
    job: TrainingJobResponse;
    color: string;
  }[];

  // Comprehensive 10-Dimension Configuration Parity Check
  const parityWarnings: string[] = [];
  if (selectedRuns.length >= 2) {
    const baseConfig = selectedRuns[0].job.config;
    for (let i = 1; i < selectedRuns.length; i++) {
      const cmp = selectedRuns[i].job.config;
      const baseModel = baseConfig.model_name ?? baseConfig.model_id;
      const cmpModel = cmp.model_name ?? cmp.model_id;
      if (baseModel !== cmpModel) {
        parityWarnings.push(`Model architecture differs: '${baseModel}' vs '${cmpModel}'.`);
      }
      const baseDataset = baseConfig.dataset ?? selectedRuns[0].job.dataset ?? 'CIFAR-10';
      const cmpDataset = cmp.dataset ?? selectedRuns[i].job.dataset ?? 'CIFAR-10';
      if (baseDataset !== cmpDataset) {
        parityWarnings.push(`Dataset differs: '${baseDataset}' vs '${cmpDataset}'.`);
      }
      if (baseConfig.max_train_samples !== cmp.max_train_samples) {
        parityWarnings.push(`Train sample budget differs: ${baseConfig.max_train_samples} vs ${cmp.max_train_samples}.`);
      }
      if (baseConfig.max_val_samples !== cmp.max_val_samples) {
        parityWarnings.push(`Validation sample budget differs: ${baseConfig.max_val_samples} vs ${cmp.max_val_samples}.`);
      }
      const basePrecision = baseConfig.precision ?? 'float32';
      const cmpPrecision = cmp.precision ?? 'float32';
      if (basePrecision !== cmpPrecision) {
        parityWarnings.push(`Precision differs: '${basePrecision}' vs '${cmpPrecision}'.`);
      }
      const baseOpt = baseConfig.optimizer ?? 'SGD';
      const cmpOpt = cmp.optimizer ?? 'SGD';
      if (baseOpt !== cmpOpt) {
        parityWarnings.push(`Optimizer differs: '${baseOpt}' vs '${cmpOpt}'.`);
      }
      if (baseConfig.learning_rate !== cmp.learning_rate) {
        parityWarnings.push(`Learning rate differs: ${baseConfig.learning_rate} vs ${cmp.learning_rate}.`);
      }
      if (baseConfig.epochs !== cmp.epochs) {
        parityWarnings.push(`Epoch count differs: ${baseConfig.epochs} vs ${cmp.epochs}.`);
      }
      if (baseConfig.batch_size !== cmp.batch_size) {
        parityWarnings.push(`Batch size differs: ${baseConfig.batch_size} vs ${cmp.batch_size}.`);
      }
      if (baseConfig.seed !== cmp.seed) {
        parityWarnings.push(`Random seed differs: ${baseConfig.seed} vs ${cmp.seed}.`);
      }
    }
  }

  const hasParityMismatch = parityWarnings.length > 0;

  // Calculate speedups relative to CPU baseline if present and parity is 100% matched
  const cpuWallTime = jobCpu?.metrics?.total_wall_time_sec;

  // Chart data for execution duration
  const timingChartData = selectedRuns.map((r) => {
    const m = r.job.metrics || {};
    const wallSec = m.total_wall_time_sec ?? 0;
    const speedup =
      !hasParityMismatch && cpuWallTime && cpuWallTime > 0 && wallSec > 0
        ? Number((cpuWallTime / wallSec).toFixed(2))
        : 1.0;

    return {
      name: r.label,
      wall_time_sec: Number(wallSec.toFixed(2)),
      speedup: r.job.backend === 'CPU_NATIVE' ? 1.0 : speedup,
      accuracy: Number((m.best_val_acc ?? 0).toFixed(2)),
    };
  });

  return (
    <div className="glass-panel p-5 sm:p-6 rounded-2xl border border-white/[0.07] space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-white/[0.06]">
        <div className="flex items-center gap-2.5">
          <GitCompare className="w-5 h-5 text-[#76b900]" />
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Three-Backend Empirical Comparison
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Side-by-side analysis of CPU_NATIVE, GPU_NATIVE (cuBLAS), and GPU_CUSTOM (CUDA Autograd)
            </p>
          </div>
        </div>
      </div>

      {/* Selectors for 3 Backends */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* CPU Selector */}
        <div className="space-y-1.5">
          <label className="text-xs font-mono font-semibold text-slate-300 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-slate-400" />
            CPU Native Baseline Run
          </label>
          <select
            value={selectedCpuId}
            onChange={(e) => setSelectedCpuId(e.target.value)}
            className="w-full px-3 py-2 rounded-xl bg-obsidian-900 border border-white/[0.08] text-xs font-mono text-white focus:outline-none cursor-pointer"
          >
            <option value="">Select CPU Run...</option>
            {completedJobs
              .filter((j) => j.backend === 'CPU_NATIVE')
              .map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  {j.job_id} ({j.model_name}, {j.epochs}ep, {(j.metrics?.total_wall_time_sec ?? 0).toFixed(2)}s)
                </option>
              ))}
          </select>
        </div>

        {/* GPU Native Selector */}
        <div className="space-y-1.5">
          <label className="text-xs font-mono font-semibold text-slate-300 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-emerald-400" />
            GPU Native (cuBLAS) Run
          </label>
          <select
            value={selectedGpuNativeId}
            onChange={(e) => setSelectedGpuNativeId(e.target.value)}
            className="w-full px-3 py-2 rounded-xl bg-obsidian-900 border border-white/[0.08] text-xs font-mono text-white focus:outline-none cursor-pointer"
          >
            <option value="">Select GPU Native Run...</option>
            {completedJobs
              .filter((j) => j.backend === 'GPU_NATIVE')
              .map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  {j.job_id} ({j.model_name}, {j.epochs}ep, {(j.metrics?.total_wall_time_sec ?? 0).toFixed(2)}s)
                </option>
              ))}
          </select>
        </div>

        {/* GPU Custom Selector */}
        <div className="space-y-1.5">
          <label className="text-xs font-mono font-semibold text-slate-300 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-[#76b900]" />
            GPU Custom CUDA Run
          </label>
          <select
            value={selectedGpuCustomId}
            onChange={(e) => setSelectedGpuCustomId(e.target.value)}
            className="w-full px-3 py-2 rounded-xl bg-obsidian-900 border border-white/[0.08] text-xs font-mono text-white focus:outline-none cursor-pointer"
          >
            <option value="">Select GPU Custom Run...</option>
            {completedJobs
              .filter((j) => j.backend === 'GPU_CUSTOM')
              .map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  {j.job_id} ({j.model_name}, {j.epochs}ep, {(j.metrics?.total_wall_time_sec ?? 0).toFixed(2)}s)
                </option>
              ))}
          </select>
        </div>
      </div>

      {/* Explicit Comparison Limitation Notice */}
      {hasParityMismatch && (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/25 space-y-1.5">
          <div className="flex items-center gap-2 text-xs font-bold text-amber-400">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            <span>Comparison Limitation Notice: Configuration Discrepancies Detected</span>
          </div>
          <p className="text-[11px] text-slate-300 leading-relaxed">
            Strict scientific benchmarking requires identical model architecture, dataset, sample budgets, preprocessing, precision, optimizer, learning rate, epochs, batch size, and seed.
          </p>
          <ul className="text-xs font-mono text-amber-200/90 list-disc list-inside space-y-0.5 pl-1">
            {parityWarnings.map((w, idx) => (
              <li key={idx}>{w}</li>
            ))}
          </ul>
          <p className="text-[11px] text-amber-300 font-semibold pt-1">
            Definitive speedup claims are withheld until all hyperparameter dimensions match identically.
          </p>
        </div>
      )}

      {selectedRuns.length === 0 ? (
        <div className="py-10 text-center text-slate-500 text-xs font-mono">
          Select completed training jobs above to generate comparative analysis.
        </div>
      ) : (
        <>
          {/* Comparative Metrics Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-white/[0.07] text-slate-400">
                  <th className="py-2.5 px-3">Backend / Attribute</th>
                  {selectedRuns.map((r) => (
                    <th key={r.job.job_id} className="py-2.5 px-3">
                      <span style={{ color: r.color }} className="font-bold">
                        {r.label}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                <tr>
                  <td className="py-2 px-3 text-slate-400">Job ID</td>
                  {selectedRuns.map((r) => (
                    <td key={r.job.job_id} className="py-2 px-3 text-white">
                      {r.job.job_id}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Target Device</td>
                  {selectedRuns.map((r) => (
                    <td key={r.job.job_id} className="py-2 px-3 text-slate-300">
                      {r.job.metrics?.device || 'cpu'}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Total Wall Time</td>
                  {selectedRuns.map((r) => (
                    <td key={r.job.job_id} className="py-2 px-3 font-bold text-white">
                      {(r.job.metrics?.total_wall_time_sec ?? 0).toFixed(2)}s
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Speedup vs CPU</td>
                  {selectedRuns.map((r) => {
                    const wallSec = r.job.metrics?.total_wall_time_sec ?? 0;
                    if (r.job.backend === 'CPU_NATIVE') {
                      return (
                        <td key={r.job.job_id} className="py-2 px-3 text-slate-400">
                          1.00× (Ref Baseline)
                        </td>
                      );
                    }
                    if (hasParityMismatch) {
                      return (
                        <td key={r.job.job_id} className="py-2 px-3 text-amber-400/90 font-mono text-[11px]">
                          Withheld (Parity Mismatch)
                        </td>
                      );
                    }
                    if (cpuWallTime && cpuWallTime > 0 && wallSec > 0) {
                      const spd = cpuWallTime / wallSec;
                      return (
                        <td key={r.job.job_id} className="py-2 px-3 font-bold text-[#76b900]">
                          {spd.toFixed(2)}× Faster (Verified Parity)
                        </td>
                      );
                    }
                    return <td key={r.job.job_id} className="py-2 px-3 text-slate-500">--</td>;
                  })}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Best Val Accuracy</td>
                  {selectedRuns.map((r) => (
                    <td key={r.job.job_id} className="py-2 px-3 text-emerald-400 font-bold">
                      {(r.job.metrics?.best_val_acc ?? 0).toFixed(2)}%
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Final Train Loss</td>
                  {selectedRuns.map((r) => (
                    <td key={r.job.job_id} className="py-2 px-3 text-white">
                      {(r.job.metrics?.final_train_loss ?? 0).toFixed(4)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Mean Step Latency</td>
                  {selectedRuns.map((r) => (
                    <td key={r.job.job_id} className="py-2 px-3 text-slate-300">
                      {(r.job.metrics?.mean_step_latency_ms ?? 0).toFixed(2)} ms
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Throughput</td>
                  {selectedRuns.map((r) => (
                    <td key={r.job.job_id} className="py-2 px-3 text-slate-300">
                      {Math.round(r.job.metrics?.throughput_samples_per_sec ?? 0)} smp/s
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Peak VRAM</td>
                  {selectedRuns.map((r) => (
                    <td key={r.job.job_id} className="py-2 px-3 text-slate-300">
                      {r.job.backend === 'CPU_NATIVE'
                        ? '0.00 MB'
                        : `${(r.job.metrics?.peak_allocated_vram_mb ?? 0).toFixed(1)} MB`}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Custom CUDA Dispatches</td>
                  {selectedRuns.map((r) => {
                    const dispatches = r.job.metrics?.custom_cuda_dispatches
                      ? Object.values(r.job.metrics.custom_cuda_dispatches).reduce((a, b) => a + b, 0)
                      : 0;
                    return (
                      <td key={r.job.job_id} className="py-2 px-3 font-mono font-bold text-[#76b900]">
                        {dispatches}
                      </td>
                    );
                  })}
                </tr>
              </tbody>
            </table>
          </div>

          {/* Wall-Time Comparison Bar Chart */}
          <div className="pt-4 border-t border-white/[0.06] space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5 font-mono">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              Total Training Duration Comparison (Seconds — Lower is Better)
            </h4>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={timingChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a0f1d',
                      borderColor: 'rgba(255,255,255,0.1)',
                      borderRadius: '0.75rem',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                    }}
                  />
                  <Bar dataKey="wall_time_sec" name="Wall Time (s)" fill="#76b900" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
