import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { 
  Flame, 
  Clock, 
  Download, 
  Zap,
  AlertCircle,
  Layers
} from 'lucide-react';
import { TrainingResponse, TrainingRecord } from '../types/api';
import { StatusBadge } from './StatusBadge';
import { WarningBanner } from './WarningBanner';
import { api } from '../services/api';

interface TrainingSectionProps {
  data: TrainingResponse | null;
  loading: boolean;
}

export const TrainingSection: React.FC<TrainingSectionProps> = ({ data, loading }) => {
  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-4 border-amber-500/20 border-t-amber-500 rounded-full animate-spin" />
          <span className="text-sm font-medium text-slate-400">Loading training telemetry...</span>
        </div>
      </div>
    );
  }

  const rows = data.rows || [];

  if (rows.length === 0) {
    return (
      <div id="section-training" className="space-y-6 animate-fadeIn">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.07]">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-extrabold text-white">PyTorch Training Engine</h2>
              <StatusBadge status={data.status} />
            </div>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              BaselineMLP Architecture • Mixed Precision (AMP) vs FP32 vs CPU Baseline
            </p>
          </div>
        </div>
        <WarningBanner warnings={data.warnings} />
        <div className="glass-panel rounded-2xl p-8 border border-white/[0.07] text-center space-y-3">
          <AlertCircle className="w-10 h-10 text-amber-400 mx-auto" />
          <h3 className="text-lg font-bold text-white">No Training Experiments Recorded</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Training experiment artifact was not found or contains no records.
          </p>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const chartData = rows.map(r => ({
    name: `${r.device.toUpperCase()} ${r.use_amp ? '(AMP FP16)' : ''}`,
    train_time_sec: Number((r.total_train_time_ms / 1000).toFixed(2)),
    speedup: Number(r.speedup_vs_cpu.toFixed(2)),
    val_accuracy: Number(r.best_val_accuracy.toFixed(1)),
    peak_vram_mb: r.peak_gpu_memory_mb,
  }));

  return (
    <div id="section-training" className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.07]">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-extrabold text-white">PyTorch Training Engine</h2>
            <StatusBadge provenance={data.source?.provenance} />
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            BaselineMLP Architecture • Automatic Mixed Precision (AMP) vs FP32 vs CPU Reference
          </p>
        </div>

        <div className="flex items-center gap-2">
          <a
            href={api.getArtifactUrl('training-csv')}
            download="training_experiments.csv"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-obsidian-900 hover:bg-obsidian-850 text-slate-200 text-xs font-semibold border border-white/[0.08] hover:border-white/[0.15] transition-all cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-amber-400" />
            <span>Download CSV</span>
          </a>
          <a
            href={api.getArtifactUrl('training-json')}
            download="training_experiments.json"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-obsidian-900 hover:bg-obsidian-850 text-slate-200 text-xs font-semibold border border-white/[0.08] hover:border-white/[0.15] transition-all cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-[#76b900]" />
            <span>Download JSON</span>
          </a>
        </div>
      </div>

      {/* Warnings Banner */}
      <WarningBanner warnings={data.warnings} />

      {/* Model Architecture Inspector Card */}
      <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-3">
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-2.5">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white">Model Specification: BaselineMLP</h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">Dataset: Fashion-MNIST (60k train / 10k val)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-2.5 rounded-lg bg-obsidian-900/60 border border-white/[0.04]">
            <span className="text-slate-400 font-sans text-[10px] uppercase block">Input Dimension</span>
            <span className="text-white font-semibold">28x28 (784 features)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-obsidian-900/60 border border-white/[0.04]">
            <span className="text-slate-400 font-sans text-[10px] uppercase block">Hidden Layers</span>
            <span className="text-white font-semibold">FC 256 -&gt; FC 128 (ReLU)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-obsidian-900/60 border border-white/[0.04]">
            <span className="text-slate-400 font-sans text-[10px] uppercase block">Total Parameters</span>
            <span className="text-amber-300 font-semibold">235,146 weights</span>
          </div>
          <div className="p-2.5 rounded-lg bg-obsidian-900/60 border border-white/[0.04]">
            <span className="text-slate-400 font-sans text-[10px] uppercase block">Loss &amp; Optimizer</span>
            <span className="text-[#76b900] font-semibold">CrossEntropy • Adam</span>
          </div>
        </div>
      </div>

      {/* Recharts Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Total Training Time Chart */}
        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-amber-400" />
              <h3 className="text-sm font-bold text-white">Total Training Time (10 Epochs)</h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400">Seconds (lower is better)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#060a14',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    borderRadius: '10px',
                    fontSize: '12px',
                    color: '#f8fafc',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="train_time_sec" name="Training Time (s)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Speedup & Accuracy */}
        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-[#76b900]" />
              <h3 className="text-sm font-bold text-white">Relative Speedup Multiplier vs Single-Threaded CPU Reference</h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400">Multiplier (x)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={10} />
                <ReferenceLine 
                  y={1.0} 
                  stroke="#64748b" 
                  strokeDasharray="3 3" 
                  label={{ value: '1.0x CPU Parity', fill: '#94a3b8', fontSize: 10, position: 'right' }} 
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#060a14',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    borderRadius: '10px',
                    fontSize: '12px',
                    color: '#f8fafc',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="speedup" name="Speedup vs 1T CPU Reference (x)" fill="#76b900" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Experiments Table */}
      <div className="glass-panel rounded-2xl border border-white/[0.07] overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-white/[0.06] flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Flame className="w-4 h-4 text-amber-400" />
            <h3 className="text-base font-bold text-white tracking-tight">Training Experiment Records</h3>
          </div>
          <span className="text-xs font-mono text-amber-400 font-semibold">
            {rows.length} Runs Recorded
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[760px]">
            <thead className="bg-obsidian-900/90 text-slate-400 uppercase font-semibold text-[10px] tracking-wider border-b border-white/[0.06]">
              <tr>
                <th className="py-3 px-4">Experiment ID</th>
                <th className="py-3 px-3">Device / Mode</th>
                <th className="py-3 px-3">Parameters</th>
                <th className="py-3 px-3 text-right">Epochs</th>
                <th className="py-3 px-3 text-right">Batch Size</th>
                <th className="py-3 px-3 text-right">Train Time (s)</th>
                <th className="py-3 px-3 text-right">Final Loss</th>
                <th className="py-3 px-3 text-right">Val Acc (%)</th>
                <th className="py-3 px-3 text-right">Speedup vs 1T CPU</th>
                <th className="py-3 px-4 text-right">Peak VRAM</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] font-mono tabular-nums">
              {rows.map((row: TrainingRecord) => (
                <tr key={row.experiment_id} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-4 font-sans font-medium text-white truncate max-w-[150px]">
                    {row.experiment_id}
                  </td>
                  <td className="py-3 px-3">
                    <span className={`px-2 py-0.5 rounded-md text-[11px] font-sans font-semibold ${
                      row.device === 'cuda'
                        ? row.use_amp 
                          ? 'bg-[#76b900]/15 text-[#76b900] border border-[#76b900]/30' 
                          : 'bg-cyan-950 text-cyan-400 border border-cyan-500/30'
                        : 'bg-slate-800 text-slate-300'
                    }`}>
                      {row.device.toUpperCase()} {row.use_amp ? '(AMP FP16)' : ''}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {row.parameters.toLocaleString()}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {row.epochs}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {row.batch_size}
                  </td>
                  <td className="py-3 px-3 text-right text-amber-300 font-semibold">
                    {(row.total_train_time_ms / 1000).toFixed(2)}s
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {row.final_train_loss.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-right font-semibold text-[#76b900]">
                    {row.best_val_accuracy.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-cyan-400">
                    {row.speedup_vs_cpu.toFixed(2)}x
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {row.peak_gpu_memory_mb ? `${row.peak_gpu_memory_mb.toFixed(1)} MB` : 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
