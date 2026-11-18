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
          <div className="w-10 h-10 border-4 border-[#76B900]/20 border-t-[#76B900] rounded-full animate-spin" />
          <span className="text-sm font-medium text-[#A6ADB7]">Loading training telemetry...</span>
        </div>
      </div>
    );
  }

  const rows = data.rows || [];

  if (rows.length === 0) {
    return (
      <div id="section-training" className="space-y-6 animate-fadeIn">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#171A1F] rounded-2xl p-6 border border-[#30353D]">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold text-[#F5F5F4] font-sans">PyTorch Training Engine</h2>
              <StatusBadge status={data.status} />
            </div>
            <p className="text-xs sm:text-sm text-[#A6ADB7] font-sans mt-1">
              BaselineMLP Architecture • Mixed Precision (AMP) vs FP32 vs CPU Baseline
            </p>
          </div>
        </div>
        <WarningBanner warnings={data.warnings} />
        <div className="bg-[#171A1F] rounded-2xl p-8 border border-[#30353D] text-center space-y-3">
          <AlertCircle className="w-10 h-10 text-[#FBBF24] mx-auto" />
          <h3 className="text-lg font-bold text-[#F5F5F4] font-sans">No Training Experiments Recorded</h3>
          <p className="text-xs text-[#A6ADB7] font-sans max-w-md mx-auto">
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

  const tooltipStyle = {
    backgroundColor: '#0D0F12',
    borderColor: '#30353D',
    borderRadius: '0.75rem',
    fontSize: '11px',
    fontFamily: 'monospace',
    color: '#F5F5F4',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.7)',
  };

  return (
    <div id="section-training" className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#171A1F] rounded-2xl p-6 border border-[#30353D]">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-[#F5F5F4] font-sans">PyTorch Training Engine</h2>
            <StatusBadge provenance={data.source?.provenance} />
          </div>
          <p className="text-xs sm:text-sm text-[#A6ADB7] font-sans mt-1">
            BaselineMLP Architecture • Automatic Mixed Precision (AMP) vs FP32 vs CPU Reference
          </p>
        </div>

        <div className="flex items-center gap-2">
          <a
            href={api.getArtifactUrl('training-csv')}
            download="training_experiments.csv"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] text-xs font-semibold font-sans border border-[#30353D] transition-all cursor-pointer shadow-none"
          >
            <Download className="w-3.5 h-3.5 text-[#76B900]" />
            <span>Download CSV</span>
          </a>
          <a
            href={api.getArtifactUrl('training-json')}
            download="training_experiments.json"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] text-xs font-semibold font-sans border border-[#30353D] transition-all cursor-pointer shadow-none"
          >
            <Download className="w-3.5 h-3.5 text-[#70D6C2]" />
            <span>Download JSON</span>
          </a>
        </div>
      </div>

      {/* Warnings Banner */}
      <WarningBanner warnings={data.warnings} />

      {/* Model Architecture Inspector Card */}
      <div className="interactive-card bg-[#171A1F] rounded-2xl p-5 border border-[#30353D] space-y-3">
        <div className="flex items-center justify-between border-b border-[#30353D] pb-2.5">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#76B900]" />
            <h3 className="text-sm font-bold text-[#F5F5F4] font-sans">Model Specification: BaselineMLP</h3>
          </div>
          <span className="text-[11px] font-mono text-[#A6ADB7]">Dataset: Fashion-MNIST (60k train / 10k val)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
            <span className="text-[#A6ADB7] font-sans text-[10px] uppercase block">Input Dimension</span>
            <span className="text-[#F5F5F4] font-semibold">28x28 (784 features)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
            <span className="text-[#A6ADB7] font-sans text-[10px] uppercase block">Hidden Layers</span>
            <span className="text-[#F5F5F4] font-semibold">FC 256 -&gt; FC 128 (ReLU)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
            <span className="text-[#A6ADB7] font-sans text-[10px] uppercase block">Total Parameters</span>
            <span className="text-[#F5F5F4] font-semibold">235,146 weights</span>
          </div>
          <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
            <span className="text-[#A6ADB7] font-sans text-[10px] uppercase block">Loss &amp; Optimizer</span>
            <span className="text-[#76B900] font-semibold">CrossEntropy • Adam</span>
          </div>
        </div>
      </div>

      {/* Recharts Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Total Training Time Chart */}
        <div className="interactive-card bg-[#171A1F] rounded-2xl p-5 border border-[#30353D] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-[#76B900]" />
              <h3 className="text-sm font-bold text-[#F5F5F4] font-sans">Total Training Time (10 Epochs)</h3>
            </div>
            <span className="text-[11px] font-mono text-[#7E8691]">Seconds (lower is better)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                <XAxis dataKey="name" stroke="#7E8691" fontSize={11} fontFamily="monospace" />
                <YAxis stroke="#7E8691" fontSize={10} fontFamily="monospace" />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px', fontFamily: 'monospace' }} />
                <Bar dataKey="train_time_sec" name="Training Time (s)" fill="#76B900" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Speedup & Accuracy */}
        <div className="interactive-card bg-[#171A1F] rounded-2xl p-5 border border-[#30353D] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-[#70D6C2]" />
              <h3 className="text-sm font-bold text-[#F5F5F4] font-sans">Relative Speedup Multiplier vs Single-Threaded CPU Reference</h3>
            </div>
            <span className="text-[11px] font-mono text-[#7E8691]">Multiplier (x)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                <XAxis dataKey="name" stroke="#7E8691" fontSize={11} fontFamily="monospace" />
                <YAxis stroke="#7E8691" fontSize={10} fontFamily="monospace" />
                <ReferenceLine 
                  y={1.0} 
                  stroke="#7E8691" 
                  strokeDasharray="3 3" 
                  label={{ value: '1.0x CPU Parity', fill: '#A6ADB7', fontSize: 10, position: 'right' }} 
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px', fontFamily: 'monospace' }} />
                <Bar dataKey="speedup" name="Speedup vs 1T CPU Reference (x)" fill="#70D6C2" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Experiments Table */}
      <div className="interactive-card bg-[#171A1F] rounded-2xl border border-[#30353D] overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-[#30353D] flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Flame className="w-4 h-4 text-[#76B900]" />
            <h3 className="text-base font-bold text-[#F5F5F4] tracking-tight font-sans">Training Experiment Records</h3>
          </div>
          <span className="text-xs font-mono text-[#76B900] font-semibold">
            {rows.length} Runs Recorded
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[760px]">
            <thead className="bg-[#121519] text-[#A6ADB7] uppercase font-semibold text-[10px] tracking-wider border-b border-[#30353D] font-sans">
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
            <tbody className="divide-y divide-[#30353D]/60 font-mono tabular-nums">
              {rows.map((row: TrainingRecord) => (
                <tr key={row.experiment_id} className="hover:bg-[#20242A]/60 transition-colors">
                  <td className="py-3 px-4 font-sans font-medium text-[#F5F5F4] truncate max-w-[150px]">
                    {row.experiment_id}
                  </td>
                  <td className="py-3 px-3">
                    <span className={`px-2 py-0.5 rounded-md text-[11px] font-sans font-semibold ${
                      row.device === 'cuda'
                        ? row.use_amp 
                          ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/30' 
                          : 'bg-[#70D6C2]/15 text-[#70D6C2] border border-[#70D6C2]/30'
                        : 'bg-[#121519] text-[#A6ADB7] border border-[#30353D]'
                    }`}>
                      {row.device.toUpperCase()} {row.use_amp ? '(AMP FP16)' : ''}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-[#A6ADB7]">
                    {row.parameters.toLocaleString()}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A6ADB7]">
                    {row.epochs}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A6ADB7]">
                    {row.batch_size}
                  </td>
                  <td className="py-3 px-3 text-right text-[#F5F5F4] font-semibold">
                    {(row.total_train_time_ms / 1000).toFixed(2)}s
                  </td>
                  <td className="py-3 px-3 text-right text-[#A6ADB7]">
                    {row.final_train_loss.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-right font-semibold text-[#76B900]">
                    {row.best_val_accuracy.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-[#70D6C2]">
                    {row.speedup_vs_cpu.toFixed(2)}x
                  </td>
                  <td className="py-3 px-4 text-right text-[#A6ADB7]">
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
