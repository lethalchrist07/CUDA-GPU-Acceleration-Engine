import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { 
  Sparkles, 
  Download, 
  Activity, 
  Cpu,
  AlertCircle,
  Layers
} from 'lucide-react';
import { InferenceResponse, InferenceRecord } from '../types/api';
import { StatusBadge } from './StatusBadge';
import { WarningBanner } from './WarningBanner';
import { api } from '../services/api';

interface InferenceSectionProps {
  data: InferenceResponse | null;
  loading: boolean;
}

export const InferenceSection: React.FC<InferenceSectionProps> = ({ data, loading }) => {
  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin" />
          <span className="text-sm font-medium text-slate-400">Loading inference telemetry...</span>
        </div>
      </div>
    );
  }

  const rows = data.rows || [];

  if (rows.length === 0) {
    return (
      <div id="section-inference" className="space-y-6 animate-fadeIn">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.07]">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-extrabold text-white">LLM Inference Telemetry</h2>
              <StatusBadge status={data.status} />
            </div>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              Quantized SLM/LLM Inference • Generation Latency, TTFT & Token Throughput
            </p>
          </div>
        </div>
        <WarningBanner warnings={data.warnings} />
        <div className="glass-panel rounded-2xl p-8 border border-white/[0.07] text-center space-y-3">
          <AlertCircle className="w-10 h-10 text-amber-400 mx-auto" />
          <h3 className="text-lg font-bold text-white">No Inference Experiments Recorded</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Inference experiment artifact was not found or contains no records.
          </p>
        </div>
      </div>
    );
  }

  // Chart data: Tokens per second
  const tpsChartData = rows.map(r => ({
    name: `${r.model_name.replace(':latest', '')} (${r.device.toUpperCase()})`,
    tps: Number(r.tokens_per_second.toFixed(2)),
    ttft_ms: Number(r.ttft_ms.toFixed(1)),
    device: r.device,
  }));

  return (
    <div id="section-inference" className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.07]">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-extrabold text-white">LLM Inference Telemetry</h2>
            <StatusBadge provenance={data.source?.provenance} />
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Quantized SLM/LLM Inference • Generation Latency, Time to First Token (TTFT) & Token Throughput
          </p>
        </div>

        <div className="flex items-center gap-2">
          <a
            href={api.getArtifactUrl('inference-csv')}
            download="inference_experiments.csv"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-obsidian-900 hover:bg-obsidian-850 text-slate-200 text-xs font-semibold border border-white/[0.08] hover:border-white/[0.15] transition-all cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-purple-400" />
            <span>Download CSV</span>
          </a>
          <a
            href={api.getArtifactUrl('inference-json')}
            download="inference_experiments.json"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-obsidian-900 hover:bg-obsidian-850 text-slate-200 text-xs font-semibold border border-white/[0.08] hover:border-white/[0.15] transition-all cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-[#76b900]" />
            <span>Download JSON</span>
          </a>
        </div>
      </div>

      {/* Warnings Banner */}
      <WarningBanner warnings={data.warnings} />

      {/* Model Spec & Quantization Card */}
      <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-3">
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-2.5">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-purple-400" />
            <h3 className="text-sm font-bold text-white">Inference Engine Specification</h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">Target Device: RTX 3050 Laptop (6GB VRAM)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-2.5 rounded-lg bg-obsidian-900/60 border border-white/[0.04]">
            <span className="text-slate-400 font-sans text-[10px] uppercase block">Evaluated Model</span>
            <span className="text-white font-semibold">Qwen 2.5 (1.5B)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-obsidian-900/60 border border-white/[0.04]">
            <span className="text-slate-400 font-sans text-[10px] uppercase block">Quantization Format</span>
            <span className="text-purple-300 font-semibold">Q4_K_M GGUF (4-bit)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-obsidian-900/60 border border-white/[0.04]">
            <span className="text-slate-400 font-sans text-[10px] uppercase block">Prefill Latency</span>
            <span className="text-cyan-300 font-semibold">TTFT Measured (ms)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-obsidian-900/60 border border-white/[0.04]">
            <span className="text-slate-400 font-sans text-[10px] uppercase block">Decode Throughput</span>
            <span className="text-[#76b900] font-semibold">Tokens / Second</span>
          </div>
        </div>
      </div>

      {/* Recharts Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Token Throughput Chart */}
        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <h3 className="text-sm font-bold text-white">Generation Throughput (Tokens / Second)</h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400">Tokens/s (higher is better)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tpsChartData} margin={{ top: 10, right: 10, left: -15, bottom: 35 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis 
                  dataKey="name" 
                  stroke="#64748b" 
                  fontSize={10} 
                  angle={-20} 
                  textAnchor="end"
                  interval={0}
                />
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
                <Bar dataKey="tps" name="Tokens / sec" fill="#a855f7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Time to First Token (TTFT) Chart */}
        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-white">Time to First Token (TTFT Latency)</h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400">Milliseconds (lower is better)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tpsChartData} margin={{ top: 10, right: 10, left: -15, bottom: 35 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis 
                  dataKey="name" 
                  stroke="#64748b" 
                  fontSize={10} 
                  angle={-20} 
                  textAnchor="end"
                  interval={0}
                />
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
                <Bar dataKey="ttft_ms" name="TTFT (ms)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Experiments Table */}
      <div className="glass-panel rounded-2xl border border-white/[0.07] overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-white/[0.06] flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-purple-400" />
            <h3 className="text-base font-bold text-white tracking-tight">Inference Experiment Logs</h3>
          </div>
          <span className="text-xs font-mono text-purple-400 font-semibold">
            {rows.length} Runs Logged
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[760px]">
            <thead className="bg-obsidian-900/90 text-slate-400 uppercase font-semibold text-[10px] tracking-wider border-b border-white/[0.06]">
              <tr>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-3">Quant / Size</th>
                <th className="py-3 px-3">Device Label</th>
                <th className="py-3 px-3 text-right">Prompt Tok</th>
                <th className="py-3 px-3 text-right">Gen Tok</th>
                <th className="py-3 px-3 text-right">TTFT (ms)</th>
                <th className="py-3 px-3 text-right">Gen Time (ms)</th>
                <th className="py-3 px-3 text-right">Tokens / sec</th>
                <th className="py-3 px-4 text-right">Global VRAM</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] font-mono tabular-nums">
              {rows.map((row: InferenceRecord) => (
                <tr key={row.experiment_id} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-4 font-sans font-medium text-white truncate max-w-[140px]">
                    {row.model_name}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    <span className="px-1.5 py-0.5 rounded bg-obsidian-900 border border-white/[0.08] text-[11px]">
                      {row.quantization} • {row.parameter_size}
                    </span>
                  </td>
                  <td className="py-3 px-3">
                    <span className={`px-2 py-0.5 rounded-md text-[11px] font-sans font-semibold ${
                      row.device === 'cuda'
                        ? 'bg-purple-950 text-purple-400 border border-purple-500/30'
                        : 'bg-slate-800 text-slate-300'
                    }`}>
                      {row.device.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {row.prompt_tokens}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {row.output_tokens}
                  </td>
                  <td className="py-3 px-3 text-right text-cyan-300 font-semibold">
                    {row.ttft_ms.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {row.generation_duration_ms.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-purple-400">
                    {row.tokens_per_second.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {row.global_vram_used_mb ? `${row.global_vram_used_mb.toFixed(1)} MB` : 'N/A'}
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
