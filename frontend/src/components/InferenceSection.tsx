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
          <div className="w-10 h-10 border-4 border-[#76B900]/20 border-t-[#76B900] rounded-full animate-spin" />
          <span className="text-sm font-medium text-[#A6ADB7]">Loading LLM inference telemetry...</span>
        </div>
      </div>
    );
  }

  const rows = data.rows || [];

  if (rows.length === 0) {
    return (
      <div id="section-inference" className="space-y-6 animate-fadeIn">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-[#30353D]">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-extrabold text-[#F5F5F4]">Edge LLM Inference Engine</h2>
              <StatusBadge status={data.status} />
            </div>
            <p className="text-xs sm:text-sm text-[#A6ADB7] mt-1">
              GGUF Quantized Local LLM Inference • RTX 3050 Laptop GPU (6GB)
            </p>
          </div>
        </div>
        <WarningBanner warnings={data.warnings} />
        <div className="glass-panel rounded-2xl p-8 border border-[#30353D] text-center space-y-3">
          <AlertCircle className="w-10 h-10 text-[#FBBF24] mx-auto" />
          <h3 className="text-lg font-bold text-[#F5F5F4]">No Inference Benchmarks Available</h3>
          <p className="text-xs text-[#A6ADB7] max-w-md mx-auto">
            Inference benchmark artifact was not found or could not be loaded.
          </p>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const tpsChartData = rows.map(r => ({
    name: `${r.model_name.replace('Qwen2.5-', '')} (${r.quantization})`,
    tps: Number(r.tokens_per_second.toFixed(2)),
    ttft_ms: Number(r.ttft_ms.toFixed(1)),
    device: r.device,
  }));

  return (
    <div id="section-inference" className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-[#30353D]">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-extrabold text-[#F5F5F4]">Edge LLM Inference Engine</h2>
            <StatusBadge provenance={data.source?.provenance} />
          </div>
          <p className="text-xs sm:text-sm text-[#A6ADB7] mt-1">
            GGUF Quantized Local LLM Inference • RTX 3050 Laptop GPU (6GB)
          </p>
        </div>

        <div className="flex items-center gap-2">
          <a
            href={api.getArtifactUrl('inference-csv')}
            download="inference_experiments.csv"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] text-xs font-semibold border border-[#30353D] hover:border-[#76B900]/40 transition-all cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-[#76B900]" />
            <span>Download CSV</span>
          </a>
          <a
            href={api.getArtifactUrl('inference-json')}
            download="inference_experiments.json"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] text-xs font-semibold border border-[#30353D] hover:border-[#70D6C2]/40 transition-all cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-[#70D6C2]" />
            <span>Download JSON</span>
          </a>
        </div>
      </div>

      {/* Warnings Banner */}
      <WarningBanner warnings={data.warnings} />

      {/* Model Spec & Quantization Card */}
      <div className="interactive-card glass-panel rounded-2xl p-5 border border-[#30353D] space-y-3">
        <div className="flex items-center justify-between border-b border-[#30353D] pb-2.5">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#76B900]" />
            <h3 className="text-sm font-bold text-[#F5F5F4]">Inference Engine Specification</h3>
          </div>
          <span className="text-[11px] font-mono text-[#A6ADB7]">Target Device: RTX 3050 Laptop (6GB VRAM)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
            <span className="text-[#A6ADB7] font-sans text-[10px] uppercase block">Evaluated Model</span>
            <span className="text-[#F5F5F4] font-semibold">Qwen 2.5 (1.5B)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
            <span className="text-[#A6ADB7] font-sans text-[10px] uppercase block">Quantization Format</span>
            <span className="text-[#76B900] font-semibold">Q4_K_M GGUF (4-bit)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
            <span className="text-[#A6ADB7] font-sans text-[10px] uppercase block">Prefill Latency</span>
            <span className="text-[#70D6C2] font-semibold">TTFT Measured (ms)</span>
          </div>
          <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
            <span className="text-[#A6ADB7] font-sans text-[10px] uppercase block">Decode Throughput</span>
            <span className="text-[#76B900] font-semibold">Tokens / Second</span>
          </div>
        </div>
      </div>

      {/* Recharts Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Token Throughput Chart */}
        <div className="interactive-card glass-panel rounded-2xl p-5 border border-[#30353D] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-[#76B900]" />
              <h3 className="text-sm font-bold text-[#F5F5F4]">Generation Throughput (Tokens / Second)</h3>
            </div>
            <span className="text-[11px] font-mono text-[#7E8691]">Tokens/s (higher is better)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tpsChartData} margin={{ top: 10, right: 10, left: -15, bottom: 35 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                <XAxis 
                  dataKey="name" 
                  stroke="#7E8691" 
                  fontSize={10} 
                  angle={-20} 
                  textAnchor="end"
                  interval={0}
                />
                <YAxis stroke="#7E8691" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0D0F12',
                    borderColor: '#30353D',
                    borderRadius: '10px',
                    fontSize: '12px',
                    color: '#F5F5F4',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.7)',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="tps" name="Tokens / sec" fill="#76B900" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Time to First Token (TTFT) Chart */}
        <div className="interactive-card glass-panel rounded-2xl p-5 border border-[#30353D] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#70D6C2]" />
              <h3 className="text-sm font-bold text-[#F5F5F4]">Time to First Token (TTFT Latency)</h3>
            </div>
            <span className="text-[11px] font-mono text-[#7E8691]">Milliseconds (lower is better)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tpsChartData} margin={{ top: 10, right: 10, left: -15, bottom: 35 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                <XAxis 
                  dataKey="name" 
                  stroke="#7E8691" 
                  fontSize={10} 
                  angle={-20} 
                  textAnchor="end"
                  interval={0}
                />
                <YAxis stroke="#7E8691" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0D0F12',
                    borderColor: '#30353D',
                    borderRadius: '10px',
                    fontSize: '12px',
                    color: '#F5F5F4',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.7)',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="ttft_ms" name="TTFT (ms)" fill="#70D6C2" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Experiments Table */}
      <div className="interactive-card glass-panel rounded-2xl border border-[#30353D] overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-[#30353D] flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[#76B900]" />
            <h3 className="text-base font-bold text-[#F5F5F4] tracking-tight">Inference Experiment Logs</h3>
          </div>
          <span className="text-xs font-mono text-[#76B900] font-semibold">
            {rows.length} Runs Logged
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[760px]">
            <thead className="bg-[#121519] text-[#A6ADB7] uppercase font-semibold text-[10px] tracking-wider border-b border-[#30353D]">
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
            <tbody className="divide-y divide-[#30353D]/60 font-mono tabular-nums">
              {rows.map((row: InferenceRecord) => (
                <tr key={row.experiment_id} className="hover:bg-[#20242A]/60 transition-colors">
                  <td className="py-3 px-4 font-sans font-medium text-[#F5F5F4] truncate max-w-[140px]">
                    {row.model_name}
                  </td>
                  <td className="py-3 px-3 text-[#A6ADB7]">
                    <span className="px-1.5 py-0.5 rounded bg-[#121519] border border-[#30353D] text-[11px]">
                      {row.quantization} • {row.parameter_size}
                    </span>
                  </td>
                  <td className="py-3 px-3">
                    <span className={`px-2 py-0.5 rounded-md text-[11px] font-sans font-semibold ${
                      row.device === 'cuda'
                        ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/30'
                        : 'bg-[#121519] text-[#A6ADB7] border border-[#30353D]'
                    }`}>
                      {row.device.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right text-[#A6ADB7]">
                    {row.prompt_tokens}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A6ADB7]">
                    {row.output_tokens}
                  </td>
                  <td className="py-3 px-3 text-right text-[#70D6C2] font-semibold">
                    {row.ttft_ms.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A6ADB7]">
                    {row.generation_duration_ms.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-[#76B900]">
                    {row.tokens_per_second.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right text-[#A6ADB7]">
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
