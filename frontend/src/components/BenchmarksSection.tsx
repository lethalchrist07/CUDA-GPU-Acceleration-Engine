import React, { useState } from 'react';
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
  CheckCircle2, 
  Zap, 
  Layers, 
  Download, 
  Filter, 
  AlertCircle 
} from 'lucide-react';
import { BenchmarkResponse, BenchmarkRecord } from '../types/api';
import { StatusBadge } from './StatusBadge';
import { WarningBanner } from './WarningBanner';
import { api } from '../services/api';

interface BenchmarksSectionProps {
  data: BenchmarkResponse | null;
  loading: boolean;
}

export const BenchmarksSection: React.FC<BenchmarksSectionProps> = ({ data, loading }) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [timeScale, setTimeScale] = useState<'log' | 'linear'>('log');

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-4 border-[#76b900]/20 border-t-[#76b900] rounded-full animate-spin" />
          <span className="text-sm font-medium text-slate-400">Loading benchmark telemetry...</span>
        </div>
      </div>
    );
  }

  const rows = data.rows || [];

  if (rows.length === 0) {
    return (
      <div id="section-benchmarks" className="space-y-6 animate-fadeIn">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.07]">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-extrabold text-white">CUDA Benchmark Verification</h2>
              <StatusBadge status={data.status} />
            </div>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              NVIDIA RTX 3050 Laptop GPU (6 GB) vs Single-Threaded CPU Reference
            </p>
          </div>
        </div>
        <WarningBanner warnings={data.warnings} />
        <div className="glass-panel rounded-2xl p-8 border border-white/[0.07] text-center space-y-3">
          <AlertCircle className="w-10 h-10 text-amber-400 mx-auto" />
          <h3 className="text-lg font-bold text-white">No Benchmark Data Available</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Benchmark results artifact was not found or could not be loaded.
          </p>
        </div>
      </div>
    );
  }

  const categories = ['ALL', ...Array.from(new Set(rows.map(r => r.category)))];

  const filteredRows = selectedCategory === 'ALL' 
    ? rows 
    : rows.filter(r => r.category === selectedCategory);

  // Prepare chart data for execution times
  const timeChartData = filteredRows.map(r => ({
    name: `${r.category === 'Vector Addition' ? 'VecAdd' : 'SGEMM'} ${r.workload_size.replace(' elements', '').replace(' matrix', '')}`,
    cpu_ms: Number(r.cpu_time_ms.toFixed(3)),
    kernel_ms: Number(r.gpu_kernel_ms.toFixed(3)),
    e2e_ms: Number(r.gpu_end_to_end_ms.toFixed(3)),
    category: r.category,
  }));

  // Prepare chart data for speedup
  const speedupChartData = filteredRows.map(r => ({
    name: `${r.category === 'Vector Addition' ? 'VecAdd' : 'SGEMM'} ${r.dimension}`,
    kernel_speedup: Number(r.kernel_speedup.toFixed(2)),
    e2e_speedup: Number(r.end_to_end_speedup.toFixed(2)),
  }));

  return (
    <div id="section-benchmarks" className="space-y-6 animate-fadeIn">
      {/* Header with Title and Download Buttons */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.07]">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-extrabold text-white">CUDA Benchmark Verification</h2>
            <StatusBadge provenance={data.source?.provenance} />
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            NVIDIA RTX 3050 Laptop GPU (6 GB) vs Single-Threaded CPU Reference
          </p>
        </div>

        <div className="flex items-center gap-2">
          <a
            href={api.getArtifactUrl('benchmark-csv')}
            download="benchmark_results.csv"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-obsidian-900 hover:bg-obsidian-850 text-slate-200 text-xs font-semibold border border-white/[0.08] hover:border-white/[0.15] transition-all cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-[#76b900]" />
            <span>Download CSV</span>
          </a>
          <a
            href={api.getArtifactUrl('benchmark-json')}
            download="benchmark_results.json"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-obsidian-900 hover:bg-obsidian-850 text-slate-200 text-xs font-semibold border border-white/[0.08] hover:border-white/[0.15] transition-all cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Download JSON</span>
          </a>
        </div>
      </div>

      {/* Warnings & Notes */}
      <WarningBanner warnings={data.warnings} />

      {/* Filter Tabs */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        <Filter className="w-4 h-4 text-slate-400 mr-1" />
        {categories.map(cat => (
          <button
            key={cat}
            id={`filter-btn-${cat.toLowerCase().replace(/[^a-z0-9]/g, '-')}`}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
              selectedCategory === cat
                ? 'bg-[#76b900]/15 text-[#76b900] border border-[#76b900]/40 font-semibold shadow-[0_0_12px_rgba(118,185,0,0.1)]'
                : 'bg-obsidian-900/60 text-slate-400 hover:text-slate-200 border border-white/[0.05]'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Recharts Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Execution Time Chart */}
        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-[#76b900]" />
              <h3 className="text-sm font-bold text-white">Execution Times: Single-Threaded CPU Ref vs GPU Kernel vs End-to-End</h3>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setTimeScale(prev => (prev === 'log' ? 'linear' : 'log'))}
                className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold uppercase bg-obsidian-900 text-slate-300 border border-white/[0.1] hover:border-white/[0.2] cursor-pointer"
                title="Toggle between logarithmic and linear scale"
              >
                Scale: {timeScale.toUpperCase()}
              </button>
              <span className="text-[11px] font-mono text-slate-400">ms</span>
            </div>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={timeChartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis 
                  dataKey="name" 
                  stroke="#64748b" 
                  fontSize={10} 
                  angle={-20} 
                  textAnchor="end"
                  interval={0}
                />
                <YAxis 
                  stroke="#64748b" 
                  fontSize={10} 
                  scale={timeScale === 'log' ? 'log' : 'linear'} 
                  domain={timeScale === 'log' ? ['auto', 'auto'] : [0, 'auto']} 
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
                <Bar dataKey="cpu_ms" name="CPU Ref (Single-Threaded) (ms)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="kernel_ms" name="GPU Kernel (ms)" fill="#76b900" radius={[4, 4, 0, 0]} />
                <Bar dataKey="e2e_ms" name="GPU End-to-End (ms)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Speedup Factor Chart */}
        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-white">Speedup Multiplier vs Single-Threaded CPU Reference</h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400">Multiplier (x)</span>
          </div>

          <div className="h-64 sm:h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={speedupChartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
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
                <Bar dataKey="kernel_speedup" name="Kernel Speedup vs 1T CPU (x)" fill="#76b900" radius={[4, 4, 0, 0]} />
                <Bar dataKey="e2e_speedup" name="End-to-End Speedup vs 1T CPU (x)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Comprehensive Benchmark Table */}
      <div className="glass-panel rounded-2xl border border-white/[0.07] overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-white/[0.06] flex justify-between items-center">
          <h3 className="text-base font-bold text-white tracking-tight">Verified Benchmark Measurements</h3>
          <span className="text-xs font-mono text-[#76b900] font-semibold">
            {filteredRows.length} Configurations Tested
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[760px]">
            <thead className="bg-obsidian-900/90 text-slate-400 uppercase font-semibold text-[10px] tracking-wider border-b border-white/[0.06]">
              <tr>
                <th className="py-3 px-4">Workload</th>
                <th className="py-3 px-3">Variant</th>
                <th className="py-3 px-3">Size / Dim</th>
                <th className="py-3 px-3 text-right">CPU Ref (1T) (ms)</th>
                <th className="py-3 px-3 text-right">Kernel (ms)</th>
                <th className="py-3 px-3 text-right">E2E (ms)</th>
                <th className="py-3 px-3 text-right">Kernel vs CPU</th>
                <th className="py-3 px-3 text-right">E2E vs CPU</th>
                <th className="py-3 px-3 text-right">Bandwidth / GFLOPS</th>
                <th className="py-3 px-4 text-center">Parity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] font-mono tabular-nums">
              {filteredRows.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-slate-400 font-sans text-xs">
                    No benchmark records found for category "{selectedCategory}".
                  </td>
                </tr>
              ) : (
                filteredRows.map((row: BenchmarkRecord) => (
                <tr key={row.id} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-4 font-sans font-medium text-white">
                    {row.category}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {row.variant.toLowerCase().includes('tiled') ? (
                      <span className="px-2 py-0.5 rounded-md bg-[#76b900]/15 text-[#76b900] border border-[#76b900]/30 font-sans font-semibold text-[11px] whitespace-nowrap">
                        16x16 Tiled SRAM
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-md bg-white/[0.05] text-slate-300 border border-white/[0.08] font-sans font-medium text-[11px] whitespace-nowrap">
                        {row.variant}
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {row.workload_size}
                  </td>
                  <td className="py-3 px-3 text-right text-amber-300 font-semibold">
                    {row.cpu_time_ms.toFixed(3)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#76b900] font-semibold">
                    {row.gpu_kernel_ms.toFixed(3)}
                  </td>
                  <td className="py-3 px-3 text-right text-cyan-300 font-semibold">
                    {row.gpu_end_to_end_ms.toFixed(3)}
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-[#76b900]">
                    {row.kernel_speedup.toFixed(2)}x
                  </td>
                  <td className="py-3 px-3 text-right text-cyan-400 font-semibold">
                    {row.end_to_end_speedup.toFixed(2)}x
                  </td>
                  <td className="py-3 px-3 text-right text-slate-200">
                    {row.category === 'Vector Addition' 
                      ? `${row.memory_bandwidth_gb_s.toFixed(1)} GB/s`
                      : `${row.gflops.toFixed(1)} GFLOPS`}
                  </td>
                  <td className="py-3 px-4 text-center">
                    {row.verified ? (
                      <span className="inline-flex items-center gap-1 text-[11px] font-sans text-[#76b900] font-semibold">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        Pass
                      </span>
                    ) : (
                      <span className="text-rose-400 font-sans font-medium">Fail</span>
                    )}
                  </td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
