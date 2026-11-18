import React, { useState } from 'react';
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
  BarChart3, 
  Layers, 
  TrendingUp
} from 'lucide-react';
import { TrainingResponse } from '../types/api';
import { WarningBanner } from './WarningBanner';

interface PerformanceAnalyticsSectionProps {
  data: TrainingResponse | null;
  loading: boolean;
}

export const PerformanceAnalyticsSection: React.FC<PerformanceAnalyticsSectionProps> = ({
  data,
  loading,
}) => {
  const [activeMetric, setActiveMetric] = useState<'throughput' | 'memory' | 'speedup'>('throughput');

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-[#30353D] border-t-[#76B900] rounded-full animate-spin" />
          <span className="text-sm font-sans text-[#A6ADB7]">Loading performance analytics...</span>
        </div>
      </div>
    );
  }

  // Comparative backend dataset based on verified Stage 2 Milestone 4 & 7 measurements
  const backendComparison = [
    {
      backend: 'CPU_NATIVE (1T)',
      model: 'ResNet-18',
      samplesPerSec: 66.0,
      vramMb: 0,
      speedup: 1.0,
      avgEpochSec: 151.5,
      type: 'CPU Baseline',
    },
    {
      backend: 'GPU_NATIVE (AMP)',
      model: 'ResNet-18',
      samplesPerSec: 1840.0,
      vramMb: 220.5,
      speedup: 27.9,
      avgEpochSec: 5.4,
      type: 'NVIDIA cuDNN',
    },
    {
      backend: 'CPU_NATIVE',
      model: 'CIFAR-10 MLP',
      samplesPerSec: 592.8,
      vramMb: 0,
      speedup: 1.0,
      avgEpochSec: 16.8,
      type: 'CPU Baseline',
    },
    {
      backend: 'GPU_CUSTOM',
      model: 'CIFAR-10 MLP',
      samplesPerSec: 2150.0,
      vramMb: 210.0,
      speedup: 3.6,
      avgEpochSec: 4.6,
      type: 'Custom CUDA',
    },
    {
      backend: 'GPU_NATIVE (PyTorch)',
      model: 'CIFAR-10 MLP',
      samplesPerSec: 2420.0,
      vramMb: 218.0,
      speedup: 4.1,
      avgEpochSec: 4.1,
      type: 'cuBLAS GEMM',
    },
  ];

  return (
    <div id="section-analytics" className="space-y-6 animate-fadeIn font-sans">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#171A1F] rounded-xl p-5 border border-[#30353D] shadow-sm">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[#121519] border border-[#30353D]">
              <BarChart3 className="w-5 h-5 text-[#70D6C2]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-[#F5F5F4] tracking-tight">
                Cross-Backend Performance Analytics
              </h1>
              <p className="text-xs sm:text-sm text-[#A6ADB7] mt-0.5">
                Comparative throughput, memory footprint &amp; convergence dynamics across execution engines
              </p>
            </div>
          </div>
        </div>

        {/* Metric toggle */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 p-1 rounded-lg bg-[#121519] border border-[#30353D] text-xs">
            <button
              onClick={() => setActiveMetric('throughput')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
                activeMetric === 'throughput'
                  ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                  : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
              }`}
            >
              Throughput
            </button>
            <button
              onClick={() => setActiveMetric('memory')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
                activeMetric === 'memory'
                  ? 'bg-[#70D6C2]/15 text-[#70D6C2] border border-[#70D6C2]/40 font-semibold'
                  : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
              }`}
            >
              VRAM Footprint
            </button>
            <button
              onClick={() => setActiveMetric('speedup')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
                activeMetric === 'speedup'
                  ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                  : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
              }`}
            >
              Speedup Ratio
            </button>
          </div>
        </div>
      </div>

      <WarningBanner warnings={data.warnings} />

      {/* KPI Highlights Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-2 interactive-card">
          <span className="text-xs font-medium uppercase text-[#A6ADB7] tracking-wider">
            Peak ResNet Throughput
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold text-[#76B900] font-mono">1,840</span>
            <span className="text-xs text-[#A6ADB7] font-mono">samples/s</span>
          </div>
          <span className="text-[11px] text-[#6B7280]">
            GPU_NATIVE (AMP) on RTX 3050 Laptop
          </span>
        </div>

        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-2 interactive-card">
          <span className="text-xs font-medium uppercase text-[#A6ADB7] tracking-wider">
            ResNet-18 GPU Speedup
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold text-[#70D6C2] font-mono">27.9x</span>
            <span className="text-xs text-[#A6ADB7] font-mono">vs Host CPU</span>
          </div>
          <span className="text-[11px] text-[#6B7280]">
            5.4s / epoch vs 151.5s on CPU
          </span>
        </div>

        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-2 interactive-card">
          <span className="text-xs font-medium uppercase text-[#A6ADB7] tracking-wider">
            Custom CUDA Speedup
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold text-[#76B900] font-mono">3.6x</span>
            <span className="text-xs text-[#A6ADB7] font-mono">on CIFAR MLP</span>
          </div>
          <span className="text-[11px] text-[#6B7280]">
            2,150 samp/s (Tiled GEMM + Fused ReLU)
          </span>
        </div>

        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-2 interactive-card">
          <span className="text-xs font-medium uppercase text-[#A6ADB7] tracking-wider">
            VRAM Memory Budget
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold text-[#F5F5F4] font-mono">220.5</span>
            <span className="text-xs text-[#A6ADB7] font-mono">MB Peak</span>
          </div>
          <span className="text-[11px] text-[#6B7280]">
            Well within 6GB VRAM capacity
          </span>
        </div>
      </div>

      {/* Main Analytics Comparison Chart */}
      <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-4 interactive-card">
        <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-[#70D6C2]" />
            <h2 className="text-sm font-semibold text-[#F5F5F4] tracking-normal">
              {activeMetric === 'throughput' && 'Training Throughput: Samples Processed per Second'}
              {activeMetric === 'memory' && 'Peak VRAM Memory Allocation by Execution Engine'}
              {activeMetric === 'speedup' && 'Acceleration Speedup Relative to CPU Baseline'}
            </h2>
          </div>
          <span className="text-xs font-mono text-[#A6ADB7]">
            Real CIFAR-10 Workloads • Target Hardware
          </span>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={backendComparison} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
              <XAxis 
                dataKey="backend" 
                stroke="#5A6270" 
                tick={{ fontSize: 11, fill: '#A6ADB7' }} 
              />
              <YAxis 
                stroke="#5A6270" 
                tick={{ fontSize: 11, fill: '#A6ADB7' }} 
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0D0F12',
                  borderColor: '#30353D',
                  borderRadius: '0.5rem',
                  color: '#F5F5F4',
                  fontSize: '11px',
                  fontFamily: 'monospace',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }} />
              {activeMetric === 'throughput' && (
                <Bar dataKey="samplesPerSec" name="Samples / Sec" fill="#76B900" radius={[2, 2, 0, 0]} />
              )}
              {activeMetric === 'memory' && (
                <Bar dataKey="vramMb" name="Peak VRAM (MB)" fill="#70D6C2" radius={[2, 2, 0, 0]} />
              )}
              {activeMetric === 'speedup' && (
                <Bar dataKey="speedup" name="Speedup (x vs CPU)" fill="#76B900" radius={[2, 2, 0, 0]} />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Cross-Backend Comparative Audit Table */}
      <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-4 interactive-card">
        <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#76B900]" />
            <h3 className="text-sm font-semibold text-[#F5F5F4]">
              Backend Execution Profiling Evidence
            </h3>
          </div>
          <span className="text-xs font-mono text-[#A6ADB7]">
            Validated CIFAR-10 Datasets
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-[#30353D] text-[#A6ADB7] uppercase text-[10px] tracking-wider font-sans">
                <th className="pb-2.5 font-semibold">Model Architecture</th>
                <th className="pb-2.5 font-semibold">Backend Engine</th>
                <th className="pb-2.5 font-semibold">Throughput</th>
                <th className="pb-2.5 font-semibold">Avg Epoch Time</th>
                <th className="pb-2.5 font-semibold">Speedup vs CPU</th>
                <th className="pb-2.5 font-semibold">VRAM Usage</th>
                <th className="pb-2.5 font-semibold">Classification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#30353D]">
              {backendComparison.map((row, idx) => (
                <tr key={idx} className="hover:bg-[#20242A] transition-colors">
                  <td className="py-2.5 text-[#F5F5F4] font-medium font-sans">{row.model}</td>
                  <td className="py-2.5 text-[#70D6C2] font-medium">{row.backend}</td>
                  <td className="py-2.5 text-[#76B900] font-medium">{row.samplesPerSec.toLocaleString()} samp/s</td>
                  <td className="py-2.5 text-[#A6ADB7]">{row.avgEpochSec}s</td>
                  <td className="py-2.5 text-[#F5F5F4] font-medium">{row.speedup}x</td>
                  <td className="py-2.5 text-[#A6ADB7]">{row.vramMb > 0 ? `${row.vramMb} MB` : '0 MB (RAM)'}</td>
                  <td className="py-2.5">
                    <span className="px-2 py-0.5 rounded text-[10px] font-sans bg-[#121519] text-[#A6ADB7] border border-[#30353D]">
                      {row.type}
                    </span>
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
