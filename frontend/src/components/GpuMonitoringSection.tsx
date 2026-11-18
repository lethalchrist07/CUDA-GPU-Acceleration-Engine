import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  RotateCw, 
  Thermometer, 
  Zap, 
  Server, 
  Radio, 
  Database
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend 
} from 'recharts';
import { HardwareResponse, GpuTelemetry } from '../types/api';
import { StatusBadge } from './StatusBadge';
import { WarningBanner } from './WarningBanner';

interface GpuMonitoringSectionProps {
  data: HardwareResponse | null;
  loading: boolean;
  onRefresh: () => void;
  autoRefresh: boolean;
  onToggleAutoRefresh: () => void;
}

interface TelemetrySample {
  timestamp: string;
  utilization: number;
  temperature: number;
  vramUsedMb: number;
  powerW?: number;
}

export const GpuMonitoringSection: React.FC<GpuMonitoringSectionProps> = ({
  data,
  loading,
  onRefresh,
  autoRefresh,
  onToggleAutoRefresh,
}) => {
  const [history, setHistory] = useState<TelemetrySample[]>([]);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const lastFetchedAtRef = useRef<string>('');

  // Accumulate genuine timestamped telemetry into session history (max 30 points)
  useEffect(() => {
    if (!data || !data.gpus || data.gpus.length === 0) return;
    if (data.fetched_at === lastFetchedAtRef.current) return;
    lastFetchedAtRef.current = data.fetched_at;

    const gpu = data.gpus[0];
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setLastUpdated(timeStr);

    const newSample: TelemetrySample = {
      timestamp: timeStr,
      utilization: gpu.utilization_percent ?? 0,
      temperature: gpu.temperature_c ?? 0,
      vramUsedMb: gpu.memory_used_mib ? Math.round(gpu.memory_used_mib) : 0,
      powerW: gpu.power_w !== null && gpu.power_w !== undefined ? Number(gpu.power_w.toFixed(1)) : undefined,
    };

    setHistory((prev) => {
      const updated = [...prev, newSample];
      if (updated.length > 30) {
        return updated.slice(updated.length - 30);
      }
      return updated;
    });
  }, [data]);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-[#30353D] border-t-[#76B900] rounded-full animate-spin" />
          <span className="text-sm font-sans text-[#A6ADB7]">Querying live NVIDIA telemetry...</span>
        </div>
      </div>
    );
  }

  const gpu: GpuTelemetry | undefined = data?.gpus?.[0];
  const host = data?.host;

  // Real memory metrics
  const totalMib = gpu?.memory_total_mib ?? 6144;
  const usedMib = gpu?.memory_used_mib ?? null;
  const freeMib = gpu?.memory_free_mib ?? null;
  const memoryPercent = (totalMib && usedMib !== null) ? Math.min(100, Math.round((usedMib / totalMib) * 100)) : 0;

  const temp = gpu?.temperature_c ?? null;
  const tempStatus = temp === null 
    ? 'Unknown' 
    : temp < 55 ? 'Cool / Optimal' : temp < 75 ? 'Nominal Load' : 'High Thermal Load';
  const tempColor = temp === null 
    ? 'text-[#A6ADB7]' 
    : temp < 55 ? 'text-[#4ADE80]' : temp < 75 ? 'text-[#FBBF24]' : 'text-[#FB7185]';

  const util = gpu?.utilization_percent ?? null;
  const powerW = gpu?.power_w !== null && gpu?.power_w !== undefined ? gpu.power_w.toFixed(1) : null;

  return (
    <div id="section-gpu-monitoring" className="space-y-6 animate-fadeIn font-sans">
      {/* Header with Live Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#171A1F] rounded-xl p-5 border border-[#30353D] shadow-lg">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-[#F5F5F4] tracking-tight">GPU Hardware Telemetry</h1>
            <StatusBadge status={data?.status} />
          </div>
          <p className="text-xs sm:text-sm text-[#A6ADB7] mt-1">
            Real-Time NVIDIA-SMI Query Engine • Ampere Mobile Architecture • Read-Only Telemetry Monitor
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {lastUpdated && (
            <span className="text-[11px] font-mono text-[#A6ADB7] hidden md:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#121519] border border-[#30353D]">
              <Radio className="w-3 h-3 text-[#70D6C2]" />
              <span>Polled: {lastUpdated}</span>
            </span>
          )}

          <button
            id="btn-toggle-autorefresh"
            onClick={onToggleAutoRefresh}
            aria-label={autoRefresh ? 'Pause automatic telemetry polling' : 'Resume automatic telemetry polling'}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all duration-200 cursor-pointer ${
              autoRefresh
                ? 'bg-[#76B900]/10 text-[#8ED11A] border-[#76B900]/30 font-semibold shadow-sm'
                : 'bg-[#121519] text-[#A6ADB7] border-[#30353D] hover:bg-[#20242A]'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-[#76B900] shadow-[0_0_6px_rgba(118,185,0,0.6)]' : 'bg-[#7A828E]'}`} />
            <span>{autoRefresh ? 'Auto-Polling (3s)' : 'Polling Paused'}</span>
          </button>

          <button
            id="btn-manual-refresh"
            onClick={onRefresh}
            aria-label="Refresh hardware telemetry immediately"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] text-xs font-medium border border-[#30353D] hover:border-[#76B900]/40 transition-all duration-200 cursor-pointer hover:-translate-y-0.5"
          >
            <RotateCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-[#76B900]' : 'text-[#A6ADB7]'}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      <WarningBanner warnings={data?.warnings || []} />

      {(!gpu || data?.status === 'cpu_only') && (
        <div className="p-4 rounded-xl bg-[#171A1F] border border-[#FBBF24]/40 flex items-start gap-3 text-xs">
          <Server className="w-5 h-5 text-[#FBBF24] shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h3 className="font-semibold text-[#F5F5F4] font-sans">CPU-Only Fallback Mode Active</h3>
            <p className="text-[#A6ADB7] leading-relaxed font-sans">
              No compatible NVIDIA GPU device was detected or active via NVML. The platform is running safely in host CPU fallback mode. All PyTorch neural network computations and benchmarks will execute using host processor threads without hardware acceleration.
            </p>
          </div>
        </div>
      )}

      {/* Main Telemetry Gauges Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Utilization */}
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-3 interactive-card-teal shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase text-[#A6ADB7] tracking-wider">
              GPU Utilization
            </span>
            <Activity className="w-4 h-4 text-[#70D6C2]" />
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-[#F5F5F4] font-mono">
                {util !== null ? `${Math.round(util)}%` : 'N/A'}
              </span>
              <span className="text-xs text-[#A6ADB7] font-mono">Core Load</span>
            </div>
            <div className="w-full h-2 bg-[#121519] rounded-full overflow-hidden mt-3 border border-[#30353D]">
              <div 
                className="h-full bg-[#70D6C2] rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(112,214,194,0.4)]"
                style={{ width: `${util !== null ? Math.min(100, Math.max(2, util)) : 0}%` }}
              />
            </div>
          </div>
          <span className="text-[11px] font-mono text-[#7A828E]">
            Source: NVML Subprocess
          </span>
        </div>

        {/* VRAM Memory */}
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-3 interactive-card shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase text-[#A6ADB7] tracking-wider">
              VRAM Allocated
            </span>
            <Database className="w-4 h-4 text-[#A6ADB7]" />
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-[#F5F5F4] font-mono">
                {usedMib !== null ? Math.round(usedMib) : 'N/A'}
              </span>
              <span className="text-xs text-[#A6ADB7] font-mono">/ {Math.round(totalMib)} MB</span>
            </div>
            <div className="w-full h-2 bg-[#121519] rounded-full overflow-hidden mt-3 border border-[#30353D]">
              <div 
                className="h-full bg-[#76B900] rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(118,185,0,0.4)]"
                style={{ width: `${Math.min(100, Math.max(2, memoryPercent))}%` }}
              />
            </div>
          </div>
          <span className="text-[11px] font-mono text-[#7A828E]">
            {freeMib !== null ? `${Math.round(freeMib)} MB Free` : 'GDDR6 VRAM'}
          </span>
        </div>

        {/* Temperature */}
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-3 interactive-card shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase text-[#A6ADB7] tracking-wider">
              Core Temperature
            </span>
            <Thermometer className="w-4 h-4 text-[#A6ADB7]" />
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <span className={`text-3xl font-bold font-mono ${tempColor}`}>
                {temp !== null ? `${Math.round(temp)}°C` : 'N/A'}
              </span>
              <span className="text-xs text-[#A6ADB7] font-mono">Die Sensor</span>
            </div>
            <div className="w-full h-2 bg-[#121519] rounded-full overflow-hidden mt-3 border border-[#30353D]">
              <div 
                className={`h-full rounded-full transition-all duration-500 ${
                  temp && temp > 75 ? 'bg-[#FB7185]' : temp && temp > 55 ? 'bg-[#FBBF24]' : 'bg-[#4ADE80]'
                }`}
                style={{ width: `${temp !== null ? Math.min(100, (temp / 90) * 100) : 0}%` }}
              />
            </div>
          </div>
          <span className="text-[11px] font-mono text-[#7A828E]">
            Status: {tempStatus}
          </span>
        </div>

        {/* Power Draw */}
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-3 interactive-card-teal shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase text-[#A6ADB7] tracking-wider">
              Power Consumption
            </span>
            <Zap className="w-4 h-4 text-[#70D6C2]" />
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-[#F5F5F4] font-mono">
                {powerW ? `${powerW} W` : '--'}
              </span>
              <span className="text-xs text-[#A6ADB7] font-mono">Active Draw</span>
            </div>
            <div className="w-full h-2 bg-[#121519] rounded-full overflow-hidden mt-3 border border-[#30353D]">
              <div 
                className="h-full bg-[#70D6C2] rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(112,214,194,0.4)]"
                style={{ width: `${powerW ? Math.min(100, (Number(powerW) / 60) * 100) : 10}%` }}
              />
            </div>
          </div>
          <span className="text-[11px] font-mono text-[#7A828E]">
            TGP: Dynamic Laptop OEM
          </span>
        </div>
      </div>

      {/* Historical Telemetry Charts (Session Time-Series) */}
      <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-4 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#30353D] pb-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#70D6C2]" />
            <h2 className="text-sm font-semibold text-[#F5F5F4] tracking-normal font-sans">
              Session Telemetry Time-Series
            </h2>
          </div>
          <span className="text-xs font-mono text-[#A6ADB7]">
            {history.length} Live Samples Recorded (Polled every 3s)
          </span>
        </div>

        {history.length > 1 ? (
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorUtil" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#70D6C2" stopOpacity={0.25}/>
                    <stop offset="95%" stopColor="#70D6C2" stopOpacity={0.0}/>
                  </linearGradient>
                  <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#76B900" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#76B900" stopOpacity={0.0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                <XAxis dataKey="timestamp" stroke="#7A828E" tick={{ fontSize: 10, fill: '#A6ADB7' }} />
                <YAxis stroke="#7A828E" tick={{ fontSize: 10, fill: '#A6ADB7' }} domain={[0, 100]} />
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
                <Area 
                  type="monotone" 
                  dataKey="utilization" 
                  name="GPU Utilization (%)" 
                  stroke="#70D6C2" 
                  strokeWidth={1.5}
                  fillOpacity={1} 
                  fill="url(#colorUtil)" 
                />
                <Area 
                  type="monotone" 
                  dataKey="temperature" 
                  name="Temperature (°C)" 
                  stroke="#76B900" 
                  strokeWidth={1.5}
                  fillOpacity={1} 
                  fill="url(#colorTemp)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-44 flex items-center justify-center text-xs font-mono text-[#7A828E] bg-[#121519] rounded-lg border border-[#30353D]">
            Accumulating live telemetry points... Keep polling active to render session curve.
          </div>
        )}
      </div>

      {/* Hardware Spec & Platform Details Table */}
      <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-4 shadow-lg">
        <div className="flex items-center gap-2 border-b border-[#30353D] pb-3">
          <Server className="w-4 h-4 text-[#A6ADB7]" />
          <h2 className="text-sm font-semibold text-[#F5F5F4]">
            Device &amp; Driver Hardware Snapshot
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 font-sans text-xs">
          <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2">
            <div className="flex justify-between">
              <span className="text-[#A6ADB7]">GPU Device Name</span>
              <span className="text-[#F5F5F4] font-medium">{gpu?.name || 'RTX 3050 Laptop GPU'}</span>
            </div>
            <div className="flex justify-between font-mono">
              <span className="text-[#A6ADB7] font-sans">Hardware UUID</span>
              <span className="text-[#A6ADB7] truncate max-w-[220px]" title={gpu?.uuid}>
                {gpu?.uuid || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#A6ADB7]">Compute Architecture</span>
              <span className="text-[#76B900] font-mono font-medium">SM 8.6 Ampere</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#A6ADB7]">Streaming Multiprocessors</span>
              <span className="text-[#F5F5F4] font-mono">20 SMs (2,560 CUDA Cores)</span>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2">
            <div className="flex justify-between">
              <span className="text-[#A6ADB7]">Host OS Platform</span>
              <span className="text-[#F5F5F4] font-medium">{host?.platform || 'Windows 11'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#A6ADB7]">Host CPU</span>
              <span className="text-[#A6ADB7] truncate max-w-[220px]" title={host?.processor}>
                {host?.processor ? host.processor.split(',')[0] : (host?.processor || 'Host CPU')}
              </span>
            </div>
            <div className="flex justify-between font-mono">
              <span className="text-[#A6ADB7] font-sans">NVIDIA Driver</span>
              <span className="text-[#A6ADB7]">v{gpu?.driver_version || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#A6ADB7]">Query Interface</span>
              <span className="text-[#70D6C2] font-mono">{data?.source || 'nvidia-smi'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
