import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  RotateCw, 
  Thermometer, 
  Zap, 
  Layers, 
  Server, 
  Cpu, 
  AlertCircle,
  Radio
} from 'lucide-react';
import { HardwareResponse, GpuTelemetry } from '../types/api';
import { StatusBadge } from './StatusBadge';
import { WarningBanner } from './WarningBanner';

interface HardwareSectionProps {
  data: HardwareResponse | null;
  loading: boolean;
  onRefresh: () => void;
  autoRefresh: boolean;
  onToggleAutoRefresh: () => void;
}

export const HardwareSection: React.FC<HardwareSectionProps> = ({
  data,
  loading,
  onRefresh,
  autoRefresh,
  onToggleAutoRefresh,
}) => {
  const [lastUpdated, setLastUpdated] = useState<string>('');

  useEffect(() => {
    if (data) {
      const now = new Date();
      setLastUpdated(now.toLocaleTimeString());
    }
  }, [data]);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-4 border-[#76b900]/20 border-t-[#76b900] rounded-full animate-spin" />
          <span className="text-sm font-medium text-slate-400">Querying live NVIDIA telemetry...</span>
        </div>
      </div>
    );
  }

  const gpu: GpuTelemetry | undefined = data?.gpus?.[0];
  const host = data?.host;

  // Calculate memory metrics
  const totalMib = gpu?.memory_total_mib ?? null;
  const usedMib = gpu?.memory_used_mib ?? null;
  const freeMib = gpu?.memory_free_mib ?? null;
  const memoryPercent = (totalMib && usedMib !== null) ? Math.min(100, Math.round((usedMib / totalMib) * 100)) : null;

  const temp = gpu?.temperature_c ?? null;
  const tempColor = temp === null 
    ? 'text-slate-400' 
    : temp < 55 ? 'text-[#76b900]' : temp < 75 ? 'text-amber-400' : 'text-rose-400';

  return (
    <div id="section-hardware" className="space-y-6 animate-fadeIn">
      {/* Header with Live Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.07]">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-extrabold text-white">Live Hardware Telemetry</h2>
            <StatusBadge status={data?.status} />
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Real-Time NVIDIA-SMI Query Engine • Read-Only Hardware Monitor
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {lastUpdated && (
            <span className="text-[11px] font-mono text-slate-400 hidden md:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-obsidian-900 border border-white/[0.05]">
              <Radio className="w-3 h-3 text-[#76b900] animate-pulse" />
              <span>Polled: {lastUpdated}</span>
            </span>
          )}

          <button
            id="btn-toggle-autorefresh"
            onClick={onToggleAutoRefresh}
            aria-label={autoRefresh ? 'Pause automatic hardware telemetry polling' : 'Resume automatic hardware telemetry polling'}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all cursor-pointer ${
              autoRefresh
                ? 'bg-[#76b900]/15 text-[#76b900] border-[#76b900]/40 shadow-[0_0_12px_rgba(118,185,0,0.15)] font-semibold'
                : 'bg-obsidian-900 text-slate-400 border-white/[0.08]'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-[#76b900] animate-pulse' : 'bg-slate-600'}`} />
            <span>{autoRefresh ? 'Auto-Polling (3s)' : 'Polling Paused'}</span>
          </button>

          <button
            id="btn-refresh-hardware"
            onClick={onRefresh}
            disabled={loading}
            aria-label="Refresh hardware telemetry now"
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-obsidian-900 hover:bg-obsidian-850 text-slate-200 text-xs font-semibold border border-white/[0.08] hover:border-white/[0.15] transition-all cursor-pointer disabled:opacity-50"
          >
            <RotateCw className={`w-3.5 h-3.5 text-cyan-400 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Warnings Banner */}
      {data?.warnings && <WarningBanner warnings={data.warnings} />}

      {/* Main GPU Live Dashboard Grid */}
      {gpu ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* GPU Utilization Card */}
            <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] border-t-2 border-t-[#76b900] space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">GPU Utilization</span>
                <Activity className="w-4 h-4 text-[#76b900]" />
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-extrabold text-white font-mono tabular-nums">
                  {gpu.utilization_percent !== null ? gpu.utilization_percent : 'N/A'}
                </span>
                {gpu.utilization_percent !== null && <span className="text-sm font-semibold text-slate-400">%</span>}
              </div>
              {/* Progress bar */}
              <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-white/[0.05]">
                <div 
                  className="bg-[#76b900] h-2 rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(118,185,0,0.5)]" 
                  style={{ width: `${gpu.utilization_percent ?? 0}%` }}
                />
              </div>
              <span className="text-[11px] text-slate-400 block truncate">Streaming Multiprocessors Active</span>
            </div>

            {/* VRAM Usage Card */}
            <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] border-t-2 border-t-cyan-400 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">VRAM Allocation</span>
                <Layers className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-extrabold text-white font-mono tabular-nums">
                  {usedMib !== null ? (usedMib / 1024).toFixed(2) : 'N/A'}
                </span>
                <span className="text-sm font-semibold text-slate-400 font-mono">
                  / {totalMib !== null ? (totalMib / 1024).toFixed(1) : 'N/A'} GB
                </span>
              </div>
              {/* Memory progress bar */}
              <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-white/[0.05]">
                <div 
                  className="bg-cyan-400 h-2 rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(6,182,212,0.5)]" 
                  style={{ width: `${memoryPercent ?? 0}%` }}
                />
              </div>
              <div className="flex justify-between text-[11px] font-mono text-slate-400">
                <span>Free: {freeMib !== null ? `${Math.round(freeMib)} MiB` : 'N/A'}</span>
                <span className="text-cyan-300 font-semibold">{memoryPercent !== null ? `${memoryPercent}%` : ''}</span>
              </div>
            </div>

            {/* Temperature Card */}
            <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] border-t-2 border-t-amber-400 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Thermal Sensor</span>
                <Thermometer className={`w-4 h-4 ${tempColor}`} />
              </div>
              <div className="flex items-baseline gap-1">
                <span className={`text-3xl font-extrabold font-mono tabular-nums ${tempColor}`}>
                  {temp !== null ? temp : 'N/A'}
                </span>
                {temp !== null && <span className="text-sm font-semibold text-slate-400">°C</span>}
              </div>
              <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-white/[0.05]">
                <div 
                  className={`h-2 rounded-full transition-all duration-500 ${
                    (temp ?? 0) < 55 ? 'bg-[#76b900]' : (temp ?? 0) < 75 ? 'bg-amber-400' : 'bg-rose-500'
                  }`}
                  style={{ width: `${Math.min(100, ((temp ?? 30) / 100) * 100)}%` }}
                />
              </div>
              <span className="text-[11px] text-slate-400 block truncate">
                GPU Core Junction • Limit: 87°C
              </span>
            </div>

            {/* Power Consumption Card */}
            <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] border-t-2 border-t-purple-400 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Power Draw</span>
                <Zap className="w-4 h-4 text-purple-400" />
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-extrabold text-white font-mono tabular-nums">
                  {gpu.power_w !== null ? gpu.power_w.toFixed(1) : 'N/A'}
                </span>
                <span className="text-sm font-semibold text-slate-400 font-mono">
                  {gpu.power_limit_w !== null ? `/ ${gpu.power_limit_w.toFixed(0)} W` : 'Watts (Real-Time)'}
                </span>
              </div>
              <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-white/[0.05]">
                <div 
                  className="bg-purple-400 h-2 rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(168,85,247,0.4)]" 
                  style={{ 
                    width: (gpu.power_w !== null && gpu.power_limit_w) 
                      ? `${Math.min(100, (gpu.power_w / gpu.power_limit_w) * 100)}%` 
                      : (gpu.power_w !== null ? `${Math.min(100, (gpu.power_w / 75) * 100)}%` : '0%')
                  }}
                />
              </div>
              <span className="text-[11px] text-slate-400 block truncate" title={gpu.power_limit_w === null ? 'Mobile Laptop GPU: VBIOS delegates power limits to Dynamic Boost' : undefined}>
                {gpu.power_limit_w === null ? 'Mobile GPU: Dynamic Boost managed (VBIOS N/A)' : 'Real-time board power draw'}
              </span>
            </div>
          </div>

          {/* GPU Device Detail Card */}
          <div className="glass-panel rounded-2xl p-6 border border-white/[0.07] space-y-4">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <div className="flex items-center gap-2.5">
                <Cpu className="w-5 h-5 text-[#76b900]" />
                <h3 className="text-base font-bold text-white tracking-tight">{gpu.name}</h3>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-[#76b900]/15 text-[#76b900] border border-[#76b900]/30">
                GPU #{gpu.index}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
              <div className="p-3.5 rounded-xl bg-obsidian-900/70 border border-white/[0.05] space-y-1">
                <span className="text-slate-400 font-sans uppercase text-[10px] tracking-wider">Driver Version</span>
                <p className="text-slate-200 font-semibold">{gpu.driver_version}</p>
              </div>
              <div className="p-3.5 rounded-xl bg-obsidian-900/70 border border-white/[0.05] space-y-1">
                <span className="text-slate-400 font-sans uppercase text-[10px] tracking-wider">Hardware UUID</span>
                <p className="text-slate-200 truncate" title={gpu.uuid}>{gpu.uuid}</p>
              </div>
              <div className="p-3.5 rounded-xl bg-obsidian-900/70 border border-white/[0.05] space-y-1">
                <span className="text-slate-400 font-sans uppercase text-[10px] tracking-wider">Total Dedicated VRAM</span>
                <p className="text-[#76b900] font-semibold">{totalMib ? `${totalMib.toLocaleString()} MiB (${(totalMib / 1024).toFixed(1)} GB GDDR6)` : 'N/A'}</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl p-8 border border-white/[0.07] text-center space-y-3">
          <AlertCircle className="w-10 h-10 text-amber-400 mx-auto" />
          <h3 className="text-lg font-bold text-white">No Active GPU Telemetry</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            nvidia-smi is either not reporting telemetry or no NVIDIA GPU was returned.
          </p>
        </div>
      )}

      {/* Host System Info */}
      {host && (
        <div className="glass-panel rounded-2xl p-6 border border-white/[0.07] space-y-3">
          <div className="flex items-center gap-2 border-b border-white/[0.06] pb-3">
            <Server className="w-4 h-4 text-slate-400" />
            <h3 className="text-sm font-bold text-white">Host System Platform</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
            <div className="p-3 rounded-xl bg-obsidian-900/50 border border-white/[0.04]">
              <span className="text-slate-400 font-sans text-[11px] block">Platform / OS</span>
              <span className="text-slate-200 font-semibold">{host.platform}</span>
            </div>
            <div className="p-3 rounded-xl bg-obsidian-900/50 border border-white/[0.04]">
              <span className="text-slate-400 font-sans text-[11px] block">Processor</span>
              <span className="text-slate-200 font-semibold">{host.processor}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
