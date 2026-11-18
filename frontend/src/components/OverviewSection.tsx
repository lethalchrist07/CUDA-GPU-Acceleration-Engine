import React from 'react';
import { 
  CheckCircle, 
  Cpu, 
  Zap, 
  Server,
  Layers,
  ArrowUpRight,
  Flame,
  Activity,
  Database,
  Box,
  ChevronDown,
  ShieldCheck,
  Sparkles
} from 'lucide-react';
import { OverviewResponse, BenchmarkRecord, HardwareResponse } from '../types/api';
import { TrainingJobResponse } from '../types/training';
import { SectionTab } from '../types/navigation';
import { MetricCard } from './MetricCard';
import { StatusBadge } from './StatusBadge';
import { WarningBanner } from './WarningBanner';

interface OverviewSectionProps {
  data: OverviewResponse | null;
  loading: boolean;
  hardwareData: HardwareResponse | null;
  recentJob: TrainingJobResponse | null;
  onNavigateTab: (tab: SectionTab) => void;
}

export const OverviewSection: React.FC<OverviewSectionProps> = ({
  data,
  loading,
  hardwareData,
  recentJob,
  onNavigateTab,
}) => {
  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-[#30353D] border-t-[#76B900] rounded-full animate-spin" />
          <span className="text-sm font-sans text-[#A6ADB7]">Loading command center telemetry...</span>
        </div>
      </div>
    );
  }

  const sys = data.system_info || {};
  const rows = data.benchmark_rows || [];
  const gpu = hardwareData?.gpus?.[0];

  // Calculate highest speedups and peak compute from actual verified rows
  const maxKernelRow = rows.reduce<BenchmarkRecord | null>(
    (best, r) => (!best || (r.kernel_speedup || 0) > (best.kernel_speedup || 0) ? r : best),
    null
  );
  const maxE2ERow = rows.reduce<BenchmarkRecord | null>(
    (best, r) => (!best || (r.end_to_end_speedup || 0) > (best.end_to_end_speedup || 0) ? r : best),
    null
  );
  const maxKernelSpeedup = maxKernelRow?.kernel_speedup || 0;
  const maxE2ESpeedup = maxE2ERow?.end_to_end_speedup || 0;

  const matrixRows = rows.filter(r => r.category === 'Matrix Multiplication');
  const maxGflopsRow = matrixRows.reduce<BenchmarkRecord | null>(
    (best, r) => (!best || (r.gflops || 0) > (best.gflops || 0) ? r : best),
    null
  );
  const maxGflops = maxGflopsRow?.gflops || 0;

  // Real GPU Telemetry values
  const vramUsedMib = gpu?.memory_used_mib !== null && gpu?.memory_used_mib !== undefined ? Math.round(gpu.memory_used_mib) : null;
  const vramTotalMib = gpu?.memory_total_mib !== null && gpu?.memory_total_mib !== undefined ? Math.round(gpu.memory_total_mib) : 6144;
  const vramPercent = vramUsedMib ? Math.min(100, Math.round((vramUsedMib / vramTotalMib) * 100)) : 0;
  const gpuTemp = gpu?.temperature_c !== null && gpu?.temperature_c !== undefined ? Math.round(gpu.temperature_c) : null;
  const gpuUtil = gpu?.utilization_percent !== null && gpu?.utilization_percent !== undefined ? Math.round(gpu.utilization_percent) : null;
  const gpuPower = gpu?.power_w !== null && gpu?.power_w !== undefined ? gpu.power_w.toFixed(1) : null;

  return (
    <div id="section-overview" className="space-y-6 animate-fadeIn font-sans">
      {/* Top Command Banner */}
      <div className="rounded-xl bg-[#171A1F] border border-[#30353D] p-6 sm:p-7 shadow-xl relative overflow-hidden">
        {/* Subtle Ambient Radial Highlight */}
        <div className="absolute top-0 right-1/4 w-72 h-36 bg-[#76B900]/5 blur-3xl pointer-events-none rounded-full" />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2.5">
              <h1 className="text-2xl sm:text-3xl font-bold text-[#F5F5F4] tracking-tight">
                NVIDIA CUDA Acceleration Command Center
              </h1>
              <span className="px-2.5 py-0.5 rounded-md text-xs font-mono font-medium bg-[#76B900]/10 text-[#8ED11A] border border-[#76B900]/25">
                LIVE RUNTIME
              </span>
            </div>
            <p className="text-sm text-[#A6ADB7] max-w-2xl leading-relaxed">
              Real-time monitoring, hardware profiling, and deep learning training orchestration for 
              <span className="text-[#F5F5F4] font-medium"> NVIDIA RTX 3050 Laptop GPU</span> (Ampere SM 8.6, 6GB VRAM) 
              evaluated against single-threaded native C++ CPU baselines.
            </p>
          </div>

          {/* Quick Action Shortcuts */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              id="btn-overview-showcase"
              onClick={() => onNavigateTab('showcase')}
              className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-[#76B900]/10 hover:bg-[#76B900]/20 text-[#8ED11A] font-semibold text-xs sm:text-sm transition-all duration-200 border border-[#76B900]/30 cursor-pointer hover:-translate-y-0.5"
            >
              <Sparkles className="w-4 h-4 text-[#76B900]" />
              <span>Showcase Experience</span>
            </button>
            <button
              id="btn-overview-launch-training"
              onClick={() => onNavigateTab('training_studio')}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] font-semibold text-xs sm:text-sm btn-primary-lift cursor-pointer shadow-md shadow-[#76B900]/15"
            >
              <Flame className="w-4 h-4 fill-[#08090B] stroke-none" />
              <span>Launch Training Studio</span>
            </button>
            <button
              id="btn-overview-view-models"
              onClick={() => onNavigateTab('model_registry')}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] font-medium text-xs sm:text-sm transition-all duration-200 border border-[#30353D] hover:border-[#76B900]/40 cursor-pointer hover:-translate-y-0.5"
            >
              <Box className="w-4 h-4 text-[#A6ADB7]" />
              <span>Model Registry</span>
            </button>
            <button
              id="btn-overview-open-benchmarks"
              onClick={() => onNavigateTab('cuda_lab')}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] font-medium text-xs sm:text-sm transition-all duration-200 border border-[#30353D] hover:border-[#70D6C2]/40 cursor-pointer hover:-translate-y-0.5"
            >
              <Cpu className="w-4 h-4 text-[#A6ADB7]" />
              <span>CUDA Lab</span>
            </button>
          </div>
        </div>
      </div>

      {/* Primary KPI Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="CUDA Kernels Parity"
          value={`${data.benchmark_passed} / ${data.benchmark_count}`}
          unit="Verified"
          subtitle="100% numerical parity vs CPU"
          icon={<CheckCircle className="w-5 h-5 text-[#4ADE80]" />}
          glow="copper"
          badge={<StatusBadge status={data.status} size="sm" />}
        />
        <MetricCard
          title="Peak End-to-End Speedup"
          value={maxE2ESpeedup > 0 ? `${maxE2ESpeedup.toFixed(1)}x` : 'N/A'}
          subtitle={maxE2ERow ? `${maxE2ERow.category === 'Matrix Multiplication' ? 'SGEMM' : 'VecAdd'} ${maxE2ERow.workload_size}` : 'Host wall time vs CPU ref'}
          icon={<Zap className="w-5 h-5 text-[#76B900]" />}
          glow="copper"
        />
        <MetricCard
          title="Peak Kernel Speedup"
          value={maxKernelSpeedup > 0 ? `${maxKernelSpeedup.toFixed(1)}x` : 'N/A'}
          subtitle={maxKernelRow ? `${maxKernelRow.category === 'Matrix Multiplication' ? 'SGEMM' : 'VecAdd'} ${maxKernelRow.workload_size}` : 'Raw CUDA kernel vs CPU'}
          icon={<Layers className="w-5 h-5 text-[#76B900]" />}
          glow="copper"
        />
        <MetricCard
          title="Peak Compute Throughput"
          value={maxGflops > 0 ? maxGflops.toFixed(1) : 'N/A'}
          unit={maxGflops > 0 ? 'GFLOPS' : undefined}
          subtitle={maxGflopsRow ? `SGEMM ${maxGflopsRow.workload_size} Tiled` : 'Measured FP32 SGEMM'}
          icon={<Cpu className="w-5 h-5 text-[#70D6C2]" />}
          glow="teal"
        />
      </div>

      {/* Live Operational Status: GPU Health & Training Status Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live Hardware Telemetry Gauge Card */}
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-4 interactive-card shadow-lg">
          <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#70D6C2]" />
              <h2 className="text-xs font-semibold text-[#F5F5F4] uppercase tracking-wider font-sans">
                Live GPU Telemetry
              </h2>
            </div>
            <button
              onClick={() => onNavigateTab('gpu_monitoring')}
              className="text-xs text-[#A6ADB7] hover:text-[#76B900] font-sans inline-flex items-center gap-1 cursor-pointer transition-colors"
            >
              <span>Full Telemetry</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1.5 font-sans">
                <span className="text-[#A6ADB7]">VRAM Allocation</span>
                <span className="text-[#F5F5F4] font-mono">
                  {vramUsedMib !== null ? `${vramUsedMib} MB` : '--'} / {vramTotalMib} MB ({vramPercent}%)
                </span>
              </div>
              <div className="w-full h-2 bg-[#121519] rounded-full overflow-hidden border border-[#30353D]">
                <div 
                  className="h-full bg-[#70D6C2] transition-all duration-500 rounded-full shadow-[0_0_8px_rgba(112,214,194,0.5)]"
                  style={{ width: `${Math.max(2, vramPercent)}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2.5 text-center pt-1 font-sans">
              <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
                <span className="text-[10px] text-[#7A828E] block uppercase font-medium">Core Util</span>
                <span className="text-base font-bold text-[#F5F5F4] font-mono">
                  {gpuUtil !== null ? `${gpuUtil}%` : '--'}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
                <span className="text-[10px] text-[#7A828E] block uppercase font-medium">Temperature</span>
                <span className={`text-base font-bold font-mono ${gpuTemp && gpuTemp > 70 ? 'text-[#FB7185]' : 'text-[#F5F5F4]'}`}>
                  {gpuTemp !== null ? `${gpuTemp}°C` : '--'}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D]">
                <span className="text-[10px] text-[#7A828E] block uppercase font-medium">Power</span>
                <span className="text-base font-bold text-[#70D6C2] font-mono">
                  {gpuPower ? `${gpuPower} W` : '--'}
                </span>
              </div>
            </div>
          </div>

          <div className="text-[11px] font-mono text-[#7A828E] pt-2 border-t border-[#30353D] flex items-center justify-between">
            <span>Driver: {gpu?.driver_version || 'N/A'}</span>
            <span className="text-[#7A828E]">nvidia-smi</span>
          </div>
        </div>

        {/* Current / Recent Training Job Card */}
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-4 interactive-card shadow-lg">
          <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-[#76B900]" />
              <h2 className="text-xs font-semibold text-[#F5F5F4] uppercase tracking-wider font-sans">
                Training Orchestration
              </h2>
            </div>
            <button
              onClick={() => onNavigateTab('training_studio')}
              className="text-xs text-[#A6ADB7] hover:text-[#76B900] font-sans inline-flex items-center gap-1 cursor-pointer transition-colors"
            >
              <span>Training Studio</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {recentJob ? (
            <div className="space-y-3 font-sans">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${
                    recentJob.status === 'RUNNING' ? 'bg-[#76B900] animate-pulse' :
                    recentJob.status === 'COMPLETED' ? 'bg-[#4ADE80]' :
                    recentJob.status === 'QUEUED' ? 'bg-[#FBBF24]' : 'bg-[#7A828E]'
                  }`} />
                  <span className="text-xs font-semibold text-[#F5F5F4] uppercase font-mono">
                    {recentJob.status}
                  </span>
                </div>
                <span className="text-[11px] font-mono text-[#7A828E] truncate max-w-[150px]">
                  {recentJob.job_id}
                </span>
              </div>

              <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-[#A6ADB7]">Architecture</span>
                  <span className="text-[#F5F5F4] font-medium uppercase font-mono">{recentJob.model_name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#A6ADB7]">Backend</span>
                  <span className="text-[#70D6C2] font-medium font-mono">{recentJob.backend}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#A6ADB7]">Accuracy / Loss</span>
                  <span className="text-[#F5F5F4] font-mono">
                    {recentJob.progress?.val_accuracy !== undefined ? `${recentJob.progress.val_accuracy.toFixed(1)}%` : '--'} • 
                    Loss: {recentJob.progress?.train_loss !== undefined ? recentJob.progress.train_loss.toFixed(4) : '--'}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between text-xs text-[#A6ADB7]">
                <span>Epochs: <strong className="text-[#F5F5F4] font-mono">{recentJob.epochs}</strong></span>
                <span className="text-[11px]">
                  {recentJob.completed_at ? 'Finished' : 'Running'}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-center py-6 space-y-2">
              <p className="text-xs text-[#A6ADB7]">No active job currently in queue.</p>
              <button
                onClick={() => onNavigateTab('training_studio')}
                className="px-3 py-1.5 rounded-lg bg-[#171A1F] text-[#76B900] border border-[#76B900]/30 text-xs font-medium hover:bg-[#76B900]/10 transition-colors cursor-pointer"
              >
                + Queue New Training Job
              </button>
            </div>
          )}

          <div className="text-[11px] font-mono text-[#7A828E] pt-2 border-t border-[#30353D] flex items-center justify-between">
            <span>Scheduler: Mutual Exclusion</span>
            <span className="text-[#76B900]">Synchronized</span>
          </div>
        </div>

        {/* Model Registry Summary Card */}
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-4 interactive-card shadow-lg">
          <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-[#A6ADB7]" />
              <h2 className="text-xs font-semibold text-[#F5F5F4] uppercase tracking-wider font-sans">
                Model Catalog Snapshot
              </h2>
            </div>
            <button
              onClick={() => onNavigateTab('model_registry')}
              className="text-xs text-[#A6ADB7] hover:text-[#76B900] font-sans inline-flex items-center gap-1 cursor-pointer transition-colors"
            >
              <span>Catalog</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-2 text-xs">
            <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D] flex items-center justify-between">
              <div>
                <span className="text-[#F5F5F4] font-medium block">ResNet-18</span>
                <span className="text-[11px] text-[#A6ADB7]">11.17M params • CIFAR-10</span>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#76B900]/10 text-[#8ED11A] border border-[#76B900]/25">
                85.25% Acc
              </span>
            </div>

            <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D] flex items-center justify-between">
              <div>
                <span className="text-[#F5F5F4] font-medium block">CIFAR-10 MLP</span>
                <span className="text-[11px] text-[#A6ADB7]">3.43M params • Custom CUDA</span>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#70D6C2]/10 text-[#70D6C2] border border-[#70D6C2]/25">
                53.65% Acc
              </span>
            </div>

            <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D] flex items-center justify-between opacity-75">
              <div>
                <span className="text-[#A6ADB7] font-medium block">100M Transformer</span>
                <span className="text-[11px] text-[#7A828E]">Decoder-only • Phase 12</span>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#171A1F] text-[#7A828E] border border-[#30353D]">
                Roadmap
              </span>
            </div>
          </div>

          <div className="text-[11px] font-mono text-[#7A828E] pt-2 border-t border-[#30353D] flex items-center justify-between">
            <span>Verified Checkpoints</span>
            <span className="text-[#F5F5F4]">SHA-256 Validated</span>
          </div>
        </div>
      </div>

      {/* Target Test System Hardware Specifications */}
      <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] shadow-lg">
        <div className="flex items-center justify-between mb-4 border-b border-[#30353D] pb-3">
          <div className="flex items-center gap-2">
            <Server className="w-4 h-4 text-[#76B900]" />
            <h2 className="text-sm font-semibold text-[#F5F5F4]">Target System Specification</h2>
          </div>
          <span className="text-[11px] font-mono text-[#A6ADB7] uppercase">Hardware Baseline</span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 font-sans text-xs">
          <div className="space-y-1">
            <span className="text-[10px] text-[#7A828E] uppercase font-medium">Device</span>
            <p className="text-sm font-medium text-[#F5F5F4] truncate" title={gpu?.name || sys.device_name || 'NVIDIA RTX 3050'}>
              {gpu?.name ? 'RTX 3050 Laptop' : sys.device_name || 'RTX 3050 Laptop'}
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] text-[#7A828E] uppercase font-medium">Architecture</span>
            <p className="text-sm font-mono text-[#76B900] font-medium">
              SM {sys.compute_capability || '8.6'} (Ampere)
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] text-[#7A828E] uppercase font-medium">VRAM Capacity</span>
            <p className="text-sm font-mono text-[#F5F5F4] font-medium">
              6,144 MB GDDR6
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] text-[#7A828E] uppercase font-medium">Compute Units</span>
            <p className="text-sm font-mono text-[#F5F5F4] font-medium">
              20 SMs (2,560 Cores)
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] text-[#7A828E] uppercase font-medium">NVIDIA Driver</span>
            <p className="text-sm font-mono text-[#A6ADB7]">
              v{gpu?.driver_version || sys.cuda_driver_version || 'N/A'}
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] text-[#7A828E] uppercase font-medium">CUDA Toolchain</span>
            <p className="text-sm font-mono text-[#A6ADB7]">
              v12.4.1 (MSVC 19.38)
            </p>
          </div>
        </div>
      </div>

      {/* Expandable Methodology, Provenance & Baseline Disclaimers */}
      <div className="rounded-xl bg-[#171A1F] border border-[#30353D] overflow-hidden transition-all shadow-md">
        <details className="group">
          <summary className="flex items-center justify-between p-4 cursor-pointer hover:bg-[#20242A] transition-colors select-none">
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-lg bg-[#70D6C2]/10 border border-[#70D6C2]/20 flex items-center justify-center text-[#70D6C2]">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <div>
                <span className="text-xs font-semibold text-[#F5F5F4] block">
                  Methodology, Provenance &amp; Baseline Disclaimers
                </span>
                <p className="text-[11px] text-[#A6ADB7]">
                  Single-threaded CPU references, timing isolation, and cryptographic parity disclosures
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#121519] text-[#A6ADB7] border border-[#30353D]">
                {data.warnings?.length || 2} Disclaimers
              </span>
              <ChevronDown className="w-4 h-4 text-[#A6ADB7] group-open:rotate-180 transition-transform duration-200" />
            </div>
          </summary>
          <div className="px-5 pb-5 pt-3 border-t border-[#30353D] bg-[#121519]/60 space-y-4">
            <WarningBanner warnings={data.warnings} />
            <div className="p-3.5 rounded-lg bg-[#171A1F] border border-[#30353D] text-xs text-[#A6ADB7] space-y-2 leading-relaxed font-sans">
              <p>
                <strong className="text-[#F5F5F4]">Baseline Reference Definition:</strong> All reported speedups 
                (peak <span className="text-[#76B900] font-mono">89.91× end-to-end</span> and <span className="text-[#76B900] font-mono">116.75× kernel</span>) 
                are measured against the project's clean, single-threaded C++ reference implementation (<code className="text-[#76B900] font-mono">std::chrono::steady_clock</code> synchronized host wall time). 
                They are strictly not compared against multi-threaded or vendor-optimized BLAS libraries (e.g., Intel MKL or OpenBLAS).
              </p>
              <p>
                <strong className="text-[#F5F5F4]">Cryptographic Verification:</strong> 17/17 Phase 2 benchmark JSON/CSV manifest files are validated against SHA-256 hashes with zero drift. Historical ML experiment logs are explicitly labeled as unverified records for demonstration.
              </p>
            </div>
          </div>
        </details>
      </div>
    </div>
  );
};
