import React, { useState } from 'react';
import { 
  Zap, 
  Flame, 
  ArrowRight, 
  CheckCircle, 
  ShieldCheck, 
  Database, 
  Activity, 
  Thermometer, 
  ExternalLink,
  ChevronRight,
  Sparkles,
  Info
} from 'lucide-react';
import { SectionTab } from '../../types/navigation';
import { OverviewResponse, BenchmarkResponse, HardwareResponse } from '../../types/api';
import { ComputeCenterpiece } from './ComputeCenterpiece';
import { useScrollReveal } from '../../hooks/useScrollReveal';

interface ShowcaseExperienceProps {
  overviewData: OverviewResponse | null;
  benchmarksData: BenchmarkResponse | null;
  hardwareData: HardwareResponse | null;
  onNavigateTab: (tab: SectionTab) => void;
}

export const ShowcaseExperience: React.FC<ShowcaseExperienceProps> = ({
  overviewData,
  benchmarksData: _benchmarksData,
  hardwareData,
  onNavigateTab,
}) => {
  // Selected workload in benchmark storytelling section
  const [selectedWorkload, setSelectedWorkload] = useState<'sgemm_2048' | 'sgemm_1024' | 'vec_25m'>('sgemm_2048');

  // Scroll reveal hooks for each story section
  const story1 = useScrollReveal();
  const story2 = useScrollReveal();
  const story3 = useScrollReveal();
  const story4 = useScrollReveal();
  const story5 = useScrollReveal();
  const story6 = useScrollReveal();

  const gpu = hardwareData?.gpus?.[0];
  const vramUsedMib = gpu?.memory_used_mib !== null && gpu?.memory_used_mib !== undefined ? Math.round(gpu.memory_used_mib) : null;
  const vramTotalMib = gpu?.memory_total_mib !== null && gpu?.memory_total_mib !== undefined ? Math.round(gpu.memory_total_mib) : 6144;
  const vramPercent = vramUsedMib ? Math.min(100, Math.round((vramUsedMib / vramTotalMib) * 100)) : 0;
  const gpuTemp = gpu?.temperature_c !== null && gpu?.temperature_c !== undefined ? Math.round(gpu.temperature_c) : null;
  const gpuUtil = gpu?.utilization_percent !== null && gpu?.utilization_percent !== undefined ? Math.round(gpu.utilization_percent) : null;
  const gpuPower = gpu?.power_w !== null && gpu?.power_w !== undefined ? gpu.power_w.toFixed(1) : null;

  // Verified benchmark data points
  const workloadDetails = {
    sgemm_2048: {
      title: 'SGEMM 2048 × 2048',
      category: 'Matrix Multiplication',
      cpuMs: '2,690.86 ms',
      gpuKernelMs: '23.05 ms',
      gpuE2EMs: '29.93 ms',
      kernelSpeedup: '116.75x',
      e2eSpeedup: '89.91x',
      throughput: '745.4 GFLOPS',
      parity: 'PASS (max err ≤ 10⁻³)',
      desc: 'Double-buffered 16×16 shared-memory tile caching reduces global memory reads by 16x.',
    },
    sgemm_1024: {
      title: 'SGEMM 1024 × 1024',
      category: 'Matrix Multiplication',
      cpuMs: '253.48 ms',
      gpuKernelMs: '2.82 ms',
      gpuE2EMs: '4.42 ms',
      kernelSpeedup: '89.77x',
      e2eSpeedup: '57.40x',
      throughput: '760.5 GFLOPS',
      parity: 'PASS (max err ≤ 10⁻³)',
      desc: 'High SM occupancy workload achieving the peak measured single-precision GFLOPS on Ampere.',
    },
    vec_25m: {
      title: 'Vector Add 25,000,000 Elements',
      category: 'Vector Arithmetic',
      cpuMs: '15.98 ms',
      gpuKernelMs: '1.93 ms',
      gpuE2EMs: '45.14 ms',
      kernelSpeedup: '8.28x',
      e2eSpeedup: '0.35x (PCIe bound)',
      throughput: '155.4 GB/s',
      parity: 'PASS (max err ≤ 10⁻⁵)',
      desc: '128-bit memory coalesced loads across 32-thread warps reaching 92.5% of hardware memory bus capacity.',
    },
  };

  const currentWl = workloadDetails[selectedWorkload];

  return (
    <div className="relative space-y-24 sm:space-y-32 pb-24 font-sans text-[#F5F5F4] select-none">
      {/* Background Ambient Illumination & Faint Technical Grid */}
      <div className="absolute inset-0 bg-tech-grid opacity-40 pointer-events-none -z-10" />
      <div className="absolute inset-0 bg-showcase-mesh pointer-events-none -z-10" />

      {/* =========================================================================
          HERO SECTION: Editorial Headline, Subhead, Badges & Primary Actions
          ========================================================================= */}
      <section className="relative pt-12 sm:pt-20 text-center max-w-4xl mx-auto px-4 sm:px-6">
        {/* Floating Architecture Pill */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#171A1F] border border-[#30353D] text-xs font-mono mb-8 animate-fadeIn shadow-sm hover:border-[#76B900]/40 transition-colors">
          <span className="w-2 h-2 rounded-full bg-[#76B900] animate-pulse" />
          <span className="text-[#F5F5F4] font-medium">
            {overviewData?.system_info?.device_name || 'NVIDIA GeForce RTX 3050 6GB Laptop GPU'}
          </span>
          <span className="text-[#30353D]">•</span>
          <span className="text-[#8ED11A] font-semibold">
            SM {overviewData?.system_info?.compute_capability || '8.6'}
          </span>
          <span className="text-[#30353D]">•</span>
          <span className="text-[#A6ADB7]">
            CUDA {overviewData?.system_info?.cuda_driver_version ? `v${overviewData.system_info.cuda_driver_version}` : '12.4'}
          </span>
        </div>

        {/* Large Editorial Headline with Gentle Shimmer */}
        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-gradient-hero mb-6 leading-[1.1] animate-fadeInUp">
          High-performance <br className="hidden sm:inline" />
          <span className="text-gradient-copper-animated">CUDA acceleration</span> &amp; AI orchestration.
        </h1>

        {/* Story Subhead */}
        <p className="text-base sm:text-lg text-[#A6ADB7] max-w-2xl mx-auto leading-relaxed mb-10 animate-fadeInUp delay-100">
          Bridging bare-metal NVIDIA CUDA C++ computing with full-stack observability. 
          Harnessing on-chip SRAM tiling, multi-backend neural training, and verifiable 
          cryptographic precision evaluated against host CPU baselines.
        </p>

        {/* Action CTAs with Subtle Hover Lift */}
        <div className="flex flex-wrap items-center justify-center gap-4 mb-16 animate-fadeInUp delay-200">
          <button
            id="hero-btn-explore-engine"
            onClick={() => onNavigateTab('cuda_lab')}
            className="inline-flex items-center gap-2.5 px-6 py-3 rounded-xl bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] font-semibold text-sm btn-primary-lift shadow-lg shadow-[#76B900]/15 cursor-pointer"
          >
            <Zap className="w-4 h-4 fill-[#08090B] stroke-none" />
            <span>Explore Engine</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            id="hero-btn-launch-training"
            onClick={() => onNavigateTab('training_studio')}
            className="inline-flex items-center gap-2.5 px-6 py-3 rounded-xl bg-[#171A1F] hover:bg-[#20242A] text-[#F5F5F4] border border-[#30353D] hover:border-[#76B900]/40 font-medium text-sm transition-all duration-200 cursor-pointer shadow-sm hover:-translate-y-0.5"
          >
            <Flame className="w-4 h-4 text-[#76B900]" />
            <span>Launch Training</span>
          </button>

          <button
            id="hero-btn-open-workspace"
            onClick={() => onNavigateTab('overview')}
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-[#0D0F12] hover:bg-[#171A1F] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D] font-mono text-xs transition-colors duration-200 cursor-pointer hover:-translate-y-0.5"
          >
            <span>Console View</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Compute Centerpiece Integration */}
        <div className="animate-fadeInUp delay-300">
          <ComputeCenterpiece onExploreEngine={() => onNavigateTab('cuda_lab')} />
        </div>
      </section>

      {/* =========================================================================
          STORYTELLING SECTION 1: Real CUDA Acceleration & Silicon Arithmetic
          ========================================================================= */}
      <section 
        ref={story1.ref} 
        className={`max-w-6xl mx-auto px-4 sm:px-6 transition-all duration-700 ${
          story1.isVisible ? 'reveal-visible' : 'reveal-hidden'
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-end justify-between mb-10 gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-[#76B900] uppercase tracking-wider mb-2">
              <Zap className="w-3.5 h-3.5" />
              <span>01 / Silicon Arithmetic</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-[#F5F5F4]">
              Real CUDA acceleration results.
            </h2>
            <p className="text-sm text-[#A6ADB7] max-w-xl mt-2 leading-relaxed">
              Measured host wall-time comparisons between native single-threaded C++ references 
              and custom CUDA kernels using <code className="text-[#F5F5F4] font-mono">16x16</code> shared memory SRAM tiling.
            </p>
          </div>

          <button
            onClick={() => onNavigateTab('cuda_lab')}
            className="inline-flex items-center gap-1.5 text-xs font-mono text-[#76B900] hover:text-[#8ED11A] self-start lg:self-auto cursor-pointer transition-colors"
          >
            <span>Open CUDA Performance Lab</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Workload Selector Tabs & Detailed Comparison Card */}
        <div className="rounded-2xl bg-[#0D0F12] border border-[#30353D] p-6 sm:p-8 space-y-6 shadow-xl">
          {/* Workload Switcher */}
          <div className="flex flex-wrap items-center gap-2 border-b border-[#30353D] pb-4">
            <button
              onClick={() => setSelectedWorkload('sgemm_2048')}
              className={`px-4 py-2 rounded-lg text-xs font-mono transition-all duration-200 cursor-pointer ${
                selectedWorkload === 'sgemm_2048'
                  ? 'bg-[#76B900]/15 text-[#8ED11A] font-semibold border border-[#76B900]/40 shadow-sm'
                  : 'bg-[#171A1F] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D]'
              }`}
            >
              SGEMM 2048×2048 (Peak 116.75x)
            </button>
            <button
              onClick={() => setSelectedWorkload('sgemm_1024')}
              className={`px-4 py-2 rounded-lg text-xs font-mono transition-all duration-200 cursor-pointer ${
                selectedWorkload === 'sgemm_1024'
                  ? 'bg-[#76B900]/15 text-[#8ED11A] font-semibold border border-[#76B900]/40 shadow-sm'
                  : 'bg-[#171A1F] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D]'
              }`}
            >
              SGEMM 1024×1024 (760.5 GFLOPS)
            </button>
            <button
              onClick={() => setSelectedWorkload('vec_25m')}
              className={`px-4 py-2 rounded-lg text-xs font-mono transition-all duration-200 cursor-pointer ${
                selectedWorkload === 'vec_25m'
                  ? 'bg-[#76B900]/15 text-[#8ED11A] font-semibold border border-[#76B900]/40 shadow-sm'
                  : 'bg-[#171A1F] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D]'
              }`}
            >
              Vector Add 25M (155.4 GB/s)
            </button>
          </div>

          {/* Workload Display Metrics Grid with Hover Elevation */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-[#171A1F] border border-[#30353D] interactive-card">
              <span className="text-[10px] font-mono uppercase text-[#A6ADB7] block mb-1">Single-Thread CPU</span>
              <span className="text-xl font-bold font-mono text-[#FB7185]">{currentWl.cpuMs}</span>
              <span className="text-[11px] text-[#7A828E] block mt-1">Host baseline wall time</span>
            </div>

            <div className="p-4 rounded-xl bg-[#171A1F] border border-[#30353D] interactive-card">
              <span className="text-[10px] font-mono uppercase text-[#76B900] block mb-1">CUDA Kernel Time</span>
              <span className="text-xl font-bold font-mono text-[#76B900]">{currentWl.gpuKernelMs}</span>
              <span className="text-[11px] text-[#7A828E] block mt-1">Dedicated GPU execution</span>
            </div>

            <div className="p-4 rounded-xl bg-[#171A1F] border border-[#76B900]/40 bg-[#76B900]/5 interactive-card">
              <span className="text-[10px] font-mono uppercase text-[#8ED11A] block mb-1">Raw Kernel Speedup</span>
              <span className="text-2xl font-bold font-mono text-[#76B900]">{currentWl.kernelSpeedup}</span>
              <span className="text-[11px] text-[#A6ADB7] block mt-1">Over 1T CPU implementation</span>
            </div>

            <div className="p-4 rounded-xl bg-[#171A1F] border border-[#30353D] interactive-card-teal">
              <span className="text-[10px] font-mono uppercase text-[#70D6C2] block mb-1">Throughput</span>
              <span className="text-xl font-bold font-mono text-[#70D6C2]">{currentWl.throughput}</span>
              <span className="text-[11px] text-[#7A828E] block mt-1">Algorithmic compute / bus</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-[#171A1F] border border-[#30353D] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-[#4ADE80] shrink-0" />
              <span className="text-[#F5F5F4] font-medium">{currentWl.parity}</span>
              <span className="text-[#30353D]">•</span>
              <span className="text-[#A6ADB7]">{currentWl.desc}</span>
            </div>
            <span className="text-[11px] font-mono text-[#7A828E] shrink-0">Synchronized E2E: {currentWl.gpuE2EMs}</span>
          </div>
        </div>
      </section>

      {/* =========================================================================
          STORYTELLING SECTION 2: GPU Hardware & Live Telemetry
          ========================================================================= */}
      <section 
        ref={story2.ref} 
        className={`max-w-6xl mx-auto px-4 sm:px-6 transition-all duration-700 ${
          story2.isVisible ? 'reveal-visible' : 'reveal-hidden'
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-end justify-between mb-10 gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-[#70D6C2] uppercase tracking-wider mb-2">
              <Activity className="w-3.5 h-3.5" />
              <span>02 / Live Telemetry</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-[#F5F5F4]">
              GPU hardware &amp; continuous monitoring.
            </h2>
            <p className="text-sm text-[#A6ADB7] max-w-xl mt-2 leading-relaxed">
              Sub-second NVML subprocess sampling reporting live core utilization, dedicated VRAM allocation, 
              die thermals, and board wattage without synthetic zeroing.
            </p>
          </div>

          <button
            onClick={() => onNavigateTab('gpu_monitoring')}
            className="inline-flex items-center gap-1.5 text-xs font-mono text-[#70D6C2] hover:text-[#8CE0CF] self-start lg:self-auto cursor-pointer transition-colors"
          >
            <span>Open GPU Monitoring</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Live Gauges Container */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Gauge 1: Utilization */}
          <div className="p-5 rounded-2xl bg-[#0D0F12] border border-[#30353D] flex flex-col justify-between interactive-card-teal shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono uppercase text-[#A6ADB7]">Core Utilization</span>
              <Activity className="w-4 h-4 text-[#70D6C2]" />
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-bold font-mono text-[#F5F5F4]">
                  {gpuUtil !== null ? gpuUtil : '--'}
                </span>
                <span className="text-sm text-[#A6ADB7] font-mono">%</span>
              </div>
              <div className="w-full h-1.5 bg-[#171A1F] rounded-full overflow-hidden border border-[#30353D]">
                <div 
                  className="h-full bg-[#70D6C2] transition-all duration-500 rounded-full shadow-[0_0_8px_rgba(112,214,194,0.5)]"
                  style={{ width: `${Math.max(2, gpuUtil || 0)}%` }}
                />
              </div>
            </div>
            <span className="text-[11px] font-mono text-[#7A828E] mt-4">2,560 CUDA Cores</span>
          </div>

          {/* Gauge 2: VRAM Allocation */}
          <div className="p-5 rounded-2xl bg-[#0D0F12] border border-[#30353D] flex flex-col justify-between interactive-card-teal shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono uppercase text-[#A6ADB7]">VRAM Allocation</span>
              <Database className="w-4 h-4 text-[#70D6C2]" />
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-bold font-mono text-[#F5F5F4]">
                  {vramUsedMib !== null ? vramUsedMib : '--'}
                </span>
                <span className="text-sm text-[#A6ADB7] font-mono">/ {vramTotalMib} MB</span>
              </div>
              <div className="w-full h-1.5 bg-[#171A1F] rounded-full overflow-hidden border border-[#30353D]">
                <div 
                  className="h-full bg-[#70D6C2] transition-all duration-500 rounded-full shadow-[0_0_8px_rgba(112,214,194,0.5)]"
                  style={{ width: `${Math.max(2, vramPercent)}%` }}
                />
              </div>
            </div>
            <span className="text-[11px] font-mono text-[#7A828E] mt-4">6,144 MB GDDR6 (96-bit)</span>
          </div>

          {/* Gauge 3: Temperature */}
          <div className="p-5 rounded-2xl bg-[#0D0F12] border border-[#30353D] flex flex-col justify-between interactive-card shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono uppercase text-[#A6ADB7]">Junction Die Temp</span>
              <Thermometer className={`w-4 h-4 ${gpuTemp && gpuTemp > 70 ? 'text-[#FB7185]' : 'text-[#70D6C2]'}`} />
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-1">
                <span className={`text-3xl font-bold font-mono ${gpuTemp && gpuTemp > 70 ? 'text-[#FB7185]' : 'text-[#F5F5F4]'}`}>
                  {gpuTemp !== null ? `${gpuTemp}` : '--'}
                </span>
                <span className="text-sm text-[#A6ADB7] font-mono">°C</span>
              </div>
              <span className="text-xs text-[#70D6C2] font-mono block">
                {gpuTemp && gpuTemp > 75 ? 'Thermal Warning' : 'Optimal Thermal Window'}
              </span>
            </div>
            <span className="text-[11px] font-mono text-[#7A828E] mt-4">Safe Limit: 87°C</span>
          </div>

          {/* Gauge 4: Power Draw */}
          <div className="p-5 rounded-2xl bg-[#0D0F12] border border-[#30353D] flex flex-col justify-between interactive-card shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono uppercase text-[#A6ADB7]">Power Draw</span>
              <Zap className="w-4 h-4 text-[#76B900]" />
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-bold font-mono text-[#F5F5F4]">
                  {gpuPower}
                </span>
                <span className="text-sm text-[#A6ADB7] font-mono">W</span>
              </div>
              <span className="text-xs text-[#A6ADB7] font-mono block">
                NVIDIA-SMI Real-Time Draw
              </span>
            </div>
            <span className="text-[11px] font-mono text-[#7A828E] mt-4">Driver: v{gpu?.driver_version || 'N/A'}</span>
          </div>
        </div>
      </section>

      {/* =========================================================================
          STORYTELLING SECTION 3: AI Model Training Orchestration
          ========================================================================= */}
      <section 
        ref={story3.ref} 
        className={`max-w-6xl mx-auto px-4 sm:px-6 transition-all duration-700 ${
          story3.isVisible ? 'reveal-visible' : 'reveal-hidden'
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-end justify-between mb-10 gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-[#76B900] uppercase tracking-wider mb-2">
              <Flame className="w-3.5 h-3.5" />
              <span>03 / Neural Training</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-[#F5F5F4]">
              Three-backend AI model training.
            </h2>
            <p className="text-sm text-[#A6ADB7] max-w-xl mt-2 leading-relaxed">
              Compare convergence and speed across <code className="text-[#F5F5F4] font-mono">CPU_NATIVE</code>, 
              <code className="text-[#70D6C2] font-mono"> GPU_NATIVE</code> (AMP FP16), and 
              <code className="text-[#8ED11A] font-mono"> GPU_CUSTOM</code> (handcrafted CUDA C++).
            </p>
          </div>

          <button
            onClick={() => onNavigateTab('training_studio')}
            className="inline-flex items-center gap-1.5 text-xs font-mono text-[#76B900] hover:text-[#8ED11A] self-start lg:self-auto cursor-pointer transition-colors"
          >
            <span>Launch Training Studio</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* 6-Step Workflow Banner & Model Cards */}
        <div className="rounded-2xl bg-[#0D0F12] border border-[#30353D] p-6 sm:p-8 space-y-6 shadow-xl">
          <div className="flex items-center justify-between text-xs font-mono text-[#A6ADB7] border-b border-[#30353D] pb-4 overflow-x-auto">
            <span className="text-[#76B900] font-semibold shrink-0">1. Select Architecture</span>
            <span className="text-[#7A828E] shrink-0">&rarr;</span>
            <span className="text-[#F5F5F4] shrink-0">2. Configure Hyperparameters</span>
            <span className="text-[#7A828E] shrink-0">&rarr;</span>
            <span className="text-[#F5F5F4] shrink-0">3. Review Config</span>
            <span className="text-[#7A828E] shrink-0">&rarr;</span>
            <span className="text-[#F5F5F4] shrink-0">4. Mutual Exclusion Launch</span>
            <span className="text-[#7A828E] shrink-0">&rarr;</span>
            <span className="text-[#F5F5F4] shrink-0">5. Real-Time Monitor</span>
            <span className="text-[#7A828E] shrink-0">&rarr;</span>
            <span className="text-[#F5F5F4] shrink-0">6. Inspect Checkpoint</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Model Card 1: ResNet-18 */}
            <div className="p-5 rounded-xl bg-[#171A1F] border border-[#76B900]/40 flex flex-col justify-between interactive-card">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#76B900]/15 text-[#8ED11A] border border-[#76B900]/25">
                    Production Vision
                  </span>
                  <span className="text-xs font-mono text-[#76B900] font-semibold">85.25% Acc</span>
                </div>
                <h3 className="text-base font-bold text-[#F5F5F4]">ResNet-18 (CIFAR-10)</h3>
                <p className="text-xs text-[#A6ADB7] mt-1.5 leading-relaxed">
                  11.17M parameters. Deep residual architecture with 8 residual blocks and BatchNorm.
                  AMP FP16 accelerated on Tensor Cores.
                </p>
              </div>
              <div className="pt-4 border-t border-[#30353D] mt-4 flex items-center justify-between text-xs font-mono">
                <span className="text-[#A6ADB7]">Checkpoint Size</span>
                <span className="text-[#F5F5F4]">134.2 MB (.pt)</span>
              </div>
            </div>

            {/* Model Card 2: CIFAR-10 Custom CUDA MLP */}
            <div className="p-5 rounded-xl bg-[#171A1F] border border-[#30353D] flex flex-col justify-between interactive-card-teal">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#70D6C2]/15 text-[#70D6C2] border border-[#70D6C2]/25">
                    Custom CUDA
                  </span>
                  <span className="text-xs font-mono text-[#70D6C2] font-semibold">53.65% Acc</span>
                </div>
                <h3 className="text-base font-bold text-[#F5F5F4]">CIFAR-10 MLP</h3>
                <p className="text-xs text-[#A6ADB7] mt-1.5 leading-relaxed">
                  3.43M parameters. Forward &amp; backward passes evaluated using handcrafted CUDA C++ 
                  GEMM and fused activation kernels.
                </p>
              </div>
              <div className="pt-4 border-t border-[#30353D] mt-4 flex items-center justify-between text-xs font-mono">
                <span className="text-[#A6ADB7]">Checkpoint Size</span>
                <span className="text-[#F5F5F4]">13.1 MB (.pt)</span>
              </div>
            </div>

            {/* Model Card 3: 100M Causal Transformer Roadmap */}
            <div className="p-5 rounded-xl bg-[#171A1F] border border-[#30353D] opacity-85 flex flex-col justify-between interactive-card">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#7A828E]/20 text-[#A6ADB7] border border-[#7A828E]/30">
                    Roadmap Phase 12
                  </span>
                  <span className="text-xs font-mono text-[#A6ADB7]">Planned</span>
                </div>
                <h3 className="text-base font-bold text-[#A6ADB7]">100M Transformer</h3>
                <p className="text-xs text-[#7A828E] mt-1.5 leading-relaxed">
                  12-layer decoder-only causal Transformer. Architecture planned for Phase 12 LLM pretraining. 
                  Clearly distinguished from active production models.
                </p>
              </div>
              <div className="pt-4 border-t border-[#30353D] mt-4 flex items-center justify-between text-xs font-mono">
                <span className="text-[#7A828E]">Target VRAM</span>
                <span className="text-[#A6ADB7]">~4.2 GB (AMP)</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          STORYTELLING SECTION 4: Architecture Catalog & Checkpoint Persistence
          ========================================================================= */}
      <section 
        ref={story4.ref} 
        className={`max-w-6xl mx-auto px-4 sm:px-6 transition-all duration-700 ${
          story4.isVisible ? 'reveal-visible' : 'reveal-hidden'
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-end justify-between mb-10 gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-[#76B900] uppercase tracking-wider mb-2">
              <Database className="w-3.5 h-3.5" />
              <span>04 / Model Registry</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-[#F5F5F4]">
              Trained models &amp; atomic checkpoints.
            </h2>
            <p className="text-sm text-[#A6ADB7] max-w-xl mt-2 leading-relaxed">
              Atomic state-dict persistence with SHA-256 cryptographic hashes, validation metrics, 
              and seamless reload for downstream classification.
            </p>
          </div>

          <button
            onClick={() => onNavigateTab('model_registry')}
            className="inline-flex items-center gap-1.5 text-xs font-mono text-[#76B900] hover:text-[#8ED11A] self-start lg:self-auto cursor-pointer transition-colors"
          >
            <span>Browse Model Registry</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Checkpoint Table Card */}
        <div className="rounded-2xl bg-[#0D0F12] border border-[#30353D] p-6 sm:p-8 space-y-4 shadow-xl">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-[#171A1F] border border-[#30353D] space-y-2 interactive-card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[#F5F5F4]">resnet18_cifar10/best_model.pt</span>
                <span className="text-[10px] font-mono text-[#8ED11A] bg-[#76B900]/10 px-2 py-0.5 rounded border border-[#76B900]/20">
                  Best Validation
                </span>
              </div>
              <p className="text-xs font-mono text-[#A6ADB7] truncate">
                SHA-256: d86e92cb7151cf388901bbd3c1264c399b9cf0e976ea6509f7a55cb49ca0447a
              </p>
              <div className="flex items-center justify-between text-xs pt-2 border-t border-[#30353D] text-[#A6ADB7]">
                <span>Accuracy: <strong className="text-[#76B900]">85.25%</strong></span>
                <span>File Size: <strong className="text-[#F5F5F4]">134.2 MB</strong></span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#171A1F] border border-[#30353D] space-y-2 interactive-card-teal">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[#F5F5F4]">cifar10_mlp_custom/checkpoint.pt</span>
                <span className="text-[10px] font-mono text-[#70D6C2] bg-[#70D6C2]/10 px-2 py-0.5 rounded border border-[#70D6C2]/20">
                  CUDA C++ Weights
                </span>
              </div>
              <p className="text-xs font-mono text-[#A6ADB7] truncate">
                SHA-256: 4fae7264858d4a6782ff9249ba65ff291f4f5a34ec08b7e2c90c743c7b65fa3a
              </p>
              <div className="flex items-center justify-between text-xs pt-2 border-t border-[#30353D] text-[#A6ADB7]">
                <span>Accuracy: <strong className="text-[#70D6C2]">53.65%</strong></span>
                <span>File Size: <strong className="text-[#F5F5F4]">13.1 MB</strong></span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          STORYTELLING SECTION 5: Verification & Engineering Credibility
          ========================================================================= */}
      <section 
        ref={story5.ref} 
        className={`max-w-6xl mx-auto px-4 sm:px-6 transition-all duration-700 ${
          story5.isVisible ? 'reveal-visible' : 'reveal-hidden'
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-end justify-between mb-10 gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-[#4ADE80] uppercase tracking-wider mb-2">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>05 / Verifiable Engineering</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-[#F5F5F4]">
              Cryptographic integrity &amp; auditability.
            </h2>
            <p className="text-sm text-[#A6ADB7] max-w-xl mt-2 leading-relaxed">
              Every reported benchmark is immutably validated against certified SHA-256 checksums. 
              Zero synthetic extrapolation and complete transparency on baseline references.
            </p>
          </div>

          <button
            onClick={() => onNavigateTab('verification')}
            className="inline-flex items-center gap-1.5 text-xs font-mono text-[#76B900] hover:text-[#8ED11A] self-start lg:self-auto cursor-pointer transition-colors"
          >
            <span>View Verification &amp; Manifest</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Verification Summary Highlights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl bg-[#0D0F12] border border-[#4ADE80]/30 space-y-3 interactive-card shadow-lg">
            <div className="flex items-center gap-2 text-xs font-mono text-[#4ADE80]">
              <CheckCircle className="w-4 h-4" />
              <span>17 / 17 Manifest Matches</span>
            </div>
            <h4 className="text-base font-bold text-[#F5F5F4]">Cryptographic Parity</h4>
            <p className="text-xs text-[#A6ADB7] leading-relaxed">
              100% of Phase 2 benchmark JSON, CSV, and log files match their reference SHA-256 hashes with zero drift.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-[#0D0F12] border border-[#30353D] space-y-3 interactive-card-teal shadow-lg">
            <div className="flex items-center gap-2 text-xs font-mono text-[#70D6C2]">
              <Info className="w-4 h-4" />
              <span>Strict Methodology</span>
            </div>
            <h4 className="text-base font-bold text-[#F5F5F4]">Single-Thread CPU Reference</h4>
            <p className="text-xs text-[#A6ADB7] leading-relaxed">
              Speedups are measured against standard single-threaded C++ references (steady_clock) rather than 
              multithreaded MKL/BLAS libraries.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-[#0D0F12] border border-[#30353D] space-y-3 interactive-card shadow-lg">
            <div className="flex items-center gap-2 text-xs font-mono text-[#A6ADB7]">
              <Database className="w-4 h-4" />
              <span>Confined Artifact Store</span>
            </div>
            <h4 className="text-base font-bold text-[#F5F5F4]">Read-Only Sandbox</h4>
            <p className="text-xs text-[#A6ADB7] leading-relaxed">
              FastAPI backend prevents path-traversal and restricts endpoints to verified read-only paths and authorized training queues.
            </p>
          </div>
        </div>
      </section>

      {/* =========================================================================
          CLOSING CANVAS: Call to Action & Console Entry
          ========================================================================= */}
      <section 
        ref={story6.ref}
        className={`max-w-4xl mx-auto px-4 sm:px-6 text-center transition-all duration-700 ${
          story6.isVisible ? 'reveal-visible' : 'reveal-hidden'
        }`}
      >
        <div className="p-8 sm:p-12 rounded-3xl bg-gradient-to-b from-[#171A1F] to-[#0D0F12] border border-[#30353D] shadow-2xl relative overflow-hidden">
          {/* Faint Radial Accent */}
          <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-80 h-40 bg-[#76B900]/8 blur-3xl pointer-events-none rounded-full" />

          <div className="w-12 h-12 rounded-2xl bg-[#76B900]/15 border border-[#76B900]/30 flex items-center justify-center text-[#76B900] mx-auto mb-6 shadow-sm">
            <Sparkles className="w-6 h-6" />
          </div>

          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-[#F5F5F4] mb-4">
            Ready to orchestrate GPU acceleration workloads?
          </h2>

          <p className="text-sm text-[#A6ADB7] max-w-xl mx-auto mb-8 leading-relaxed">
            Enter the full engineering workspace to configure custom training jobs, profile live hardware 
            telemetry, inspect checkpoints, or test CIFAR-10 inference.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={() => onNavigateTab('training_studio')}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] font-semibold text-sm btn-primary-lift shadow-lg shadow-[#76B900]/15 cursor-pointer"
            >
              <Flame className="w-4 h-4 fill-[#08090B] stroke-none" />
              <span>Launch Training Studio</span>
            </button>

            <button
              onClick={() => onNavigateTab('overview')}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-[#08090B] hover:bg-[#171A1F] text-[#F5F5F4] border border-[#30353D] hover:border-[#76B900]/40 font-medium text-sm transition-all duration-200 cursor-pointer shadow-sm hover:-translate-y-0.5"
            >
              <span>Enter Engineering Console</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
