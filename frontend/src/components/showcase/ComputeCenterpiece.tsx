import React, { useState } from 'react';
import { 
  Cpu, 
  Layers, 
  Zap, 
  ArrowRight, 
  Play, 
  CheckCircle,
  Database,
  Info,
  Activity
} from 'lucide-react';

export type CenterpieceMode = 'tiled_sgemm' | 'vector_add' | 'tensor_core';

interface ComputeCenterpieceProps {
  onExploreEngine?: () => void;
}

export const ComputeCenterpiece: React.FC<ComputeCenterpieceProps> = ({ onExploreEngine }) => {
  const [mode, setMode] = useState<CenterpieceMode>('tiled_sgemm');
  const [isSimulating, setIsSimulating] = useState(false);
  const [activeStage, setActiveStage] = useState<number>(0);
  const [hoveredSubsystem, setHoveredSubsystem] = useState<string | null>(null);

  const runSimulation = () => {
    setIsSimulating(true);
    setActiveStage(1);
    
    // Smooth step-by-step dispatch progression
    setTimeout(() => setActiveStage(2), 450);
    setTimeout(() => setActiveStage(3), 900);
    setTimeout(() => setActiveStage(4), 1350);
    setTimeout(() => {
      setIsSimulating(false);
      setActiveStage(0);
    }, 1850);
  };

  const subsystemDetails: Record<string, { title: string; spec: string; desc: string }> = {
    host: {
      title: 'Host CPU Subsystem',
      spec: 'Intel Core i5-13450HX (10 Cores, DDR5-4800)',
      desc: 'Origin of uncoalesced training data and single-threaded CPU reference validation baselines.'
    },
    pcie: {
      title: 'PCIe 4.0 x8 Bus',
      spec: 'Theoretical 15.75 GB/s bidirectional DMA',
      desc: 'DMA memory transfer bridge between host RAM and dedicated GPU device GDDR6 buffers.'
    },
    vram: {
      title: 'Global GDDR6 VRAM',
      spec: '6,144 MB GDDR6 (96-bit bus @ 168 GB/s)',
      desc: 'Dedicated high-bandwidth device memory staging large tensor weights and operational buffers.'
    },
    l2cache: {
      title: 'Unified L2 Cache',
      spec: '2,048 KB on-die crossbar cache',
      desc: 'Ultra-low latency buffer servicing memory requests across all 20 Streaming Multiprocessors.'
    },
    sm: {
      title: '20 Streaming Multiprocessors (SMs)',
      spec: '2,560 CUDA FP32 Cores + 80 3rd Gen Tensor Cores',
      desc: 'Parallel execution engine executing warps of 32 threads with independent integer and FP math pipelines.'
    },
    sram: {
      title: 'Configurable L1 / Shared SRAM',
      spec: '128 KB per SM (up to 48 KB hardware Shared Memory)',
      desc: 'Sub-nanosecond programmer-managed scratchpad enabling 16x16 tiled SGEMM matrix caching.'
    }
  };

  return (
    <div className="relative w-full max-w-5xl mx-auto rounded-2xl bg-[#0D0F12] border border-[#30353D] shadow-2xl overflow-hidden transition-all duration-300">
      {/* Top Header & Interactive Mode Selectors */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 px-6 py-4 bg-[#171A1F] border-b border-[#30353D]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#76B900]/10 border border-[#76B900]/25 flex items-center justify-center text-[#76B900]">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-semibold text-[#F5F5F4] uppercase tracking-wider">
                Ampere Silicon Compute Pipeline
              </span>
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-[#76B900]/10 text-[#8ED11A] border border-[#76B900]/20">
                GA107 • SM 8.6
              </span>
            </div>
            <p className="text-[11px] text-[#A6ADB7]">
              Interactive schematic of on-chip memory hierarchy, SRAM caching &amp; kernel dispatch
            </p>
          </div>
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-[#08090B] border border-[#30353D] text-xs font-mono self-start sm:self-auto">
          <button
            onClick={() => setMode('tiled_sgemm')}
            className={`px-3 py-1.5 rounded-lg transition-all duration-200 cursor-pointer ${
              mode === 'tiled_sgemm'
                ? 'bg-[#171A1F] text-[#76B900] font-semibold shadow-sm border border-[#76B900]/30'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
            }`}
          >
            Tiled SGEMM
          </button>
          <button
            onClick={() => setMode('vector_add')}
            className={`px-3 py-1.5 rounded-lg transition-all duration-200 cursor-pointer ${
              mode === 'vector_add'
                ? 'bg-[#171A1F] text-[#70D6C2] font-semibold shadow-sm border border-[#70D6C2]/30'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
            }`}
          >
            Vector Coalescing
          </button>
          <button
            onClick={() => setMode('tensor_core')}
            className={`px-3 py-1.5 rounded-lg transition-all duration-200 cursor-pointer ${
              mode === 'tensor_core'
                ? 'bg-[#171A1F] text-[#76B900] font-semibold shadow-sm border border-[#76B900]/30'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
            }`}
          >
            AMP FP16
          </button>
        </div>
      </div>

      {/* Main Interactive Stage */}
      <div className="p-6 lg:p-8 space-y-6">
        {/* Dynamic Architectural Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Node 1: Host / CPU Baseline */}
          <div 
            onMouseEnter={() => setHoveredSubsystem('host')}
            onMouseLeave={() => setHoveredSubsystem(null)}
            className="p-4 rounded-xl bg-[#171A1F] border border-[#30353D] relative overflow-hidden group hover:border-[#76B900]/40 transition-all duration-200 interactive-card"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-mono uppercase text-[#A6ADB7] tracking-wider">
                Reference Host
              </span>
              <span className="text-[11px] font-mono text-[#7A828E]">Single-Thread</span>
            </div>
            <h4 className="text-sm font-semibold text-[#F5F5F4] mb-1">Intel Core i5-13450HX</h4>
            <p className="text-xs text-[#A6ADB7] mb-3 leading-relaxed">
              {mode === 'tiled_sgemm' && 'Iterative 3-loop single-threaded C++ baseline (chrono steady_clock).'}
              {mode === 'vector_add' && 'Linear sequential CPU array accumulation over 25M elements.'}
              {mode === 'tensor_core' && 'FP32 standard IEEE-754 reference matrix multiplication.'}
            </p>
            <div className="pt-2 border-t border-[#30353D] flex items-center justify-between text-xs font-mono">
              <span className="text-[#A6ADB7]">Baseline Latency</span>
              <span className="text-[#FB7185] font-semibold">
                {mode === 'tiled_sgemm' && '2,690.8 ms'}
                {mode === 'vector_add' && '15.98 ms'}
                {mode === 'tensor_core' && '1.00x Baseline'}
              </span>
            </div>
          </div>

          {/* Node 2: Memory Hierarchy & SRAM */}
          <div 
            onMouseEnter={() => setHoveredSubsystem('sram')}
            onMouseLeave={() => setHoveredSubsystem(null)}
            className="p-4 rounded-xl bg-[#171A1F] border border-[#30353D] relative overflow-hidden group hover:border-[#76B900]/40 transition-all duration-200 interactive-card"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-mono uppercase text-[#76B900] tracking-wider">
                Silicon Acceleration
              </span>
              <span className="text-[11px] font-mono text-[#8ED11A]">On-Chip SRAM</span>
            </div>
            <h4 className="text-sm font-semibold text-[#F5F5F4] mb-1">
              {mode === 'tiled_sgemm' && '16x16 Shared Memory Tiling'}
              {mode === 'vector_add' && 'Coalesced Memory Strides'}
              {mode === 'tensor_core' && 'Tensor Cores + Dynamic Scaler'}
            </h4>
            <p className="text-xs text-[#A6ADB7] mb-3 leading-relaxed">
              {mode === 'tiled_sgemm' && '48KB SRAM per SM block reduces global VRAM memory requests by 16x via __syncthreads().'}
              {mode === 'vector_add' && '128-byte transaction coalescing across 32-thread warps maximizes bus saturation.'}
              {mode === 'tensor_core' && 'Half-precision FP16 matrix math with full FP32 accumulator retention.'}
            </p>
            <div className="pt-2 border-t border-[#30353D] flex items-center justify-between text-xs font-mono">
              <span className="text-[#A6ADB7]">Algorithmic Peak</span>
              <span className="text-[#76B900] font-semibold">
                {mode === 'tiled_sgemm' && '760.5 GFLOPS'}
                {mode === 'vector_add' && '155.4 GB/s'}
                {mode === 'tensor_core' && 'AMP Scaled'}
              </span>
            </div>
          </div>

          {/* Node 3: Real Speedup & Verified Accuracy */}
          <div 
            onMouseEnter={() => setHoveredSubsystem('sm')}
            onMouseLeave={() => setHoveredSubsystem(null)}
            className="p-4 rounded-xl bg-[#171A1F] border border-[#76B900]/30 relative overflow-hidden interactive-card"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-mono uppercase text-[#70D6C2] tracking-wider">
                Verified Outcome
              </span>
              <span className="flex items-center gap-1 text-[11px] font-mono text-[#4ADE80]">
                <CheckCircle className="w-3.5 h-3.5" /> 100% Parity
              </span>
            </div>
            <div className="flex items-baseline gap-2 mb-1">
              <span className="text-2xl font-bold font-mono text-[#76B900]">
                {mode === 'tiled_sgemm' && '116.75x'}
                {mode === 'vector_add' && '8.28x'}
                {mode === 'tensor_core' && '3.20x'}
              </span>
              <span className="text-xs text-[#A6ADB7]">
                {mode === 'tiled_sgemm' && 'Kernel Speedup'}
                {mode === 'vector_add' && 'Kernel Speedup'}
                {mode === 'tensor_core' && 'Throughput Boost'}
              </span>
            </div>
            <p className="text-xs text-[#A6ADB7] mb-3 leading-relaxed">
              {mode === 'tiled_sgemm' && '23.05 ms GPU vs 2,690.8 ms CPU for 2048x2048 FP32 GEMM.'}
              {mode === 'vector_add' && '1.93 ms GPU vs 15.98 ms CPU for 25,000,000 floats.'}
              {mode === 'tensor_core' && 'Automatic Mixed Precision FP16 on RTX 3050 Tensor Cores.'}
            </p>
            <div className="pt-2 border-t border-[#30353D] flex items-center justify-between text-xs font-mono">
              <span className="text-[#A6ADB7]">Error Tolerance</span>
              <span className="text-[#F5F5F4]">max |err| &le; 10⁻³</span>
            </div>
          </div>
        </div>

        {/* Silicon Pipeline Flow Diagram with Animated Data Path */}
        <div className="relative p-6 rounded-xl bg-[#08090B] border border-[#30353D] overflow-hidden">
          {/* Subtle Ambient Background Glow */}
          <div className="absolute inset-0 bg-tech-grid opacity-60 pointer-events-none" />
          <div className="absolute -top-12 left-1/2 -translate-x-1/2 w-96 h-24 bg-[#76B900]/5 blur-3xl pointer-events-none rounded-full" />

          {/* Pipeline Stages */}
          <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-4 text-xs font-mono">
            {/* Step 1: Host Memory */}
            <div 
              onMouseEnter={() => setHoveredSubsystem('host')}
              onMouseLeave={() => setHoveredSubsystem(null)}
              className={`flex items-center gap-3 p-3.5 rounded-xl bg-[#171A1F] border w-full md:w-auto transition-all duration-300 cursor-pointer ${
                activeStage === 1 || isSimulating
                  ? 'border-[#76B900] shadow-[0_0_15px_rgba(118,185,0,0.25)] scale-105'
                  : 'border-[#30353D] hover:border-[#76B900]/40'
              }`}
            >
              <Database className={`w-4 h-4 ${activeStage === 1 ? 'text-[#76B900]' : 'text-[#A6ADB7]'}`} />
              <div>
                <span className="text-[#A6ADB7] block text-[10px] uppercase">1. Host DDR5</span>
                <span className="text-[#F5F5F4] font-semibold">16 GB Host RAM</span>
              </div>
            </div>

            <ArrowRight className={`w-4 h-4 hidden md:block shrink-0 transition-all duration-300 ${activeStage >= 1 ? 'text-[#76B900] translate-x-1' : 'text-[#7A828E]'}`} />

            {/* Step 2: PCIe Transfer */}
            <div 
              onMouseEnter={() => setHoveredSubsystem('pcie')}
              onMouseLeave={() => setHoveredSubsystem(null)}
              className={`p-3 rounded-xl bg-[#171A1F] border text-center w-full md:w-auto transition-all duration-300 cursor-pointer ${
                activeStage === 2
                  ? 'border-[#70D6C2] shadow-[0_0_15px_rgba(112,214,194,0.25)] scale-105'
                  : 'border-[#30353D] hover:border-[#70D6C2]/40'
              }`}
            >
              <span className="text-[10px] text-[#70D6C2] block font-semibold uppercase">2. H2D DMA Transfer</span>
              <span className="text-[#F5F5F4] text-[11px]">PCIe 4.0 x8 Bus</span>
            </div>

            <ArrowRight className={`w-4 h-4 hidden md:block shrink-0 transition-all duration-300 ${activeStage >= 2 ? 'text-[#70D6C2] translate-x-1' : 'text-[#7A828E]'}`} />

            {/* Step 3: Global VRAM */}
            <div 
              onMouseEnter={() => setHoveredSubsystem('vram')}
              onMouseLeave={() => setHoveredSubsystem(null)}
              className={`flex items-center gap-3 p-3.5 rounded-xl bg-[#171A1F] border w-full md:w-auto transition-all duration-300 cursor-pointer ${
                activeStage === 3
                  ? 'border-[#70D6C2] shadow-[0_0_15px_rgba(112,214,194,0.25)] scale-105'
                  : 'border-[#30353D] hover:border-[#70D6C2]/40'
              }`}
            >
              <Layers className={`w-4 h-4 ${activeStage === 3 ? 'text-[#70D6C2]' : 'text-[#70D6C2]'}`} />
              <div>
                <span className="text-[#70D6C2] block text-[10px] uppercase">3. Device VRAM</span>
                <span className="text-[#F5F5F4] font-semibold">6GB GDDR6 (168 GB/s)</span>
              </div>
            </div>

            <ArrowRight className={`w-4 h-4 hidden md:block shrink-0 transition-all duration-300 ${activeStage >= 3 ? 'text-[#76B900] translate-x-1' : 'text-[#7A828E]'}`} />

            {/* Step 4: 20 SMs + Shared Memory */}
            <div 
              onMouseEnter={() => setHoveredSubsystem('sram')}
              onMouseLeave={() => setHoveredSubsystem(null)}
              className={`p-3.5 rounded-xl bg-[#76B900]/10 border w-full md:w-auto transition-all duration-300 cursor-pointer ${
                activeStage === 4
                  ? 'border-[#76B900] bg-[#76B900]/25 shadow-[0_0_20px_rgba(118,185,0,0.35)] scale-105'
                  : 'border-[#76B900]/30 hover:border-[#76B900]/50'
              }`}
            >
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-[#76B900]" />
                <span className="text-[#8ED11A] font-bold text-[11px]">4. 20 SMs (2,560 Cores)</span>
              </div>
              <span className="text-[10px] text-[#A6ADB7] block mt-0.5">16x16 SRAM Block Tiling</span>
            </div>
          </div>

          {/* Subsystem Explanatory Popout */}
          <div className="mt-4 pt-3 border-t border-[#30353D]/80 flex items-center justify-between text-xs">
            {hoveredSubsystem && subsystemDetails[hoveredSubsystem] ? (
              <div className="flex items-center gap-2 text-[#F5F5F4] animate-fadeIn">
                <Info className="w-3.5 h-3.5 text-[#76B900] shrink-0" />
                <span className="font-semibold text-[#8ED11A]">{subsystemDetails[hoveredSubsystem].title}:</span>
                <span className="text-[#A6ADB7]">{subsystemDetails[hoveredSubsystem].desc}</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-[#7A828E] text-[11px] font-mono">
                <Activity className="w-3.5 h-3.5 text-[#70D6C2]" />
                <span>Hover over any architecture block to inspect on-die hardware roles &amp; memory characteristics.</span>
              </div>
            )}
          </div>

          {/* Interactive Simulation Trigger & Distinction Disclaimer */}
          <div className="mt-4 pt-3 border-t border-[#30353D] flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-[#A6ADB7]">
              <span className={`w-2 h-2 rounded-full ${isSimulating ? 'bg-[#76B900] animate-ping' : 'bg-[#76B900]'}`} />
              <span className="text-[11px] font-mono">
                {isSimulating ? 'Simulating memory-coalesced warp dispatch sequence...' : 'Conceptual Architecture Simulation • Hardware Telemetry Displayed Separately'}
              </span>
            </div>

            <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
              <button
                onClick={runSimulation}
                disabled={isSimulating}
                className="inline-flex items-center justify-center gap-2 px-3.5 py-1.5 rounded-lg bg-[#171A1F] hover:bg-[#20242A] text-[#F5F5F4] border border-[#30353D] hover:border-[#76B900]/40 text-xs font-mono transition-all duration-200 cursor-pointer disabled:opacity-50 hover:-translate-y-0.5"
              >
                <Play className={`w-3.5 h-3.5 text-[#76B900] ${isSimulating ? 'animate-spin' : ''}`} />
                <span>Simulate Dispatch</span>
              </button>

              {onExploreEngine && (
                <button
                  onClick={onExploreEngine}
                  className="inline-flex items-center justify-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] text-xs font-semibold font-mono btn-primary-lift shadow-sm shadow-[#76B900]/15 cursor-pointer"
                >
                  <span>Explore Benchmarks</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
