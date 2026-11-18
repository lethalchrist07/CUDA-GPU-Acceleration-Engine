import React from 'react';
import { 
  Menu, 
  RotateCw, 
  Flame, 
  XCircle, 
  Activity, 
  Thermometer, 
  Database,
  Sparkles,
  LayoutDashboard,
  Server
} from 'lucide-react';
import { SectionTab } from '../types/navigation';
import { HealthResponse, HardwareResponse } from '../types/api';

interface CommandBarProps {
  activeTab: SectionTab;
  onSelectTab: (tab: SectionTab) => void;
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  health: HealthResponse | null;
  healthError: boolean;
  hardwareData: HardwareResponse | null;
  onRefreshHardware: () => void;
  isRefreshing?: boolean;
}

const SECTION_TITLES: Record<SectionTab, { title: string; subtitle: string }> = {
  showcase: { title: 'Product Showcase & Story', subtitle: 'Interactive motion experience & bare-metal CUDA storytelling' },
  overview: { title: 'Overview / Command Center', subtitle: 'Operational landing & executive system snapshot' },
  gpu_monitoring: { title: 'GPU Monitoring & Telemetry', subtitle: 'Real-time NVIDIA-SMI hardware telemetry & thermals' },
  training_studio: { title: 'AI Training Studio', subtitle: 'Three-backend CIFAR-10 training & ResNet-18 pipeline' },
  model_registry: { title: 'Model Registry & Checkpoints', subtitle: 'Verified model weights, architecture catalog & checkpoints' },
  cuda_lab: { title: 'CUDA Performance Laboratory', subtitle: 'Custom CUDA C++ kernels, tiled GEMM & CPU speedups' },
  analytics: { title: 'Performance Analytics', subtitle: 'Cross-backend throughput, convergence & VRAM profiling' },
  inference: { title: 'Inference Playground', subtitle: 'Real CIFAR-10 classification & roadmap LLM architecture' },
  verification: { title: 'Verification & Artifacts', subtitle: 'Phase 2 SHA-256 manifest integrity & test audit' },
};

export const CommandBar: React.FC<CommandBarProps> = ({
  activeTab,
  onSelectTab,
  sidebarCollapsed,
  onToggleSidebar,
  health,
  healthError,
  hardwareData,
  onRefreshHardware,
  isRefreshing = false,
}) => {
  const currentSection = SECTION_TITLES[activeTab] || SECTION_TITLES.overview;
  const gpu = hardwareData?.gpus?.[0];

  const vramUsed = gpu?.memory_used_mib !== null && gpu?.memory_used_mib !== undefined ? Math.round(gpu.memory_used_mib) : null;
  const vramTotal = gpu?.memory_total_mib !== null && gpu?.memory_total_mib !== undefined ? Math.round(gpu.memory_total_mib) : 6144;
  const util = gpu?.utilization_percent !== null && gpu?.utilization_percent !== undefined ? Math.round(gpu.utilization_percent) : null;
  const temp = gpu?.temperature_c !== null && gpu?.temperature_c !== undefined ? Math.round(gpu.temperature_c) : null;

  return (
    <header 
      id="top-command-bar"
      className="sticky top-0 z-30 flex items-center justify-between h-16 px-4 sm:px-6 bg-[#0D0F12]/95 backdrop-blur-md border-b border-[#30353D] transition-colors"
    >
      {/* Left: Sidebar Toggle & Section Title Breadcrumb */}
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={onToggleSidebar}
          aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="p-1.5 rounded-lg text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A] border border-transparent hover:border-[#30353D] transition-all cursor-pointer"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex flex-col min-w-0">
          <div className="flex items-center gap-2 text-xs font-sans">
            <span className="text-[#A6ADB7] hidden sm:inline font-mono">CUDA ENGINE</span>
            <span className="text-[#30353D] hidden sm:inline">/</span>
            <span className="text-[#F5F5F4] font-semibold truncate tracking-tight">{currentSection.title}</span>
          </div>
          <span className="text-xs text-[#A6ADB7] font-normal truncate hidden md:block">
            {currentSection.subtitle}
          </span>
        </div>
      </div>

      {/* Right: Live Telemetry Snippet + Backend Health + Action Buttons */}
      <div className="flex items-center gap-2 sm:gap-3 shrink-0">
        {/* Live GPU Telemetry Pills (desktop view) */}
        {gpu ? (
          <div className="hidden lg:flex items-center gap-3 px-3 py-1.5 rounded-lg bg-[#171A1F] border border-[#30353D] text-[11px] font-mono shadow-sm">
            <div className="flex items-center gap-1.5 text-[#F5F5F4]">
              <Thermometer className={`w-3.5 h-3.5 ${temp && temp > 70 ? 'text-[#FB7185]' : 'text-[#70D6C2]'}`} />
              <span>{temp !== null ? `${temp}°C` : '--'}</span>
            </div>
            <span className="text-[#30353D]">•</span>
            <div className="flex items-center gap-1.5 text-[#F5F5F4]">
              <Activity className="w-3.5 h-3.5 text-[#70D6C2]" />
              <span>{util !== null ? `${util}%` : '--'}</span>
            </div>
            <span className="text-[#30353D]">•</span>
            <div className="flex items-center gap-1.5 text-[#F5F5F4]">
              <Database className="w-3.5 h-3.5 text-[#A6ADB7]" />
              <span>{vramUsed !== null ? `${vramUsed} / ${vramTotal} MB` : '--'}</span>
            </div>
          </div>
        ) : (
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#171A1F] border border-[#30353D] text-[11px] font-mono shadow-sm text-[#FBBF24]">
            <Server className="w-3.5 h-3.5 text-[#FBBF24]" />
            <span>CPU Fallback Mode</span>
          </div>
        )}

        {/* Backend Status Pill */}
        <div 
          id="backend-status-pill"
          className="flex items-center gap-2 px-2.5 sm:px-3 py-1.5 rounded-lg bg-[#171A1F] border border-[#30353D] text-xs font-mono shadow-sm"
        >
          {healthError ? (
            <div className="flex items-center gap-1.5 text-[#FB7185]">
              <XCircle className="w-3.5 h-3.5" />
              <span className="hidden sm:inline font-medium">Backend Offline</span>
            </div>
          ) : health ? (
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#76B900] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#76B900]"></span>
              </span>
              <span className="text-[#F5F5F4] hidden sm:inline">v{health.version}</span>
              <span className="text-[10px] text-[#76B900] font-semibold tracking-wider">ONLINE</span>
            </div>
          ) : (
            <span className="text-[#FBBF24] text-[11px]">Connecting...</span>
          )}
        </div>

        {/* Layer Switcher: Showcase vs Workspace */}
        {activeTab === 'showcase' ? (
          <button
            id="btn-switch-to-workspace"
            onClick={() => onSelectTab('overview')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#171A1F] hover:bg-[#20242A] text-[#F5F5F4] border border-[#30353D] hover:border-[#76B900]/40 text-xs font-medium font-sans transition-all duration-200 cursor-pointer shadow-sm hover:-translate-y-0.5"
          >
            <LayoutDashboard className="w-3.5 h-3.5 text-[#76B900]" />
            <span className="hidden sm:inline">Workspace Console</span>
          </button>
        ) : (
          <button
            id="btn-switch-to-showcase"
            onClick={() => onSelectTab('showcase')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#171A1F] hover:bg-[#20242A] text-[#8ED11A] border border-[#30353D] hover:border-[#76B900]/40 text-xs font-medium font-sans transition-all duration-200 cursor-pointer shadow-sm hover:-translate-y-0.5"
          >
            <Sparkles className="w-3.5 h-3.5 text-[#76B900]" />
            <span className="hidden sm:inline">Product Story</span>
          </button>
        )}

        {/* Quick Launch CTA Button */}
        <button
          id="btn-quick-launch-training"
          onClick={() => onSelectTab('training_studio')}
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] text-xs font-semibold font-sans btn-primary-lift shadow-sm shadow-[#76B900]/15 cursor-pointer"
        >
          <Flame className="w-3.5 h-3.5 fill-[#08090B] stroke-none" />
          <span className="hidden sm:inline">Launch Training</span>
          <span className="sm:hidden">Train</span>
        </button>

        {/* Telemetry Refresh Button */}
        <button
          id="btn-command-refresh-telemetry"
          onClick={onRefreshHardware}
          disabled={isRefreshing}
          title="Query live NVIDIA GPU telemetry via NVML subprocess"
          aria-label="Refresh GPU telemetry"
          className="p-1.5 rounded-lg bg-[#171A1F] hover:bg-[#20242A] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D] hover:border-[#76B900]/40 transition-all duration-200 cursor-pointer disabled:opacity-50"
        >
          <RotateCw className={`w-4 h-4 text-[#70D6C2] ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </header>
  );
};
