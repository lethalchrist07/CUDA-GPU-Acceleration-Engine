import React from 'react';
import { 
  Cpu, 
  BarChart3, 
  Flame, 
  Sparkles, 
  Activity, 
  ShieldCheck, 
  Zap,
  XCircle
} from 'lucide-react';
import { HealthResponse } from '../types/api';

export type SectionTab = 'overview' | 'training_studio' | 'benchmarks' | 'training' | 'inference' | 'hardware' | 'verification';

interface NavbarProps {
  activeTab: SectionTab;
  onSelectTab: (tab: SectionTab) => void;
  health: HealthResponse | null;
  healthError: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onSelectTab,
  health,
  healthError,
}) => {
  const tabs: { id: SectionTab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <Zap className="w-4 h-4" /> },
    { id: 'training_studio', label: 'AI Training Studio', icon: <Flame className="w-4 h-4 text-[#76B900]" /> },
    { id: 'benchmarks', label: 'CUDA Benchmarks', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'training', label: 'Historical Benchmarks', icon: <Flame className="w-4 h-4 text-[#858D98]" /> },
    { id: 'inference', label: 'LLM Inference', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'hardware', label: 'Live Telemetry', icon: <Activity className="w-4 h-4" /> },
    { id: 'verification', label: 'Verification & Manifest', icon: <ShieldCheck className="w-4 h-4" /> },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-[#30353D] bg-[#0D0F12]/90 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Hardware Summary */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#76B900] to-[#8ED11A] flex items-center justify-center shadow-md shadow-[#76B900]/20 text-[#08090B] font-bold border border-[#76B900]/40">
              <Cpu className="w-5 h-5 text-[#08090B] stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-base tracking-tight text-[#F5F5F4]">
                  CUDA GPU Acceleration Engine
                </span>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-md bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/30 font-semibold tracking-wide">
                  RTX 3050 Laptop • SM 8.6 Ampere
                </span>
                <span className="hidden xl:inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[#171A1F] text-[#A6ADB7] border border-[#30353D] text-[10px] font-mono">
                  Independent Portfolio Project
                </span>
              </div>
              <p className="text-xs text-[#A6ADB7] hidden sm:block">
                Independent GPU Systems Engineering Portfolio • Read-Only Telemetry Engine
              </p>
            </div>
          </div>

          {/* Backend Status Pill */}
          <div className="flex items-center gap-2">
            <div 
              id="backend-health-indicator"
              className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono border border-[#30353D] bg-[#121519] shadow-inner"
            >
              {healthError ? (
                <>
                  <XCircle className="w-3.5 h-3.5 text-[#FB7185]" />
                  <span className="text-[#FB7185]">Backend Offline</span>
                </>
              ) : health ? (
                <>
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4ADE80] opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[#4ADE80]"></span>
                  </span>
                  <span className="text-[#F5F5F4]">FastAPI v{health.version}</span>
                  <span className="text-[10px] text-[#4ADE80] font-semibold uppercase tracking-wider">ONLINE</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-[#FBBF24] animate-pulse" />
                  <span className="text-[#FBBF24]">Connecting...</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex space-x-1 sm:space-x-1.5 overflow-x-auto py-2 scrollbar-none border-t border-[#30353D]" aria-label="Dashboard Section Navigation">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                id={`nav-tab-${tab.id}`}
                onClick={() => onSelectTab(tab.id)}
                aria-label={`Navigate to ${tab.label}`}
                aria-selected={isActive}
                role="tab"
                className={`relative flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-[#76B900]/15 text-[#F5F5F4] border border-[#76B900]/40 shadow-[0_0_15px_rgba(118,185,0,0.12)] font-semibold'
                    : 'text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A] border border-transparent'
                }`}
              >
                <span className={isActive ? 'text-[#76B900]' : 'text-[#6B7280]'}>
                  {tab.icon}
                </span>
                <span>{tab.label}</span>
                {isActive && (
                  <span className="absolute bottom-0 left-2 right-2 h-[2px] bg-[#76B900] rounded-full shadow-[0_0_8px_#76B900]" />
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
