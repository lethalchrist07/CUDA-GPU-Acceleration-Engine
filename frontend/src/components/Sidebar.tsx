import React from 'react';
import { 
  Zap, 
  Activity, 
  Flame, 
  Database, 
  Cpu, 
  BarChart3, 
  Sparkles, 
  ShieldCheck, 
  ChevronLeft, 
  ChevronRight,
  Server
} from 'lucide-react';
import { SectionTab } from '../types/navigation';
import { HealthResponse } from '../types/api';

interface SidebarProps {
  activeTab: SectionTab;
  onSelectTab: (tab: SectionTab) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  health: HealthResponse | null;
  healthError: boolean;
  gpuName?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  collapsed,
  onToggleCollapse,
  health,
  healthError,
  gpuName = 'RTX 3050 Laptop (6GB)',
}) => {
  const showcaseItem = { 
    id: 'showcase' as SectionTab, 
    label: 'Product Showcase', 
    icon: Sparkles, 
    num: 0, 
    badge: 'Experience' 
  };

  const navItems: { id: SectionTab; label: string; icon: React.ComponentType<{ className?: string }>; num: number; badge?: string }[] = [
    { id: 'overview', label: 'Overview / Command', icon: Zap, num: 1 },
    { id: 'gpu_monitoring', label: 'GPU Monitoring', icon: Activity, num: 2, badge: 'Live' },
    { id: 'training_studio', label: 'AI Training Studio', icon: Flame, num: 3, badge: 'ResNet' },
    { id: 'model_registry', label: 'Model Registry', icon: Database, num: 4 },
    { id: 'cuda_lab', label: 'CUDA Performance Lab', icon: Cpu, num: 5 },
    { id: 'analytics', label: 'Performance Analytics', icon: BarChart3, num: 6 },
    { id: 'inference', label: 'Inference Playground', icon: Sparkles, num: 7 },
    { id: 'verification', label: 'Verification & Artifacts', icon: ShieldCheck, num: 8, badge: '17/17' },
  ];

  return (
    <aside
      id="app-sidebar"
      className={`fixed inset-y-0 left-0 z-40 flex flex-col bg-[#0D0F12] border-r border-[#30353D] transition-all duration-300 ease-in-out select-none ${
        collapsed ? 'w-20' : 'w-72'
      }`}
    >
      {/* Top Branding Section */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-[#30353D] bg-[#0D0F12]">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="w-9 h-9 rounded-lg bg-[#171A1F] flex items-center justify-center shrink-0 border border-[#30353D] shadow-sm">
            <Cpu className="w-5 h-5 text-[#76B900]" />
          </div>
          {!collapsed && (
            <div className="flex flex-col min-w-0 animate-fadeIn">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm tracking-tight text-[#F5F5F4] truncate font-sans">
                  CUDA ENGINE
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#76B900]/10 text-[#8ED11A] font-medium border border-[#76B900]/25">
                  {health?.version ? `v${health.version}` : 'v1.0.0'}
                </span>
              </div>
              <span className="text-[11px] font-mono text-[#A6ADB7] truncate">
                Ampere SM 8.6 • 6GB
              </span>
            </div>
          )}
        </div>

        <button
          onClick={onToggleCollapse}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="p-1.5 rounded-md text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A] border border-transparent hover:border-[#30353D] transition-all cursor-pointer"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation List */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1 scrollbar-thin">
        {/* Showcase / Experience Item */}
        <div className="mb-3">
          <button
            id={`nav-item-${showcaseItem.id}`}
            onClick={() => onSelectTab(showcaseItem.id)}
            title={collapsed ? showcaseItem.label : undefined}
            className={`w-full group relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium font-sans transition-all duration-200 cursor-pointer text-left ${
              activeTab === showcaseItem.id
                ? 'bg-gradient-to-r from-[#76B900]/20 via-[#76B900]/10 to-transparent text-[#F5F5F4] font-semibold border border-[#76B900]/40 shadow-sm shadow-[#76B900]/10'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A] border border-[#30353D]/50 hover:border-[#76B900]/30'
            }`}
          >
            {activeTab === showcaseItem.id && (
              <span className="absolute left-0 top-2 bottom-2 w-1 bg-[#76B900] rounded-r shadow-[0_0_8px_rgba(118,185,0,0.5)] transition-all" />
            )}
            <Sparkles
              className={`w-4 h-4 shrink-0 transition-transform duration-200 group-hover:scale-110 ${
                activeTab === showcaseItem.id
                  ? 'text-[#76B900]'
                  : 'text-[#76B900] group-hover:text-[#8ED11A]'
              }`}
            />
            {!collapsed && (
              <div className="flex items-center justify-between w-full min-w-0">
                <span className="truncate font-semibold tracking-tight">{showcaseItem.label}</span>
                <div className="flex items-center gap-1.5 shrink-0 ml-2">
                  <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-[#76B900]/15 text-[#8ED11A] font-medium border border-[#76B900]/25">
                    {showcaseItem.badge}
                  </span>
                  <span className="text-[10px] font-mono text-[#7A828E] group-hover:text-[#A6ADB7]">
                    [0]
                  </span>
                </div>
              </div>
            )}
          </button>
        </div>

        {!collapsed && (
          <div className="px-3 pt-2 pb-1.5 text-[10px] font-mono uppercase font-semibold text-[#7A828E] tracking-wider border-t border-[#30353D]">
            Engineering Workspace
          </div>
        )}

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;

          return (
            <button
              key={item.id}
              id={`nav-item-${item.id}`}
              onClick={() => onSelectTab(item.id)}
              title={collapsed ? item.label : undefined}
              className={`w-full group relative flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium font-sans transition-all duration-200 cursor-pointer text-left ${
                isActive
                  ? 'bg-[#76B900]/10 text-[#F5F5F4] font-semibold border border-[#76B900]/30 shadow-[0_0_12px_rgba(118,185,0,0.06)]'
                  : 'text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A] border border-transparent'
              }`}
            >
              {/* Active left indicator line */}
              {isActive && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-1 bg-[#76B900] rounded-r shadow-[0_0_8px_rgba(118,185,0,0.5)] transition-all" />
              )}

              <Icon
                className={`w-4 h-4 shrink-0 transition-transform duration-200 group-hover:scale-110 ${
                  isActive
                    ? 'text-[#76B900]'
                    : 'text-[#A6ADB7] group-hover:text-[#F5F5F4]'
                }`}
              />

              {!collapsed && (
                <div className="flex items-center justify-between w-full min-w-0">
                  <span className="truncate">{item.label}</span>
                  <div className="flex items-center gap-1.5 shrink-0 ml-2">
                    {item.badge && (
                      <span className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded ${
                        isActive 
                          ? 'bg-[#76B900]/15 text-[#8ED11A] font-medium' 
                          : 'bg-[#20242A] text-[#A6ADB7] border border-[#30353D]'
                      }`}>
                        {item.badge}
                      </span>
                    )}
                    <span className="text-[10px] font-mono text-[#7A828E] group-hover:text-[#A6ADB7]">
                      [{item.num}]
                    </span>
                  </div>
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Hardware & Backend Status Footer */}
      <div className="p-3 border-t border-[#30353D] bg-[#0D0F12] space-y-2">
        {!collapsed ? (
          <div className="p-2.5 rounded-lg bg-[#171A1F] border border-[#30353D] space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase text-[#A6ADB7] font-medium tracking-wider">
                Target GPU
              </span>
              <span className="flex items-center gap-1.5 text-[10px] font-mono text-[#76B900]">
                <span className="w-1.5 h-1.5 rounded-full bg-[#76B900] shadow-[0_0_6px_rgba(118,185,0,0.5)]" />
                Active
              </span>
            </div>
            <p className="text-xs font-medium text-[#F5F5F4] truncate font-sans" title={gpuName}>
              {gpuName}
            </p>
            <div className="flex items-center justify-between pt-1.5 border-t border-[#30353D] text-[11px] font-mono">
              <span className="text-[#A6ADB7] flex items-center gap-1.5">
                <Server className="w-3 h-3 text-[#70D6C2]" />
                FastAPI
              </span>
              {healthError ? (
                <span className="text-[#FB7185] font-medium">Offline</span>
              ) : health ? (
                <span className="text-[#76B900] font-medium">Online</span>
              ) : (
                <span className="text-[#FBBF24]">Polling...</span>
              )}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 py-1">
            <div 
              title={`GPU: ${gpuName} • FastAPI Online`}
              className="w-8 h-8 rounded-md bg-[#171A1F] border border-[#30353D] flex items-center justify-center text-[#70D6C2]"
            >
              <Server className="w-4 h-4" />
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
