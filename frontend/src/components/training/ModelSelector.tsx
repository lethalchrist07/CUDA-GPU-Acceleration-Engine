import React from 'react';
import { ModelCapabilityInfo, BackendType } from '../../types/training';
import { Cpu, Zap, Box, CheckCircle2, AlertTriangle } from 'lucide-react';

interface ModelSelectorProps {
  models: ModelCapabilityInfo[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  selectedBackend: BackendType;
  onSelectBackend: (backend: BackendType) => void;
  disabled?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  selectedBackend,
  onSelectBackend,
  disabled = false,
}) => {
  const currentModel = models.find((m) => m.model_id === selectedModelId) || models[0];

  const backends: {
    id: BackendType;
    label: string;
    sublabel: string;
    icon: React.ReactNode;
    requiresCustomCuda?: boolean;
  }[] = [
    {
      id: 'CPU_NATIVE',
      label: 'CPU Native',
      sublabel: 'PyTorch CPU Engine (Baseline Reference)',
      icon: <Cpu className="w-4 h-4 text-slate-300" />,
    },
    {
      id: 'GPU_NATIVE',
      label: 'GPU Native',
      sublabel: 'cuBLAS & cuDNN Hardware Acceleration',
      icon: <Zap className="w-4 h-4 text-emerald-400" />,
    },
    {
      id: 'GPU_CUSTOM',
      label: 'GPU Custom CUDA',
      sublabel: 'Handcrafted CUDA Autograd Kernels',
      icon: <Zap className="w-4 h-4 text-[#76b900]" />,
      requiresCustomCuda: true,
    },
  ];

  return (
    <div className="space-y-4">
      {/* 1. Model Architecture Selection */}
      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
          1. Select Model Architecture
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {models.map((m) => {
            const isSelected = m.model_id === selectedModelId;
            return (
              <div
                key={m.model_id}
                onClick={() => !disabled && onSelectModel(m.model_id)}
                className={`relative p-4 rounded-xl border transition-all cursor-pointer ${
                  disabled ? 'opacity-60 cursor-not-allowed' : ''
                } ${
                  isSelected
                    ? 'bg-[#76b900]/10 border-[#76b900]/60 shadow-[0_0_20px_rgba(118,185,0,0.15)]'
                    : 'bg-obsidian-900/90 hover:bg-obsidian-850 border-white/[0.07] hover:border-white/[0.15]'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <div
                      className={`p-2 rounded-lg ${
                        isSelected
                          ? 'bg-[#76b900]/20 text-[#76b900]'
                          : 'bg-white/[0.05] text-slate-400'
                      }`}
                    >
                      <Box className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-white tracking-tight">
                        {m.name}
                      </h4>
                      <span className="text-[11px] font-mono text-slate-400">
                        {m.dataset}
                      </span>
                    </div>
                  </div>
                  {isSelected && (
                    <CheckCircle2 className="w-4 h-4 text-[#76b900] flex-shrink-0" />
                  )}
                </div>

                <p className="text-xs text-slate-400 mt-2 line-clamp-2">
                  {m.description}
                </p>

                {/* Specs Pill Summary */}
                <div className="flex flex-wrap items-center gap-1.5 mt-3 pt-2.5 border-t border-white/[0.05] text-[10px] font-mono">
                  <span className="px-2 py-0.5 rounded bg-white/[0.05] text-slate-300 border border-white/[0.06]">
                    {(m.parameters / 1e6).toFixed(2)}M params
                  </span>
                  <span className="px-2 py-0.5 rounded bg-white/[0.05] text-slate-300 border border-white/[0.06]">
                    {m.classes} classes
                  </span>
                  {m.custom_cuda_supported ? (
                    <span className="px-2 py-0.5 rounded bg-[#76b900]/15 text-[#76b900] border border-[#76b900]/30 font-semibold">
                      Custom CUDA Ready
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      Standard PyTorch
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 2. Execution Backend Selection */}
      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
          2. Select Execution Engine
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {backends.map((b) => {
            const isSupported = currentModel
              ? currentModel.supported_backends.includes(b.id)
              : true;
            const isSelected = selectedBackend === b.id;

            return (
              <div
                key={b.id}
                onClick={() => {
                  if (!disabled && isSupported) {
                    onSelectBackend(b.id);
                  }
                }}
                className={`relative p-3.5 rounded-xl border transition-all cursor-pointer ${
                  !isSupported || disabled
                    ? 'opacity-40 cursor-not-allowed bg-obsidian-950/60 border-white/[0.04]'
                    : isSelected
                    ? 'bg-[#76b900]/10 border-[#76b900]/60 shadow-[0_0_20px_rgba(118,185,0,0.12)]'
                    : 'bg-obsidian-900/90 hover:bg-obsidian-850 border-white/[0.07] hover:border-white/[0.15]'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <div
                      className={`p-1.5 rounded-lg ${
                        isSelected
                          ? 'bg-[#76b900]/20 text-[#76b900]'
                          : 'bg-white/[0.05] text-slate-400'
                      }`}
                    >
                      {b.icon}
                    </div>
                    <span className="text-sm font-bold text-white font-mono">
                      {b.id}
                    </span>
                  </div>
                  {isSelected && (
                    <CheckCircle2 className="w-4 h-4 text-[#76b900] flex-shrink-0" />
                  )}
                </div>

                <div className="text-xs text-slate-400 mt-1.5 font-medium">
                  {b.label}
                </div>
                <p className="text-[11px] text-slate-500 mt-0.5 line-clamp-1">
                  {b.sublabel}
                </p>

                {!isSupported && (
                  <div className="flex items-center gap-1 mt-2 text-[10px] text-amber-400 font-mono">
                    <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                    <span>Unsupported on {currentModel?.name}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
