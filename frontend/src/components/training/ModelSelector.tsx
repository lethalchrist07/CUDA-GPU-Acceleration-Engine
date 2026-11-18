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
      icon: <Cpu className="w-4 h-4 text-[#A6ADB7]" />,
    },
    {
      id: 'GPU_NATIVE',
      label: 'GPU Native',
      sublabel: 'cuBLAS & cuDNN Hardware Acceleration',
      icon: <Zap className="w-4 h-4 text-[#70D6C2]" />,
    },
    {
      id: 'GPU_CUSTOM',
      label: 'GPU Custom CUDA',
      sublabel: 'Handcrafted CUDA Autograd Kernels',
      icon: <Zap className="w-4 h-4 text-[#76B900]" />,
      requiresCustomCuda: true,
    },
  ];

  return (
    <div className="space-y-4">
      {/* 1. Model Architecture Selection */}
      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-[#A6ADB7] font-sans mb-2">
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
                  disabled ? 'opacity-50 cursor-not-allowed' : ''
                } ${
                  isSelected
                    ? 'bg-[#76B900]/10 border-[#76B900]/40'
                    : 'bg-[#121519] hover:bg-[#20242A] border-[#30353D] hover:border-[#76B900]/30'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <div
                      className={`p-2 rounded-lg ${
                        isSelected
                          ? 'bg-[#76B900]/20 text-[#76B900]'
                          : 'bg-[#171A1F] text-[#A6ADB7] border border-[#30353D]'
                      }`}
                    >
                      <Box className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-[#F5F5F4] tracking-tight font-sans">
                        {m.name}
                      </h4>
                      <span className="text-[11px] font-mono text-[#A6ADB7]">
                        {m.dataset}
                      </span>
                    </div>
                  </div>
                  {isSelected && (
                    <CheckCircle2 className="w-4 h-4 text-[#76B900] flex-shrink-0" />
                  )}
                </div>

                <p className="text-xs text-[#A6ADB7] font-sans mt-2 line-clamp-2">
                  {m.description}
                </p>

                {/* Specs Pill Summary */}
                <div className="flex flex-wrap items-center gap-1.5 mt-3 pt-2.5 border-t border-[#30353D] text-[10px] font-mono">
                  <span className="px-2 py-0.5 rounded bg-[#171A1F] text-[#A6ADB7] border border-[#30353D]">
                    {(m.parameters / 1e6).toFixed(2)}M params
                  </span>
                  <span className="px-2 py-0.5 rounded bg-[#171A1F] text-[#A6ADB7] border border-[#30353D]">
                    {m.task === 'CausalLanguageModeling' || m.model_id.includes('transformer')
                      ? '1,024 Vocab'
                      : `${m.classes} classes`}
                  </span>
                  {m.custom_cuda_supported ? (
                    <span className="px-2 py-0.5 rounded bg-[#76B900]/10 text-[#76B900] border border-[#76B900]/30 font-medium">
                      Custom CUDA Ready
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30 font-medium">
                      Native PyTorch SDPA
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
        <label className="block text-xs font-semibold uppercase tracking-wider text-[#A6ADB7] font-sans mb-2">
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
                    ? 'opacity-40 cursor-not-allowed bg-[#08090B] border-[#30353D]'
                    : isSelected
                    ? 'bg-[#76B900]/10 border-[#76B900]/40'
                    : 'bg-[#121519] hover:bg-[#20242A] border-[#30353D] hover:border-[#76B900]/30'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <div
                      className={`p-1.5 rounded-lg ${
                        isSelected
                          ? 'bg-[#76B900]/20 text-[#76B900]'
                          : 'bg-[#171A1F] text-[#A6ADB7] border border-[#30353D]'
                      }`}
                    >
                      {b.icon}
                    </div>
                    <span className="text-sm font-bold text-[#F5F5F4] font-mono">
                      {b.id}
                    </span>
                  </div>
                  {isSelected && (
                    <CheckCircle2 className="w-4 h-4 text-[#76B900] flex-shrink-0" />
                  )}
                </div>

                <div className="text-xs text-[#A6ADB7] font-sans mt-1.5 font-medium">
                  {b.label}
                </div>
                <p className="text-[11px] text-[#6B7280] font-sans mt-0.5 line-clamp-1">
                  {b.sublabel}
                </p>

                {!isSupported && (
                  <div className="flex items-center gap-1 mt-2 text-[10px] text-[#FBBF24] font-mono">
                    <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                    <span>
                      {b.id === 'GPU_CUSTOM' && (currentModel?.model_id === 'resnet18' || currentModel?.model_id.includes('transformer'))
                        ? 'Custom CUDA autograd GEMM implemented for MLP only'
                        : `Unsupported on ${currentModel?.name}`}
                    </span>
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
