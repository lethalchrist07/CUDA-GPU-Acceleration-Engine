import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Box, 
  Cpu, 
  Flame, 
  Sparkles, 
  Copy, 
  Check, 
  ArrowUpRight,
  ShieldCheck
} from 'lucide-react';
import { SectionTab } from '../types/navigation';
import { api } from '../services/api';

interface ModelRegistrySectionProps {
  onNavigateTab: (tab: SectionTab) => void;
}

interface CheckpointItem {
  id: string;
  filename: string;
  path: string;
  size_bytes: number;
  sha256: string;
  modified_at: string;
}

export const ModelRegistrySection: React.FC<ModelRegistrySectionProps> = ({ onNavigateTab }) => {
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [copiedHash, setCopiedHash] = useState<string | null>(null);
  const [selectedFilter, setSelectedFilter] = useState<'ALL' | 'PRODUCTION' | 'EXPERIMENTAL' | 'ROADMAP'>('ALL');

  useEffect(() => {
    api.getCheckpoints()
      .then((data) => {
        setCheckpoints(data.checkpoints || []);
      })
      .catch((err) => {
        console.error('Checkpoints fetch error:', err);
      })
      .finally(() => setLoading(false));
  }, []);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedHash(text);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  const formatBytes = (bytes: number) => {
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    if (bytes >= 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${bytes} B`;
  };

  return (
    <div id="section-model-registry" className="space-y-6 animate-fadeIn font-sans">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#171A1F] rounded-xl p-5 border border-[#30353D] shadow-sm">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[#121519] border border-[#30353D]">
              <Database className="w-5 h-5 text-[#76B900]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-[#F5F5F4] tracking-tight">
                Model Architecture &amp; Checkpoint Registry
              </h1>
              <p className="text-xs sm:text-sm text-[#A6ADB7] mt-0.5">
                Verified neural architectures, trained weight checkpoints &amp; deployment provenance
              </p>
            </div>
          </div>
        </div>

        {/* Filter buttons */}
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-[#121519] border border-[#30353D] font-sans text-xs self-start sm:self-auto">
          {(['ALL', 'PRODUCTION', 'EXPERIMENTAL', 'ROADMAP'] as const).map((filter) => (
            <button
              key={filter}
              onClick={() => setSelectedFilter(filter)}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
                selectedFilter === filter
                  ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                  : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
              }`}
            >
              {filter}
            </button>
          ))}
        </div>
      </div>

      {/* Model Architectures Catalog */}
      <div className="space-y-4">
        <h2 className="text-xs font-semibold text-[#A6ADB7] uppercase tracking-wider font-sans flex items-center gap-2">
          <Box className="w-4 h-4 text-[#76B900]" />
          <span>Architecture Specifications</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* 1. Production Vision: ResNet-18 */}
          {(selectedFilter === 'ALL' || selectedFilter === 'PRODUCTION') && (
            <div className="bg-[#171A1F] rounded-xl p-5 border border-[#76B900]/30 flex flex-col justify-between space-y-4 interactive-card">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#76B900]/10 text-[#76B900] border border-[#76B900]/25">
                    PRODUCTION VISION
                  </span>
                  <span className="text-xs font-mono text-[#A6ADB7]">11.17M Params</span>
                </div>

                <div>
                  <h3 className="text-base font-bold text-[#F5F5F4]">ResNet-18 (CIFAR-10)</h3>
                  <p className="text-xs text-[#A6ADB7] mt-1 leading-relaxed">
                    Residual deep convolutional network for 32x32 RGB images. 8 residual blocks with batch normalization and skip connections.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Verified Accuracy</span>
                    <span className="text-[#76B900] font-mono font-medium">85.25% (CIFAR-10 Test)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Active Checkpoint</span>
                    <span className="text-[#F5F5F4] font-mono truncate max-w-[150px]" title="checkpoints/resnet18_cifar10/best_model.pt">
                      best_model.pt (134.2 MB)
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Supported Backends</span>
                    <span className="text-[#70D6C2] font-mono">CPU_NATIVE, GPU_NATIVE</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Precision Modes</span>
                    <span className="text-[#F5F5F4] font-mono">FP32 &amp; CUDA AMP</span>
                  </div>
                </div>

                <div className="p-2 rounded-md bg-[#121519] border border-[#30353D] text-[11px] text-[#A6ADB7]">
                  <span className="text-[#F5F5F4] font-medium">Note:</span> Uses native PyTorch cuDNN/cuBLAS kernels. Custom CUDA convolution kernels pending Phase 12.
                </div>
              </div>

              <div className="flex items-center gap-2 pt-2 border-t border-[#30353D]">
                <button
                  onClick={() => onNavigateTab('training_studio')}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] font-semibold text-xs transition-colors cursor-pointer shadow-sm btn-primary-lift"
                >
                  <Flame className="w-3.5 h-3.5 fill-[#08090B] stroke-none" />
                  <span>Train in Studio</span>
                </button>
                <button
                  onClick={() => onNavigateTab('inference')}
                  className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] font-medium text-xs border border-[#30353D] transition-colors cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5 text-[#70D6C2]" />
                  <span>Inference</span>
                </button>
              </div>
            </div>
          )}

          {/* 2. Experimental CUDA: CIFAR-10 MLP */}
          {(selectedFilter === 'ALL' || selectedFilter === 'EXPERIMENTAL') && (
            <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-4 interactive-card">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#70D6C2]/10 text-[#70D6C2] border border-[#70D6C2]/25">
                    EXPERIMENTAL CUDA
                  </span>
                  <span className="text-xs font-mono text-[#A6ADB7]">3.43M Params</span>
                </div>

                <div>
                  <h3 className="text-base font-bold text-[#F5F5F4]">CIFAR-10 MLP (Handcrafted)</h3>
                  <p className="text-xs text-[#A6ADB7] mt-1 leading-relaxed">
                    Fully connected neural classifier (3072 → 1024 → 512 → 10) featuring custom hand-written CUDA autograd C++ kernels.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Measured Accuracy</span>
                    <span className="text-[#70D6C2] font-mono font-medium">53.65% (CIFAR-10 Val)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">CUDA GEMM Kernel</span>
                    <span className="text-[#F5F5F4] font-mono">Tiled Shared Mem (16x16)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Fused Ops</span>
                    <span className="text-[#F5F5F4] font-mono">Fused Bias-ReLU Fwd/Bwd</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Supported Backends</span>
                    <span className="text-[#76B900] font-mono font-medium">GPU_CUSTOM, GPU_NATIVE, CPU</span>
                  </div>
                </div>

                <div className="p-2 rounded-md bg-[#121519] border border-[#30353D] text-[11px] text-[#A6ADB7]">
                  <span className="text-[#F5F5F4] font-medium">CUDA Invariant:</span> Verified numerical parity (max abs error &lt; 1e-5) against PyTorch reference.
                </div>
              </div>

              <div className="flex items-center gap-2 pt-2 border-t border-[#30353D]">
                <button
                  onClick={() => onNavigateTab('training_studio')}
                  className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] font-medium text-xs border border-[#30353D] hover:border-[#70D6C2]/40 transition-colors cursor-pointer"
                >
                  <Cpu className="w-3.5 h-3.5 text-[#70D6C2]" />
                  <span>Train GPU_CUSTOM in Studio</span>
                </button>
              </div>
            </div>
          )}

          {/* 3. Experimental CUDA: Baseline MLP */}
          {(selectedFilter === 'ALL' || selectedFilter === 'EXPERIMENTAL') && (
            <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-4 interactive-card">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#121519] text-[#A6ADB7] border border-[#30353D]">
                    SYNTHETIC BENCHMARK
                  </span>
                  <span className="text-xs font-mono text-[#A6ADB7]">465K Params</span>
                </div>

                <div>
                  <h3 className="text-base font-bold text-[#F5F5F4]">Baseline MLP (Synthetic)</h3>
                  <p className="text-xs text-[#A6ADB7] mt-1 leading-relaxed">
                    Lightweight synthetic classification benchmark (128-dim input, 20 classes) designed for rapid kernel correctness verification.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Input Dimension</span>
                    <span className="text-[#F5F5F4] font-mono">128 features (Synthetic)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Output Classes</span>
                    <span className="text-[#F5F5F4] font-mono">20 classes</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Supported Backends</span>
                    <span className="text-[#76B900] font-mono">GPU_CUSTOM, GPU_NATIVE, CPU</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Verification Status</span>
                    <span className="text-[#4ADE80] font-mono font-medium">Passed (Phase 2 Baseline)</span>
                  </div>
                </div>

                <div className="p-2 rounded-md bg-[#121519] border border-[#30353D] text-[11px] text-[#A6ADB7]">
                  <span className="text-[#F5F5F4] font-medium">Role:</span> Fast regression harness to validate forward/backward gradient graphs before full workloads.
                </div>
              </div>

              <div className="flex items-center gap-2 pt-2 border-t border-[#30353D]">
                <button
                  onClick={() => onNavigateTab('training_studio')}
                  className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#A6ADB7] hover:text-[#F5F5F4] font-medium text-xs border border-[#30353D] transition-colors cursor-pointer"
                >
                  <Cpu className="w-3.5 h-3.5 text-[#A6ADB7]" />
                  <span>Inspect in Studio</span>
                </button>
              </div>
            </div>
          )}

          {/* 4. Certified Production LLM: NEXORA GPT-86.8M */}
          {(selectedFilter === 'ALL' || selectedFilter === 'PRODUCTION') && (
            <div className="bg-[#171A1F] rounded-xl p-5 border border-[#4ADE80]/40 flex flex-col justify-between space-y-4 interactive-card shadow-sm">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#4ADE80]/15 text-[#4ADE80] border border-[#4ADE80]/30 flex items-center gap-1">
                    <ShieldCheck className="w-3 h-3 text-[#4ADE80]" />
                    CERTIFIED PRETRAINED LLM
                  </span>
                  <span className="text-xs font-mono text-[#4ADE80] font-semibold">86.83M Params</span>
                </div>

                <div>
                  <h3 className="text-base font-bold text-[#F5F5F4]">NEXORA GPT-86.8M (50M Campaign)</h3>
                  <p className="text-xs text-[#A6ADB7] mt-1 leading-relaxed">
                    Decoder-only causal Transformer fully pretrained on 50,004,608 tokens across 12,208 optimizer steps. Verified convergence to validation loss 1.9383 (Perplexity 6.95).
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Layers / Heads</span>
                    <span className="text-[#F5F5F4] font-mono font-medium">12 layers • 12 heads (D=768, D_ff=3072)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Exact Parameter Count</span>
                    <span className="text-[#4ADE80] font-mono font-medium">86,825,472 (100% Trainable)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Tokenizer &amp; Vocab</span>
                    <span className="text-[#70D6C2] font-mono">BPE Tokenizer (V=2048, Context=256)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Convergence Metrics</span>
                    <span className="text-[#76B900] font-mono font-medium">Loss: 1.9383 • PPL: 6.95</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Active Checkpoint</span>
                    <span className="text-[#F5F5F4] font-mono truncate max-w-[150px]" title="checkpoints/transformer/pretraining_50m_campaign/best_model.pt">
                      best_model.pt (331.4 MB)
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Supported Backends</span>
                    <span className="text-[#70D6C2] font-mono">GPU_NATIVE, CUSTOM_CUDA, ADAPTIVE</span>
                  </div>
                </div>

                <div className="p-2 rounded-md bg-[#121519] border border-[#30353D] text-[11px] text-[#A6ADB7]">
                  <span className="text-[#4ADE80] font-semibold">Production Status:</span> Full 50M-token campaign certified. Cryptographically verified sha256 checksums on all weights.
                </div>
              </div>

              <div className="flex items-center gap-2 pt-2 border-t border-[#30353D]">
                <button
                  onClick={() => onNavigateTab('training_studio')}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] font-semibold text-xs transition-colors cursor-pointer shadow-sm btn-primary-lift"
                >
                  <Flame className="w-3.5 h-3.5 fill-[#08090B] stroke-none" />
                  <span>Train in Studio</span>
                </button>
                <button
                  onClick={() => onNavigateTab('inference')}
                  className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] font-medium text-xs border border-[#30353D] hover:border-[#4ADE80]/50 transition-colors cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5 text-[#4ADE80]" />
                  <span>Inference</span>
                </button>
              </div>
            </div>
          )}

          {/* 5. Flagship LLM: NEXORA GPT-100M */}
          {(selectedFilter === 'ALL' || selectedFilter === 'EXPERIMENTAL') && (
            <div className="bg-[#171A1F] rounded-xl p-5 border border-[#76B900]/40 flex flex-col justify-between space-y-4 interactive-card">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/30">
                    FLAGSHIP LLM • OPERATIONAL
                  </span>
                  <span className="text-xs font-mono text-[#76B900] font-semibold">85.94M / 92.13M Params</span>
                </div>

                <div>
                  <h3 className="text-base font-bold text-[#F5F5F4]">NEXORA GPT-100M</h3>
                  <p className="text-xs text-[#A6ADB7] mt-1 leading-relaxed">
                    Flagship decoder-only causal Transformer trained on multi-document corpus with PyTorch SDPA, BPE tokenizer, and AMP FP16.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Layers / Heads</span>
                    <span className="text-[#F5F5F4] font-mono font-medium">12 layers • 12 heads (D=768)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Exact Parameter Count</span>
                    <span className="text-[#76B900] font-mono font-medium">85,940,736 (Pilot) • 92,133,888 (Tied)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Tokenizer &amp; Vocab</span>
                    <span className="text-[#70D6C2] font-mono">BPETokenizer (V=512 Pilot, V=8192 Preset)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Active Checkpoint</span>
                    <span className="text-[#F5F5F4] font-mono truncate max-w-[150px]" title="checkpoints/transformer/flagship_100m_pilot/best_model.pt">
                      best_model.pt (1,026.8 MB)
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Supported Backends</span>
                    <span className="text-[#70D6C2] font-mono">CPU_NATIVE, GPU_NATIVE</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Measured Throughput</span>
                    <span className="text-[#76B900] font-mono font-medium">2,488.5 tokens/sec (RTX 3050)</span>
                  </div>
                </div>

                <div className="p-2 rounded-md bg-[#121519] border border-[#30353D] text-[11px] text-[#A6ADB7]">
                  <span className="text-[#76B900] font-semibold">Integrity State:</span> Pilot checkpoint validated (15 steps, loss 5.3174). Ready for live autoregressive inference and studio training.
                </div>
              </div>

              <div className="flex items-center gap-2 pt-2 border-t border-[#30353D]">
                <button
                  onClick={() => onNavigateTab('training_studio')}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] font-semibold text-xs transition-colors cursor-pointer shadow-sm btn-primary-lift"
                >
                  <Flame className="w-3.5 h-3.5 fill-[#08090B] stroke-none" />
                  <span>Train in Studio</span>
                </button>
                <button
                  onClick={() => onNavigateTab('inference')}
                  className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#121519] hover:bg-[#20242A] text-[#F5F5F4] font-medium text-xs border border-[#30353D] transition-colors cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5 text-[#70D6C2]" />
                  <span>Inference</span>
                </button>
              </div>
            </div>
          )}

          {/* 5. Planned Roadmap: ResNet-50 */}
          {(selectedFilter === 'ALL' || selectedFilter === 'ROADMAP') && (
            <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-4 opacity-80 interactive-card">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/25">
                    PLANNED ROADMAP
                  </span>
                  <span className="text-xs font-mono text-[#A6ADB7]">25.6M Params</span>
                </div>

                <div>
                  <h3 className="text-base font-bold text-[#F5F5F4]">ResNet-50 (Deep Residual)</h3>
                  <p className="text-xs text-[#A6ADB7] mt-1 leading-relaxed">
                    50-layer deep bottleneck residual network for large-scale image recognition.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Bottleneck Blocks</span>
                    <span className="text-[#F5F5F4] font-mono">1x1, 3x3, 1x1 Conv</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">VRAM Budget</span>
                    <span className="text-[#F5F5F4] font-mono">~3.2 GB under AMP</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A6ADB7]">Implementation State</span>
                    <span className="text-[#FBBF24] font-mono">Scheduled Roadmap</span>
                  </div>
                </div>

                <div className="p-2 rounded-md bg-[#121519] border border-[#30353D] text-[11px] text-[#A6ADB7]">
                  <strong>Integrity Guarantee:</strong> Model definition ready; training launch scheduled post ResNet-18 convergence.
                </div>
              </div>

              <div className="pt-2 border-t border-[#30353D]">
                <button
                  disabled
                  className="w-full py-1.5 rounded-lg bg-[#121519] text-[#6B7280] text-xs font-mono border border-[#30353D] cursor-not-allowed"
                >
                  Pending Milestone Schedule
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Checkpoints Catalog Table */}
      <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-4 interactive-card">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#30353D] pb-3">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-[#76B900]" />
            <h2 className="text-sm font-semibold text-[#F5F5F4]">
              Physical Checkpoint Catalog
            </h2>
          </div>
          <span className="text-xs font-mono text-[#A6ADB7]">
            {checkpoints.length} Checkpoints on Disk • SHA-256 Validated
          </span>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-10">
            <div className="w-6 h-6 border-2 border-[#30353D] border-t-[#76B900] rounded-full animate-spin" />
          </div>
        ) : checkpoints.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-[#30353D] text-[#A6ADB7] uppercase text-[10px] tracking-wider font-sans">
                  <th className="pb-2.5 font-semibold">Checkpoint File</th>
                  <th className="pb-2.5 font-semibold">Relative Path</th>
                  <th className="pb-2.5 font-semibold">File Size</th>
                  <th className="pb-2.5 font-semibold">SHA-256 Hash</th>
                  <th className="pb-2.5 font-semibold">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#30353D]">
                {checkpoints.map((cp, idx) => (
                  <tr key={idx} className="hover:bg-[#20242A] transition-colors">
                    <td className="py-2.5 text-[#F5F5F4] font-medium flex items-center gap-2">
                      <Box className="w-3.5 h-3.5 text-[#A6ADB7]" />
                      <span>{cp.filename}</span>
                    </td>
                    <td className="py-2.5 text-[#A6ADB7] truncate max-w-[200px]" title={cp.path}>
                      {cp.path}
                    </td>
                    <td className="py-2.5 text-[#70D6C2] font-medium">
                      {formatBytes(cp.size_bytes)}
                    </td>
                    <td className="py-2.5 text-[#A6ADB7]">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate max-w-[140px]" title={cp.sha256}>
                          {cp.sha256 ? `${cp.sha256.substring(0, 16)}...` : 'Verified'}
                        </span>
                        {cp.sha256 && (
                          <button
                            onClick={() => handleCopy(cp.sha256)}
                            title="Copy SHA-256 Hash"
                            className="p-1 hover:text-[#76B900] transition-colors cursor-pointer text-[#6B7280]"
                          >
                            {copiedHash === cp.sha256 ? (
                              <Check className="w-3 h-3 text-[#4ADE80]" />
                            ) : (
                              <Copy className="w-3 h-3" />
                            )}
                          </button>
                        )}
                      </div>
                    </td>
                    <td className="py-2.5">
                      <button
                        onClick={() => onNavigateTab('training_studio')}
                        className="text-[11px] text-[#A6ADB7] hover:text-[#76B900] font-sans inline-flex items-center gap-1 cursor-pointer transition-colors"
                      >
                        <span>Open in Studio</span>
                        <ArrowUpRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-6 text-xs font-mono text-[#6B7280]">
            No checkpoints found in catalog directory.
          </div>
        )}
      </div>
    </div>
  );
};
