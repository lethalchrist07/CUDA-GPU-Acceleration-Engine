import React, { useState, useEffect } from 'react';
import { BackendType, TrainingJobCreateRequest, AdaptiveRecommendationResponse } from '../../types/training';
import { trainingApi } from '../../services/trainingApi';
import { Play, Sliders, Shield, RefreshCw, Zap, Settings2, AlertTriangle, Cpu, Layers, Sparkles } from 'lucide-react';

interface TrainingConfigFormProps {
  modelId: string;
  backend: BackendType;
  onSubmit: (request: TrainingJobCreateRequest) => Promise<void>;
  isSubmitting: boolean;
  disabled?: boolean;
}

export const TrainingConfigForm: React.FC<TrainingConfigFormProps> = ({
  modelId,
  backend,
  onSubmit,
  isSubmitting,
  disabled = false,
}) => {
  const isTransformer =
    modelId.toLowerCase().includes('transformer') ||
    modelId.toLowerCase().includes('gpt');

  // Vision / Classical Model State
  const [epochs, setEpochs] = useState<number>(3);
  const [batchSize, setBatchSize] = useState<number>(128);
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [seed, setSeed] = useState<number>(42);
  const [isFastPilot, setIsFastPilot] = useState<boolean>(false);
  const [maxTrainSamples, setMaxTrainSamples] = useState<number>(10000);
  const [maxValSamples, setMaxValSamples] = useState<number>(2000);
  const [maxTestSamples, setMaxTestSamples] = useState<number>(2000);
  const [precision, setPrecision] = useState<'float32' | 'amp'>('float32');
  const [lrScheduler, setLrScheduler] = useState<'none' | 'cosine' | 'step'>('none');
  const [gradAccumSteps, setGradAccumSteps] = useState<number>(1);
  const [resumeCheckpoint, setResumeCheckpoint] = useState<string>('');
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // Transformer Model State (Language Modeling)
  const [maxSteps, setMaxSteps] = useState<number>(15);
  const [seqLen, setSeqLen] = useState<number>(128);
  const [microbatch, setMicrobatch] = useState<number>(2);
  const [transformerGradAccum, setTransformerGradAccum] = useState<number>(2);
  const [transformerLr, setTransformerLr] = useState<number>(0.0003);
  const [warmupSteps, setWarmupSteps] = useState<number>(5);
  const [weightDecay, setWeightDecay] = useState<number>(0.1);
  const [gradClip, setGradClip] = useState<number>(1.0);
  const [tokenizerType, setTokenizerType] = useState<'bpe' | 'byte' | 'char'>('bpe');
  const [vocabSize, setVocabSize] = useState<number>(1024);
  const [transformerDataset, setTransformerDataset] = useState<string>('Tiny Shakespeare (1.1MB Text Corpus)');
  const [datasetPath, setDatasetPath] = useState<string>('data/text/tiny_shakespeare.txt');
  const [transformerPrecision, setTransformerPrecision] = useState<'float32' | 'amp'>('amp');

  // Adaptive Engine Optimization State
  const [adaptiveMode, setAdaptiveMode] = useState<boolean>(true);
  const [autoBatchSize, setAutoBatchSize] = useState<boolean>(false);
  const [adaptiveRec, setAdaptiveRec] = useState<AdaptiveRecommendationResponse | null>(null);
  const [isLoadingRec, setIsLoadingRec] = useState<boolean>(false);

  useEffect(() => {
    if (!adaptiveMode) return;
    let isCancelled = false;
    setIsLoadingRec(true);

    const currentBatch = isTransformer ? microbatch : batchSize;
    const currentPrecision = isTransformer
      ? (backend === 'CPU_NATIVE' ? 'float32' : transformerPrecision)
      : precision;

    trainingApi.getAdaptiveRecommendation({
      model_name: modelId,
      backend,
      batch_size: autoBatchSize ? undefined : currentBatch,
      precision: currentPrecision,
      seq_len: isTransformer ? seqLen : undefined,
      gradient_accumulation_steps: isTransformer ? transformerGradAccum : gradAccumSteps,
    }).then((res) => {
      if (!isCancelled) {
        setAdaptiveRec(res);
        setIsLoadingRec(false);
      }
    }).catch(() => {
      if (!isCancelled) setIsLoadingRec(false);
    });

    return () => {
      isCancelled = true;
    };
  }, [adaptiveMode, autoBatchSize, modelId, backend, batchSize, microbatch, seqLen, precision, transformerPrecision, transformerGradAccum, gradAccumSteps]);

  const effectiveTokens = seqLen * microbatch * transformerGradAccum;
  const totalBudgetTokens = maxSteps * effectiveTokens;

  // GPU Memory and Backend Safety Checks
  const isBackendIncompatible = isTransformer && backend === 'GPU_CUSTOM';
  const isHighVramRisk =
    isTransformer &&
    backend === 'GPU_NATIVE' &&
    seqLen >= 512 &&
    microbatch >= 4 &&
    transformerPrecision === 'float32';

  const handleFastPilotToggle = (checked: boolean) => {
    setIsFastPilot(checked);
    if (checked) {
      setMaxTrainSamples(256);
      setMaxValSamples(64);
      setMaxTestSamples(64);
    } else {
      setMaxTrainSamples(10000);
      setMaxValSamples(2000);
      setMaxTestSamples(2000);
    }
  };

  const handleTransformerPreset = (type: 'pilot' | 'balanced' | 'extended') => {
    if (type === 'pilot') {
      setMaxSteps(15);
      setSeqLen(128);
      setMicrobatch(2);
      setTransformerGradAccum(2);
      setTransformerPrecision('amp');
    } else if (type === 'balanced') {
      setMaxSteps(30);
      setSeqLen(256);
      setMicrobatch(2);
      setTransformerGradAccum(4);
      setTransformerPrecision('amp');
    } else {
      setMaxSteps(50);
      setSeqLen(256);
      setMicrobatch(4);
      setTransformerGradAccum(4);
      setTransformerPrecision('amp');
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (disabled || isSubmitting || isBackendIncompatible) return;

    if (isTransformer) {
      const request: TrainingJobCreateRequest = {
        model_name: modelId,
        backend,
        dataset: transformerDataset,
        dataset_path: datasetPath,
        max_steps: maxSteps,
        epochs: 1,
        batch_size: microbatch,
        seq_len: seqLen,
        learning_rate: transformerLr,
        seed,
        precision: backend === 'CPU_NATIVE' ? 'float32' : transformerPrecision,
        lr_scheduler: 'cosine',
        warmup_steps: warmupSteps,
        weight_decay: weightDecay,
        grad_clip: gradClip,
        tokenizer_type: tokenizerType,
        vocab_size: vocabSize,
        gradient_accumulation_steps: transformerGradAccum,
        resume_checkpoint_path: resumeCheckpoint.trim() ? resumeCheckpoint.trim() : undefined,
        adaptive_mode: adaptiveMode,
        auto_batch_size: autoBatchSize,
      };
      onSubmit(request);
    } else {
      const request: TrainingJobCreateRequest = {
        model_name: modelId,
        backend,
        dataset: 'CIFAR-10',
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        seed,
        max_train_samples: maxTrainSamples,
        max_val_samples: maxValSamples,
        max_test_samples: maxTestSamples,
        precision,
        lr_scheduler: lrScheduler,
        gradient_accumulation_steps: gradAccumSteps,
        resume_checkpoint_path: resumeCheckpoint.trim() ? resumeCheckpoint.trim() : undefined,
        adaptive_mode: adaptiveMode,
        auto_batch_size: autoBatchSize,
      };
      onSubmit(request);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-[#171A1F] p-5 sm:p-6 rounded-2xl border border-[#30353D] space-y-5"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-[#30353D]">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-[#76B900]" />
          <h3 className="text-base font-bold text-[#F5F5F4] tracking-tight font-sans">
            3. Configure Training Hyperparameters
          </h3>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs font-sans text-[#A6ADB7] hover:text-[#F5F5F4] px-2.5 py-1 rounded-lg bg-[#121519] border border-[#30353D] hover:border-[#76B900]/40 transition-colors"
          >
            <Settings2 className="w-3.5 h-3.5 text-[#76B900]" />
            <span>{showAdvanced ? 'Hide Advanced Controls' : 'Advanced Controls'}</span>
          </button>
          {!isTransformer && (
            <label className="flex items-center gap-2 cursor-pointer text-xs font-sans text-[#A6ADB7] hover:text-[#F5F5F4] px-2.5 py-1 rounded-lg bg-[#121519] border border-[#30353D] hover:border-[#76B900]/40">
              <input
                type="checkbox"
                checked={isFastPilot}
                onChange={(e) => handleFastPilotToggle(e.target.checked)}
                disabled={disabled || isSubmitting}
                className="rounded bg-[#08090B] border-[#30353D] text-[#76B900] focus:ring-[#76B900]"
              />
              <Zap className="w-3.5 h-3.5 text-[#FBBF24]" />
              <span>Fast Pilot Subset (256 samples)</span>
            </label>
          )}
        </div>
      </div>

      {/* Backend Incompatibility Guard Warning */}
      {isBackendIncompatible && (
        <div className="flex items-start gap-3 p-3.5 rounded-xl bg-[#FB7185]/10 border border-[#FB7185]/30 text-xs text-[#FB7185]">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold">Backend Incompatibility Guard:</span>
            <p className="text-[11px] leading-relaxed">
              Transformer models require PyTorch native FlashAttention/SDPA execution via <strong className="font-mono text-[#F5F5F4]">GPU_NATIVE</strong> or <strong className="font-mono text-[#F5F5F4]">CPU_NATIVE</strong>. Custom C++ CUDA autograd (<strong className="font-mono">GPU_CUSTOM</strong>) is currently implemented for MLP architectures only.
            </p>
          </div>
        </div>
      )}

      {/* High VRAM Safety Warning */}
      {isHighVramRisk && (
        <div className="flex items-start gap-3 p-3.5 rounded-xl bg-[#FBBF24]/10 border border-[#FBBF24]/30 text-xs text-[#FBBF24]">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold">6GB VRAM Allocation Warning:</span>
            <p className="text-[11px] leading-relaxed">
              Sequence length ≥ 512 with microbatch ≥ 4 under unscaled FP32 precision approaches the 6GB VRAM limit on the RTX 3050 Laptop GPU. We recommend enabling CUDA AMP (FP16 autocast) or setting microbatch to 2.
            </p>
          </div>
        </div>
      )}

      {/* ADAPTIVE TRAINING OPTIMIZATION PANEL */}
      <div className="p-4 rounded-xl bg-[#121519] border border-[#30353D] space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-[#70D6C2]" />
            <span className="text-xs font-bold font-sans text-[#F5F5F4]">
              Adaptive Training Engine (RTX 3050 6GB VRAM Safe Mode)
            </span>
            <span className={`px-2 py-0.5 rounded text-[10px] font-mono ${
              adaptiveMode ? 'bg-[#70D6C2]/15 text-[#70D6C2] border border-[#70D6C2]/30' : 'bg-[#30353D] text-[#A6ADB7]'
            }`}>
              {adaptiveMode ? 'ENABLED' : 'MANUAL OVERRIDE'}
            </span>
            {adaptiveMode && isLoadingRec && (
              <RefreshCw className="w-3 h-3 text-[#70D6C2] animate-spin" />
            )}
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer text-xs font-sans text-[#A6ADB7] hover:text-[#F5F5F4]">
              <input
                type="checkbox"
                checked={adaptiveMode}
                onChange={(e) => setAdaptiveMode(e.target.checked)}
                disabled={disabled || isSubmitting}
                className="rounded bg-[#08090B] border-[#30353D] text-[#70D6C2] focus:ring-[#70D6C2]"
              />
              <span>Enable Adaptive Optimization</span>
            </label>
            {adaptiveMode && (
              <label className="flex items-center gap-2 cursor-pointer text-xs font-sans text-[#A6ADB7] hover:text-[#F5F5F4]">
                <input
                  type="checkbox"
                  checked={autoBatchSize}
                  onChange={(e) => setAutoBatchSize(e.target.checked)}
                  disabled={disabled || isSubmitting}
                  className="rounded bg-[#08090B] border-[#30353D] text-[#70D6C2] focus:ring-[#70D6C2]"
                />
                <span>Auto Batch-Size</span>
              </label>
            )}
          </div>
        </div>

        {adaptiveMode && adaptiveRec && (
          <div className="pt-2 border-t border-[#30353D]/50 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="p-2 rounded-lg bg-[#171A1F] border border-[#30353D]/70">
              <span className="text-[10px] text-[#A6ADB7] block">Selected Batch Size</span>
              <span className="font-mono font-bold text-[#F5F5F4]">
                {adaptiveRec.applied_batch_size}
                {adaptiveRec.gradient_accumulation_steps > 1 && (
                  <span className="text-[#70D6C2] font-normal text-[10px] ml-1">
                    (x{adaptiveRec.gradient_accumulation_steps} accum = {adaptiveRec.effective_batch_size})
                  </span>
                )}
              </span>
            </div>
            <div className="p-2 rounded-lg bg-[#171A1F] border border-[#30353D]/70">
              <span className="text-[10px] text-[#A6ADB7] block">Selected Precision</span>
              <span className="font-mono font-bold text-[#70D6C2]">
                {adaptiveRec.applied_precision.toUpperCase()}
              </span>
            </div>
            <div className="p-2 rounded-lg bg-[#171A1F] border border-[#30353D]/70">
              <span className="text-[10px] text-[#A6ADB7] block">Estimated Peak VRAM</span>
              <span className="font-mono font-bold text-[#F5F5F4]">
                {adaptiveRec.estimated_peak_vram_mb.toFixed(0)} MB
                <span className="text-[10px] text-[#A6ADB7] ml-1">
                  ({(adaptiveRec.vram_safety_ratio * 100).toFixed(0)}% budget)
                </span>
              </span>
            </div>
            <div className="p-2 rounded-lg bg-[#171A1F] border border-[#30353D]/70">
              <span className="text-[10px] text-[#A6ADB7] block">Safety Headroom</span>
              <span className={`font-mono font-bold ${adaptiveRec.is_safe ? 'text-[#4ADE80]' : 'text-[#FB7185]'}`}>
                {adaptiveRec.safety_margin_mb.toFixed(0)} MB
              </span>
            </div>
            {adaptiveRec.adjustments_made.length > 0 && (
              <div className="col-span-2 sm:col-span-4 mt-1 p-2 rounded bg-[#70D6C2]/10 border border-[#70D6C2]/20 text-[11px] text-[#70D6C2] font-mono">
                {adaptiveRec.adjustments_made.join(' ')}
              </div>
            )}
          </div>
        )}
      </div>

      {/* TRANSFORMER HYPERPARAMETERS CONTROLS */}
      {isTransformer ? (
        <div className="space-y-4">
          {/* Quick Presets Bar */}
          <div className="flex flex-wrap items-center justify-between gap-2 p-3 rounded-xl bg-[#121519] border border-[#30353D]">
            <div className="flex items-center gap-2 text-xs font-sans text-[#A6ADB7]">
              <Layers className="w-4 h-4 text-[#76B900]" />
              <span>Quick Presets:</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => handleTransformerPreset('pilot')}
                className={`px-3 py-1 rounded-lg text-xs font-mono transition-colors ${
                  maxSteps === 15 && seqLen === 128
                    ? 'bg-[#76B900] text-[#08090B] font-bold'
                    : 'bg-[#171A1F] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D]'
                }`}
              >
                Fast Pilot (15 steps, 128 ctx)
              </button>
              <button
                type="button"
                onClick={() => handleTransformerPreset('balanced')}
                className={`px-3 py-1 rounded-lg text-xs font-mono transition-colors ${
                  maxSteps === 30 && seqLen === 256
                    ? 'bg-[#76B900] text-[#08090B] font-bold'
                    : 'bg-[#171A1F] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D]'
                }`}
              >
                Balanced (30 steps, 256 ctx)
              </button>
              <button
                type="button"
                onClick={() => handleTransformerPreset('extended')}
                className={`px-3 py-1 rounded-lg text-xs font-mono transition-colors ${
                  maxSteps === 50 && seqLen === 256
                    ? 'bg-[#76B900] text-[#08090B] font-bold'
                    : 'bg-[#171A1F] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D]'
                }`}
              >
                Extended (50 steps, 256 ctx)
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Optimizer Steps */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                  Optimizer Steps
                </label>
                <span className="text-xs font-mono text-[#76B900] font-semibold">
                  {maxSteps} steps
                </span>
              </div>
              <input
                type="number"
                min={1}
                max={500}
                value={maxSteps}
                onChange={(e) => setMaxSteps(Math.max(1, Math.min(500, parseInt(e.target.value) || 1)))}
                disabled={disabled || isSubmitting}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none"
              />
            </div>

            {/* Sequence Length */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                  Sequence Length (Tokens)
                </label>
                <span className="text-xs font-mono text-[#76B900] font-semibold">
                  {seqLen}
                </span>
              </div>
              <select
                value={seqLen}
                onChange={(e) => setSeqLen(parseInt(e.target.value))}
                disabled={disabled || isSubmitting}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none cursor-pointer"
              >
                <option value={64}>64 (Ultra-Fast)</option>
                <option value={128}>128 (Standard Pilot)</option>
                <option value={256}>256 (Balanced)</option>
                <option value={512}>512 (Extended Context)</option>
                <option value={1024}>1024 (Full Architectural Max)</option>
              </select>
            </div>

            {/* Microbatch Size */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                  Microbatch Size
                </label>
                <span className="text-xs font-mono text-[#76B900] font-semibold">
                  {microbatch}
                </span>
              </div>
              <select
                value={microbatch}
                onChange={(e) => setMicrobatch(parseInt(e.target.value))}
                disabled={disabled || isSubmitting}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none cursor-pointer"
              >
                <option value={1}>1 (Safe Low VRAM)</option>
                <option value={2}>2 (Recommended RTX 3050)</option>
                <option value={4}>4 (Moderate VRAM)</option>
                <option value={8}>8 (High Throughput)</option>
              </select>
            </div>

            {/* Gradient Accumulation */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                  Grad Accumulation
                </label>
                <span className="text-xs font-mono text-[#76B900] font-semibold">
                  {transformerGradAccum}x
                </span>
              </div>
              <select
                value={transformerGradAccum}
                onChange={(e) => setTransformerGradAccum(parseInt(e.target.value))}
                disabled={disabled || isSubmitting}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none cursor-pointer"
              >
                <option value={1}>1x (Microbatch = Step)</option>
                <option value={2}>2x (Balanced Accumulation)</option>
                <option value={4}>4x (Higher Stability)</option>
                <option value={8}>8x (Large Effective Batch)</option>
                <option value={16}>16x (Max Accumulation)</option>
              </select>
            </div>
          </div>

          {/* Effective Batch Accounting Display */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-[#08090B] border border-[#30353D] text-xs font-mono">
            <div className="flex items-center gap-2 text-[#A6ADB7]">
              <Cpu className="w-4 h-4 text-[#70D6C2]" />
              <span>Effective Batch Tokens:</span>
              <strong className="text-[#70D6C2]">{effectiveTokens.toLocaleString()} tokens/step</strong>
            </div>
            <div className="flex items-center gap-2 text-[#A6ADB7]">
              <span>Total Token Budget:</span>
              <strong className="text-[#76B900]">{totalBudgetTokens.toLocaleString()} tokens</strong>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Learning Rate */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                  Learning Rate (AdamW)
                </label>
                <span className="text-xs font-mono text-[#76B900] font-semibold">
                  {transformerLr}
                </span>
              </div>
              <input
                type="number"
                step="any"
                min="0.000001"
                max="0.01"
                value={transformerLr}
                onChange={(e) => setTransformerLr(parseFloat(e.target.value) || 0.0003)}
                disabled={disabled || isSubmitting}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none"
              />
            </div>

            {/* Tokenizer */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                Tokenizer Type
              </label>
              <select
                value={tokenizerType}
                onChange={(e) => setTokenizerType(e.target.value as any)}
                disabled={disabled || isSubmitting}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none cursor-pointer"
              >
                <option value="bpe">BPE Tokenizer (1,024 Vocab — Recommended)</option>
                <option value="byte">Byte Tokenizer (256 ASCII Bytes)</option>
                <option value="char">Character Tokenizer (Single UTF-8)</option>
              </select>
            </div>

            {/* Dataset Corpus */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                Dataset Corpus
              </label>
              <select
                value={datasetPath}
                onChange={(e) => {
                  setDatasetPath(e.target.value);
                  setTransformerDataset(
                    e.target.value.includes('shakespeare')
                      ? 'Tiny Shakespeare (1.1MB Text Corpus)'
                      : 'Technical Corpus (GPU & AI Architecture)'
                  );
                }}
                disabled={disabled || isSubmitting}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none cursor-pointer"
              >
                <option value="data/text/tiny_shakespeare.txt">Tiny Shakespeare (1.1MB)</option>
                <option value="data/text/corpus">Technical Corpus</option>
              </select>
            </div>

            {/* Precision Policy */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                Precision Policy
              </label>
              <select
                value={transformerPrecision}
                onChange={(e) => setTransformerPrecision(e.target.value as any)}
                disabled={disabled || isSubmitting || backend === 'CPU_NATIVE'}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
              >
                <option value="amp">CUDA AMP (Mixed Precision FP16)</option>
                <option value="float32">FP32 (Single Precision)</option>
              </select>
            </div>
          </div>
        </div>
      ) : (
        /* VISION HYPERPARAMETERS CONTROLS (ResNet-18 & CIFAR10MLP) */
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Epochs */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                Epochs (1 - 50)
              </label>
              <span className="text-xs font-mono text-[#76B900] font-semibold">
                {epochs}
              </span>
            </div>
            <input
              type="number"
              min={1}
              max={50}
              value={epochs}
              onChange={(e) => setEpochs(Math.max(1, Math.min(50, parseInt(e.target.value) || 1)))}
              disabled={disabled || isSubmitting}
              className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none"
            />
          </div>

          {/* Batch Size */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                Batch Size (16 - 512)
              </label>
              <span className="text-xs font-mono text-[#76B900] font-semibold">
                {batchSize}
              </span>
            </div>
            <select
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              disabled={disabled || isSubmitting}
              className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none cursor-pointer"
            >
              <option value={16}>16 (Low Memory)</option>
              <option value={32}>32</option>
              <option value={64}>64 (Balanced)</option>
              <option value={128}>128 (Standard)</option>
              <option value={256}>256 (High Throughput)</option>
              <option value={512}>512 (Max GPU)</option>
            </select>
          </div>

          {/* Learning Rate */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                Learning Rate (SGD)
              </label>
              <span className="text-xs font-mono text-[#76B900] font-semibold">
                {learningRate}
              </span>
            </div>
            <input
              type="number"
              step="any"
              min="0.000001"
              max="1.0"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
              disabled={disabled || isSubmitting}
              className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none"
            />
          </div>

          {/* Seed */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                Random Seed (Fairness)
              </label>
              <span className="text-xs font-mono text-[#76B900] font-semibold">
                {seed}
              </span>
            </div>
            <input
              type="number"
              min={0}
              max={2147483647}
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
              disabled={disabled || isSubmitting}
              className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-sm font-mono focus:border-[#76B900] focus:outline-none"
            />
          </div>
        </div>
      )}

      {/* Advanced Hyperparameter Section */}
      {showAdvanced && (
        <div className="pt-4 border-t border-[#30353D] space-y-4 animate-fadeIn">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-[#76B900]" />
            <h4 className="text-xs font-bold text-[#A6ADB7] uppercase tracking-wider font-sans">
              Advanced Training &amp; Checkpoint Options
            </h4>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {isTransformer ? (
              <>
                {/* Warmup Steps */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                    Warmup Steps
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={warmupSteps}
                    onChange={(e) => setWarmupSteps(parseInt(e.target.value) || 0)}
                    disabled={disabled || isSubmitting}
                    className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  />
                </div>

                {/* Weight Decay */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                    Weight Decay
                  </label>
                  <input
                    type="number"
                    step="any"
                    min={0}
                    max={1.0}
                    value={weightDecay}
                    onChange={(e) => setWeightDecay(parseFloat(e.target.value) || 0.1)}
                    disabled={disabled || isSubmitting}
                    className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  />
                </div>

                {/* Gradient Clipping */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                    Gradient Clip Norm
                  </label>
                  <input
                    type="number"
                    step="any"
                    min={0.1}
                    max={10.0}
                    value={gradClip}
                    onChange={(e) => setGradClip(parseFloat(e.target.value) || 1.0)}
                    disabled={disabled || isSubmitting}
                    className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  />
                </div>

                {/* Random Seed for Transformer */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                    Random Seed
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={2147483647}
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                    disabled={disabled || isSubmitting}
                    className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  />
                </div>

                {/* BPE Vocab Size */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                    BPE Vocab Size
                  </label>
                  <input
                    type="number"
                    min={256}
                    max={8192}
                    value={vocabSize}
                    onChange={(e) => setVocabSize(parseInt(e.target.value) || 1024)}
                    disabled={disabled || isSubmitting || tokenizerType !== 'bpe'}
                    className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  />
                </div>
              </>
            ) : (
              <>
                {/* Precision (FP32 vs AMP) */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                    Precision Policy
                  </label>
                  <select
                    value={precision}
                    onChange={(e) => setPrecision(e.target.value as any)}
                    disabled={disabled || isSubmitting || backend === 'CPU_NATIVE'}
                    className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  >
                    <option value="float32">FP32 (Single Precision)</option>
                    <option value="amp">CUDA AMP (Mixed Precision FP16)</option>
                  </select>
                </div>

                {/* LR Scheduler */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                    LR Decay Schedule
                  </label>
                  <select
                    value={lrScheduler}
                    onChange={(e) => setLrScheduler(e.target.value as any)}
                    disabled={disabled || isSubmitting}
                    className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  >
                    <option value="none">Constant (No Schedule)</option>
                    <option value="cosine">Cosine Annealing</option>
                    <option value="step">Step LR (0.1x at 50% epochs)</option>
                  </select>
                </div>

                {/* Gradient Accumulation Steps */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                      Gradient Accumulation
                    </label>
                    <span className="text-xs font-mono text-[#76B900] font-semibold">
                      {gradAccumSteps}x
                    </span>
                  </div>
                  <input
                    type="number"
                    min={1}
                    max={16}
                    value={gradAccumSteps}
                    onChange={(e) => setGradAccumSteps(Math.max(1, Math.min(16, parseInt(e.target.value) || 1)))}
                    disabled={disabled || isSubmitting}
                    className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  />
                </div>
              </>
            )}

            {/* Resume Checkpoint */}
            <div className="space-y-1.5 sm:col-span-2">
              <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                Resume from Checkpoint Path
              </label>
              <input
                type="text"
                placeholder="checkpoints/transformer/flagship_100m_pilot/best_model.pt"
                value={resumeCheckpoint}
                onChange={(e) => setResumeCheckpoint(e.target.value)}
                disabled={disabled || isSubmitting}
                className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none placeholder:text-[#6B7280]"
              />
            </div>
          </div>
        </div>
      )}

      {/* Dataset Subsample Bounds (Vision Only) */}
      {!isTransformer && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3 border-t border-[#30353D]">
          <div className="space-y-1">
            <label className="text-xs text-[#A6ADB7] font-sans">
              Train Samples ({maxTrainSamples.toLocaleString()} max)
            </label>
            <input
              type="number"
              min={10}
              max={50000}
              value={maxTrainSamples}
              onChange={(e) => setMaxTrainSamples(parseInt(e.target.value) || 256)}
              disabled={disabled || isSubmitting || isFastPilot}
              className="w-full px-3 py-1.5 rounded-lg bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-[#A6ADB7] font-sans">
              Validation Samples ({maxValSamples.toLocaleString()} max)
            </label>
            <input
              type="number"
              min={10}
              max={10000}
              value={maxValSamples}
              onChange={(e) => setMaxValSamples(parseInt(e.target.value) || 64)}
              disabled={disabled || isSubmitting || isFastPilot}
              className="w-full px-3 py-1.5 rounded-lg bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-[#A6ADB7] font-sans">
              Test Samples ({maxTestSamples.toLocaleString()} max)
            </label>
            <input
              type="number"
              min={10}
              max={10000}
              value={maxTestSamples}
              onChange={(e) => setMaxTestSamples(parseInt(e.target.value) || 64)}
              disabled={disabled || isSubmitting || isFastPilot}
              className="w-full px-3 py-1.5 rounded-lg bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono"
            />
          </div>
        </div>
      )}

      {/* Action Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-[#30353D]">
        <div className="flex items-center gap-2 text-xs font-sans text-[#A6ADB7]">
          <Shield className="w-4 h-4 text-[#76B900]" />
          <span>
            {backend === 'CPU_NATIVE'
              ? 'CPU Native execution • Zero GPU VRAM allocated'
              : 'GPU Mutual Exclusion Lock active • 1 concurrent job on RTX 3050'}
          </span>
        </div>

        <button
          type="submit"
          id="btn-launch-training-job"
          disabled={disabled || isSubmitting || isBackendIncompatible}
          className={`flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl font-bold font-sans text-sm transition-all cursor-pointer ${
            disabled || isSubmitting || isBackendIncompatible
              ? 'bg-[#121519] text-[#A6ADB7]/50 border border-[#30353D] cursor-not-allowed'
              : 'bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] shadow-sm btn-primary-lift'
          }`}
        >
          {isSubmitting ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-[#08090B]" />
              <span>Queueing Job...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current text-[#08090B]" />
              <span>{isTransformer ? 'Launch Transformer Job' : 'Launch Training Job'}</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
};
