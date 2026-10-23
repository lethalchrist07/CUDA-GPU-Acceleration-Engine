import React, { useState } from 'react';
import { BackendType, TrainingJobCreateRequest } from '../../types/training';
import { Play, Sliders, Shield, RefreshCw, Zap } from 'lucide-react';

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
  const [epochs, setEpochs] = useState<number>(3);
  const [batchSize, setBatchSize] = useState<number>(128);
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [seed, setSeed] = useState<number>(42);
  const [isFastPilot, setIsFastPilot] = useState<boolean>(false);
  const [maxTrainSamples, setMaxTrainSamples] = useState<number>(10000);
  const [maxValSamples, setMaxValSamples] = useState<number>(2000);
  const [maxTestSamples, setMaxTestSamples] = useState<number>(2000);

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (disabled || isSubmitting) return;

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
      precision: 'float32',
    };

    onSubmit(request);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="glass-panel p-5 sm:p-6 rounded-2xl border border-white/[0.07] space-y-5"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-[#76b900]" />
          <h3 className="text-base font-bold text-white tracking-tight">
            3. Configure Training Hyperparameters
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 cursor-pointer text-xs font-mono text-slate-300 px-2.5 py-1 rounded-lg bg-white/[0.04] border border-white/[0.08] hover:border-white/[0.15]">
            <input
              type="checkbox"
              checked={isFastPilot}
              onChange={(e) => handleFastPilotToggle(e.target.checked)}
              disabled={disabled || isSubmitting}
              className="rounded bg-obsidian-900 border-white/[0.2] text-[#76b900] focus:ring-[#76b900]"
            />
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span>Fast Pilot Subset (256 samples)</span>
          </label>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Epochs */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-slate-300">
              Epochs (1 - 50)
            </label>
            <span className="text-xs font-mono text-[#76b900] font-semibold">
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
            className="w-full px-3 py-2 rounded-xl bg-obsidian-900 border border-white/[0.09] text-white text-sm font-mono focus:border-[#76b900] focus:outline-none"
          />
        </div>

        {/* Batch Size */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-slate-300">
              Batch Size (16 - 512)
            </label>
            <span className="text-xs font-mono text-[#76b900] font-semibold">
              {batchSize}
            </span>
          </div>
          <select
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            disabled={disabled || isSubmitting}
            className="w-full px-3 py-2 rounded-xl bg-obsidian-900 border border-white/[0.09] text-white text-sm font-mono focus:border-[#76b900] focus:outline-none cursor-pointer"
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
            <label className="text-xs font-medium text-slate-300">
              Learning Rate (SGD)
            </label>
            <span className="text-xs font-mono text-[#76b900] font-semibold">
              {learningRate}
            </span>
          </div>
          <input
            type="number"
            step="0.001"
            min="0.0001"
            max="1.0"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
            disabled={disabled || isSubmitting}
            className="w-full px-3 py-2 rounded-xl bg-obsidian-900 border border-white/[0.09] text-white text-sm font-mono focus:border-[#76b900] focus:outline-none"
          />
        </div>

        {/* Seed */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-slate-300">
              Random Seed (Fairness)
            </label>
            <span className="text-xs font-mono text-[#76b900] font-semibold">
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
            className="w-full px-3 py-2 rounded-xl bg-obsidian-900 border border-white/[0.09] text-white text-sm font-mono focus:border-[#76b900] focus:outline-none"
          />
        </div>
      </div>

      {/* Dataset Subsample Bounds */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3 border-t border-white/[0.04]">
        <div className="space-y-1">
          <label className="text-xs text-slate-400 font-mono">
            Train Samples ({maxTrainSamples.toLocaleString()} max)
          </label>
          <input
            type="number"
            min={10}
            max={50000}
            value={maxTrainSamples}
            onChange={(e) => setMaxTrainSamples(parseInt(e.target.value) || 256)}
            disabled={disabled || isSubmitting || isFastPilot}
            className="w-full px-3 py-1.5 rounded-lg bg-obsidian-900 border border-white/[0.07] text-white text-xs font-mono"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-slate-400 font-mono">
            Validation Samples ({maxValSamples.toLocaleString()} max)
          </label>
          <input
            type="number"
            min={10}
            max={10000}
            value={maxValSamples}
            onChange={(e) => setMaxValSamples(parseInt(e.target.value) || 64)}
            disabled={disabled || isSubmitting || isFastPilot}
            className="w-full px-3 py-1.5 rounded-lg bg-obsidian-900 border border-white/[0.07] text-white text-xs font-mono"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-slate-400 font-mono">
            Test Samples ({maxTestSamples.toLocaleString()} max)
          </label>
          <input
            type="number"
            min={10}
            max={10000}
            value={maxTestSamples}
            onChange={(e) => setMaxTestSamples(parseInt(e.target.value) || 64)}
            disabled={disabled || isSubmitting || isFastPilot}
            className="w-full px-3 py-1.5 rounded-lg bg-obsidian-900 border border-white/[0.07] text-white text-xs font-mono"
          />
        </div>
      </div>

      {/* Action Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-white/[0.07]">
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <Shield className="w-4 h-4 text-[#76b900]" />
          <span>
            {backend === 'CPU_NATIVE'
              ? 'CPU Native execution • Zero GPU VRAM allocated'
              : 'GPU Mutual Exclusion Lock active • 1 concurrent job on RTX 3050'}
          </span>
        </div>

        <button
          type="submit"
          id="btn-launch-training-job"
          disabled={disabled || isSubmitting}
          className={`flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl font-bold text-sm transition-all shadow-md cursor-pointer ${
            disabled || isSubmitting
              ? 'bg-obsidian-800 text-slate-500 border border-white/[0.05] cursor-not-allowed'
              : 'bg-[#76b900] hover:bg-[#86d200] text-slate-950 shadow-[#76b900]/25 hover:shadow-[#76b900]/40'
          }`}
        >
          {isSubmitting ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-slate-950" />
              <span>Queueing Job...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Launch Training Job</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
};
