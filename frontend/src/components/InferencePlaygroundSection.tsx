import React, { useState, useEffect } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { 
  Sparkles, 
  Download, 
  Activity, 
  CheckCircle2, 
  Clock, 
  Terminal, 
  RotateCw, 
  Image as ImageIcon,
  Play,
  Zap,
  Cpu,
  AlertTriangle,
  HardDrive
} from 'lucide-react';
import { InferenceResponse, GenerateTextResponse } from '../types/api';
import { WarningBanner } from './WarningBanner';
import { api } from '../services/api';

interface InferencePlaygroundSectionProps {
  data: InferenceResponse | null;
  loading: boolean;
}

interface PredictionResult {
  status: string;
  model: string;
  device: string;
  latency_ms: number;
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  correct: boolean;
  top_predictions: {
    class_name: string;
    class_index: number;
    confidence: number;
    confidence_percent: number;
  }[];
  metadata: {
    source: string;
    sample_index: number;
    ground_truth_class: string;
    ground_truth_index: number;
  };
}

const CIFAR_SAMPLES = [
  { index: 0, label: 'Sample #0', groundTruth: 'cat', task: 'CIFAR-10 Test Sample 0' },
  { index: 1, label: 'Sample #1', groundTruth: 'ship', task: 'CIFAR-10 Test Sample 1' },
  { index: 2, label: 'Sample #2', groundTruth: 'ship', task: 'CIFAR-10 Test Sample 2' },
  { index: 3, label: 'Sample #3', groundTruth: 'airplane', task: 'CIFAR-10 Test Sample 3' },
  { index: 4, label: 'Sample #4', groundTruth: 'frog', task: 'CIFAR-10 Test Sample 4' },
  { index: 5, label: 'Sample #5', groundTruth: 'frog', task: 'CIFAR-10 Test Sample 5' },
  { index: 6, label: 'Sample #6', groundTruth: 'automobile', task: 'CIFAR-10 Test Sample 6' },
  { index: 7, label: 'Sample #7', groundTruth: 'frog', task: 'CIFAR-10 Test Sample 7' },
  { index: 8, label: 'Sample #8', groundTruth: 'cat', task: 'CIFAR-10 Test Sample 8' },
  { index: 9, label: 'Sample #9', groundTruth: 'automobile', task: 'CIFAR-10 Test Sample 9' },
];

const PROMPT_PRESETS = [
  'Technical Foundations of GPU Computing:',
  'Deep neural networks accelerate',
  'Once upon a time in a digital world,',
  'In the beginning of artificial intelligence,',
];

export const InferencePlaygroundSection: React.FC<InferencePlaygroundSectionProps> = ({
  data,
  loading,
}) => {
  const [activeTab, setActiveTab] = useState<'vision' | 'history' | 'transformer'>('vision');
  const [selectedSample, setSelectedSample] = useState<number>(0);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Transformer Text Generation State
  const [prompt, setPrompt] = useState<string>('Technical Foundations of GPU Computing:');
  const [maxNewTokens, setMaxNewTokens] = useState<number>(50);
  const [temperature, setTemperature] = useState<number>(0.8);
  const [topK, setTopK] = useState<number>(40);
  const [topP, setTopP] = useState<number>(0.95);
  const [seed, setSeed] = useState<number>(42);
  const [device, setDevice] = useState<string>('cuda:0');
  const [checkpoint, setCheckpoint] = useState<string>('checkpoints/transformer/pretraining_50m_campaign/best_model.pt');
  const [generationResult, setGenerationResult] = useState<GenerateTextResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  // Auto-run first vision prediction on mount
  useEffect(() => {
    handleRunInference(0);
  }, []);

  const handleRunInference = async (sampleIdx: number) => {
    setSelectedSample(sampleIdx);
    setIsPredicting(true);
    setPredictError(null);

    try {
      await api.loadModel({ model_name: 'ResNet18' });
      const result = await api.predictVision({ sample_index: sampleIdx });
      setPrediction(result);
    } catch (err: any) {
      console.error('Inference error:', err);
      setPredictError(err.message || 'Failed to execute inference');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleGenerateText = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setGenerateError(null);

    try {
      const res = await api.generateText({
        prompt,
        max_new_tokens: maxNewTokens,
        temperature,
        top_k: topK,
        top_p: topP,
        seed,
        device,
        checkpoint: checkpoint.trim() || undefined,
      });
      setGenerationResult(res);
    } catch (err: any) {
      console.error('Text generation error:', err);
      const msg = err.message || '';
      if (msg.includes('409') || msg.toLowerCase().includes('training') || msg.toLowerCase().includes('contention')) {
        setGenerateError(
          'GPU Mutual Exclusion Active: A training job is currently running on the RTX 3050. Inference is paused to prevent VRAM collision.'
        );
      } else {
        setGenerateError(msg || 'Failed to execute autoregressive generation.');
      }
    } finally {
      setIsGenerating(false);
    }
  };

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-[#30353D] border-t-[#76B900] rounded-full animate-spin" />
          <span className="text-sm font-sans text-[#A6ADB7]">Loading inference playground...</span>
        </div>
      </div>
    );
  }

  const rows = data.rows || [];

  return (
    <div id="section-inference" className="space-y-6 animate-fadeIn font-sans">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#171A1F] rounded-xl p-5 border border-[#30353D] shadow-sm">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[#121519] border border-[#30353D]">
              <Sparkles className="w-5 h-5 text-[#76B900]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-[#F5F5F4] tracking-tight">
                Inference Playground &amp; Model Serving
              </h1>
              <p className="text-xs sm:text-sm text-[#A6ADB7] mt-0.5">
                Real-time GPU inference, confidence scoring &amp; live autoregressive Transformer generation
              </p>
            </div>
          </div>
        </div>

        {/* Tab Switchers */}
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-[#121519] border border-[#30353D] font-sans text-xs self-start sm:self-auto">
          <button
            onClick={() => setActiveTab('vision')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
              activeTab === 'vision'
                ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
            }`}
          >
            Vision Classifier (Live)
          </button>
          <button
            onClick={() => setActiveTab('transformer')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
              activeTab === 'transformer'
                ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
            }`}
          >
            Generative Transformer (Live)
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
              activeTab === 'history'
                ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                : 'text-[#A6ADB7] hover:text-[#F5F5F4]'
            }`}
          >
            Historical Benchmarks
          </button>
        </div>
      </div>

      <WarningBanner warnings={data.warnings} />

      {/* TAB 1: LIVE VISION INFERENCE PLAYGROUND */}
      {activeTab === 'vision' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {/* Left: Sample Selector */}
            <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-3 interactive-card">
              <div className="flex items-center justify-between border-b border-[#30353D] pb-2.5">
                <div className="flex items-center gap-2">
                  <ImageIcon className="w-4 h-4 text-[#A6ADB7]" />
                  <h2 className="text-xs font-semibold text-[#F5F5F4] uppercase tracking-wider font-sans">
                    CIFAR-10 Test Samples
                  </h2>
                </div>
                <span className="text-[10px] font-mono text-[#A6ADB7]">32x32 RGB</span>
              </div>

              <div className="space-y-1.5">
                {CIFAR_SAMPLES.map((sample) => (
                  <button
                    key={sample.index}
                    onClick={() => handleRunInference(sample.index)}
                    disabled={isPredicting}
                    className={`w-full flex items-center justify-between p-2 rounded-lg text-xs font-mono transition-colors cursor-pointer text-left ${
                      selectedSample === sample.index
                        ? 'bg-[#76B900]/15 text-[#F5F5F4] font-semibold border border-[#76B900]/40'
                        : 'bg-[#121519] text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A] border border-[#30353D]'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full ${selectedSample === sample.index ? 'bg-[#76B900]' : 'bg-[#6B7280]'}`} />
                      <span>{sample.label}</span>
                    </div>
                    <span className="text-[10px] uppercase font-mono text-[#A6ADB7] px-1.5 py-0.5 rounded bg-[#171A1F]">
                      Truth: {sample.groundTruth}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Center/Right: Live Prediction Instrument */}
            <div className="lg:col-span-2 bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-4 flex flex-col justify-between interactive-card">
              <div>
                <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-[#70D6C2]" />
                    <h2 className="text-xs font-semibold text-[#F5F5F4] uppercase tracking-wider font-sans">
                      Inference Engine Response
                    </h2>
                  </div>
                  <button
                    onClick={() => handleRunInference(selectedSample)}
                    disabled={isPredicting}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] font-semibold font-sans text-xs transition-colors cursor-pointer btn-primary-lift"
                  >
                    <RotateCw className={`w-3.5 h-3.5 ${isPredicting ? 'animate-spin' : ''}`} />
                    <span>Run Inference</span>
                  </button>
                </div>

                {predictError && (
                  <div className="p-3 rounded-lg bg-[#FB7185]/10 border border-[#FB7185]/25 text-[#FB7185] text-xs font-mono my-3">
                    {predictError}
                  </div>
                )}

                {prediction ? (
                  <div className="space-y-4 pt-3">
                    {/* Top Result Card */}
                    <div className="p-4 rounded-lg bg-[#121519] border border-[#30353D] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      <div>
                        <span className="text-[10px] font-sans text-[#A6ADB7] uppercase tracking-wider block font-medium">
                          Predicted Class
                        </span>
                        <div className="flex items-baseline gap-3 mt-1">
                          <span className="text-2xl sm:text-3xl font-bold text-[#F5F5F4] uppercase font-mono tracking-tight">
                            {prediction.predicted_class}
                          </span>
                          <span className="text-base font-semibold text-[#76B900] font-mono">
                            {(prediction.confidence * 100).toFixed(1)}% Confidence
                          </span>
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        {prediction.correct ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#4ADE80]/10 text-[#4ADE80] border border-[#4ADE80]/25 font-sans text-xs font-medium">
                            <CheckCircle2 className="w-3.5 h-3.5 text-[#4ADE80]" />
                            Correct Match
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/25 font-sans text-xs font-medium">
                            Incorrect Match
                          </span>
                        )}
                        <span className="px-2.5 py-1 rounded-md bg-[#171A1F] text-[#A6ADB7] border border-[#30353D] font-mono text-xs">
                          {prediction.latency_ms.toFixed(1)} ms
                        </span>
                      </div>
                    </div>

                    {/* Probability Distribution Chart */}
                    <div className="space-y-2">
                      <span className="text-xs font-sans uppercase text-[#A6ADB7] font-medium tracking-wider block">
                        Softmax Class Probabilities
                      </span>
                      <div className="h-44 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart 
                            data={prediction.top_predictions.slice(0, 5)} 
                            layout="vertical"
                            margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                            <XAxis type="number" domain={[0, 100]} stroke="#5A6270" tick={{ fontSize: 10, fill: '#A6ADB7' }} />
                            <YAxis type="category" dataKey="class_name" stroke="#5A6270" tick={{ fontSize: 11, fill: '#F5F5F4' }} />
                            <Tooltip
                              formatter={(val: any) => [`${Number(val).toFixed(2)}%`, 'Confidence']}
                              contentStyle={{
                                backgroundColor: '#0D0F12',
                                borderColor: '#30353D',
                                borderRadius: '0.5rem',
                                color: '#F5F5F4',
                                fontSize: '11px',
                                fontFamily: 'monospace',
                              }}
                            />
                            <Bar dataKey="confidence_percent" fill="#76B900" radius={[0, 2, 2, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-10 text-center text-xs font-mono text-[#6B7280]">
                    Executing live inference on active GPU model...
                  </div>
                )}
              </div>

              <div className="pt-2.5 border-t border-[#30353D] flex items-center justify-between text-xs font-mono text-[#A6ADB7]">
                <span>Model: ResNet-18 (11.17M Parameters)</span>
                <span className="text-[#70D6C2]">Device: {prediction?.device || 'cuda:0'}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: LIVE GENERATIVE TRANSFORMER TEXT GENERATION */}
      {activeTab === 'transformer' && (
        <div className="space-y-6 animate-fadeIn">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            {/* Left Controls Column (5 cols) */}
            <div className="lg:col-span-5 bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-4 interactive-card">
              <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-[#76B900]" />
                  <h2 className="text-sm font-semibold text-[#F5F5F4]">
                    Generation Controls
                  </h2>
                </div>
                <span className="text-[10px] font-mono text-[#70D6C2] px-2 py-0.5 rounded bg-[#70D6C2]/10 border border-[#70D6C2]/30">
                  {checkpoint.includes('50m') ? 'NEXORA GPT-86.8M' : 'NEXORA GPT-100M'}
                </span>
              </div>

              {/* Checkpoint Selection */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-[#A6ADB7] font-sans flex items-center gap-1.5">
                    <HardDrive className="w-3.5 h-3.5 text-[#76B900]" />
                    <span>Model Checkpoint</span>
                  </label>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => setCheckpoint('checkpoints/transformer/pretraining_50m_campaign/best_model.pt')}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                        checkpoint.includes('pretraining_50m_campaign')
                          ? 'bg-[#76B900]/20 text-[#76B900] border border-[#76B900]/50 font-bold'
                          : 'bg-[#121519] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D]'
                      }`}
                    >
                      50M Campaign (Certified)
                    </button>
                    <button
                      type="button"
                      onClick={() => setCheckpoint('checkpoints/transformer/flagship_100m_pilot/best_model.pt')}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                        checkpoint.includes('flagship_100m_pilot')
                          ? 'bg-[#70D6C2]/20 text-[#70D6C2] border border-[#70D6C2]/50 font-bold'
                          : 'bg-[#121519] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D]'
                      }`}
                    >
                      100M Pilot
                    </button>
                  </div>
                </div>
                <input
                  type="text"
                  value={checkpoint}
                  onChange={(e) => setCheckpoint(e.target.value)}
                  placeholder="checkpoints/transformer/pretraining_50m_campaign/best_model.pt"
                  className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                />
                <span className="text-[10px] text-[#6B7280] font-mono">
                  {checkpoint.includes('50m')
                    ? 'Certified 50M Campaign: 86.83M params, 12,208 steps, val loss 1.9383'
                    : 'Pilot Checkpoint: 15 optimizer steps, baseline exploration'}
                </span>
              </div>

              {/* Execution Device */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[#A6ADB7] font-sans flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-[#70D6C2]" />
                  <span>Execution Backend</span>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setDevice('cuda:0')}
                    className={`py-2 px-3 rounded-lg text-xs font-mono transition-colors text-center ${
                      device.startsWith('cuda')
                        ? 'bg-[#76B900]/15 text-[#76B900] border border-[#76B900]/40 font-semibold'
                        : 'bg-[#121519] text-[#A6ADB7] border border-[#30353D]'
                    }`}
                  >
                    CUDA (RTX 3050)
                  </button>
                  <button
                    type="button"
                    onClick={() => setDevice('cpu')}
                    className={`py-2 px-3 rounded-lg text-xs font-mono transition-colors text-center ${
                      device === 'cpu'
                        ? 'bg-[#70D6C2]/15 text-[#70D6C2] border border-[#70D6C2]/40 font-semibold'
                        : 'bg-[#121519] text-[#A6ADB7] border border-[#30353D]'
                    }`}
                  >
                    CPU Fallback
                  </button>
                </div>
              </div>

              {/* Prompt Input */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[#A6ADB7] font-sans">
                  Input Prompt
                </label>
                <textarea
                  rows={3}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter initial prompt tokens..."
                  className="w-full px-3 py-2 rounded-xl bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none resize-none"
                />
              </div>

              {/* Quick Prompt Presets */}
              <div className="space-y-1">
                <span className="text-[10px] text-[#A6ADB7] uppercase font-semibold">Suggested Prompts:</span>
                <div className="flex flex-wrap gap-1.5">
                  {PROMPT_PRESETS.map((p, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setPrompt(p)}
                      className="px-2 py-1 rounded-md text-[11px] font-sans bg-[#121519] text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A] border border-[#30353D] transition-colors"
                    >
                      {p.slice(0, 24)}...
                    </button>
                  ))}
                </div>
              </div>

              {/* Generation Hyperparameters Grid */}
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-[#30353D]">
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#A6ADB7]">Max New Tokens</span>
                    <span className="font-mono text-[#76B900] font-bold">{maxNewTokens}</span>
                  </div>
                  <input
                    type="range"
                    min={8}
                    max={128}
                    step={4}
                    value={maxNewTokens}
                    onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
                    className="w-full accent-[#76B900]"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#A6ADB7]">Temperature</span>
                    <span className="font-mono text-[#76B900] font-bold">{temperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={1.5}
                    step={0.05}
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-[#76B900]"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#A6ADB7]">Top-K Sampling</span>
                    <span className="font-mono text-[#70D6C2] font-bold">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={100}
                    step={1}
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="w-full accent-[#70D6C2]"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#A6ADB7]">Top-P Nucleus</span>
                    <span className="font-mono text-[#70D6C2] font-bold">{topP.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={1.0}
                    step={0.05}
                    value={topP}
                    onChange={(e) => setTopP(parseFloat(e.target.value))}
                    className="w-full accent-[#70D6C2]"
                  />
                </div>

                <div className="space-y-1 col-span-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#A6ADB7]">RNG Seed</span>
                    <span className="font-mono text-[#76B900] font-bold">{seed}</span>
                  </div>
                  <input
                    type="number"
                    min={0}
                    max={2147483647}
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                    className="w-full px-2.5 py-1 rounded-lg bg-[#121519] border border-[#30353D] text-[#F5F5F4] text-xs font-mono focus:border-[#76B900] focus:outline-none"
                  />
                </div>
              </div>

              {/* Generate Button */}
              <button
                type="button"
                onClick={handleGenerateText}
                disabled={isGenerating || !prompt.trim()}
                id="btn-generate-transformer"
                className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl font-bold font-sans text-xs transition-all cursor-pointer ${
                  isGenerating || !prompt.trim()
                    ? 'bg-[#121519] text-[#A6ADB7]/50 border border-[#30353D] cursor-not-allowed'
                    : 'bg-[#76B900] hover:bg-[#8ED11A] text-[#08090B] shadow-sm btn-primary-lift'
                }`}
              >
                {isGenerating ? (
                  <>
                    <RotateCw className="w-4 h-4 animate-spin text-[#08090B]" />
                    <span>Executing Autoregressive Decoding...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current text-[#08090B]" />
                    <span>Generate Text ({maxNewTokens} Tokens)</span>
                  </>
                )}
              </button>
            </div>

            {/* Right Output & Telemetry Column (7 cols) */}
            <div className="lg:col-span-7 bg-[#171A1F] rounded-xl p-5 border border-[#30353D] flex flex-col justify-between space-y-4 interactive-card">
              <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-[#70D6C2]" />
                    <h2 className="text-xs font-semibold text-[#F5F5F4] uppercase tracking-wider font-sans">
                      Autoregressive Generation Stream
                    </h2>
                  </div>
                  {generationResult && (
                    <span className="text-[10px] font-mono text-[#76B900]">
                      {generationResult.tokens_per_second.toFixed(1)} tok/s
                    </span>
                  )}
                </div>

                {/* Error Banner */}
                {generateError && (
                  <div className="p-3.5 rounded-xl bg-[#FB7185]/10 border border-[#FB7185]/30 text-xs text-[#FB7185] flex items-start gap-2.5">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="block font-bold">Inference Dispatch Error:</strong>
                      <span className="font-mono text-[11px]">{generateError}</span>
                    </div>
                  </div>
                )}

                {/* Live Output Box */}
                <div className="p-4 rounded-xl bg-[#08090B] border border-[#30353D] min-h-[220px] font-mono text-xs leading-relaxed space-y-3">
                  {generationResult ? (
                    <div>
                      <span className="text-[#76B900] font-semibold">{generationResult.prompt}</span>
                      <span className="text-[#70D6C2] animate-fadeIn">{generationResult.generated_text}</span>
                    </div>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center py-16 text-[#6B7280] space-y-2 text-center">
                      <Terminal className="w-8 h-8 opacity-40 mx-auto" />
                      <p className="text-xs">
                        Configure generation parameters on the left and click <strong className="text-[#F5F5F4]">Generate Text</strong> to run real local inference on your trained model checkpoint.
                      </p>
                    </div>
                  )}
                </div>

                {/* Telemetry Metrics Bar */}
                {generationResult && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                    <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D] space-y-0.5">
                      <span className="text-[10px] text-[#A6ADB7] font-sans">Generation Latency</span>
                      <p className="text-xs font-mono font-bold text-[#F5F5F4]">
                        {generationResult.generation_duration_ms.toFixed(1)} ms
                      </p>
                    </div>

                    <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D] space-y-0.5">
                      <span className="text-[10px] text-[#A6ADB7] font-sans">Throughput</span>
                      <p className="text-xs font-mono font-bold text-[#70D6C2]">
                        {generationResult.tokens_per_second.toFixed(1)} tok/s
                      </p>
                    </div>

                    <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D] space-y-0.5">
                      <span className="text-[10px] text-[#A6ADB7] font-sans">Token Count</span>
                      <p className="text-xs font-mono font-bold text-[#76B900]">
                        {generationResult.prompt_tokens} in / {generationResult.generated_tokens} out
                      </p>
                    </div>

                    <div className="p-2.5 rounded-lg bg-[#121519] border border-[#30353D] space-y-0.5">
                      <span className="text-[10px] text-[#A6ADB7] font-sans">Peak VRAM</span>
                      <p className="text-xs font-mono font-bold text-[#F5F5F4]">
                        {generationResult.peak_vram_mb.toFixed(1)} MB
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Truthfulness & Scientific Notice Banner */}
              <div className="pt-3 border-t border-[#30353D] space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[#A6ADB7]">
                  <span className="flex items-center gap-1.5">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        generationResult
                          ? !generationResult.is_partially_trained
                            ? 'bg-[#4ADE80]'
                            : 'bg-[#FBBF24]'
                          : checkpoint.includes('50m')
                          ? 'bg-[#4ADE80]'
                          : 'bg-[#FBBF24]'
                      }`}
                    />
                    <span>
                      Status:{' '}
                      <strong
                        className={
                          generationResult
                            ? !generationResult.is_partially_trained
                              ? 'text-[#4ADE80]'
                              : 'text-[#FBBF24]'
                            : checkpoint.includes('50m')
                            ? 'text-[#4ADE80]'
                            : 'text-[#FBBF24]'
                        }
                      >
                        {generationResult?.status_label ||
                          (checkpoint.includes('50m')
                            ? 'Production Pretrained (50M Campaign Certified — 12,208 Steps)'
                            : 'Experimental / Partially Trained (Pilot Checkpoint — 15 Optimizer Steps)')}
                      </strong>
                    </span>
                  </span>
                  <span>
                    Execution: <strong className="text-[#70D6C2]">{generationResult?.device || device}</strong>
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-[#121519] border border-[#30353D] text-[11px] text-[#A6ADB7] leading-relaxed">
                  <strong className="text-[#F5F5F4]">Scientific Transparency Notice:</strong>{' '}
                  {checkpoint.includes('50m') || (generationResult && !generationResult.is_partially_trained)
                    ? 'All generated text originates from genuine PyTorch forward passes executed on local GPU memory with BPE tokenization. The 50M campaign model completed 12,208 optimizer steps (50,004,608 tokens, final val loss 1.9383, perplexity 6.95). No synthetic tokens or canned responses.'
                    : 'All generated text originates from genuine PyTorch forward passes executed on local GPU memory with BPE tokenization. The pilot checkpoint was trained for 15 optimizer steps in Milestone 10; output exhibits initial token probability convergence without advanced conversational English mastery. No synthetic tokens, canned responses, or cloud AI APIs are used.'}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: HISTORICAL LLM BENCHMARKS */}
      {activeTab === 'history' && (
        <div className="bg-[#171A1F] rounded-xl p-5 border border-[#30353D] space-y-4 interactive-card">
          <div className="flex items-center justify-between border-b border-[#30353D] pb-3">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-[#70D6C2]" />
              <h2 className="text-sm font-semibold text-[#F5F5F4]">
                Historical SLM / LLM Benchmark Records
              </h2>
            </div>
            <a
              href={api.getArtifactUrl('inference-csv')}
              className="flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-sans bg-[#121519] hover:bg-[#20242A] text-[#A6ADB7] hover:text-[#F5F5F4] border border-[#30353D] transition-colors"
            >
              <Download className="w-3.5 h-3.5 text-[#76B900]" />
              <span>CSV</span>
            </a>
          </div>

          {rows.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-[#30353D] text-[#A6ADB7] uppercase text-[10px] tracking-wider font-sans">
                    <th className="pb-2.5 font-semibold">Model Name</th>
                    <th className="pb-2.5 font-semibold">Device</th>
                    <th className="pb-2.5 font-semibold">Tokens / Sec</th>
                    <th className="pb-2.5 font-semibold">TTFT (ms)</th>
                    <th className="pb-2.5 font-semibold">VRAM Used</th>
                    <th className="pb-2.5 font-semibold">Tokens (In / Out)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#30353D]">
                  {rows.map((r, idx) => (
                    <tr key={idx} className="hover:bg-[#20242A] transition-colors">
                      <td className="py-2.5 text-[#F5F5F4] font-medium font-sans">{r.model_name}</td>
                      <td className="py-2.5 text-[#70D6C2]">{r.device_name}</td>
                      <td className="py-2.5 text-[#76B900] font-medium">{r.tokens_per_second.toFixed(1)} tok/s</td>
                      <td className="py-2.5 text-[#A6ADB7]">{r.ttft_ms.toFixed(1)} ms</td>
                      <td className="py-2.5 text-[#F5F5F4]">{r.backend_vram_mb} MB</td>
                      <td className="py-2.5 text-[#A6ADB7]">{r.prompt_tokens} / {r.output_tokens}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-6 text-center text-xs font-mono text-[#6B7280]">
              No historical inference logs recorded.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
