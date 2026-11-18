import React from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { TrainingJobResponse } from '../../types/training';
import { Flame, Award, Clock, Activity, Zap } from 'lucide-react';

interface TrainingChartsProps {
  job: TrainingJobResponse;
}

export const TrainingCharts: React.FC<TrainingChartsProps> = ({ job }) => {
  const metrics = (job.metrics || {}) as any;
  const isTransformer =
    job.model_name?.toLowerCase().includes('transformer') ||
    job.model_name?.toLowerCase().includes('gpt') ||
    job.config?.model_id?.toLowerCase().includes('transformer');

  const stepList: any[] = metrics.step_records || metrics.step_history || [];
  const epochList: any[] = metrics.epoch_metrics || metrics.epoch_records || [];

  // Build chart series from completed step records (transformer) or epoch records (vision)
  const chartData =
    isTransformer && stepList.length > 0
      ? stepList.map((st: any, idx: number) => {
          const isLast = idx === stepList.length - 1;
          const hasStepVal = st.val_perplexity !== undefined && st.val_perplexity !== null && st.val_perplexity > 0;
          const valPpl = hasStepVal
            ? Number(Number(st.val_perplexity).toFixed(2))
            : (isLast && metrics.final_val_perplexity !== undefined && metrics.final_val_perplexity !== null && metrics.final_val_perplexity > 0
                ? Number(Number(metrics.final_val_perplexity).toFixed(2))
                : undefined);
          const hasStepValLoss = st.val_loss !== undefined && st.val_loss !== null && st.val_loss > 0;
          const valLoss = hasStepValLoss
            ? Number(Number(st.val_loss).toFixed(4))
            : (isLast && metrics.final_val_loss !== undefined && metrics.final_val_loss !== null && metrics.final_val_loss > 0
                ? Number(Number(metrics.final_val_loss).toFixed(4))
                : undefined);

          return {
            name: `Step ${st.step}`,
            step: st.step,
            train_loss: Number((st.train_loss ?? 0).toFixed(4)),
            val_loss: valLoss,
            val_perplexity: valPpl,
            is_final_eval: Boolean(st.is_final_eval || (isLast && !hasStepVal && valPpl !== undefined)),
            duration_sec: Number((st.step_duration_sec ?? 0).toFixed(3)),
            throughput: Math.round(st.tokens_per_sec ?? 0),
          };
        })
      : epochList.length > 0
      ? epochList.map((em: any) => ({
          name: `Epoch ${em.epoch}`,
          epoch: em.epoch,
          train_loss: Number((em.train_loss ?? 0).toFixed(4)),
          val_loss: Number((em.val_loss ?? 0).toFixed(4)),
          val_acc: Number((em.val_acc ?? em.val_accuracy ?? 0).toFixed(2)),
          duration_sec: Number((em.duration_sec ?? em.epoch_time_sec ?? 0).toFixed(2)),
          throughput: Math.round(em.samples_per_sec ?? 0),
        }))
      : [];

  const rawDispatches: Record<string, number> =
    metrics.custom_cuda_dispatches ||
    (metrics.dispatch_summary
      ? {
          ...(metrics.dispatch_summary.custom_forward_dispatches || {}),
          ...(metrics.dispatch_summary.custom_backward_dispatches || {}),
        }
      : {});
  const dispatchData = Object.entries(rawDispatches).map(([name, count]) => ({
    kernel: name.replace(/_/g, ' '),
    dispatches: count,
  }));
  const hasDispatches = dispatchData.length > 0;

  if (chartData.length === 0) {
    return (
      <div className="bg-[#171A1F] p-6 rounded-2xl border border-[#30353D] text-center space-y-2">
        <Activity className="w-8 h-8 text-[#6B7280] mx-auto animate-pulse" />
        <h4 className="text-sm font-bold text-[#F5F5F4] font-sans">
          {isTransformer ? 'Awaiting Optimizer Step Telemetry' : 'Awaiting Epoch Completion Telemetry'}
        </h4>
        <p className="text-xs text-[#A6ADB7] font-sans max-w-sm mx-auto">
          {isTransformer
            ? 'Scientific loss, perplexity, and token throughput curves will render automatically as optimizer steps complete.'
            : 'Scientific loss and accuracy curves will render automatically as training epochs complete.'}
        </p>
      </div>
    );
  }

  const tooltipStyle = {
    backgroundColor: '#0D0F12',
    borderColor: '#30353D',
    borderRadius: '0.5rem',
    fontSize: '11px',
    fontFamily: 'monospace',
    color: '#F5F5F4',
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 1. Loss Curves Chart */}
        <div className="bg-[#171A1F] p-5 rounded-2xl border border-[#30353D] space-y-3 interactive-card">
          <div className="flex items-center gap-2">
            <Flame className="w-4 h-4 text-[#76B900]" />
            <h4 className="text-sm font-bold text-[#F5F5F4] tracking-tight font-sans">
              {isTransformer ? 'Loss Progression per Optimizer Step' : 'Loss Progression per Epoch'}
            </h4>
          </div>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                <XAxis dataKey="name" stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} />
                <YAxis stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px', fontFamily: 'monospace' }} />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  name="Train Loss"
                  stroke="#76B900"
                  strokeWidth={2}
                  dot={{ r: 4, fill: '#76B900' }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  name="Val Loss"
                  stroke="#FB7185"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 3, fill: '#FB7185' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 2. Validation Accuracy or Perplexity Curve */}
        <div className="bg-[#171A1F] p-5 rounded-2xl border border-[#30353D] space-y-3 interactive-card">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Award className="w-4 h-4 text-[#70D6C2]" />
              <h4 className="text-sm font-bold text-[#F5F5F4] tracking-tight font-sans">
                {isTransformer ? 'Validation Perplexity Progression' : 'Validation Accuracy Progression (%)'}
              </h4>
            </div>
            {isTransformer && metrics.final_val_perplexity !== undefined && (
              <span className="text-[11px] font-mono text-[#70D6C2] bg-[#70D6C2]/10 px-2 py-0.5 rounded border border-[#70D6C2]/20">
                Final: {Number(metrics.final_val_perplexity).toFixed(2)}
              </span>
            )}
          </div>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                <XAxis dataKey="name" stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} />
                <YAxis stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} domain={isTransformer ? ['auto', 'auto'] : [0, 100]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px', fontFamily: 'monospace' }} />
                <Line
                  type="monotone"
                  dataKey={isTransformer ? "val_perplexity" : "val_acc"}
                  name={isTransformer ? "Val Perplexity" : "Validation Accuracy (%)"}
                  stroke="#70D6C2"
                  strokeWidth={2.5}
                  connectNulls={true}
                  dot={{ r: 5, fill: '#70D6C2' }}
                  activeDot={{ r: 7 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 3. Duration Latency Bar Chart */}
        <div className="bg-[#171A1F] p-5 rounded-2xl border border-[#30353D] space-y-3 interactive-card">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-[#70D6C2]" />
            <h4 className="text-sm font-bold text-[#F5F5F4] tracking-tight font-sans">
              {isTransformer ? 'Per-Step Duration (Seconds)' : 'Per-Epoch Duration (Seconds)'}
            </h4>
          </div>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                <XAxis dataKey="name" stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} />
                <YAxis stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="duration_sec" name="Duration (s)" fill="#70D6C2" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. Custom CUDA Kernel Dispatches Breakdown (if applicable) or Throughput */}
        {hasDispatches ? (
          <div className="bg-[#171A1F] p-5 rounded-2xl border border-[#30353D] space-y-3 interactive-card">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-[#76B900]" />
              <h4 className="text-sm font-bold text-[#F5F5F4] tracking-tight font-sans">
                Verified Custom CUDA Kernel Invocations
              </h4>
            </div>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dispatchData} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                  <XAxis type="number" stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} />
                  <YAxis type="category" dataKey="kernel" stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 10, fontFamily: 'monospace' }} width={90} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="dispatches" name="Dispatches" fill="#76B900" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="bg-[#171A1F] p-5 rounded-2xl border border-[#30353D] space-y-3 interactive-card">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-[#70D6C2]" />
              <h4 className="text-sm font-bold text-[#F5F5F4] tracking-tight font-sans">
                {isTransformer ? 'Training Throughput (Tokens / Sec)' : 'Training Throughput (Samples / Sec)'}
              </h4>
            </div>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30353D" />
                  <XAxis dataKey="name" stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} />
                  <YAxis stroke="#5A6270" tick={{ fill: '#A6ADB7', fontSize: 11, fontFamily: 'monospace' }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="throughput" name={isTransformer ? "Tokens/s" : "Samples/s"} fill="#70D6C2" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
