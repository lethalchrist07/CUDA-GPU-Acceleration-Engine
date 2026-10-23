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
  const epochList: any[] = metrics.epoch_metrics || metrics.epoch_records || [];

  // Build chart series from completed epoch records or metrics
  const chartData =
    epochList.length > 0
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
      <div className="glass-panel p-6 rounded-2xl border border-white/[0.07] text-center space-y-2">
        <Activity className="w-8 h-8 text-slate-500 mx-auto animate-pulse" />
        <h4 className="text-sm font-bold text-white">Awaiting Epoch Completion Telemetry</h4>
        <p className="text-xs text-slate-400 max-w-sm mx-auto">
          Scientific loss and accuracy curves will render automatically as training epochs complete.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 1. Loss Curves Chart */}
        <div className="glass-panel p-5 rounded-2xl border border-white/[0.07] space-y-3">
          <div className="flex items-center gap-2">
            <Flame className="w-4 h-4 text-amber-400" />
            <h4 className="text-sm font-bold text-white tracking-tight">
              Loss Progression per Epoch
            </h4>
          </div>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0a0f1d',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderRadius: '0.75rem',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  name="Train Loss"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 4, fill: '#f59e0b' }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  name="Val Loss"
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 3, fill: '#ef4444' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 2. Validation Accuracy Curve */}
        <div className="glass-panel p-5 rounded-2xl border border-white/[0.07] space-y-3">
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-emerald-400" />
            <h4 className="text-sm font-bold text-white tracking-tight">
              Validation Accuracy Progression (%)
            </h4>
          </div>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0a0f1d',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderRadius: '0.75rem',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Line
                  type="monotone"
                  dataKey="val_acc"
                  name="Validation Accuracy (%)"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  dot={{ r: 4, fill: '#10b981' }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 3. Epoch Duration Latency Bar Chart */}
        <div className="glass-panel p-5 rounded-2xl border border-white/[0.07] space-y-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <h4 className="text-sm font-bold text-white tracking-tight">
              Per-Epoch Duration (Seconds)
            </h4>
          </div>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0a0f1d',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderRadius: '0.75rem',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                />
                <Bar dataKey="duration_sec" name="Duration (s)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. Custom CUDA Kernel Dispatches Breakdown (if applicable) */}
        {hasDispatches ? (
          <div className="glass-panel p-5 rounded-2xl border border-white/[0.07] space-y-3">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-[#76b900]" />
              <h4 className="text-sm font-bold text-white tracking-tight">
                Verified Custom CUDA Kernel Invocations
              </h4>
            </div>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dispatchData} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="kernel" stroke="#64748b" tick={{ fontSize: 10 }} width={90} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a0f1d',
                      borderColor: 'rgba(255,255,255,0.1)',
                      borderRadius: '0.75rem',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                    }}
                  />
                  <Bar dataKey="dispatches" name="Dispatches" fill="#76b900" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="glass-panel p-5 rounded-2xl border border-white/[0.07] space-y-3">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-slate-400" />
              <h4 className="text-sm font-bold text-white tracking-tight">
                Training Throughput (Samples / Sec)
              </h4>
            </div>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a0f1d',
                      borderColor: 'rgba(255,255,255,0.1)',
                      borderRadius: '0.75rem',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                    }}
                  />
                  <Bar dataKey="throughput" name="Samples/s" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
