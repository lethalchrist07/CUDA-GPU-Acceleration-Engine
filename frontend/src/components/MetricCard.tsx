import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  glow?: 'emerald' | 'cyan' | 'amber' | 'purple' | 'none';
  badge?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  icon,
  glow = 'none',
  badge,
}) => {
  const accentStyle = {
    emerald: {
      border: 'border-white/[0.07] border-t-2 border-t-[#76b900] hover:border-white/[0.15]',
      glow: 'group-hover:shadow-[0_0_20px_rgba(118,185,0,0.12)]',
      iconHover: 'group-hover:text-[#76b900]',
      topGradient: 'bg-gradient-to-b from-[#76b900]/10 via-transparent to-transparent',
    },
    cyan: {
      border: 'border-white/[0.07] border-t-2 border-t-cyan-400 hover:border-white/[0.15]',
      glow: 'group-hover:shadow-[0_0_20px_rgba(6,182,212,0.12)]',
      iconHover: 'group-hover:text-cyan-400',
      topGradient: 'bg-gradient-to-b from-cyan-500/10 via-transparent to-transparent',
    },
    amber: {
      border: 'border-white/[0.07] border-t-2 border-t-amber-400 hover:border-white/[0.15]',
      glow: 'group-hover:shadow-[0_0_20px_rgba(245,158,11,0.12)]',
      iconHover: 'group-hover:text-amber-400',
      topGradient: 'bg-gradient-to-b from-amber-500/10 via-transparent to-transparent',
    },
    purple: {
      border: 'border-white/[0.07] border-t-2 border-t-purple-400 hover:border-white/[0.15]',
      glow: 'group-hover:shadow-[0_0_20px_rgba(168,85,247,0.12)]',
      iconHover: 'group-hover:text-purple-400',
      topGradient: 'bg-gradient-to-b from-purple-500/10 via-transparent to-transparent',
    },
    none: {
      border: 'border-white/[0.07] hover:border-white/[0.15]',
      glow: '',
      iconHover: 'group-hover:text-slate-200',
      topGradient: '',
    },
  }[glow];

  return (
    <div className={`glass-panel rounded-2xl p-5 border transition-all duration-200 relative overflow-hidden group ${accentStyle.border} ${accentStyle.glow}`}>
      {accentStyle.topGradient && (
        <div className={`absolute inset-x-0 top-0 h-10 ${accentStyle.topGradient} pointer-events-none`} />
      )}
      <div className="flex items-center justify-between mb-3 relative z-10">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">{title}</span>
        {icon && <div className={`text-slate-400 transition-colors ${accentStyle.iconHover}`}>{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2 mb-1.5 relative z-10">
        <span className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white font-mono tabular-nums">
          {value}
        </span>
        {unit && <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">{unit}</span>}
      </div>
      <div className="flex items-center justify-between mt-2 relative z-10">
        {subtitle && <p className="text-xs text-slate-400 truncate max-w-[85%]">{subtitle}</p>}
        {badge}
      </div>
    </div>
  );
};
