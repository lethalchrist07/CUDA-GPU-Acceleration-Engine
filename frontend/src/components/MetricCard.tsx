import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  glow?: 'nvidia' | 'copper' | 'teal' | 'amber' | 'emerald' | 'cyan' | 'purple' | 'none';
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
  const accentBorder = {
    nvidia: 'border-t-[#76B900]',
    copper: 'border-t-[#76B900]',
    teal: 'border-t-[#70D6C2]',
    amber: 'border-t-[#FBBF24]',
    emerald: 'border-t-[#76B900]',
    cyan: 'border-t-[#70D6C2]',
    purple: 'border-t-[#76B900]',
    none: 'border-t-[#30353D]',
  }[glow];

  return (
    <div className={`bg-[#171A1F] rounded-xl p-5 border border-[#30353D] border-t-2 ${accentBorder} interactive-card relative overflow-hidden flex flex-col justify-between shadow-sm`}>
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium text-[#A6ADB7] font-sans tracking-normal">{title}</span>
          {icon && <div className="text-[#A6ADB7] transition-transform duration-200 group-hover:scale-105">{icon}</div>}
        </div>
        <div className="flex items-baseline gap-2 mb-1.5">
          <span className="text-2xl sm:text-3xl font-bold tracking-tight text-[#F5F5F4] font-mono tabular-nums">
            {value}
          </span>
          {unit && <span className="text-xs font-normal text-[#A6ADB7] font-mono">{unit}</span>}
        </div>
      </div>
      <div className="flex items-center justify-between mt-2 pt-1.5 border-t border-[#30353D]/60">
        {subtitle && <p className="text-xs text-[#7A828E] font-sans truncate max-w-[85%]">{subtitle}</p>}
        {badge}
      </div>
    </div>
  );
};
