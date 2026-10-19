import React from 'react';
import { AlertTriangle, Info } from 'lucide-react';

interface WarningBannerProps {
  warnings: string[];
  type?: 'warning' | 'info';
}

export const WarningBanner: React.FC<WarningBannerProps> = ({ warnings, type = 'warning' }) => {
  if (!warnings || warnings.length === 0) return null;

  const isWarning = type === 'warning';

  return (
    <div className={`rounded-xl border p-4 mb-6 transition-all ${
      isWarning 
        ? 'bg-amber-950/30 border-amber-500/30 text-amber-200' 
        : 'bg-cyan-950/30 border-cyan-500/30 text-cyan-200'
    }`}>
      <div className="flex items-start gap-3">
        {isWarning ? (
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        ) : (
          <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
        )}
        <div className="space-y-1">
          <h4 className="text-sm font-semibold text-white tracking-wide">
            {isWarning ? 'Measurement Disclaimers & Provenance Notes' : 'System Information'}
          </h4>
          <ul className="text-xs space-y-1 opacity-90 list-disc list-inside">
            {warnings.map((w, idx) => (
              <li key={idx} className="leading-relaxed">{w}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
