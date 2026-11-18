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
    <div className={`rounded-xl border p-4 mb-6 transition-all bg-[#171A1F] shadow-sm ${
      isWarning 
        ? 'border-[#FBBF24]/30 text-[#F5F5F4]' 
        : 'border-[#70D6C2]/30 text-[#F5F5F4]'
    }`}>
      <div className="flex items-start gap-3">
        {isWarning ? (
          <AlertTriangle className="w-5 h-5 text-[#FBBF24] shrink-0 mt-0.5" />
        ) : (
          <Info className="w-5 h-5 text-[#70D6C2] shrink-0 mt-0.5" />
        )}
        <div className="space-y-1">
          <h4 className="text-sm font-medium text-[#F5F5F4] font-sans">
            {isWarning ? 'Measurement Disclaimers & Provenance Notes' : 'System Information'}
          </h4>
          <ul className="text-xs text-[#A6ADB7] font-sans space-y-1 list-disc list-inside">
            {warnings.map((w, idx) => (
              <li key={idx} className="leading-relaxed">{w}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
