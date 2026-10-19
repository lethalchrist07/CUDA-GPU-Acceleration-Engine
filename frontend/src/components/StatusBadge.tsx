import React from 'react';
import { ShieldCheck, AlertTriangle, Clock, AlertCircle } from 'lucide-react';

interface StatusBadgeProps {
  status?: string;
  provenance?: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, provenance, size = 'sm' }) => {
  const isSm = size === 'sm';
  const sizeClasses = isSm ? 'text-xs px-2.5 py-0.5' : 'text-sm px-3.5 py-1';

  if (provenance) {
    switch (provenance) {
      case 'phase2_hash_verified':
        return (
          <span className={`inline-flex items-center gap-1.5 font-semibold rounded-full bg-[#76b900]/10 text-[#76b900] border border-[#76b900]/30 ${sizeClasses} glow-nvidia`}>
            <ShieldCheck className={isSm ? "w-3.5 h-3.5" : "w-4 h-4"} />
            <span>SHA-256 Verified</span>
          </span>
        );
      case 'historical_unverified':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-amber-950/70 text-amber-300 border border-amber-500/30 ${sizeClasses} glow-amber`}>
            <AlertTriangle className={isSm ? "w-3.5 h-3.5" : "w-4 h-4"} />
            <span>Historical Record (Unverified)</span>
          </span>
        );
      case 'hash_mismatch':
        return (
          <span className={`inline-flex items-center gap-1.5 font-semibold rounded-full bg-red-950/80 text-rose-300 border border-rose-500/40 ${sizeClasses}`}>
            <AlertCircle className={isSm ? "w-3.5 h-3.5" : "w-4 h-4"} />
            <span>Hash Mismatch</span>
          </span>
        );
      case 'recorded_execution':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-cyan-950/70 text-cyan-300 border border-cyan-500/30 ${sizeClasses} glow-cyan`}>
            <Clock className={isSm ? "w-3.5 h-3.5" : "w-4 h-4"} />
            <span>Recorded Execution</span>
          </span>
        );
      default:
        return (
          <span className={`inline-flex items-center gap-1 font-mono rounded-full bg-slate-800 text-slate-300 border border-slate-700 ${sizeClasses}`}>
            {provenance}
          </span>
        );
    }
  }

  if (status) {
    switch (status) {
      case 'available':
      case 'matched':
      case 'ok':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-500/30 ${sizeClasses}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Available
          </span>
        );
      case 'degraded':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-amber-950/80 text-amber-300 border border-amber-500/30 ${sizeClasses}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
            Degraded
          </span>
        );
      case 'unavailable':
      case 'mismatch':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-rose-950/80 text-rose-300 border border-rose-500/30 ${sizeClasses}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
            Unavailable
          </span>
        );
      default:
        return (
          <span className={`inline-flex items-center font-medium rounded-full bg-slate-800 text-slate-300 border border-slate-700 ${sizeClasses}`}>
            {status}
          </span>
        );
    }
  }

  return null;
};
