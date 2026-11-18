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
          <span className={`inline-flex items-center gap-1.5 font-medium font-sans rounded-md bg-[#4ADE80]/10 text-[#4ADE80] border border-[#4ADE80]/30 ${sizeClasses}`}>
            <ShieldCheck className={isSm ? "w-3.5 h-3.5" : "w-4 h-4"} />
            <span>SHA-256 Verified</span>
          </span>
        );
      case 'historical_unverified':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium font-sans rounded-md bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30 ${sizeClasses}`}>
            <AlertTriangle className={isSm ? "w-3.5 h-3.5" : "w-4 h-4"} />
            <span>Historical Record (Unverified)</span>
          </span>
        );
      case 'hash_mismatch':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium font-sans rounded-md bg-[#FB7185]/10 text-[#FB7185] border border-[#FB7185]/30 ${sizeClasses}`}>
            <AlertCircle className={isSm ? "w-3.5 h-3.5" : "w-4 h-4"} />
            <span>Hash Mismatch</span>
          </span>
        );
      case 'recorded_execution':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium font-sans rounded-md bg-[#70D6C2]/10 text-[#70D6C2] border border-[#70D6C2]/30 ${sizeClasses}`}>
            <Clock className={isSm ? "w-3.5 h-3.5" : "w-4 h-4"} />
            <span>Recorded Execution</span>
          </span>
        );
      default:
        return (
          <span className={`inline-flex items-center gap-1 font-mono rounded-md bg-[#171A1F] text-[#A6ADB7] border border-[#30353D] ${sizeClasses}`}>
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
          <span className={`inline-flex items-center gap-1.5 font-medium font-sans rounded-md bg-[#76B900]/10 text-[#76B900] border border-[#76B900]/30 ${sizeClasses}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-[#76B900] shadow-[0_0_6px_rgba(118,185,0,0.5)]" />
            Available
          </span>
        );
      case 'cpu_only':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium font-sans rounded-md bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30 ${sizeClasses}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-[#FBBF24] shadow-[0_0_6px_rgba(251,191,36,0.5)]" />
            CPU Fallback
          </span>
        );
      case 'degraded':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium font-sans rounded-md bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30 ${sizeClasses}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-[#FBBF24] shadow-[0_0_6px_rgba(251,191,36,0.5)]" />
            Degraded
          </span>
        );
      case 'unavailable':
      case 'mismatch':
        return (
          <span className={`inline-flex items-center gap-1.5 font-medium font-sans rounded-md bg-[#FB7185]/10 text-[#FB7185] border border-[#FB7185]/30 ${sizeClasses}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-[#FB7185] shadow-[0_0_6px_rgba(251,113,133,0.5)]" />
            Unavailable
          </span>
        );
      default:
        return (
          <span className={`inline-flex items-center font-medium font-sans rounded-md bg-[#171A1F] text-[#A6ADB7] border border-[#30353D] ${sizeClasses}`}>
            {status}
          </span>
        );
    }
  }

  return null;
};
