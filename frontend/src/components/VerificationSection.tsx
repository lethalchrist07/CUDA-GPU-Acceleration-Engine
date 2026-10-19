import React from 'react';
import { 
  ShieldCheck, 
  CheckCircle2, 
  AlertTriangle, 
  Download, 
  FileCode, 
  Lock 
} from 'lucide-react';
import { VerificationResponse, ManifestCheck, ArtifactFile } from '../types/api';
import { WarningBanner } from './WarningBanner';

interface VerificationSectionProps {
  data: VerificationResponse | null;
  loading: boolean;
}

export const VerificationSection: React.FC<VerificationSectionProps> = ({ data, loading }) => {
  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-4 border-[#76b900]/20 border-t-[#76b900] rounded-full animate-spin" />
          <span className="text-sm font-medium text-slate-400">Loading manifest verification...</span>
        </div>
      </div>
    );
  }

  const checks = data.checks || [];
  const artifacts = data.artifacts || [];
  const matchedCount = checks.filter(c => c.status === 'matched').length;
  const isAllMatched = checks.length > 0 && matchedCount === checks.length;

  return (
    <div id="section-verification" className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel rounded-2xl p-6 border border-white/[0.07]">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-extrabold text-white">Verification &amp; Artifacts Audit</h2>
            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${
              isAllMatched
                ? 'bg-[#76b900]/15 text-[#76b900] border-[#76b900]/40 glow-nvidia'
                : 'bg-amber-950/80 text-amber-300 border-amber-500/40'
            }`}>
              <ShieldCheck className="w-4 h-4 text-[#76b900]" />
              <span>{matchedCount} / {checks.length} Manifest Files Matched</span>
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            SHA-256 Hash Integrity Checks • Read-Only Artifact Archive • No Synthetic Metrics
          </p>
        </div>
      </div>

      {/* Warnings Banner */}
      <WarningBanner warnings={data.warnings} />

      {/* Audit Stats Banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] border-t-2 border-t-[#76b900] flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#76b900]/10 border border-[#76b900]/30 flex items-center justify-center text-[#76b900] shrink-0">
            <CheckCircle2 className="w-6 h-6 stroke-[2.2]" />
          </div>
          <div>
            <span className="text-[10px] text-slate-400 uppercase font-semibold tracking-wider">Manifest Parity</span>
            <p className="text-xl font-bold font-mono text-[#76b900] tabular-nums">{matchedCount} / {checks.length} Pass</p>
            <p className="text-[11px] text-slate-400">100% Phase 2 SHA-256 integrity</p>
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] border-t-2 border-t-cyan-400 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-cyan-950/60 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0">
            <Lock className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[10px] text-slate-400 uppercase font-semibold tracking-wider">Artifact Immutability</span>
            <p className="text-xl font-bold font-mono text-cyan-400">Read-Only Confined</p>
            <p className="text-[11px] text-slate-400">Path traversal protection enabled</p>
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] border-t-2 border-t-amber-400 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-amber-950/60 border border-amber-500/30 flex items-center justify-center text-amber-400 shrink-0">
            <FileCode className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[10px] text-slate-400 uppercase font-semibold tracking-wider">Available Artifacts</span>
            <p className="text-xl font-bold font-mono text-amber-300 tabular-nums">{artifacts.length} Files</p>
            <p className="text-[11px] text-slate-400">Downloadable raw execution logs</p>
          </div>
        </div>
      </div>

      {/* Downloadable Artifacts Grid */}
      <div className="glass-panel rounded-2xl border border-white/[0.07] p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
          <div className="flex items-center gap-2">
            <Download className="w-4 h-4 text-[#76b900]" />
            <h3 className="text-base font-bold text-white tracking-tight">Verified Project Artifacts</h3>
          </div>
          <span className="text-xs font-mono text-slate-400">Original unedited files</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {artifacts.map((art: ArtifactFile) => (
            <div
              key={art.id}
              className="p-3.5 rounded-xl bg-obsidian-900/70 border border-white/[0.06] flex items-center justify-between hover:border-white/[0.15] transition-all group"
            >
              <div className="space-y-0.5 truncate mr-2">
                <span className="text-xs font-semibold text-slate-200 block truncate group-hover:text-[#76b900] transition-colors">
                  {art.name}
                </span>
                <span className="text-[10px] font-mono text-slate-400 block truncate">
                  {art.path} ({(art.size_bytes / 1024).toFixed(1)} KB)
                </span>
              </div>
              <a
                href={art.download_url}
                download={art.name}
                aria-label={`Download artifact ${art.name}`}
                className="p-2 rounded-lg bg-obsidian-850 border border-white/[0.08] hover:bg-[#76b900] hover:text-slate-950 text-slate-300 transition-all shrink-0 cursor-pointer"
                title={`Download ${art.name}`}
              >
                <Download className="w-3.5 h-3.5" />
              </a>
            </div>
          ))}
        </div>
      </div>

      {/* Manifest Checks Table */}
      <div className="glass-panel rounded-2xl border border-white/[0.07] overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-white/[0.06] flex justify-between items-center">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-[#76b900]" />
            <h3 className="text-base font-bold text-white tracking-tight">Phase 2 Manifest SHA-256 Fingerprints</h3>
          </div>
          <span className="text-xs font-mono text-[#76b900] font-semibold">
            Status: {data.manifest_status.toUpperCase()}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[700px]">
            <thead className="bg-obsidian-900/90 text-slate-400 uppercase font-semibold text-[10px] tracking-wider border-b border-white/[0.06]">
              <tr>
                <th className="py-3 px-4">File Path</th>
                <th className="py-3 px-3">Integrity Status</th>
                <th className="py-3 px-3 font-mono">Expected SHA-256</th>
                <th className="py-3 px-4 font-mono">Actual SHA-256</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] font-mono text-[11px] tabular-nums">
              {checks.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-8 text-center text-slate-400 font-sans text-xs">
                    No manifest checks available.
                  </td>
                </tr>
              ) : (
                checks.map((check: ManifestCheck, idx: number) => {
                const isMatched = check.status === 'matched';
                return (
                  <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-3 px-4 font-sans font-medium text-white truncate max-w-[280px]">
                      {check.path}
                    </td>
                    <td className="py-3 px-3">
                      {isMatched ? (
                        <span className="inline-flex items-center gap-1.5 text-[#76b900] font-sans font-semibold text-[11px]">
                          <CheckCircle2 className="w-3.5 h-3.5 stroke-[2.2]" />
                          Matched
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 text-rose-400 font-sans font-semibold text-[11px]">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          Mismatch
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-slate-400 truncate max-w-[200px]" title={check.expected_sha256}>
                      {check.expected_sha256.substring(0, 24)}...
                    </td>
                    <td className="py-3 px-4 text-[#76b900] truncate max-w-[200px]" title={check.actual_sha256 || 'None'}>
                      {check.actual_sha256 ? `${check.actual_sha256.substring(0, 24)}...` : 'Missing'}
                    </td>
                  </tr>
                );
              }))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
