import React, { useEffect, useState } from 'react';
import { CheckpointMetadata } from '../../types/training';
import { trainingApi } from '../../services/trainingApi';
import { 
  X, 
  HardDrive, 
  ShieldCheck, 
  Copy, 
  Check, 
  FileCheck, 
  RefreshCw,
  AlertCircle
} from 'lucide-react';

interface CheckpointInspectorProps {
  jobId: string;
  onClose: () => void;
}

export const CheckpointInspector: React.FC<CheckpointInspectorProps> = ({
  jobId,
  onClose,
}) => {
  const [checkpoints, setCheckpoints] = useState<CheckpointMetadata[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedHash, setCopiedHash] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    trainingApi
      .getJobCheckpoints(jobId)
      .then((res) => {
        if (isMounted) {
          setCheckpoints(res.checkpoints);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to load checkpoints');
        }
      })
      .finally(() => {
        if (isMounted) {
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [jobId]);

  const handleCopyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopiedHash(hash);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-obsidian-950/80 backdrop-blur-md animate-fadeIn">
      <div className="glass-panel w-full max-w-2xl rounded-2xl border border-white/[0.1] shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-5 border-b border-white/[0.08] bg-obsidian-900/90">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-[#76b900]/15 text-[#76b900]">
              <HardDrive className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-tight">
                Model Checkpoint Inspector
              </h3>
              <p className="text-xs font-mono text-slate-400 mt-0.5">
                Job ID: {jobId}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.05] transition-all cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-5 space-y-4 overflow-y-auto flex-1">
          {/* Security & Quarantine Badge */}
          <div className="flex items-center gap-2 p-3 rounded-xl bg-[#76b900]/10 border border-[#76b900]/25 text-xs text-slate-200">
            <ShieldCheck className="w-4 h-4 text-[#76b900] flex-shrink-0" />
            <span>
              <strong>Atomic Checkpoint Isolation:</strong> Serialized directly to isolated job subdirectory. Protected Phase 9 ResNet and Milestone 4 weights remain 100% unaltered.
            </span>
          </div>

          {loading ? (
            <div className="py-12 flex flex-col items-center justify-center gap-3">
              <RefreshCw className="w-7 h-7 text-[#76b900] animate-spin" />
              <span className="text-xs font-mono text-slate-400">Loading checkpoint metadata...</span>
            </div>
          ) : error ? (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/25 flex items-center gap-2 text-xs text-rose-300">
              <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
              <span>{error}</span>
            </div>
          ) : checkpoints.length === 0 ? (
            <div className="py-10 text-center space-y-2">
              <FileCheck className="w-8 h-8 text-slate-600 mx-auto" />
              <p className="text-sm font-semibold text-slate-300">No Checkpoints Saved</p>
              <p className="text-xs text-slate-500">
                This job either did not produce a best_model.pt or was terminated before checkpointing.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {checkpoints.map((ckpt) => (
                <div
                  key={ckpt.id}
                  className="p-4 rounded-xl bg-obsidian-900 border border-white/[0.06] space-y-3"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-[#76b900]/15 text-[#76b900] border border-[#76b900]/30 text-xs font-bold font-mono">
                        Epoch {ckpt.epoch}
                      </span>
                      <span className="text-xs font-mono text-emerald-400 font-semibold">
                        Accuracy: {ckpt.accuracy.toFixed(2)}%
                      </span>
                    </div>
                    <span className="text-xs font-mono text-slate-400">
                      {(ckpt.size_bytes / (1024 * 1024)).toFixed(2)} MB ({ckpt.size_bytes.toLocaleString()} bytes)
                    </span>
                  </div>

                  {/* Path */}
                  <div className="space-y-1">
                    <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                      Filesystem Path (Host Isolated)
                    </label>
                    <div className="p-2 rounded-lg bg-obsidian-950 border border-white/[0.05] text-xs font-mono text-slate-300 break-all select-all">
                      {ckpt.checkpoint_path}
                    </div>
                  </div>

                  {/* SHA-256 Checksum */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                        Cryptographic SHA-256 Digest
                      </label>
                      <button
                        onClick={() => handleCopyHash(ckpt.sha256)}
                        className="inline-flex items-center gap-1 text-[11px] font-mono text-[#76b900] hover:underline cursor-pointer"
                      >
                        {copiedHash === ckpt.sha256 ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-400" />
                            <span>Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>Copy Hash</span>
                          </>
                        )}
                      </button>
                    </div>
                    <div className="p-2 rounded-lg bg-obsidian-950 border border-white/[0.05] text-[11px] font-mono text-slate-300 break-all select-all">
                      {ckpt.sha256}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-white/[0.08] bg-obsidian-900/90 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl text-xs font-bold bg-white/[0.08] hover:bg-white/[0.12] text-white transition-all cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
