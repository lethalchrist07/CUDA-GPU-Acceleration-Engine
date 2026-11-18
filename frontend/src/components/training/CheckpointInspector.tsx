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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#08090B]/85 backdrop-blur-sm animate-fadeIn">
      <div className="bg-[#171A1F] w-full max-w-2xl rounded-2xl border border-[#30353D] shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-5 border-b border-[#30353D] bg-[#0D0F12]">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-[#76B900]/10 text-[#76B900] border border-[#76B900]/30">
              <HardDrive className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-[#F5F5F4] tracking-tight font-sans">
                Model Checkpoint Inspector
              </h3>
              <p className="text-xs font-mono text-[#A6ADB7] mt-0.5">
                Job ID: {jobId}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-[#A6ADB7] hover:text-[#F5F5F4] hover:bg-[#20242A] transition-all cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-5 space-y-4 overflow-y-auto flex-1">
          {/* Security & Quarantine Badge */}
          <div className="flex items-center gap-2 p-3 rounded-xl bg-[#76B900]/10 border border-[#76B900]/25 text-xs text-[#F5F5F4] font-sans">
            <ShieldCheck className="w-4 h-4 text-[#76B900] flex-shrink-0" />
            <span>
              <strong className="text-[#76B900]">Atomic Checkpoint Isolation:</strong> Serialized directly to isolated job subdirectory. Protected Phase 9 ResNet and Milestone 4 weights remain 100% unaltered.
            </span>
          </div>

          {loading ? (
            <div className="py-12 flex flex-col items-center justify-center gap-3">
              <RefreshCw className="w-7 h-7 text-[#76B900] animate-spin" />
              <span className="text-xs font-mono text-[#A6ADB7]">Loading checkpoint metadata...</span>
            </div>
          ) : error ? (
            <div className="p-4 rounded-xl bg-[#171A1F] border border-[#FB7185]/30 flex items-center gap-2 text-xs text-[#FB7185]">
              <AlertCircle className="w-4 h-4 text-[#FB7185] flex-shrink-0" />
              <span>{error}</span>
            </div>
          ) : checkpoints.length === 0 ? (
            <div className="py-10 text-center space-y-2">
              <FileCheck className="w-8 h-8 text-[#6B7280] mx-auto" />
              <p className="text-sm font-semibold text-[#F5F5F4] font-sans">No Checkpoints Saved</p>
              <p className="text-xs text-[#A6ADB7] font-sans">
                This job either did not produce a best_model.pt or was terminated before checkpointing.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {checkpoints.map((ckpt) => (
                <div
                  key={ckpt.id}
                  className="p-4 rounded-xl bg-[#121519] border border-[#30353D] space-y-3 hover:border-[#76B900]/30 transition-colors"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-[#76B900]/10 text-[#76B900] border border-[#76B900]/30 text-xs font-bold font-mono">
                        {ckpt.checkpoint_path.includes('transformer') || jobId.includes('transformer')
                          ? `Step / Epoch ${ckpt.epoch}`
                          : `Epoch ${ckpt.epoch}`}
                      </span>
                      <span className="text-xs font-mono text-[#76B900] font-semibold">
                        {ckpt.checkpoint_path.includes('transformer') || jobId.includes('transformer')
                          ? `Val Perplexity: ${ckpt.accuracy > 0 ? ckpt.accuracy.toFixed(2) : '--'}`
                          : `Accuracy: ${ckpt.accuracy.toFixed(2)}%`}
                      </span>
                    </div>
                    <span className="text-xs font-mono text-[#A6ADB7]">
                      {(ckpt.size_bytes / (1024 * 1024)).toFixed(2)} MB ({ckpt.size_bytes.toLocaleString()} bytes)
                    </span>
                  </div>

                  {/* Path */}
                  <div className="space-y-1">
                    <label className="text-[11px] font-sans text-[#A6ADB7] uppercase tracking-wider">
                      Filesystem Path (Host Isolated)
                    </label>
                    <div className="p-2 rounded-lg bg-[#08090B] border border-[#30353D] text-xs font-mono text-[#A6ADB7] break-all select-all">
                      {ckpt.checkpoint_path}
                    </div>
                  </div>

                  {/* SHA-256 Checksum */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <label className="text-[11px] font-sans text-[#A6ADB7] uppercase tracking-wider">
                        Cryptographic SHA-256 Digest
                      </label>
                      <button
                        onClick={() => handleCopyHash(ckpt.sha256)}
                        className="inline-flex items-center gap-1 text-[11px] font-mono text-[#76B900] hover:underline cursor-pointer"
                      >
                        {copiedHash === ckpt.sha256 ? (
                          <>
                            <Check className="w-3 h-3 text-[#76B900]" />
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
                    <div className="p-2 rounded-lg bg-[#08090B] border border-[#30353D] text-[11px] font-mono text-[#A6ADB7] break-all select-all">
                      {ckpt.sha256}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-[#30353D] bg-[#0D0F12] flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl text-xs font-semibold font-sans bg-[#171A1F] hover:bg-[#20242A] text-[#F5F5F4] border border-[#30353D] transition-all cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
