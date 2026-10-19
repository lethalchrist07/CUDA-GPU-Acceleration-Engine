# Phase 10 kernel benchmarks

Status: NOT STARTED; Phase 10 is BLOCKED at Stage 0. No custom kernel performance
or speedup measurements exist. Required fused bias/ReLU, reductions, LayerNorm,
RMSNorm, tiled GEMM/experimental WMMA, and fused Conv2D remain unimplemented.

The existing native regression executable passed. It is a correctness check of
historical kernels, not a Phase 10 performance benchmark. No JSON/CSV performance
dataset has been generated. No numerical parity or gradient results are claimed.

Environment, executed commands, test totals/skips, hardware constraints and
reproduction instructions are in [the implementation report](PHASE10_IMPLEMENTATION_REPORT.md).
Raw environment/preservation data is under `results/phase10/audit/`.
After the gate is resolved, validate each operation before collecting warm-up and
synchronized latency distributions against matched precision/shape references.
Kernel-only and end-to-end results must remain separate.
