# Phase 10 NVIDIA profiling report

Status: tools discovered; workload profiling NOT STARTED. Phase 10 is BLOCKED.
`nvidia-smi` and tool version queries are genuine executed probes. Initial idle
temperature was 53 C, reported utilization 0%, and driver 617.42 in WDDM mode.
These snapshots are not representative workload measurements.

Nsight Compute 2026.3.1.0 and Nsight Systems 2026.3.2.476 are installed. The x64
Systems executable is in `target-windows-x64`; the ARM executable cannot run here.
Counter permissions, occupancy, register pressure, bandwidth, warp efficiency,
Tensor Core activity, launch and synchronization overhead remain UNMEASURED.
No permission denial is asserted: profiling access has not yet been tested.

Environment, commands, regression test results/skips and reproducibility steps
are in [the implementation report](PHASE10_IMPLEMENTATION_REPORT.md).
Raw telemetry appears in `results/phase10/audit/20261019T090320375514Z.json`.
There are no profiler traces or kernel timing distributions. Next: resolve the
build gate, validate a bounded kernel workload, then collect genuine traces and
counter data where permitted; keep unsupported metrics explicitly unavailable.
