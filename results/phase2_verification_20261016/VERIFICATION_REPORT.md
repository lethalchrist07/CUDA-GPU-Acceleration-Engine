**Phase 2: CUDA Benchmark Verification — ready for approval with the limitations below.**

Verified on 2026-10-16 using NVIDIA GeForce RTX 3050 6GB Laptop GPU, compute capability 8.6, 20 SMs, driver 617.42, CUDA toolkit/runtime 13.4, and Intel Core i5-13450HX. NVIDIA-SMI reports 6144 MiB VRAM; the CUDA runtime reports approximately 6143.5 MiB. The final Release build uses MSVC 19.51, CUDA 13.4.92, C++17, sm_86, host /O2, and the existing --use_fast_math/-O3 settings. No frontend or architectural changes were made.

The original executable was copied, hashed, and executed before modification: all 12 cases passed against CPU references, exit 0. It ignored `--help` and ran the entire fixed suite; inspection confirmed no argument parsing. The rebuilt executable supports `--help`/`-h` and `--output-dir DIRECTORY`. GPU selection, sizes, and iteration counts remain fixed; unsupported options now fail instead of being ignored.

**Final measured results**

These are the actual final execution's medians from [final_run/benchmark_results.json](final_run/benchmark_results.json), also exported to [CSV](final_run/benchmark_results.csv). CPU and GPU workloads were run serially, without overlapping test execution. All 12 cases passed, with zero mismatched elements.

| Workload | Variant | CPU ms | GPU kernel ms | GPU E2E ms | Kernel speedup | E2E speedup |
|---|---|---:|---:|---:|---:|---:|
| Vector 100,000 | Standard | 0.0091 | 0.0287 | 0.3527 | 0.317x | 0.026x |
| Vector 1,000,000 | Standard | 0.5162 | 0.1065 | 2.2527 | 4.846x | 0.229x |
| Vector 10,000,000 | Standard | 6.5588 | 0.7944 | 14.8087 | 8.257x | 0.443x |
| Vector 25,000,000 | Standard | 15.9809 | 1.9304 | 45.1440 | 8.279x | 0.354x |
| Matrix 256x256 | Basic | 3.9081 | 0.0813 | 0.2513 | 48.063x | 15.552x |
| Matrix 256x256 | Tiled | 3.9081 | 0.0687 | 0.2306 | 56.883x | 16.948x |
| Matrix 512x512 | Basic | 31.7281 | 0.4784 | 1.1036 | 66.321x | 28.750x |
| Matrix 512x512 | Tiled | 31.7281 | 0.3707 | 0.9510 | 85.585x | 33.363x |
| Matrix 1024x1024 | Basic | 253.4822 | 3.5257 | 5.0719 | 71.895x | 49.978x |
| Matrix 1024x1024 | Tiled | 253.4822 | 2.8237 | 4.4160 | 89.770x | 57.401x |
| Matrix 2048x2048 | Basic | 2690.8590 | 36.4785 | 43.0002 | 73.766x | 62.578x |
| Matrix 2048x2048 | Tiled | 2690.8590 | 23.0480 | 29.9297 | 116.750x | 89.906x |

Vector maximum absolute error was zero (tolerance 1e-5). Matrix maximum absolute errors, for both variants, were 4.172325134e-7, 4.768371582e-7, 1.490116119e-6, and 1.952052116e-6 for sizes 256, 512, 1024, and 2048 respectively (tolerance 1e-3). Every output element from the final timed iteration is compared; this is deterministic-input verification, not exhaustive numerical validation of every possible input. Native regression tests also exercised vector tails, partial tiles, and mixed-sign matrix inputs against an independent double-precision reference.

GPU warmup/timed counts are 3/5 for vectors and 2/3 for matrices. CPU uses one warmup and five vector samples, three matrix samples, except two at 2048. The matrix CPU median is shared between the two variants. Raw samples and the distinct CPU sample counts are saved in JSON.

**Corrections and interpretation**

- Replaced the old sum of three CUDA-event intervals with synchronized host wall time from H2D through D2H completion. This E2E time includes launch, transfers, synchronization, and timing instrumentation, but excludes allocation, input initialization, verification, and process/context startup. Kernel and transfer metrics remain CUDA-event intervals; pageable buffers and scheduling affect them. They are not isolated profiler durations. This distinction follows [NVIDIA's timing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing).
- Fixed false-success paths for missing/skipped results and failed exports, checked warmups and kernel launches, and rejected non-finite numerical outputs. JSON emits null for non-finite error statistics on failed verification. CUDA version formatting now says 13.4 instead of 13.40. Console speedups no longer truncate small nonzero values to 0.0x or call ratios below one “Faster.”
- Milliseconds convert to seconds by dividing by 1000. Speedups are CPU median / GPU median; GFLOPS are N or 2*N^3 divided by kernel seconds and 1e9. CSV/JSON values and all new raw-sample medians were independently recomputed and passed, allowing serialization rounding. Sums of separate medians need not equal a median of totals.
- The legacy `memory_mb`/`MemoryMB` fields use MiB (bytes / 2^20), now explicitly documented. `memory_bandwidth_gb_s` is estimated requested global-memory traffic / kernel seconds / 1e9, including cache reuse. It is **not measured DRAM bandwidth**; historical values such as 1881.58 GB/s cannot be compared directly with the device's physical bandwidth. The tiled traffic estimate now also handles partial tiles.
- All vector cases are slower end-to-end than CPU execution. Matrix speedups compare the project's single-threaded CPU reference, not optimized multithreaded BLAS. Kernel-only ratios are not application speedups. No hardware-counter bandwidth, allocation-inclusive total latency, peak-memory measurement, power measurement, confidence intervals, or optimized-library comparison is provided. Timing varies between runs; the changes are verification/reporting fixes, not claimed performance optimizations.

**Historical files and execution evidence**

All six original top-level CSV/JSON files remain byte-for-byte unchanged and are also preserved in `prior_results/`. Their paired contents and arithmetic passed: 12 native benchmark rows, three training rows, and six inference rows. Historical training speedups, average epoch times, and inference tokens/second are consistent with their saved inputs. Those files lack raw timings and executable/build provenance, so their original executions cannot be authenticated retrospectively. Inference rows labeled CUDA report zero backend VRAM and do not independently prove GPU offload; no LLM or training performance claim is being approved here.

Current executions are supported by captured process output/exit codes, direct CUDA device detection, numerical checks, raw samples, build logs, and SHA-256 hashes. See [final_run.log](final_run.log), [artifact_audit-final.log](artifact_audit-final.log), [manifest.json](manifest.json), and [changes.patch](changes.patch). The original binary/run and the first corrected run remain separately preserved. The current approved measurements are under `final_run/`, not the historical top-level files.

**Failures and final validation**

- CUDA numerical failures: **0/12**, missing benchmarks: **0**; final benchmark exit **0**.
- Existing automated tests rerun after the final source change: **18/18 passed in 3.19 s**, no skips; foundation script **6/6 passed**, exit 0. See [pytest-final.log](pytest-final.log) and [foundation-final.log](foundation-final.log).
- New CTest suite: **3/3 passed in 2.68 s**. Four negative executable checks returned the expected failure codes: unknown/missing arguments 2, invalid output directory 1, and no visible CUDA device 1. See [ctest-final.log](ctest-final.log) and [failure_paths.log](failure_paths.log).
- The Visual Studio/MSBuild build failed with FileTracker `E_ACCESSDENIED`; the isolated CMake/Ninja Release build succeeded. The standard Visual Studio build path remains environment-blocked in this session. The PowerShell test wrapper was blocked by execution policy; its two underlying Python commands were run directly without changing policy. A CIM CPU query was also denied; CPU identity was read from the registry. These are environment/tooling failures, not numerical benchmark failures.

**Approval assessment:** Phase 2 CUDA correctness and benchmark verification are ready for approval, using the final artifacts and the stated timing/baseline limits. This does not approve historical performance provenance or certify the environment-blocked Visual Studio wrapper. No frontend development has begun.

The exact benchmark/build/test/audit commands, working directories, and log paths are recorded in [COMMANDS.md](COMMANDS.md); compiler invocations are in [build-ninja-final.log](build-ninja-final.log), with the complete initial rebuild in [build-ninja.log](build-ninja.log).
