# Phase 10 implementation report

Status: BLOCKED at the Stage 0 preservation/toolchain gate (2026-10-19).
Stages 1–8 have not been executed. No custom kernels or backend endpoints were added.
No frontend source, Phase 9 report, checkpoint, or protected Phase 2 file was edited.

## Repository preservation

The starting repository is unborn: `git rev-parse HEAD` fails because no commit exists.
`git status --short` showed all project files untracked (160 nonignored files before
this audit). Local configuration contains only ordinary core settings; no active
non-sample hooks were found. Neither author name nor email is configured. No files
were staged and no commit was made. Author identity is requested before preparing
an explicitly reviewed source/artifact allowlist for the baseline commit.
Do not use `git add .`: the existing ignore file intentionally permits historical
reference executables. Datasets, weights, build outputs, and temporary binaries
must remain excluded. This audit does not certify every candidate file for commit.

## Environment observed

- Python 3.11.9; PyTorch 2.6.0+cu124; PyTorch runtime CUDA 12.4.
- nvcc 13.4.92; driver 617.42; Windows WDDM.
- MSVC 19.51.36260; installed toolset directories 14.44.35207 and 14.51.36231.
- CMake 4.3.1-msvc1 is installed in the VS BuildTools tree, absent from default PATH.
- RTX 3050 6GB Laptop GPU, sm_86, 20 SMs; 6,441,926,656 total CUDA-visible bytes.
- Initial CUDA free memory: 5,411,700,736 bytes; initial temperature: 53 C.
- Disk free at the audit probe: 275,371,909,120 bytes.
- Nsight Compute 2026.3.1.0 and Systems 2026.3.2.476 installed.

`torch.utils.cpp_extension._check_cuda_version` raises the exact error:

```
The detected CUDA version (13.4) mismatches the version that was used to compile
PyTorch (12.4). Please make sure to use the same CUDA versions.
```

This is a preflight guard failure, not a claimed compiler/linker failure. No
unsupported-version override or dependency installation was attempted. A proposed
fix is side-by-side CUDA 12.4 plus an MSVC 19.3x toolset; NVIDIA's archived
[CUDA 12.4 Windows compatibility table](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-microsoft-windows/index.html)
lists MSVC 193x for VS 2022. Neither observed installed toolset is 19.3x.
Approval was requested for the dependency setup under the user's execution rules.
Installation feasibility and a minimal compile/link/execute check remain required.

## Architecture and evidence

Existing native infrastructure is `src/vector_add.cu`, `src/matmul.cu`,
`src/benchmark_main.cu`, `include/`, and `tests/benchmark_regression.cu`.
The root CMake configuration and native sources are protected manifest entries;
the existing build writes directly into protected `bin/`. Future Phase 10 builds
must use separate sources and output directories, leaving these files intact.

Read the Phase 9 implementation, verification, final training audit, and the
training/performance report summaries. Historical 90.92% validation, 90.31% test,
787.17 seconds training and 57 passed/1 skipped are reference results, not Phase 10
training measurements. No new training or accuracy evaluation was run.

Added `scripts/audit_phase10_baseline.py`: bounded read-only inspection of existing
artifacts, weights-only checkpoint loading, hashes, toolchain guard and telemetry;
it writes exclusively created timestamped JSON under `results/phase10/audit/`.
The first record is `20261019T090320375514Z.json`. It also records hashes of
nonignored untracked files for subsequent preservation checks. Those hashes were
captured after baseline tests and frontend build, not before them.

## Commands and validation

Run from the repository root unless stated otherwise:

```powershell
git status --short
git rev-parse HEAD
git config --local --list
git config --get user.name
git config --get user.email
nvcc --version
nvidia-smi
ncu --version
.\.venv\Scripts\python.exe -m pytest backend/tests engine/tests -q -rs
.\.venv\Scripts\python.exe -m pytest scripts/test_phase4_validation.py -q -k 'manifest_preservation or production_build_assets'
.\bin\benchmark_regression.exe
.\.venv\Scripts\python.exe scripts/audit_phase10_baseline.py
```

Version commands were also executed using absolute paths for CMake and MSVC under
`C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools`, and x64 Nsight
Systems under `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2`.
An initial attempt to invoke the discovered ARM Nsight executable failed on x64;
the x64 executable worked. `npm.cmd run build` ran inside `frontend`.

Measured validation: backend/engine 57 passed, 1 skipped, 39.33 seconds; two selected
artifact/build checks passed in 0.04 seconds (five live-service checks deselected).
Native existing regression executable passed; its intentional invalid export
messages are expected negative tests. Frontend build passed (Vite 6.4.4, 2218
modules, reported build duration 26.65 seconds), with a >500 KB chunk advisory.
No redesign was made. FastAPI emitted a Starlette/httpx deprecation warning.
Ollama skip: no local runtime on port 11434.

All 17 historical SHA-256 entries match. Best/latest/initial ResNet checkpoints
load with `weights_only=True` on CPU; per-file sizes, epochs and SHA-256 hashes are
in the raw audit. This confirms readability, not a new model accuracy measurement.

## Limits and next step

No Phase 10 numerical parity, gradients, memory/race checks, kernel timings,
speedups, hardware counters, model replacement, or extension execution exists yet.
Preserve the audit, resolve author identity and compatible build dependencies,
create the reviewed local baseline commit, then begin Stage 1 incrementally.
No push, dependency upgrade, long training run, or Phase 11 work occurred.
