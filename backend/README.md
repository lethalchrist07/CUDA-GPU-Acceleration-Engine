**Phase 3 backend — 23 tests passed; all six data routes verified over local HTTP.**

The backend is isolated from the verified `.venv` and existing CUDA/Python pipelines. It reads actual result files and queries NVIDIA-SMI. It does not launch benchmarks, train models, change checkpoints, or rewrite result artifacts.

From the project directory, in a terminal with package-network access:

```powershell
.\scripts\install_phase3.cmd
.\.venv-phase3\Scripts\python.exe -m pytest backend/tests -v
.\.venv-phase3\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The backend has been started at `http://127.0.0.1:8000`; API documentation is exposed at `/docs`. All six data routes returned HTTP 200 with available data. The frontend remains blocked by npm registry access (`EACCES`); see `results/phase3_verification/CONTINUATION_REPORT.md`.

| Route | Data |
|---|---|
| `/api/health` | API health and read-only mode |
| `/api/overview` | Counts and actual benchmark rows |
| `/api/benchmarks` | Phase 2 final results, hashes and measurement limits |
| `/api/training` | Historical training records, explicitly unverified provenance |
| `/api/inference` | Historical inference records; no unsupported GPU-offload claim |
| `/api/hardware` | Current NVIDIA-SMI telemetry, unavailable fields represented as null |
| `/api/verification` | Phase 2 manifest checks and downloadable evidence |
| `/api/artifacts/{artifact_id}` | Allowlisted original artifacts, served without modification |

Missing or malformed data produces an explicit unavailable state. No synthetic benchmark data is used. The React dashboard stage must follow successful backend API tests.
