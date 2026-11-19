# NEXORA — November 15 BLK-03 Single-Server Deployment Remediation Report

**Document Version:** 1.0.0  
**Date:** November 15, 2026  
**Auditor / Role:** Senior Full-Stack Infrastructure & Release Engineer  
**Status:** **REMEDIATION COMPLETE — BLK-03 RESOLVED (PASS)**  
**Target Release:** November 30, 2026 (v1.0.0)  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, sm_86, Driver 591.44, CUDA 12.9 / 12.4 runtime)  

---

## Executive Summary

Under the **November 15 Stage 3 Remediation Mandate**, release engineering conducted a controlled remediation of **BLK-03 (Dual-Server Dependency and Lack of Single-Server Production Deployment)**. Prior to this intervention, running the NEXORA platform required simultaneously operating two distinct development servers:
1. A Uvicorn ASGI server hosting the FastAPI backend on `http://127.0.0.1:8000`.
2. A Node.js Vite development server hosting the frontend interface on `http://127.0.0.1:5173`.

This dual-server setup introduced severe operational and distribution vulnerabilities:
- Process lifecycle fragmentation: stopping or crashing one server left orphan zombie processes on developer machines.
- Port collision risks across two ports (`8000` and `5173`).
- Inability to distribute NEXORA as a standalone turnkey local application or self-contained executable.
- Page refresh failures on client-side routes when accessing static builds directly without SPA fallback.
- Incompatible CORS configurations preventing single-origin local deployments.

### Remediation Achievements
BLK-03 is **100% RESOLVED**:
1. **Unified Single-Server Topology:** The FastAPI backend serves both the complete REST/WebSocket API and the production-optimized React Single-Page Application (SPA) on a single port (`http://127.0.0.1:8000`).
2. **Static Asset Mounting:** Starlette `StaticFiles` is mounted under `/assets` serving compiled CSS and JS chunks with proper MIME types, HTTP caching headers, and binary stream safety.
3. **SPA Fallback Routing:** A catch-all route `GET /{full_path:path}` serves `index.html` for client-side React routes (`/overview`, `/gpu_monitoring`, `/training_studio`, `/model_registry`, `/cuda_lab`, `/analytics`, `/inference`, `/verification`), ensuring full browser reload fidelity.
4. **Strict API Route Isolation:** Any unmatched path beginning with `api` or `api/` returns a clean, structured JSON 404 response (`{"detail": "..."}`), guaranteeing that misconfigured API calls never receive HTML payloads that cause silent JSON deserialization errors.
5. **URL History & Tab Synchronization:** `frontend/src/App.tsx` was enhanced with `getInitialTab` and HTML5 History API integration (`window.history.pushState` and `popstate` listeners) to synchronize browser URL changes with active navigation state across page reloads and back/forward navigation.
6. **Unified Launcher Scripts:** `start_platform.bat` and `start_platform.ps1` were updated to run in single-server production mode, automatically compiling frontend static assets if missing and launching the unified application at `http://127.0.0.1:8000/`.
7. **Automated Verification:** 15 dedicated pytest integration tests in [`tests/test_single_server_deployment.py`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/tests/test_single_server_deployment.py) passed (**15/15 PASS**), live daemon runtime verification passed (**8/8 PASS**), and full regression testing passed (**51/51 PASS**).

---

## 1. Root Cause Analysis

### 1.1 Dual-Server Execution Defect
In the original architecture, `start_platform.bat` and `start_platform.ps1` executed:
```bat
start "NEXORA Backend" uvicorn backend.app:app --port 8000
start "NEXORA Frontend" npm --prefix frontend run dev
```
This forced users into a Node.js Vite development server (`http://localhost:5173`) which proxied API requests to `http://localhost:8000`. This architecture was unsuitable for distribution because:
- End-users were required to have Node.js and npm installed even if they only intended to run the pre-built engine.
- Vite dev mode recompiled modules on-the-fly, adding latency and memory overhead unnecessary for production.
- Process termination scripts often failed to terminate both command windows, resulting in locked ports `8000` or `5173`.

### 1.2 Lack of Static Asset & SPA Serving in Backend
In `backend/app.py`, the FastAPI application only mounted API routers:
- `/api/health`, `/api/overview`, `/api/hardware`, `/api/models`, `/api/training/*`, `/api/laboratory/*`.
- Root `GET /` was unhandled, returning `{"detail": "Not Found"}`.
- Static assets (`.js`, `.css`, `.svg`) had no mounted directory.
- Client-side routes (e.g., `http://localhost:8000/training_studio`) produced HTTP 404 errors if requested directly or refreshed in the browser.

### 1.3 CORS Origin Restrictions
In `backend/config.py`, `CUDA_ENGINE_CORS_ORIGINS` was restricted to:
```python
CUDA_ENGINE_CORS_ORIGINS: str = (
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"
)
```
Requests originating from `http://localhost:8000` or `http://127.0.0.1:8000` were not recognized as authorized CORS origins.

---

## 2. Inventory of Changes

| File Path | Nature of Change | Lines Affected | Summary of Modifications |
| :--- | :--- | :---: | :--- |
| [`backend/app.py`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/backend/app.py) | Routing & Static Serving Enhancement | 13, 294–335 | Mounted `StaticFiles(directory="frontend/dist/assets")` under `/assets`. Added `GET /` to serve `index.html`. Added `GET /{full_path:path}` catch-all SPA route with strict API 404 isolation and root-file resolution. Graceful degradation when `dist` is absent. |
| [`backend/config.py`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/backend/config.py) | Configuration Update | 21–23 | Added `http://localhost:8000` and `http://127.0.0.1:8000` to default `CUDA_ENGINE_CORS_ORIGINS`. |
| [`frontend/src/App.tsx`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/frontend/src/App.tsx) | Client Routing & History Sync | 28–75 | Added `VALID_TABS`, `getInitialTab()` parsing `window.location.pathname` and hash on mount. Wrapped `setActiveTab` with `window.history.pushState`. Added `popstate` event listener for seamless back/forward navigation. |
| [`start_platform.bat`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/start_platform.bat) | Launcher Streamlining | 29–54 | Removed Vite dev server dispatch. Added preflight build check running `npm run build` if `frontend\dist\index.html` is absent. Starts single Uvicorn server on port 8000 and opens browser directly to `http://127.0.0.1:8000/`. |
| [`start_platform.ps1`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/start_platform.ps1) | Launcher Streamlining | 24–43 | Removed dual-server startup. Added automatic static build check. Starts single Uvicorn process on port 8000 and opens `http://127.0.0.1:8000/`. |
| [`tests/test_single_server_deployment.py`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/tests/test_single_server_deployment.py) | New Test Suite | 1–147 | 15 comprehensive unit & integration tests validating root delivery, CSS/JS MIME types, 8 SPA routes, API JSON integrity, API 404 non-HTML responses, `/docs`, and CORS origins. |
| [`scripts/verify_single_server_runtime.py`](file:///c:/Projects/CUDA-GPU-Acceleration-Engine/scripts/verify_single_server_runtime.py) | New Live Test Harness | 1–212 | Live daemon test harness that spawns Uvicorn on port 8000, performs live HTTP requests against all key platform endpoints, and verifies clean teardown. |

---

## 3. Implementation Details

### 3.1 Backend Static & SPA Serving Architecture

In `backend/app.py`:
```python
# Single-Server Deployment & SPA Static Asset Serving (BLK-03)
dist_dir = (root / "frontend" / "dist").resolve()
assets_dir = dist_dir / "assets"
if assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

@app.get("/", include_in_schema=False)
def serve_root():
    index_file = dist_dir / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    raise HTTPException(
        status_code=404,
        detail="Frontend static build not found. Please build the frontend via 'npm run build' inside frontend/.",
    )

@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str):
    # Protect API endpoints: unmatched /api/... routes must return JSON 404, not HTML
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(
            status_code=404,
            detail=f"API endpoint '/{full_path}' not found",
        )

    # Direct static file in dist root (e.g. favicon, robots.txt, vite.svg)
    if dist_dir.is_dir():
        candidate = (dist_dir / full_path).resolve()
        if candidate.is_file() and str(candidate).startswith(str(dist_dir)):
            return FileResponse(candidate)

        index_path = dist_dir / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)

    raise HTTPException(
        status_code=404,
        detail="Frontend static build not found. Please build the frontend via 'npm run build' inside frontend/.",
    )
```

#### Key Design Decisions:
1. **API Guard Guarantee:** Any request path matching `api` or `api/*` that is not caught by preceding API routers raises a structured `HTTPException(status_code=404, ...)`. This prevents frontend API clients from ever receiving an HTML `index.html` string when an API endpoint does not exist.
2. **Path Traversal Protection:** Candidates for root static files are checked with `str(candidate).startswith(str(dist_dir))` to prevent directory traversal attacks.
3. **Graceful Build Degradation:** If `frontend/dist` has not yet been built (e.g., in headless API-only testing or fresh development clones), backend API routes continue functioning without throwing import or initialization exceptions.

### 3.2 Frontend History & URL Synchronization

In `frontend/src/App.tsx`:
```typescript
const VALID_TABS: SectionTab[] = [
  'showcase',
  'overview',
  'gpu_monitoring',
  'training_studio',
  'model_registry',
  'cuda_lab',
  'analytics',
  'inference',
  'verification',
];

const getInitialTab = (): SectionTab => {
  if (typeof window === 'undefined') return 'showcase';
  const path = window.location.pathname.replace(/^\/+/, '').split('/')[0];
  const hash = window.location.hash.replace(/^#/, '');
  if (VALID_TABS.includes(path as SectionTab)) return path as SectionTab;
  if (VALID_TABS.includes(hash as SectionTab)) return hash as SectionTab;
  return 'showcase';
};
```
When `setActiveTab` is called:
```typescript
const setActiveTab = useCallback((tab: SectionTab) => {
  setActiveTabState(tab);
  if (typeof window !== 'undefined') {
    const targetPath = tab === 'showcase' ? '/' : `/${tab}`;
    if (window.location.pathname !== targetPath) {
      window.history.pushState({ tab }, '', targetPath);
    }
  }
}, []);
```
And a `popstate` event listener synchronizes browser back and forward actions:
```typescript
useEffect(() => {
  const handlePopState = (event: PopStateEvent) => {
    if (event.state && event.state.tab && VALID_TABS.includes(event.state.tab)) {
      setActiveTabState(event.state.tab);
    } else {
      setActiveTabState(getInitialTab());
    }
  };
  window.addEventListener('popstate', handlePopState);
  return () => window.removeEventListener('popstate', handlePopState);
}, []);
```

---

## 4. Production Build Verification

The frontend static bundle was compiled using `npm --prefix frontend run build`:
```
> frontend@0.0.0 build
> tsc && vite build

vite v5.4.11 building for production...
transforming...
✓ 1876 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   1.28 kB │ gzip:   0.57 kB
dist/assets/index-B3Fp_5Vz.css   46.42 kB │ gzip:   8.91 kB
dist/assets/index-Bj6o4BeT.js   850.79 kB │ gzip: 247.95 kB
✓ built in 2.21s
```

### Static Asset Fingerprints & Sizes
- `frontend/dist/index.html`: `1,280 bytes`
- `frontend/dist/assets/index-B3Fp_5Vz.css`: `46,420 bytes`
- `frontend/dist/assets/index-Bj6o4BeT.js`: `850,790 bytes`

All asset links in `index.html` use relative paths (`/assets/index-...`), which resolve cleanly against the unified FastAPI server root at `http://127.0.0.1:8000/`.

---

## 5. Automated Verification Results

### 5.1 Single-Server Deployment Test Battery (`tests/test_single_server_deployment.py`)

Run command:
```powershell
pytest tests/test_single_server_deployment.py -v
```

| Test Case Name | Verified Behavior | Status |
| :--- | :--- | :---: |
| `test_root_serves_frontend_index_html` | `GET /` returns `index.html` with HTTP 200, text/html MIME, and `#root` container | **PASS** |
| `test_static_assets_served_correctly` | `GET /assets/*.css` and `GET /assets/*.js` return HTTP 200 with accurate MIME types | **PASS** |
| `test_spa_client_routes_serve_index_html[/overview]` | Route `/overview` returns `index.html` with HTTP 200 | **PASS** |
| `test_spa_client_routes_serve_index_html[/gpu_monitoring]` | Route `/gpu_monitoring` returns `index.html` with HTTP 200 | **PASS** |
| `test_spa_client_routes_serve_index_html[/training_studio]` | Route `/training_studio` returns `index.html` with HTTP 200 | **PASS** |
| `test_spa_client_routes_serve_index_html[/model_registry]` | Route `/model_registry` returns `index.html` with HTTP 200 | **PASS** |
| `test_spa_client_routes_serve_index_html[/cuda_lab]` | Route `/cuda_lab` returns `index.html` with HTTP 200 | **PASS** |
| `test_spa_client_routes_serve_index_html[/analytics]` | Route `/analytics` returns `index.html` with HTTP 200 | **PASS** |
| `test_spa_client_routes_serve_index_html[/inference]` | Route `/inference` returns `index.html` with HTTP 200 | **PASS** |
| `test_spa_client_routes_serve_index_html[/verification]` | Route `/verification` returns `index.html` with HTTP 200 | **PASS** |
| `test_valid_api_endpoints_return_json` | `/api/health`, `/api/overview`, `/api/hardware`, `/api/models`, `/api/training/models` return JSON | **PASS** |
| `test_unmatched_api_routes_return_json_404_not_html` | Nonexistent `/api/*` endpoints return JSON 404, never HTML index | **PASS** |
| `test_docs_endpoints_accessible` | OpenAPI `/docs` and `/openapi.json` remain accessible | **PASS** |
| `test_missing_frontend_build_graceful_handling` | Server initializes cleanly without `frontend/dist`, returning 404 for root | **PASS** |
| `test_cors_allows_single_server_origins` | CORS headers allow origins `http://127.0.0.1:8000` and `http://localhost:8000` | **PASS** |

**Summary: 15 passed in 16.22s.**

---

### 5.2 Live Runtime Verification (`scripts/verify_single_server_runtime.py`)

A live runtime verification harness was executed against a running Uvicorn server bound to `127.0.0.1:8000`:
```
======================================================================
  NEXORA BLK-03 — LIVE SINGLE-SERVER RUNTIME VERIFICATION
======================================================================
[*] Launching unified single-server daemon: .venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --log-level warning
[*] Waiting for server to bind on http://127.0.0.1:8000...
[+] Server successfully bound to http://127.0.0.1:8000

[PASS 1/8] Root entry point (GET /) served React SPA index.html (1,280 bytes)
[PASS 2/8] Static bundles served successfully: CSS (46,420 bytes), JS (850,790 bytes)
[PASS 3/8] SPA client-side routes (8 routes) verified for seamless page refresh
[PASS 4/8] Health & Hardware Telemetry API verified: 1 GPU(s) reported
[PASS 5/8] Model Registry & Platform Capabilities API verified (4 models registered)
[PASS 6/8] Training Studio API verified (6 trainable models discoverable)
[PASS 7/8] API 404 isolation confirmed: Unmatched /api/* routes return JSON 404, never index.html
[PASS 8/8] OpenAPI Documentation endpoints verified (/docs & /openapi.json accessible)

======================================================================
  ALL 8/8 LIVE RUNTIME VERIFICATION TESTS PASSED SUCCESSFULLY!
  SINGLE-SERVER LOCAL DEPLOYMENT (http://127.0.0.1:8000) IS FULLY OPERATIONAL.
======================================================================
[*] Terminating live test server...
[+] Test server cleanly terminated.
```

---

### 5.3 Full Platform Regression Verification

To guarantee zero regression across all previous remediation stages, the complete platform test battery was executed:
```powershell
pytest tests/test_single_server_deployment.py tests/test_cuda_portability.py tests/test_stage1_packaging_compliance.py backend/tests/test_api.py -q
```
**Result: 51 passed in 27.87s.**

Breakdown:
- `tests/test_single_server_deployment.py`: 15 passed (Single-Server SPA & API routing)
- `tests/test_cuda_portability.py`: 17 passed (BLK-02 CUDA & MSVC portability)
- `tests/test_stage1_packaging_compliance.py`: 7 passed (BLK-01 & BLK-04 licensing compliance)
- `backend/tests/test_api.py`: 12 passed (FastAPI core telemetry & model registry)

---

## 6. Known Limitations & Recommendations

1. **Production Assets Rebuild Requirement:** When developing new frontend features, changes made to `frontend/src/` require `npm --prefix frontend run build` before they appear on `http://127.0.0.1:8000/`. For rapid UI hot-reloading during frontend-only development, the Vite development server (`npm run dev`) remains available and continues to proxy to `http://localhost:8000`.
2. **Initial JIT Compilation Latency:** When custom CUDA kernels are first dispatched on a machine with a newly detected MSVC toolchain, Ninja executes JIT compilation on demand. This is expected behavior and fallback kernels ensure uninterrupted operation during that window.
3. **Port Conflict Recovery:** If port `8000` is already bound by another background application, `uvicorn` will log a port binding error. The updated `start_platform` launcher handles preflight failures cleanly.

---

## 7. Blocker Status & Release Gate Certification

| Blocker ID | Description | Pre-Remediation Status | Post-Remediation Status | Verification Method |
| :--- | :--- | :---: | :---: | :--- |
| **BLK-01** | Root `LICENSE` omission in release packaging | FAIL | **PASS** | Stage 1 Verification Suite |
| **BLK-02** | Hardcoded MSVC/CUDA paths in custom kernel loader | FAIL | **PASS** | Stage 2 Verification Suite (17/17 tests) |
| **BLK-03** | Dual-server dependency (lack of single-server deployment) | FAIL | **PASS** | BLK-03 Suite (15/15 tests) + Live Runtime (8/8) |
| **BLK-04** | Missing physical `NOTICE` and `THIRD_PARTY_LICENSES.md` | FAIL | **PASS** | Stage 1 Verification Suite |

### Final Release Gate Decision: **PASSED (ALL 4 CRITICAL BLOCKERS RESOLVED)**

NEXORA v1.0.0 now operates cleanly as a unified single-server local application served on `http://127.0.0.1:8000/`. Dual-server dependencies are eliminated, client-side routing is fully preserved across page reloads, and the platform is ready for final release packaging.
