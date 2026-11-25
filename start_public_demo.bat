@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo   NEXORA v1.0.0 — PUBLIC DEMO LAUNCHER (SECURE SANDBOX)
echo   Target: Cloudflare Quick Tunnel Demonstration ($0 Hosting Cost)
echo   Security: CUDA_ENGINE_DEMO_MODE=1 (Hardware Safeguards Active)
echo ======================================================================

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: Activate Demo Mode environment variables
set "CUDA_ENGINE_DEMO_MODE=1"
set "CUDA_ENGINE_HOST=127.0.0.1"
set "CUDA_ENGINE_PORT=8000"

:: Locate Python in virtual environment
if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
) else if exist "%PROJECT_ROOT%.venv-phase3\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%.venv-phase3\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [*] Using Python: %PYTHON_EXE%
echo [*] Public Demo Security Mode: ACTIVE (CUDA_ENGINE_DEMO_MODE=1)

"%PYTHON_EXE%" -c "import torch; print('[+] PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [-] Python environment check failed.
    pause
    exit /b 1
)

:: Ensure frontend static production bundle is built
if not exist "%PROJECT_ROOT%frontend\dist\index.html" (
    echo [*] Production frontend bundle not detected. Building static assets...
    call npm --prefix "%PROJECT_ROOT%frontend" run build
    if errorlevel 1 (
        echo [-] Frontend build failed!
        pause
        exit /b 1
    )
)

echo [*] Starting NEXORA Demo Single-Server Daemon on http://127.0.0.1:8000...
start "NEXORA - Public Demo Server (Secure Mode)" cmd /c "cd /d %PROJECT_ROOT% && set PYTHONPATH=%PROJECT_ROOT% && set CUDA_ENGINE_DEMO_MODE=1 && "%PYTHON_EXE%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

:: Brief wait for server port binding
timeout /t 3 /nobreak >nul

echo ======================================================================
echo [OK] NEXORA Public Demo Mode Active:
echo   - Local Web Dashboard: http://127.0.0.1:8000/
echo   - Health Check:        http://127.0.0.1:8000/api/health
echo   - Security Policy:     RESTRICTED (Public Demo Mode)
echo ======================================================================

start http://127.0.0.1:8000/

exit /b 0
