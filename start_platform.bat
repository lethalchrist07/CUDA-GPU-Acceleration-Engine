@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo   CUDA GPU ACCELERATION ENGINE — PLATFORM LAUNCHER
echo   Target: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere SM 8.6)
echo ======================================================================

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: Locate Python in virtual environment
if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
) else if exist "%PROJECT_ROOT%.venv-phase3\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%.venv-phase3\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [*] Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" -c "import torch; print('[+] PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
if errorlevel 1 (
    echo [-] Python environment check failed. Please ensure dependencies are installed via requirements.txt.
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

echo [*] Starting Single-Server Production Daemon on http://127.0.0.1:8000...
start "CUDA Acceleration Engine - Single Server" cmd /c "cd /d %PROJECT_ROOT% && set PYTHONPATH=%PROJECT_ROOT% && "%PYTHON_EXE%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

:: Brief wait for server port binding
timeout /t 3 /nobreak >nul

echo ======================================================================
echo [OK] Unified Single-Server Deployment Active:
echo   - Web Dashboard:  http://127.0.0.1:8000/
echo   - API Root:       http://127.0.0.1:8000/api/health
echo   - API Swagger:    http://127.0.0.1:8000/docs
echo ======================================================================

:: Open unified single-server entrypoint in default browser
start http://127.0.0.1:8000/

exit /b 0

