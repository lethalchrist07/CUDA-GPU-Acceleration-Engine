# start_platform.ps1 - CUDA GPU Acceleration Engine Platform Launcher
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  CUDA GPU ACCELERATION ENGINE — PLATFORM LAUNCHER" -ForegroundColor Cyan
Write-Host "  Target: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere SM 8.6)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Locate Python
$pythonExe = "$projectRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "$projectRoot\.venv-phase3\Scripts\python.exe"
}
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Host "[*] Checking Python environment: $pythonExe" -ForegroundColor Yellow
& $pythonExe -c "import torch; print('[+] PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Check if frontend production bundle is built
$distIndex = "$projectRoot\frontend\dist\index.html"
if (-not (Test-Path $distIndex)) {
    Write-Host "[*] Building frontend static bundle for production..." -ForegroundColor Yellow
    npm --prefix "$projectRoot\frontend" run build
}

# 1. Start Single-Server Production Service
Write-Host "[*] Starting Unified Single-Server Deployment on http://127.0.0.1:8000..." -ForegroundColor Yellow
$backendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c cd /d `"$projectRoot`" && set PYTHONPATH=`"$projectRoot`" && `"$pythonExe`" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000" -PassThru

Start-Sleep -Seconds 3

Write-Host "`n[OK] Unified Single-Server Platform Active:" -ForegroundColor Green
Write-Host "  - Web Dashboard:    http://127.0.0.1:8000/" -ForegroundColor Green
Write-Host "  - Backend API:      http://127.0.0.1:8000/api/health" -ForegroundColor Green
Write-Host "  - API Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor Green

Start-Process "http://127.0.0.1:8000/"

