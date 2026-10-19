# scripts/run_tests.ps1 - Automated Test Runner
$ErrorActionPreference = "Stop"

$projectDir = "C:\Projects\CUDA-GPU-Acceleration-Engine"
$venvPython = Join-Path $projectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  CUDA AI Acceleration Engine - Automated Tests" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# 1. Run PyTest (Engine + Backend)
Write-Host "`n[*] Running PyTest test suites (Engine + Backend)..." -ForegroundColor Yellow
& $venvPython -m pytest "$projectDir\engine\tests" "$projectDir\backend\tests" -v
$pytestExit = $LASTEXITCODE

# 2. Run Verification Script
Write-Host "`n[*] Running End-to-End Foundation Verification..." -ForegroundColor Yellow
& $venvPython "$projectDir\scripts\verify_ai_foundation.py"
$verifyExit = $LASTEXITCODE

if ($pytestExit -eq 0 -and $verifyExit -eq 0) {
    Write-Host "`n[+] All tests and verifications PASSED." -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n[-] Test failures encountered." -ForegroundColor Red
    exit 1
}
