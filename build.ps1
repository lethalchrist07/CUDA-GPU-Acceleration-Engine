# build.ps1 - CUDA GPU Acceleration Engine Build & Benchmark Pipeline
param (
    [string]$Target = "benchmark" # Options: "benchmark", "check", "all"
)

$ErrorActionPreference = "Stop"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  CUDA GPU Acceleration Engine - Build Pipeline" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# 1. Environment Configuration
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
$env:CUDA_PATH = $cudaPath
$env:CUDA_PATH_V13_4 = $cudaPath
$env:PATH = "$cudaPath\bin;$env:PATH"

$cmakePath = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
if (Test-Path "$cmakePath\cmake.exe") {
    $env:PATH = "$cmakePath;$env:PATH"
}

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "MSVC vcvars64.bat not found at $vcvars"
}

$projectDir = "C:\Projects\CUDA-GPU-Acceleration-Engine"
$buildDir = Join-Path $projectDir "build"
$binDir = Join-Path $projectDir "bin"

if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
}

# 2. Build via CMake
Write-Host "`n[*] Configuring CMake & Building Release targets..." -ForegroundColor Yellow
& "$cmakePath\cmake.exe" -S $projectDir -B $buildDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] CMake configuration failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

& "$cmakePath\cmake.exe" --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] CMake build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[+] Build Successful!" -ForegroundColor Green

# 3. Execution
if ($Target -eq "check") {
    $exe = Join-Path $binDir "gpu_check.exe"
    Write-Host "`n[*] Executing GPU Diagnostics ($exe)...`n" -ForegroundColor Yellow
    & $exe
} else {
    $exe = Join-Path $binDir "benchmark_suite.exe"
    Write-Host "`n[*] Launching Benchmark Suite ($exe)...`n" -ForegroundColor Yellow
    & $exe
}

exit $LASTEXITCODE
