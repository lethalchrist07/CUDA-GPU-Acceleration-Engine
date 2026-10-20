# scripts/activate_cuda124.ps1 - Process-Scoped Environment Activator
# Sets process-local variables for CUDA 12.4 + MSVC 14.38 without modifying global system environment.

$cudaDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if (-not (Test-Path $cudaDir)) {
    $cudaDir = Join-Path $env:TEMP "cuda_12.4_staging"
}

$msvcDir = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.38.33130"
$msvcBin = Join-Path $msvcDir "bin\Hostx64\x64"
$vcvarsBat = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if (Test-Path $vcvarsBat) {
    Write-Host "[*] Extracting MSVC 14.38 environment..." -ForegroundColor Cyan
    cmd /c "`"$vcvarsBat`" -vcvars_ver=14.38 >nul && set" | ForEach-Object {
        if ($_ -match "^(INCLUDE|LIB|LIBPATH)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

$env:CUDA_HOME = $cudaDir
$env:CUDA_PATH = $cudaDir
$env:VCToolsInstallDir = "$msvcDir\"
$env:PATH = "$cudaDir\bin;$msvcBin;$env:PATH"

Write-Host "[+] Activated process-local CUDA 12.4 + MSVC 14.38 environment:" -ForegroundColor Green
Write-Host "    CUDA_HOME:          $env:CUDA_HOME"
Write-Host "    VCToolsInstallDir:  $env:VCToolsInstallDir"
Write-Host "    NVCC Version:       $(& "$cudaDir\bin\nvcc.exe" --version | Select-String 'release')"
