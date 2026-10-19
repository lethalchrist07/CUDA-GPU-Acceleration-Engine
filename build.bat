@echo off
echo ===================================================
echo   CUDA Acceleration Engine - Build Script
echo ===================================================
echo [*] Initializing MSVC x64 build environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 goto :fail_env

set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"

echo [*] Building project via CMake (Release)...
cmake -S . -B build
if errorlevel 1 goto :fail_compile

cmake --build build --config Release
if errorlevel 1 goto :fail_compile

echo [+] Build SUCCESSFUL.
echo [*] Executing bin\benchmark_suite.exe on NVIDIA RTX 3050 Laptop GPU...
"bin\benchmark_suite.exe"
if errorlevel 1 goto :fail_run

echo [+] Benchmark execution completed successfully.
exit /b 0

:fail_env
echo [-] Failed to initialize MSVC x64 environment!
exit /b 1

:fail_compile
echo [-] Compilation FAILED!
exit /b 1

:fail_run
echo [-] Benchmark execution FAILED!
exit /b 1
