@echo off
setlocal
cd /d C:\Projects\CUDA-GPU-Acceleration-Engine
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cmake -S . -B build/phase2-ninja -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_ARCHITECTURES=86 -D BUILD_TESTING=ON
if errorlevel 1 exit /b 1
cmake --build build/phase2-ninja --target benchmark_suite benchmark_regression --verbose -j 2
exit /b %errorlevel%
