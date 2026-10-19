@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv-phase3\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m venv .venv-phase3
    if errorlevel 1 exit /b 1
)
".venv-phase3\Scripts\python.exe" -m pip install -r backend\requirements.txt --disable-pip-version-check
exit /b %errorlevel%
