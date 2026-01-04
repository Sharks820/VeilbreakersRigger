@echo off
title VEILBREAKERS Monster Rigger v4.0
cd /d "%~dp0"

echo.
echo ========================================================================
echo                     VEILBREAKERS MONSTER RIGGER v4.0
echo                   Powered by Florence-2 PRO + CUDA
echo ========================================================================
echo.

:: Use the Python 3.12 virtual environment with CUDA support
if exist "venv312\Scripts\python.exe" (
    echo Using Python 3.12 venv with GPU acceleration...
    venv312\Scripts\python.exe run.py
) else (
    echo WARNING: venv312 not found, falling back to system Python
    python run.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to launch.
    echo Make sure you have installed dependencies:
    echo   venv312\Scripts\pip.exe install -r requirements.txt
    echo.
    pause
)
