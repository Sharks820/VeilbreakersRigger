@echo off
title VEILBREAKERS Monster Rigger
cd /d "%~dp0"

echo.
echo ========================================================================
echo                     VEILBREAKERS MONSTER RIGGER
echo                        Launching UI...
echo ========================================================================
echo.

python run.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to launch. Make sure Python is installed.
    echo.
    pause
)
