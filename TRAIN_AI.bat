@echo off
echo.
echo ================================================================
echo       VEILBREAKERS MONSTER AI TRAINING SYSTEM
echo ================================================================
echo.
echo Choose an option:
echo.
echo   1. ACTIVE LEARNING (Correct AI + Train)
echo   2. LABEL TRAINING DATA (Manual labeling)
echo   3. TRAIN MODEL (After labeling)
echo   4. RUN RIGGER (Use the AI)
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo Starting Active Learning UI...
    python active_learning.py
)
if "%choice%"=="2" (
    echo Starting Labeling Tool...
    python label_training_data.py
)
if "%choice%"=="3" (
    echo Starting Professional Training...
    python train_florence2_pro.py
)
if "%choice%"=="4" (
    echo Starting Rigger UI...
    python working_ui.py
)

pause
