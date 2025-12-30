@echo off
setlocal enabledelayedexpansion
title VEILBREAKERS Monster Rigger v3.0
color 0A

:HEADER
cls
echo.
echo  ============================================================================
echo  ^|                                                                          ^|
echo  ^|     ##   ## ###### #### ##      ####  ####  ######  ####  ##  ## ######  ^|
echo  ^|     ##   ## ##      ##  ##      ##  ## ##  ## ##     ##  ## ## ## ##      ^|
echo  ^|      ## ##  ####    ##  ##      ####  ####  ####   ###### ####  ######   ^|
echo  ^|       ###   ##      ##  ##      ##  ## ## ##  ##     ##  ## ## ##     ##  ^|
echo  ^|        #    ###### #### ###### ####  ##  ## ###### ##  ## ##  ## ######  ^|
echo  ^|                                                                          ^|
echo  ^|                    MONSTER RIGGER v3.0 - AI POWERED                      ^|
echo  ============================================================================
echo.

:CHECK_PYTHON
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo       Python %PYVER% found.

:CHECK_DEPS
echo [2/4] Checking core dependencies...
python -c "import torch; print(f'       PyTorch {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo [ERROR] PyTorch not installed! Run: pip install torch
    pause
    exit /b 1
)

python -c "import gradio; print(f'       Gradio {gradio.__version__}')" 2>nul
if errorlevel 1 (
    echo [ERROR] Gradio not installed! Run: pip install gradio
    pause
    exit /b 1
)

python -c "import transformers; print(f'       Transformers {transformers.__version__}')" 2>nul
if errorlevel 1 (
    echo [ERROR] Transformers not installed! Run: pip install transformers
    pause
    exit /b 1
)

:CHECK_FLORENCE
echo [3/4] Checking Florence-2 model access...
python -c "from transformers import AutoProcessor; print('       Florence-2 ready')" 2>nul
if errorlevel 1 (
    echo [WARNING] Florence-2 may need to download on first use (~4GB)
)

:CHECK_FILES
echo [4/4] Checking rigger files...
if not exist "%~dp0veilbreakers_rigger.py" (
    echo [ERROR] veilbreakers_rigger.py not found!
    pause
    exit /b 1
)
if not exist "%~dp0working_ui.py" (
    echo [ERROR] working_ui.py not found!
    pause
    exit /b 1
)
if not exist "%~dp0active_learning.py" (
    echo [ERROR] active_learning.py not found!
    pause
    exit /b 1
)
echo       All files present.
echo.
echo  ============================================================================
echo       ALL CHECKS PASSED - SYSTEM READY
echo  ============================================================================
echo.

:MENU
echo  Choose an option:
echo.
echo    [1] MONSTER RIGGER         - Rig monsters with AI detection
echo    [2] ACTIVE LEARNING        - Correct AI + Train (RECOMMENDED)
echo    [3] MANUAL LABELING        - Label training data by hand
echo    [4] TRAIN MODEL            - Train with existing labels
echo    [5] RUN QUICK TEST         - Verify everything works
echo.
echo    [Q] QUIT
echo.
set /p choice="  Enter choice (1-5 or Q): "

if /i "%choice%"=="1" goto RIGGER
if /i "%choice%"=="2" goto ACTIVE
if /i "%choice%"=="3" goto LABEL
if /i "%choice%"=="4" goto TRAIN
if /i "%choice%"=="5" goto TEST
if /i "%choice%"=="Q" goto END
echo  Invalid choice. Try again.
timeout /t 2 >nul
goto HEADER

:RIGGER
cls
echo.
echo  ============================================================================
echo                        LAUNCHING MONSTER RIGGER
echo  ============================================================================
echo.
echo  Starting Gradio UI on http://127.0.0.1:7860
echo.
echo  Tips:
echo    - Upload a monster PNG image
echo    - Click "Smart Detect (Florence-2)" to find body parts
echo    - Adjust boxes as needed
echo    - Export your rig!
echo.
echo  Press Ctrl+C to stop the server.
echo  ============================================================================
echo.
cd /d "%~dp0"
python working_ui.py
if errorlevel 1 (
    echo.
    echo [ERROR] Rigger crashed! Check the error above.
    pause
)
goto HEADER

:ACTIVE
cls
echo.
echo  ============================================================================
echo                       LAUNCHING ACTIVE LEARNING
echo  ============================================================================
echo.
echo  This is the BEST way to improve the AI!
echo.
echo  Workflow:
echo    1. Upload image - AI detects body parts
echo    2. CORRECT mistakes (add missing, fix wrong boxes)
echo    3. Click SAVE - correction stored as training data
echo    4. After 10+ corrections, click TRAIN
echo    5. AI learns from YOUR expertise!
echo.
echo  Press Ctrl+C to stop.
echo  ============================================================================
echo.
cd /d "%~dp0"
python active_learning.py
if errorlevel 1 (
    echo.
    echo [ERROR] Active Learning crashed! Check the error above.
    pause
)
goto HEADER

:LABEL
cls
echo.
echo  ============================================================================
echo                        LAUNCHING MANUAL LABELER
echo  ============================================================================
echo.
cd /d "%~dp0"
if exist "%~dp0label_training_data.py" (
    python label_training_data.py
) else (
    echo [ERROR] label_training_data.py not found!
    pause
)
goto HEADER

:TRAIN
cls
echo.
echo  ============================================================================
echo                         TRAINING FLORENCE-2
echo  ============================================================================
echo.
echo  This will train the model with your labeled data.
echo  Training can take 10-60 minutes depending on data size.
echo.
cd /d "%~dp0"

REM Check for training data
if not exist "%~dp0training_data\labels.json" (
    echo [ERROR] No training data found!
    echo        Use Active Learning or Manual Labeling first.
    pause
    goto HEADER
)

REM Count samples
python -c "import json; data=json.load(open('training_data/labels.json')); print(f'Found {len(data)} training samples')"
echo.
set /p confirm="  Start training? (Y/N): "
if /i not "%confirm%"=="Y" goto HEADER

if exist "%~dp0train_florence2_pro.py" (
    echo.
    echo  Using PROFESSIONAL training (LoRA + augmentation)...
    python train_florence2_pro.py
) else if exist "%~dp0train_florence2.py" (
    echo.
    echo  Using basic training...
    python train_florence2.py
) else (
    echo [ERROR] No training script found!
)
pause
goto HEADER

:TEST
cls
echo.
echo  ============================================================================
echo                           RUNNING QUICK TEST
echo  ============================================================================
echo.
cd /d "%~dp0"
python -c "
import sys
print('Testing imports...')
try:
    print('  [1/6] torch...', end=' ')
    import torch
    print(f'OK ({torch.__version__})')

    print('  [2/6] transformers...', end=' ')
    import transformers
    print(f'OK ({transformers.__version__})')

    print('  [3/6] gradio...', end=' ')
    import gradio
    print(f'OK ({gradio.__version__})')

    print('  [4/6] PIL...', end=' ')
    from PIL import Image
    print('OK')

    print('  [5/6] numpy...', end=' ')
    import numpy
    print(f'OK ({numpy.__version__})')

    print('  [6/6] veilbreakers_rigger...', end=' ')
    from veilbreakers_rigger import VeilbreakersRigger
    print('OK')

    print()
    print('Testing Florence-2 model loading...')
    rigger = VeilbreakersRigger()
    print('  Loading Florence-2 (this may take a minute first time)...')

    # Try to load model
    from transformers import AutoProcessor, AutoModelForCausalLM
    from pathlib import Path

    finetuned = Path('florence2_finetuned/final')
    if finetuned.exists():
        model_id = str(finetuned)
        print(f'  Using FINE-TUNED model from {model_id}')
    else:
        model_id = 'microsoft/Florence-2-large'
        print(f'  Using base model: {model_id}')

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation='eager'
    )
    print('  Model loaded successfully!')

    print()
    print('=' * 60)
    print('  ALL TESTS PASSED - SYSTEM IS READY!')
    print('=' * 60)

except Exception as e:
    print(f'FAILED!')
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
echo.
pause
goto HEADER

:END
echo.
echo  Thanks for using VEILBREAKERS Monster Rigger!
echo.
timeout /t 2 >nul
exit /b 0
