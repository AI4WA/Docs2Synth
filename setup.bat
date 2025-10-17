@echo off
REM Setup script for Windows
REM Usage: setup.bat [--gpu]
REM Options: --gpu (install GPU-enabled PyTorch)

setlocal enabledelayedexpansion

SET INSTALL_GPU=false

REM Parse arguments
:parse_args
IF "%~1"=="" GOTO args_done
IF /I "%~1"=="--gpu" (
    SET INSTALL_GPU=true
    SHIFT
    GOTO parse_args
)
echo [ERROR] Unknown argument: %~1
echo [INFO] Usage: setup.bat [--gpu]
exit /b 1

:args_done

echo [INFO] Starting Docs2Synth development environment setup...
echo [INFO] Setup method: Python venv

REM Check if Python is installed
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    echo [INFO] Visit: https://www.python.org/downloads/
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python %PYTHON_VERSION% detected

REM Detect GPU if not manually specified
IF "%INSTALL_GPU%"=="false" (
    echo [INFO] Detecting GPU availability...
    where nvidia-smi >nul 2>nul
    IF %ERRORLEVEL% EQU 0 (
        nvidia-smi >nul 2>nul
        IF !ERRORLEVEL! EQU 0 (
            echo [INFO] NVIDIA GPU detected!
            set /p GPU_CHOICE="Install GPU-enabled PyTorch? (Y/n): "
            IF /I "!GPU_CHOICE!"=="" SET GPU_CHOICE=Y
            IF /I "!GPU_CHOICE!"=="Y" SET INSTALL_GPU=true
        )
    )
)

IF "%INSTALL_GPU%"=="false" (
    echo [INFO] No compatible GPU detected. Installing CPU version.
)

IF "%INSTALL_GPU%"=="true" (
    echo [INFO] Will install GPU-enabled PyTorch (CUDA 11.8^)
    SET TORCH_REQUIREMENTS=requirements-gpu.txt
) ELSE (
    echo [INFO] Will install CPU-only PyTorch
    SET TORCH_REQUIREMENTS=requirements-cpu.txt
)

call :setup_venv
goto :end

:setup_venv
echo [INFO] Setting up with Python venv...

REM Check if virtual environment exists
IF NOT EXIST ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
) ELSE (
    echo [WARN] Virtual environment already exists, skipping creation...
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CPU or GPU)
echo [INFO] Installing PyTorch...
pip install -r %TORCH_REQUIREMENTS%

REM Install base requirements
echo [INFO] Installing base requirements...
pip install -r requirements.txt

REM Install development dependencies
echo [INFO] Installing development dependencies...
pip install -r requirements-dev.txt

REM Install the package in editable mode
echo [INFO] Installing docs2synth in editable mode...
pip install -e .

echo.
echo ==========================================
echo Setup complete!
IF "%INSTALL_GPU%"=="true" (
    echo GPU-enabled PyTorch installed.
) ELSE (
    echo CPU-only PyTorch installed.
)
echo To activate the environment, run:
echo   .venv\Scripts\activate.bat
echo ==========================================

REM Verify installation
echo.
echo [INFO] Verifying installation...
python -c "import docs2synth" 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo [INFO] Package 'docs2synth' successfully installed!
) ELSE (
    echo [WARN] Package import test failed. Please check the installation.
)

REM Verify PyTorch installation
echo.
echo [INFO] Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
IF %ERRORLEVEL% EQU 0 (
    IF "%INSTALL_GPU%"=="true" (
        python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
        IF !ERRORLEVEL! EQU 0 (
            echo [INFO] GPU support verified successfully!
        ) ELSE (
            echo [WARN] GPU was requested but CUDA is not available. Please check your CUDA installation.
        )
    )
) ELSE (
    echo [WARN] PyTorch verification failed. Please check the installation.
)

goto :eof

:end
echo.
echo [INFO] To run code quality checks, use:
echo   scripts\check.sh (in Git Bash/WSL) or run individual commands
endlocal
