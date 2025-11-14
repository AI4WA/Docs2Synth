@echo off
REM Setup script for Windows
REM Usage: setup.bat [--gpu|--cpu]
REM Options: --gpu (force GPU-enabled PyTorch), --cpu (force CPU-only PyTorch)

setlocal enabledelayedexpansion

SET INSTALL_GPU=

REM Parse arguments
:parse_args
IF "%~1"=="" GOTO args_done
IF /I "%~1"=="--gpu" (
    SET INSTALL_GPU=true
    SHIFT
    GOTO parse_args
)
IF /I "%~1"=="--cpu" (
    SET INSTALL_GPU=false
    SHIFT
    GOTO parse_args
)
echo [ERROR] Unknown argument: %~1
echo [INFO] Usage: setup.bat [--gpu|--cpu]
exit /b 1

:args_done

echo [INFO] Starting Docs2Synth development environment setup...
echo [INFO] Setup method: uv with Python venv

REM Check if uv is installed
where uv >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] uv is not installed. Installing uv...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    where uv >nul 2>nul
    IF !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to install uv. Please install manually: https://github.com/astral-sh/uv
        exit /b 1
    )
    echo [INFO] uv installed successfully!
) ELSE (
    echo [INFO] uv is already installed
)

REM Check if Python is installed
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.11 or higher.
    echo [INFO] Visit: https://www.python.org/downloads/
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python %PYTHON_VERSION% detected

REM Detect GPU if not manually specified
IF "!INSTALL_GPU!"=="" (
    echo [INFO] Detecting GPU availability...
    where nvidia-smi >nul 2>nul
    IF %ERRORLEVEL% EQU 0 (
        nvidia-smi >nul 2>nul
        IF !ERRORLEVEL! EQU 0 (
            echo [INFO] NVIDIA GPU detected!
            set /p GPU_CHOICE="Install GPU-enabled PyTorch? (Y/n): "
            IF /I "!GPU_CHOICE!"=="" SET GPU_CHOICE=Y
            IF /I "!GPU_CHOICE!"=="Y" (
                SET INSTALL_GPU=true
            ) ELSE (
                SET INSTALL_GPU=false
            )
        ) ELSE (
            SET INSTALL_GPU=false
        )
    ) ELSE (
        echo [INFO] No compatible GPU detected. Installing CPU version.
        SET INSTALL_GPU=false
    )
)

IF "!INSTALL_GPU!"=="true" (
    echo [INFO] Will install GPU-enabled PyTorch (CUDA 12.8^)
) ELSE (
    echo [INFO] Will install CPU-only PyTorch
)

call :setup_venv
goto :end

:setup_venv
echo [INFO] Setting up with uv and Python venv...

REM Check if virtual environment exists
IF NOT EXIST ".venv" (
    echo [INFO] Creating virtual environment with uv...
    uv venv
) ELSE (
    echo [WARN] Virtual environment already exists, skipping creation...
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install package with appropriate extras
IF "!INSTALL_GPU!"=="true" (
    echo [INFO] Installing GPU-enabled PyTorch (CUDA 12.8^)...
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    echo [INFO] Installing Docs2Synth with GPU extras and dev tools...
    uv pip install -e ".[gpu,dev]"
) ELSE (
    echo [INFO] Installing Docs2Synth with CPU extras and dev tools...
    uv pip install -e ".[cpu,dev]"
)

REM Uninstall paddlex if it was installed
echo [INFO] Uninstalling paddlex...
uv pip uninstall -y paddlex >nul 2>nul

echo.
echo ==========================================
echo Setup complete!
IF "!INSTALL_GPU!"=="true" (
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
    IF "!INSTALL_GPU!"=="true" (
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
