@echo off
REM Setup script for Windows
REM Usage: setup.bat [method]
REM Methods: venv, conda (default: conda)

setlocal enabledelayedexpansion

SET SETUP_METHOD=%1
IF "%SETUP_METHOD%"=="" SET SETUP_METHOD=conda

echo [INFO] Starting Docs2Synth development environment setup...
echo [INFO] Setup method: %SETUP_METHOD%

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

IF "%SETUP_METHOD%"=="venv" (
    call :setup_venv
) ELSE IF "%SETUP_METHOD%"=="conda" (
    call :setup_conda
) ELSE (
    echo [ERROR] Unknown setup method: %SETUP_METHOD%
    echo [INFO] Usage: setup.bat [venv^|conda]
    exit /b 1
)

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

REM Install the package with all dependencies
echo [INFO] Installing docs2synth with development dependencies...
pip install -e ".[dev,datasets,qa,retriever]"

echo.
echo ==========================================
echo Setup complete!
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

goto :eof

:setup_conda
echo [INFO] Setting up with Conda...

REM Check if conda is installed
where conda >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found. Please install Miniconda or Anaconda first.
    echo [INFO] Visit: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr /C:"Docs2Synth" >nul
IF %ERRORLEVEL% EQU 0 (
    echo [WARN] Conda environment 'Docs2Synth' already exists.
    set /p RECREATE="Do you want to remove and recreate it? (y/N): "
    IF /I "!RECREATE!"=="y" (
        echo [INFO] Removing existing environment...
        call conda env remove -n Docs2Synth -y
    ) ELSE (
        echo [INFO] Updating existing environment...
        call conda env update -f environment.yml --prune
        echo.
        echo ==========================================
        echo Setup complete!
        echo To activate the environment, run:
        echo   conda activate Docs2Synth
        echo ==========================================
        goto :verify_conda
    )
)

REM Create conda environment
echo [INFO] Creating conda environment from environment.yml...
call conda env create -f environment.yml

REM Install package
echo [INFO] Installing docs2synth package...
call conda run -n Docs2Synth pip install -e ".[dev,datasets,qa,retriever]"

echo.
echo ==========================================
echo Setup complete!
echo To activate the environment, run:
echo   conda activate Docs2Synth
echo ==========================================

:verify_conda
REM Verify installation
echo.
echo [INFO] Verifying installation...
call conda run -n Docs2Synth python -c "import docs2synth" 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo [INFO] Package 'docs2synth' successfully installed!
) ELSE (
    echo [WARN] Package import test failed. Please check the installation.
)

goto :eof

:end
echo.
echo [INFO] To run code quality checks, use:
echo   scripts\check.sh (in Git Bash/WSL) or run individual commands
endlocal
