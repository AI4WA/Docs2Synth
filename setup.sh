#!/bin/bash
# Setup script for Unix-based systems (macOS, Linux, WSL)
# Usage: ./setup.sh [method] [--gpu]
# Methods: venv, conda (default: conda)
# Options: --gpu (install GPU-enabled PyTorch)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
SETUP_METHOD="conda"
INSTALL_GPU=false

for arg in "$@"; do
    case $arg in
        venv|conda)
            SETUP_METHOD=$arg
            ;;
        --gpu)
            INSTALL_GPU=true
            ;;
        *)
            echo_error "Unknown argument: $arg"
            echo_info "Usage: ./setup.sh [venv|conda] [--gpu]"
            exit 1
            ;;
    esac
done

echo_info "Starting Docs2Synth development environment setup..."
echo_info "Setup method: $SETUP_METHOD"

# Detect GPU availability
detect_gpu() {
    echo_info "Detecting GPU availability..."

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo_info "NVIDIA GPU detected!"
            return 0
        fi
    fi

    # Check for AMD GPU on Linux
    if [ -d "/sys/class/drm" ] && ls /sys/class/drm/ | grep -q "^card[0-9]"; then
        if lspci 2>/dev/null | grep -i "vga\|3d\|display" | grep -iq "amd\|radeon"; then
            echo_warn "AMD GPU detected, but PyTorch GPU support is optimized for NVIDIA CUDA."
            echo_warn "Defaulting to CPU installation. For AMD GPU, consider using ROCm separately."
            return 1
        fi
    fi

    echo_info "No compatible GPU detected. Installing CPU version."
    return 1
}

# Auto-detect GPU if not manually specified
if [ "$INSTALL_GPU" = false ]; then
    if detect_gpu; then
        read -p "GPU detected. Install GPU-enabled PyTorch? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            INSTALL_GPU=true
        fi
    fi
fi

if [ "$INSTALL_GPU" = true ]; then
    echo_info "Will install GPU-enabled PyTorch (CUDA 11.8)"
    TORCH_REQUIREMENTS="requirements-gpu.txt"
else
    echo_info "Will install CPU-only PyTorch"
    TORCH_REQUIREMENTS="requirements-cpu.txt"
fi

# Check Python version
check_python_version() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            echo_info "Python $PYTHON_VERSION detected (minimum 3.8 required)"
            return 0
        else
            echo_error "Python 3.8+ required, but found $PYTHON_VERSION"
            return 1
        fi
    else
        echo_error "Python 3 not found. Please install Python 3.8 or higher."
        return 1
    fi
}

# Setup using venv
setup_venv() {
    echo_info "Setting up with Python venv..."

    if ! check_python_version; then
        exit 1
    fi

    # Create virtual environment
    if [ ! -d ".venv" ]; then
        echo_info "Creating virtual environment..."
        python3 -m venv .venv
    else
        echo_warn "Virtual environment already exists, skipping creation..."
    fi

    # Activate virtual environment
    echo_info "Activating virtual environment..."
    source .venv/bin/activate

    # Upgrade pip
    echo_info "Upgrading pip..."
    pip install --upgrade pip

    # Install PyTorch (CPU or GPU)
    echo_info "Installing PyTorch..."
    pip install -r "$TORCH_REQUIREMENTS"

    # Install base requirements
    echo_info "Installing base requirements..."
    pip install -r requirements.txt

    # Install development dependencies
    echo_info "Installing development dependencies..."
    pip install -r requirements-dev.txt

    # Install the package in editable mode
    echo_info "Installing docs2synth in editable mode..."
    pip install -e .

    echo_info ""
    echo_info "=========================================="
    echo_info "Setup complete!"
    if [ "$INSTALL_GPU" = true ]; then
        echo_info "GPU-enabled PyTorch installed."
    else
        echo_info "CPU-only PyTorch installed."
    fi
    echo_info "To activate the environment, run:"
    echo_info "  source .venv/bin/activate"
    echo_info "=========================================="
}

# Setup using conda
setup_conda() {
    echo_info "Setting up with Conda..."

    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        echo_error "Conda not found. Please install Miniconda or Anaconda first."
        echo_info "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    # Check if environment already exists
    if conda env list | grep -q "^Docs2Synth "; then
        echo_warn "Conda environment 'Docs2Synth' already exists."
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo_info "Removing existing environment..."
            conda env remove -n Docs2Synth -y
        else
            echo_info "Updating existing environment..."
            conda env update -f environment.yml --prune

            # Install PyTorch based on GPU availability
            echo_info "Installing PyTorch..."
            conda run -n Docs2Synth pip install -r "$TORCH_REQUIREMENTS"

            # Install requirements
            conda run -n Docs2Synth pip install -r requirements.txt
            conda run -n Docs2Synth pip install -r requirements-dev.txt
            conda run -n Docs2Synth pip install -e .

            echo_info ""
            echo_info "=========================================="
            echo_info "Setup complete!"
            if [ "$INSTALL_GPU" = true ]; then
                echo_info "GPU-enabled PyTorch installed."
            else
                echo_info "CPU-only PyTorch installed."
            fi
            echo_info "To activate the environment, run:"
            echo_info "  conda activate Docs2Synth"
            echo_info "=========================================="
            return 0
        fi
    fi

    # Create conda environment
    echo_info "Creating conda environment from environment.yml..."
    conda env create -f environment.yml

    # Install PyTorch based on GPU availability
    echo_info "Installing PyTorch..."
    conda run -n Docs2Synth pip install -r "$TORCH_REQUIREMENTS"

    # Install requirements
    echo_info "Installing requirements..."
    conda run -n Docs2Synth pip install -r requirements.txt
    conda run -n Docs2Synth pip install -r requirements-dev.txt

    # Install package
    echo_info "Installing docs2synth package..."
    conda run -n Docs2Synth pip install -e .

    echo_info ""
    echo_info "=========================================="
    echo_info "Setup complete!"
    if [ "$INSTALL_GPU" = true ]; then
        echo_info "GPU-enabled PyTorch installed."
    else
        echo_info "CPU-only PyTorch installed."
    fi
    echo_info "To activate the environment, run:"
    echo_info "  conda activate Docs2Synth"
    echo_info "=========================================="
}

# Main setup logic
case $SETUP_METHOD in
    venv)
        setup_venv
        ;;
    conda)
        setup_conda
        ;;
    *)
        echo_error "Unknown setup method: $SETUP_METHOD"
        echo_info "Usage: ./setup.sh [venv|conda] [--gpu]"
        exit 1
        ;;
esac

# Verify installation
echo_info ""
echo_info "Verifying installation..."
if [ "$SETUP_METHOD" == "venv" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
else
    eval "$(conda shell.bash hook)"
    conda activate Docs2Synth
fi

if python -c "import docs2synth" 2>/dev/null; then
    echo_info "Package 'docs2synth' successfully installed!"
else
    echo_warn "Package import test failed. Please check the installation."
fi

# Verify PyTorch installation
echo_info ""
echo_info "Verifying PyTorch installation..."
if python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    if [ "$INSTALL_GPU" = true ]; then
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            echo_info "GPU support verified successfully!"
        else
            echo_warn "GPU was requested but CUDA is not available. Please check your CUDA installation."
        fi
    fi
else
    echo_warn "PyTorch verification failed. Please check the installation."
fi

echo_info ""
echo_info "To run code quality checks, use:"
echo_info "  ./scripts/check.sh"
