#!/bin/bash
# Setup script for Unix-based systems (macOS, Linux, WSL)
# Usage: ./setup.sh [--gpu]
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
INSTALL_GPU=false

for arg in "$@"; do
    case $arg in
        --gpu)
            INSTALL_GPU=true
            ;;
        *)
            echo_error "Unknown argument: $arg"
            echo_info "Usage: ./setup.sh [--gpu]"
            exit 1
            ;;
    esac
done

echo_info "Starting Docs2Synth development environment setup..."
echo_info "Setup method: uv with Python venv"

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo_warn "uv is not installed. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            echo_error "Failed to install uv. Please install manually: https://github.com/astral-sh/uv"
            exit 1
        fi
        echo_info "uv installed successfully!"
    else
        echo_info "uv is already installed"
    fi
}

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
else
    echo_info "Will install CPU-only PyTorch"
fi

# Check Python version
check_python_version() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            echo_info "Python $PYTHON_VERSION detected (minimum 3.11 required)"
            return 0
        else
            echo_error "Python 3.11+ required, but found $PYTHON_VERSION"
            return 1
        fi
    else
        echo_error "Python 3 not found. Please install Python 3.11 or higher."
        return 1
    fi
}

# Setup using uv and venv
setup_venv() {
    echo_info "Setting up with uv and Python venv..."

    if ! check_python_version; then
        exit 1
    fi

    # Check and install uv if needed
    check_uv

    # Create virtual environment using uv
    if [ ! -d ".venv" ]; then
        echo_info "Creating virtual environment with uv..."
        uv venv
    else
        echo_warn "Virtual environment already exists, skipping creation..."
    fi

    # Activate virtual environment
    echo_info "Activating virtual environment..."
    source .venv/bin/activate

    # Install PyTorch (CPU or GPU)
    echo_info "Installing PyTorch with uv..."
    if [ "$INSTALL_GPU" = true ]; then
        uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
    else
        uv pip install -r requirements-cpu.txt
    fi

    # Install the package in editable mode with dev dependencies
    echo_info "Installing docs2synth with dev dependencies..."
    uv pip install -e ".[dev]"

    # Uninstall paddlex if it was installed
    echo_info "Uninstalling paddlex..."
    uv pip uninstall paddlex

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

# Main setup
setup_venv

# Verify installation
echo_info ""
echo_info "Verifying installation..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
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
