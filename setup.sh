#!/bin/bash
# Setup script for Unix-based systems (macOS, Linux, WSL)
# Usage: ./setup.sh [method]
# Methods: venv, conda (default: conda)

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

# Detect the setup method
SETUP_METHOD=${1:-conda}

echo_info "Starting Docs2Synth development environment setup..."
echo_info "Setup method: $SETUP_METHOD"

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

    # Install the package with all dependencies
    echo_info "Installing docs2synth with development dependencies..."
    pip install -e ".[dev,datasets,qa,retriever]"

    echo_info ""
    echo_info "=========================================="
    echo_info "Setup complete!"
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
            echo_info ""
            echo_info "=========================================="
            echo_info "Setup complete!"
            echo_info "To activate the environment, run:"
            echo_info "  conda activate Docs2Synth"
            echo_info "=========================================="
            return 0
        fi
    fi

    # Create conda environment
    echo_info "Creating conda environment from environment.yml..."
    conda env create -f environment.yml

    # Activate and install package
    echo_info "Installing docs2synth package..."
    conda run -n Docs2Synth pip install -e ".[dev,datasets,qa,retriever]"

    echo_info ""
    echo_info "=========================================="
    echo_info "Setup complete!"
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
        echo_info "Usage: ./setup.sh [venv|conda]"
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

echo_info ""
echo_info "To run code quality checks, use:"
echo_info "  ./scripts/check.sh"
