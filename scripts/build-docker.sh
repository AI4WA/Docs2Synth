#!/bin/bash
# Docker build and test script for Docs2Synth
# Usage: ./scripts/build-docker.sh [cpu|gpu|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Parse arguments
BUILD_TYPE="${1:-all}"

if [[ ! "$BUILD_TYPE" =~ ^(cpu|gpu|all)$ ]]; then
    echo_error "Invalid build type: $BUILD_TYPE"
    echo_info "Usage: $0 [cpu|gpu|all]"
    exit 1
fi

echo_info "Docker Build Script for Docs2Synth"
echo_info "Build type: $BUILD_TYPE"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check disk space
available_space=$(df -h . | awk 'NR==2 {print $4}')
echo_info "Available disk space: $available_space"

# Function to build and test CPU image
build_cpu() {
    echo_step "Building CPU Docker image..."
    docker build --build-arg BUILD_TYPE=cpu -t docs2synth:cpu .

    echo_step "Testing CPU image..."
    echo "  → Checking Python version"
    docker run --rm docs2synth:cpu python --version

    echo "  → Testing package import"
    docker run --rm docs2synth:cpu python -c "import docs2synth; print('✓ docs2synth imported successfully')"

    echo "  → Testing CLI"
    docker run --rm docs2synth:cpu docs2synth --help > /dev/null

    echo "  → Running basic tests"
    docker run --rm docs2synth:cpu bash -c "pytest tests/ -v --tb=short || exit 1"

    echo_info "✓ CPU image built and tested successfully!"
    echo_info "Image size: $(docker images docs2synth:cpu --format '{{.Size}}')"
}

# Function to build and test GPU image
build_gpu() {
    echo_step "Building GPU Docker image..."
    echo_warn "GPU image is large (~11GB) and may take significant time and disk space"

    docker build --build-arg BUILD_TYPE=gpu -t docs2synth:gpu .

    echo_step "Testing GPU image..."
    echo "  → Checking Python version"
    docker run --rm docs2synth:gpu python --version

    echo "  → Testing package import"
    docker run --rm docs2synth:gpu python -c "import docs2synth; print('✓ docs2synth imported successfully')"

    echo "  → Testing PyTorch CUDA availability"
    docker run --rm docs2synth:gpu python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

    echo "  → Testing CLI"
    docker run --rm docs2synth:gpu docs2synth --help > /dev/null

    # Only run tests if GPU is available
    if docker run --rm --gpus all docs2synth:gpu python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "  → Running tests with GPU"
        docker run --rm --gpus all docs2synth:gpu bash -c "pytest tests/ -v --tb=short || exit 1"
    else
        echo_warn "GPU not available, skipping GPU tests"
        echo "  → Running basic tests (CPU mode)"
        docker run --rm docs2synth:gpu bash -c "pytest tests/ -v --tb=short || exit 1"
    fi

    echo_info "✓ GPU image built and tested successfully!"
    echo_info "Image size: $(docker images docs2synth:gpu --format '{{.Size}}')"
}

# Main build process
case $BUILD_TYPE in
    cpu)
        build_cpu
        ;;
    gpu)
        build_gpu
        ;;
    all)
        build_cpu
        echo ""
        build_gpu
        ;;
esac

echo ""
echo_info "=========================================="
echo_info "Docker build completed successfully!"
echo ""
echo_info "Available images:"
docker images docs2synth --format "  - {{.Repository}}:{{.Tag}} ({{.Size}})"
echo ""
echo_info "To run an image:"
echo_info "  CPU: docker run -it docs2synth:cpu"
echo_info "  GPU: docker run --gpus all -it docs2synth:gpu"
echo_info "=========================================="
