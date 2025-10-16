# Unified Dockerfile for Docs2Synth
# Supports CPU-only, GPU, and minimal builds using build arguments
#
# Usage:
#   CPU (full):     docker build --build-arg BUILD_TYPE=cpu -t docs2synth:cpu .
#   CPU (minimal):  docker build --build-arg BUILD_TYPE=cpu-minimal -t docs2synth:cpu-minimal .
#   GPU:            docker build --build-arg BUILD_TYPE=gpu -t docs2synth:gpu .

# =============================================================================
# Build Arguments
# =============================================================================
ARG BUILD_TYPE=cpu
ARG PYTHON_VERSION=3.10

# =============================================================================
# Base Image Selection
# =============================================================================
# For GPU builds, use NVIDIA CUDA base image; otherwise use Python slim
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-gpu
FROM python:${PYTHON_VERSION}-slim AS base-cpu
FROM python:${PYTHON_VERSION}-slim AS base-cpu-minimal

# Select the appropriate base
FROM base-${BUILD_TYPE} AS base

# =============================================================================
# System Setup
# =============================================================================
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# =============================================================================
# System Dependencies - GPU specific setup
# =============================================================================
FROM base AS system-gpu

# Install Python and dependencies on Ubuntu (GPU base image doesn't have Python)
# PaddleOCR requires OpenCV which needs graphics libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        git \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        wget && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# System Dependencies - CPU full build
# =============================================================================
FROM base AS system-cpu

# Full dependencies for production
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# System Dependencies - CPU minimal build
# =============================================================================
FROM base AS system-cpu-minimal

# Minimal dependencies for development
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        gcc \
        g++ \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# Final Stage - Common for all build types
# =============================================================================
FROM system-${BUILD_TYPE} AS final

# Re-declare build arg for this stage
ARG BUILD_TYPE=cpu

# Copy all requirements files
COPY requirements*.txt ./

# Copy source code
COPY pyproject.toml README.md ./
COPY docs2synth/ ./docs2synth/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Install Python dependencies
# Select the appropriate torch requirements based on BUILD_TYPE
RUN pip install --upgrade pip && \
    if [ -f "requirements-${BUILD_TYPE}.txt" ]; then \
        echo "Using requirements-${BUILD_TYPE}.txt for PyTorch"; \
        pip install -r "requirements-${BUILD_TYPE}.txt"; \
    else \
        echo "Using requirements-cpu.txt for PyTorch (fallback)"; \
        pip install -r requirements-cpu.txt; \
    fi && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt && \
    pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Add build type label
LABEL build_type="${BUILD_TYPE}"
LABEL maintainer="AI4WA"
LABEL description="Docs2Synth - Document processing and synthesis toolkit"

# Default command
CMD ["/bin/bash"]
