# Unified Dockerfile for Docs2Synth
# Supports CPU-only and GPU builds using build arguments
#
# Usage:
#   CPU:  docker build --build-arg BUILD_TYPE=cpu -t docs2synth:cpu .
#   GPU:  docker build --build-arg BUILD_TYPE=gpu -t docs2synth:gpu .

# =============================================================================
# Build Arguments
# =============================================================================
ARG BUILD_TYPE=cpu
ARG PYTHON_VERSION=3.11

# =============================================================================
# Base Image Selection
# =============================================================================
# For GPU builds, use NVIDIA CUDA base image; otherwise use Python slim
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-gpu
FROM python:${PYTHON_VERSION}-slim AS base-cpu

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
        python3 \
        python3-dev \
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
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# System Dependencies - CPU build
# =============================================================================
FROM base AS system-cpu

# Dependencies for CPU-based processing
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
    pip install -e ".[dev]"

# Create directories for data, logs, and PaddleX cache
# Set permissions to allow any user to write (for non-root users)
RUN mkdir -p /app/data /app/logs /app/.paddlex /app/.cache /tmp && \
    chmod -R 777 /app/data /app/logs /app/.paddlex /app/.cache /tmp

# Set PaddleX cache directories to writable locations
ENV PADDLEX_HOME=/app/.paddlex \
    XDG_CACHE_HOME=/app/.cache \
    HOME=/app

# Add build type label
LABEL build_type="${BUILD_TYPE}"
LABEL maintainer="AI4WA"
LABEL description="Docs2Synth - Document processing and synthesis toolkit"

# Default command
CMD ["/bin/bash"]
