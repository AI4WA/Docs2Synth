FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglvnd0 \
    libglx0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Copy source code
COPY docs2synth/ ./docs2synth/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Install Python dependencies
# Install CPU-only version of PyTorch
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e ".[dev,datasets,qa,retriever]"

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Expose port if needed (for future web services)
# EXPOSE 8000

# Default command
CMD ["/bin/bash"]
