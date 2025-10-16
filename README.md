# Docs2Synth

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://ai4wa.github.io/Docs2Synth/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Docs2Synth** is a Python package aimed at helping you convert, synthesise and train a retriever for your document datasets.

The workflow typically involves:

- Document processing
  - MinerU or Other OCR methods
- Agent based QA generation
  - QA Pair generation
    - For now, given content, generate a question and answer
    - Later we can extend to multiple page contexts in a single QA pair.
  - Two step verification
    - Meaningful verifier
    - Correctness checker
  - Human judgement
    - Human quickly annotate keep/discard
- Retriever training
  - Train LayoutLMv3 as a retriever
  - Extend to bert, etc
- RAG Path
  - Other out-of-box retriever strageties without training
- Framework pipeline
  - Combine all the above steps into a single pipeline and control flow based on parameters
  - Benchmarking
    - Retrieveral Hit@K Performance
    - End to end latency metrics

## Installation

```bash
pip install git+https://github.com/AI4WA/Docs2Synth.git
```

or, once released on PyPI [To be released]:

```bash
pip install docs2synth
```

## Development Setup

We provide two approaches for setting up your development environment:

1. **Local Setup** (Recommended for most users) - Automated scripts with GPU auto-detection
2. **Docker Setup** (Alternative) - Containerized environment for difficult setups or production deployment

### Option 1: Local Setup (Recommended)

We provide automated setup scripts for all major platforms that automatically detect GPU availability and install the appropriate PyTorch version.

#### Unix-based Systems (macOS, Linux, WSL)

```bash
# Clone repository
git clone https://github.com/AI4WA/Docs2Synth.git
cd Docs2Synth

# Run setup script with conda (recommended) - auto-detects GPU
./setup.sh conda

# Or use venv
./setup.sh venv

# Force GPU installation (if you have NVIDIA GPU)
./setup.sh conda --gpu

# Or force CPU-only installation
./setup.sh conda  # (will prompt, choose 'n' for CPU)
```

**The setup script will:**
- Automatically detect NVIDIA GPU availability
- Prompt you to install GPU-enabled PyTorch if GPU is detected
- Install CPU-only PyTorch if no GPU is found
- Install all project dependencies from requirements.txt files
- Verify the installation

#### Windows

```batch
REM Clone repository
git clone https://github.com/AI4WA/Docs2Synth.git
cd Docs2Synth

REM Run setup script with conda (recommended) - auto-detects GPU
setup.bat conda

REM Or use venv
setup.bat venv

REM Force GPU installation (if you have NVIDIA GPU)
setup.bat conda --gpu
```

### Option 2: Docker Setup (Alternative)

**When to use Docker:**
- Difficult to set up the local development environment (dependency conflicts, OS limitations)
- Need consistent environment across team members
- Preparing for production deployment
- Want isolated environment without affecting system Python

We provide a unified Dockerfile that supports multiple build configurations through build arguments.

#### Available Docker Images

| Image Type | Use Case | Size | Build Command |
|------------|----------|------|---------------|
| `cpu-minimal` | Development, fastest builds | ~2.5 GB | `./docker-build.sh cpu-minimal` |
| `cpu` | Production CPU deployment | ~2.5 GB | `./docker-build.sh cpu` |
| `gpu` | Training/GPU inference | ~11 GB | `./docker-build.sh gpu` |

#### Quick Start with Docker

**Using docker-compose (Recommended):**

```bash
# For development (CPU, minimal dependencies)
docker compose up -d docs2synth-cpu-minimal
docker exec -it docs2synth-cpu-minimal /bin/bash

# For production (CPU, full dependencies)
docker compose up -d docs2synth-cpu
docker exec -it docs2synth-cpu /bin/bash

# For GPU workloads (requires NVIDIA GPU + nvidia-docker)
docker compose up -d docs2synth-gpu
docker exec -it docs2synth-gpu /bin/bash
```

**Using the helper script:**

```bash
# Make script executable
chmod +x docker-build.sh

# Build CPU minimal (default, fastest)
./docker-build.sh cpu-minimal

# Build CPU full
./docker-build.sh cpu

# Build GPU image
./docker-build.sh gpu

# Build all variants
./docker-build.sh all

# Run the built image
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  docs2synth:cpu-minimal
```

**Direct Docker build:**

```bash
# CPU minimal (development)
docker build --build-arg BUILD_TYPE=cpu-minimal -t docs2synth:cpu-minimal .

# CPU full (production)
docker build --build-arg BUILD_TYPE=cpu -t docs2synth:cpu .

# GPU (requires linux/amd64)
docker build --build-arg BUILD_TYPE=gpu --platform linux/amd64 -t docs2synth:gpu .
```

#### GPU Support in Docker

**Requirements:**
- Linux host with NVIDIA GPU
- NVIDIA drivers installed
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

**Enable GPU in docker-compose.yml:**

Uncomment the `deploy` section under `docs2synth-gpu`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Run with GPU:**

```bash
# Using docker-compose
docker compose up -d docs2synth-gpu
docker exec -it docs2synth-gpu python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Using docker run
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  docs2synth:gpu
```

#### Production Deployment with Docker

Docker images are ideal for production deployment:

```bash
# Build on your CI/CD pipeline (x86_64 runners)
docker build --build-arg BUILD_TYPE=cpu -t your-registry/docs2synth:cpu .
docker push your-registry/docs2synth:cpu

# Or for GPU workloads
docker build --build-arg BUILD_TYPE=gpu --platform linux/amd64 -t your-registry/docs2synth:gpu .
docker push your-registry/docs2synth:gpu

# Deploy on production servers
docker pull your-registry/docs2synth:cpu
docker run -d \
  --name docs2synth-prod \
  -v /data:/app/data \
  -v /logs:/app/logs \
  your-registry/docs2synth:cpu \
  python -m docs2synth.cli process --config /app/config.yml
```

**Platform Notes:**
- **macOS (Apple Silicon)**: Can build all images, but GPU images use emulation and are slower. GPU containers cannot use GPU acceleration on Mac.
- **macOS (Intel)**: Same as Apple Silicon.
- **Linux (x86_64)**: Full support for all images including GPU.
- **Windows (WSL2)**: Full support with proper nvidia-docker setup.

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment (Python ≥3.8)
python -m venv .venv && source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose CPU or GPU version)
# For CPU:
pip install -r requirements-cpu.txt
# For GPU (CUDA 11.8):
# pip install -r requirements-gpu.txt

# Install project dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

**Or with conda:**

```bash
# Create conda environment
conda env create -f environment.yml
conda activate Docs2Synth

# Install PyTorch (CPU or GPU)
pip install -r requirements-cpu.txt  # or requirements-gpu.txt

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package
pip install -e .
```

### Dependency Management

The project uses separate requirements files for better control:

- **requirements.txt**: Core dependencies (PaddleOCR, etc.)
- **requirements-cpu.txt**: CPU-only PyTorch
- **requirements-gpu.txt**: GPU-enabled PyTorch (CUDA 11.8)
- **requirements-dev.txt**: Development tools (pytest, black, etc.)

### Code Quality Checks

Before pushing, run all checks:

```bash

./scripts/check.sh
```

This will run:
- **isort** - Sort imports
- **black** - Format code
- **flake8** - Lint code
- **pytest** - Run tests


## Documentation

Full documentation is available at: **https://ai4wa.github.io/Docs2Synth/**

Topics covered:
- [Quick Start Guide](https://ai4wa.github.io/Docs2Synth/)
- [Document Processing](https://ai4wa.github.io/Docs2Synth/workflow/document-processing/)
- [QA Generation](https://ai4wa.github.io/Docs2Synth/workflow/qa-generation/)
- [Retriever Training](https://ai4wa.github.io/Docs2Synth/workflow/retriever-training/)
- [RAG Path](https://ai4wa.github.io/Docs2Synth/workflow/rag-path/)
- [API Reference](https://ai4wa.github.io/Docs2Synth/api-reference/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT](LICENSE)
