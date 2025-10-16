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

## Development setup

We provide automated setup scripts for all major platforms to make development environment setup as smooth as possible.

### Quick Setup

#### Unix-based Systems (macOS, Linux, WSL)

```bash
# Clone repository
git clone https://github.com/AI4WA/Docs2Synth.git
cd Docs2Synth

# Run setup script with conda (recommended)
./setup.sh conda

# Or use venv
./setup.sh venv
```

#### Windows

```batch
REM Clone repository
git clone https://github.com/AI4WA/Docs2Synth.git
cd Docs2Synth

REM Run setup script with conda (recommended)
setup.bat conda

REM Or use venv
setup.bat venv
```

### Docker Setup

For containerized development and deployment, we provide Docker configurations for both CPU-only and GPU-enabled environments.

#### CPU-only Container

```bash
# Build and run CPU-only container
docker compose up -d docs2synth-cpu

# Enter the container
docker exec -it docs2synth-cpu /bin/bash
```

#### GPU-enabled Container

For GPU support, you need:
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

```bash
# Build and run GPU-enabled container
docker-compose up -d docs2synth-gpu

# Enter the container
docker exec -it docs2synth-gpu /bin/bash

# Verify GPU access inside container
python -c "import torch; print(torch.cuda.is_available())"
```

To enable GPU support, uncomment the `deploy` section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

#### Building Docker Images Manually

```bash
# Build CPU-only image
docker build -t docs2synth:cpu -f Dockerfile .

# Build GPU-enabled image
docker build -t docs2synth:gpu -f Dockerfile.gpu .

# Run container with volume mounts
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs docs2synth:cpu
```

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment (Python ≥3.8)
python -m venv .venv && source .venv/bin/activate
# Install editable package with all dependencies
pip install -e ".[dev,datasets,qa,retriever]"

# Or use conda
conda env create -f environment.yml
conda activate Docs2Synth
pip install -e ".[dev,datasets,qa,retriever]"
```

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
