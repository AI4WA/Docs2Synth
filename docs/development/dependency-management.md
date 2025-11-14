# Dependency Management

Modern dependency management with `uv` and `pyproject.toml`.

## Tools

- **[uv](https://github.com/astral-sh/uv)**: Fast Python package installer (10-100x faster than pip)
- **pyproject.toml**: Single source of truth for all dependencies
- **Optional extras**: Feature-based dependency groups

---

## Dependency Structure

All dependencies are managed in `pyproject.toml`:

```toml
[project]
dependencies = [
    # Minimal core dependencies
    "click>=8.0",
    "pandas>=1.5.0",
    "pillow>=10.0.0",
    # ...
]

[project.optional-dependencies]
# CPU installation (includes PyTorch CPU + all features + MCP)
cpu = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    # ... all ML/OCR/MCP dependencies
]

# GPU installation (excludes PyTorch, includes all features + MCP)
gpu = [
    "vllm>=0.11.0",  # GPU-only
    "faiss-cpu>=1.8.0",
    "transformers>=4.30.0",
    # ... all ML/OCR/MCP dependencies
]

# Development tools
dev = [
    "pytest>=7.0",
    "black==24.8.0",
    "isort>=5.12",
    # ...
]
```

---

## Setup

### Automated Setup (Recommended)

Use the setup script to install everything automatically:

```bash
./setup.sh         # Unix/macOS/WSL
# setup.bat        # Windows
```

The script:
- Installs uv
- Creates virtual environment
- Detects GPU and installs appropriate version
- Installs package with [cpu,dev] or [gpu,dev] extras

### Manual Setup

**CPU Version:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create venv
uv venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows

# Install with CPU extras
uv pip install -e ".[cpu,dev]"

# Setup config
cp config.example.yml config.yml
```

**GPU Version:**
```bash
# Create venv
uv venv
source .venv/bin/activate

# Install PyTorch GPU first (CUDA 12.8)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install with GPU extras
uv pip install -e ".[gpu,dev]"

# Setup config
cp config.example.yml config.yml
```

---

## Adding Dependencies

### Core Dependency (Required for Everyone)

Edit `pyproject.toml`:
```toml
[project]
dependencies = [
    "click>=8.0",
    "new-package>=1.0.0",  # Add here
]
```

Reinstall:
```bash
uv pip install -e ".[cpu,dev]"  # or [gpu,dev]
```

### Feature-Specific Dependency

Add to the appropriate extra in `pyproject.toml`:

**For both CPU and GPU users:**
```toml
[project.optional-dependencies]
cpu = [
    "torch>=2.0.0",
    "new-ml-library>=1.0.0",  # Add here
    # ...
]

gpu = [
    "vllm>=0.11.0",
    "new-ml-library>=1.0.0",  # Add here too
    # ...
]
```

**For development only:**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "new-dev-tool>=2.0",  # Add here
]
```

Reinstall:
```bash
uv pip install -e ".[cpu,dev]"  # or [gpu,dev]
```

---

## Updating Dependencies

### Update All Dependencies

```bash
# Remove venv and reinstall
rm -rf .venv
uv venv
source .venv/bin/activate

# CPU
uv pip install -e ".[cpu,dev]"

# GPU
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[gpu,dev]"
```

### Update Specific Package

```bash
# Update package to latest version
uv pip install --upgrade package-name

# Or specify version in pyproject.toml and reinstall
uv pip install -e ".[cpu,dev]"
```

### Update PyTorch

**CPU version:**
```bash
# PyTorch is in pyproject.toml [cpu] extra
# Edit version constraint if needed, then:
uv pip install -e ".[cpu,dev]"
```

**GPU version:**
```bash
# Install specific PyTorch version
uv pip install torch==2.x.x torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then reinstall extras
uv pip install -e ".[gpu,dev]"
```

---

## Installation Options

Install only what you need:

```bash
# CPU with all features (includes MCP)
uv pip install -e ".[cpu]"

# GPU with all features (includes MCP)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[gpu]"

# CPU + development tools
uv pip install -e ".[cpu,dev]"

# GPU + development tools
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[gpu,dev]"

# Minimal (core only, no ML/MCP)
uv pip install -e "."
```

---

## Why uv?

- **Fast**: 10-100x faster than pip
- **Reliable**: Better dependency resolution
- **Modern**: Built with Rust
- **Compatible**: Drop-in pip replacement

---

## Troubleshooting

### uv not found

```bash
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Dependency conflicts

```bash
# Clear cache and reinstall
uv cache clean
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[cpu,dev]"
```

### GPU dependencies fail

```bash
# Check CUDA version
nvidia-smi

# Use matching PyTorch version
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8

# Then install extras
uv pip install -e ".[gpu,dev]"
```

### Import errors after installation

```bash
# Verify installation
python -c "import docs2synth; print('OK')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# If still failing, reinstall
uv pip uninstall docs2synth
uv pip install -e ".[cpu,dev]"
```
