# Dependency Management

Modern dependency management with `uv` and lockfiles.

## Tools

- **[uv](https://github.com/astral-sh/uv)**: Fast Python package installer (10-100x faster than pip)
- **pyproject.toml**: Core dependencies
- **Lockfiles**: Pinned versions for reproducibility

---

## File Structure

**Source files (edit these):**
- `pyproject.toml` - Core project dependencies
- `requirements-cpu.in` - PyTorch CPU specs
- `requirements-gpu.in` - PyTorch GPU specs
- `requirements-dev.in` - Dev tools

**Lockfiles (auto-generated, don't edit):**
- `requirements-cpu.txt` - Locked CPU dependencies
- `requirements-dev.txt` - Locked dev dependencies

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
- Installs dependencies (prompts for CPU or GPU)
- Sets up config.yml

### Manual Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create venv
uv venv
source .venv/bin/activate

# CPU
uv pip install -r requirements-cpu.txt
uv pip install -e ".[dev]"

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
uv pip install -e ".[dev]"

# Setup config
cp config.example.yml config.yml
```

---

## Adding Dependencies

### Core Dependency

Edit `pyproject.toml`:
```toml
dependencies = [
    "click>=8.0",
    "new-package>=1.0.0",
]
```

Reinstall:
```bash
uv pip install -e ".[dev]"
```

### Dev Dependency

Edit `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "new-tool>=2.0",
]
```

Reinstall:
```bash
uv pip install -e ".[dev]"
```

---

## Updating Dependencies

### Regenerate Lockfiles

```bash
# CPU requirements
uv pip compile requirements-cpu.in -o requirements-cpu.txt

# Dev requirements
uv pip compile requirements-dev.in -o requirements-dev.txt

# Upgrade all to latest
uv pip compile requirements-cpu.in -o requirements-cpu.txt --upgrade
uv pip compile requirements-dev.in -o requirements-dev.txt --upgrade
```

### Update PyTorch

**CPU version:**
```bash
# Edit requirements-cpu.in with new version
uv pip compile requirements-cpu.in -o requirements-cpu.txt
```

**GPU version:**
```bash
# Edit requirements-gpu.in with instructions
pip install torch==2.x.x --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## Optional Dependencies

Install specific feature sets:

```bash
# QA generation
uv pip install -e ".[qa]"

# Retriever training
uv pip install -e ".[retriever]"

# RAG
uv pip install -e ".[rag]"

# MCP server
uv pip install -e ".[mcp]"

# All features
uv pip install -e ".[dev,qa,retriever,rag,mcp]"
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
```

### Dependency conflicts

```bash
# Clear cache and reinstall
uv cache clean
rm -rf .venv
uv venv
uv pip install -r requirements-cpu.txt
uv pip install -e ".[dev]"
```

### GPU dependencies fail

```bash
# Check CUDA version
nvidia-smi

# Use matching PyTorch version
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```
