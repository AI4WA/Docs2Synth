# Docs2Synth

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://ai4wa.github.io/Docs2Synth/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Docs2Synth** converts, synthesizes, and trains retrievers for document datasets.

## Workflow

```
Documents → Preprocess → QA Generation → Verification →
Human Annotation → Retriever Training → RAG Deployment
```

### 🚀 Quick Start: Automated Pipeline

Run the complete end-to-end pipeline with a single command:

```bash
docs2synth run
```

This automatically chains: preprocessing → QA generation → verification → retriever training → validation → RAG deployment, skipping the manual annotation UI.

### Manual Step-by-Step Workflow

For more control, run each step individually:

```bash
# 1. Preprocess documents
docs2synth preprocess data/raw/my_documents/

# 2. Generate QA pairs
docs2synth qa batch

# 3. Verify quality
docs2synth verify batch

# 4. Annotate (opens UI)
docs2synth annotate

# 5. Train retriever
docs2synth retriever preprocess
docs2synth retriever train --mode standard --lr 1e-5 --epochs 10

# 6. Deploy RAG
docs2synth rag ingest
docs2synth rag app
```

[Complete Workflow Guide →](https://ai4wa.github.io/Docs2Synth/workflow/complete-workflow/)

---

## Installation

### PyPI Installation (Recommended)

**CPU Version (includes all features + MCP server):**
```bash
pip install docs2synth[cpu]
```

**GPU Version (includes all features + MCP server):**
```bash
# Standard GPU installation (no vLLM)
pip install docs2synth[gpu]

# With vLLM for local LLM inference (requires CUDA GPU)
# 1. Install PyTorch with CUDA first:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install docs2synth with vLLM:
pip install docs2synth[gpu,vllm]

# 3. Uninstall paddlex to avoid conflicts with vLLM:
pip uninstall -y paddlex
```

> **Note**: PaddleX conflicts with vLLM. If you need vLLM support, you must uninstall paddlex after installation.

**Minimal Install (CLI only, no ML/MCP features):**
```bash
pip install docs2synth
```

### Development Setup

**Use the setup script (installs uv + dependencies automatically):**

```bash
# Clone
git clone https://github.com/AI4WA/Docs2Synth.git
cd Docs2Synth

# Run setup script
./setup.sh         # Unix/macOS/WSL
# setup.bat        # Windows
```

The script:
- Installs [uv](https://github.com/astral-sh/uv) (fast package manager)
- Creates virtual environment
- Installs dependencies (CPU or GPU)
- Sets up config

**Manual development setup:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and setup
git clone https://github.com/AI4WA/Docs2Synth.git
cd Docs2Synth
uv venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows

# Install for development
uv pip install -e ".[cpu,dev]"  # or [gpu,dev] for GPU

# Setup config
cp config.example.yml config.yml
# Edit config.yml and add your API keys
```

---

## Features

- **Document Processing**: Extract text/layout with Docling, PaddleOCR, PDFPlumber
- **QA Generation**: Automatic question-answer pair generation with LLMs
- **Verification**: Built-in meaningful and correctness verifiers
- **Human Annotation**: Streamlit UI for manual review
- **Retriever Training**: Train LayoutLMv3-based retrievers
- **RAG Deployment**: Deploy with naive or iterative strategies
- **MCP Integration**: Expose as Model Context Protocol server

---

## Configuration

Create `config.yml` from `config.example.yml`:

```yaml
# API keys (config.yml is in .gitignore)
agent:
  keys:
    openai_api_key: "sk-..."
    anthropic_api_key: "sk-ant-..."

# Document processing
preprocess:
  processor: docling
  input_dir: ./data/raw/
  output_dir: ./data/processed/

# QA generation
qa:
  strategies:
    - strategy: semantic
      provider: openai
      model: gpt-4o-mini

# Retriever training
retriever:
  learning_rate: 1e-5
  epochs: 10

# RAG
rag:
  embedding:
    model: sentence-transformers/all-MiniLM-L6-v2
```

---

## Docker

```bash
# CPU
./scripts/build-docker.sh cpu

# GPU
./scripts/build-docker.sh gpu
```

See [Docker Builds](https://ai4wa.github.io/Docs2Synth/development/docker-builds/)

---

## Documentation

Full documentation: **https://ai4wa.github.io/Docs2Synth/**

- [Complete Workflow Guide](https://ai4wa.github.io/Docs2Synth/workflow/complete-workflow/)
- [CLI Reference](https://ai4wa.github.io/Docs2Synth/cli-reference/)
- [Document Processing](https://ai4wa.github.io/Docs2Synth/workflow/document-processing/)
- [QA Generation](https://ai4wa.github.io/Docs2Synth/workflow/qa-generation/)
- [Retriever Training](https://ai4wa.github.io/Docs2Synth/workflow/retriever-training/)
- [RAG Deployment](https://ai4wa.github.io/Docs2Synth/workflow/rag-path/)

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Run code quality checks: `./scripts/check.sh`
6. Submit a pull request

See [Dependency Management](https://ai4wa.github.io/Docs2Synth/development/dependency-management/) for dev setup details.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

If you use Docs2Synth in your research, please cite:

```bibtex
@software{docs2synth2024,
  title = {Docs2Synth: A Synthetic Data Tuned Retriever Framework for \\ Scanned Visually Rich Documents Understanding},
  author = {AI4WA Team},
  year = {2025},
  url = {https://github.com/AI4WA/Docs2Synth}
}
```

---

## Support

- **Documentation**: https://ai4wa.github.io/Docs2Synth/
- **Issues**: https://github.com/AI4WA/Docs2Synth/issues
- **Discussions**: https://github.com/AI4WA/Docs2Synth/discussions
