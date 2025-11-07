# Docs2Synth

<div style="text-align: center; margin: 2rem 0;">
  <h2 style="font-size: 2rem; font-weight: 700; margin-bottom: 1rem;">
    Document Processing & Retriever Training Toolkit
  </h2>
  <p style="font-size: 1.2rem; color: var(--md-default-fg-color--light);">
    A complete pipeline for converting, synthesizing, and training retrievers for your document datasets
  </p>
</div>

---

## :sparkles: Key Features

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### :page_facing_up: Document Processing

Convert raw documents using **MinerU** or other OCR methods. Supports PDFs, images, and complex layouts.

[Learn more →](workflow/document-processing.md){ .md-button .md-button--primary }

</div>

<div class="feature-card" markdown>

### :robot: Agent-Based QA Generation

Automatically generate high-quality **question-answer pairs** with two-step verification (meaningfulness + correctness).

[Learn more →](workflow/qa-generation.md){ .md-button .md-button--primary }

</div>

<div class="feature-card" markdown>

### :brain: Retriever Training

Train custom retrievers using **LayoutLMv3**, BERT, or sentence transformers on your domain-specific data.

[Learn more →](workflow/retriever-training.md){ .md-button .md-button--primary }

</div>

<div class="feature-card" markdown>

### :rocket: RAG Path

Deploy immediately with **out-of-box strategies** (BM25, dense, hybrid) - no training required.

[Learn more →](workflow/rag-path.md){ .md-button .md-button--primary }

</div>

<div class="feature-card" markdown>

### :chart_with_upwards_trend: Benchmarking

Track retrieval performance with **Hit@K**, MRR, NDCG metrics and monitor end-to-end latency.

[Learn more →](workflow/retriever-training.md#evaluation){ .md-button .md-button--primary }

</div>

<div class="feature-card" markdown>

### :gear: Extensible Pipeline

Combine all steps into a **unified pipeline** with flexible control flow based on your requirements.

[Get Started →](#quick-start){ .md-button .md-button--primary }

</div>

<div class="feature-card" markdown>

### :electric_plug: MCP Integration

Expose functionality as **Model Context Protocol (MCP)** server for AI agents like Claude, ChatGPT, and Cursor.

[Learn more →](mcp-integration.md){ .md-button .md-button--primary }

</div>

</div>

## Installation

### From GitHub

```bash
pip install git+https://github.com/AI4WA/Docs2Synth.git
```

### From PyPI (once released)

```bash
pip install docs2synth
```

## Quick Start

### Development Setup

#### Recommended: Using uv (Fast & Modern)

```bash
# Clone repository
git clone https://github.com/AI4WA/Docs2Synth.git
cd Docs2Synth

# Install uv (if not already installed)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements-cpu.txt  # or requirements-gpu.txt for GPU
uv pip install -e ".[dev]"
```

#### Alternative: Automated Setup Script

```bash
# Run setup script (auto-installs uv and dependencies)
./setup.sh  # On Unix/macOS/WSL
# setup.bat  # On Windows
```

#### Alternative: Traditional pip

```bash
# Create virtual environment (Python ≥3.11)
python -m venv .venv && source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU or GPU)
pip install -r requirements-cpu.txt  # or requirements-gpu.txt for GPU

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest
```

## Workflow

The typical Docs2Synth workflow follows these stages:

1. **[Document Processing](workflow/document-processing.md)**: OCR and preprocessing
2. **[QA Generation](workflow/qa-generation.md)**: Generate and verify question-answer pairs
3. **[Retriever Training](workflow/retriever-training.md)**: Train custom retrieval models
4. **[RAG Path](workflow/rag-path.md)**: Deploy without training using pre-built strategies

## Architecture

```
Docs2Synth/
├── integration/    # Integration utilities
├── preprocess/     # Document preprocessing
├── qa/            # QA generation and verification
├── retriever/     # Retriever training and inference
├── rag/           # RAG strategies
└── utils/         # Logging, timing, and utilities
```

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/AI4WA/Docs2Synth) for guidelines.

## License

This project is licensed under the [MIT License](https://github.com/AI4WA/Docs2Synth/blob/main/LICENSE).

## Support

- Report issues: [GitHub Issues](https://github.com/AI4WA/Docs2Synth/issues)
- Documentation: [Full documentation](https://github.com/AI4WA/Docs2Synth)
