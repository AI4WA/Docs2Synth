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

## Quick CLI Examples

```bash
# Download datasets
docs2synth datasets list
docs2synth datasets download funsd

# Process documents
docs2synth preprocess document.png --processor paddleocr

# Generate text with LLM agents
docs2synth agent generate "Explain quantum computing" --provider openai

# Chat with history
docs2synth agent chat "Hello" --history-file chat.json

# Generate QA pairs
docs2synth qa semantic "Form contains name field" "John Doe"
docs2synth qa layout "What is the address?" --image doc.png
docs2synth qa generate "Context" "Target" --image doc.png
```

[View Full CLI Reference →](cli-reference.md){ .md-button .md-button--primary }

## Complete End-to-End Workflow

The typical Docs2Synth workflow consists of these stages:

```mermaid
graph LR
    A[Documents] --> B[Preprocess]
    B --> C[QA Generation]
    C --> D[Verification]
    D --> E[Human Annotation]
    E --> F[Retriever Training]
    F --> G[RAG Deployment]
```

### Step-by-Step Guide

1. **Setup Data Folder** - Organize documents and configure `config.yml`
2. **[Document Processing](workflow/document-processing.md)** - Extract text and layout with OCR
   ```bash
   docs2synth preprocess data/raw/my_documents/
   ```

3. **[QA Generation](workflow/qa-generation.md)** - Generate question-answer pairs with LLMs
   ```bash
   docs2synth qa batch
   ```

4. **Verification** - Automatically verify QA quality
   ```bash
   docs2synth verify batch
   ```

5. **Human Annotation** - Manual review via Streamlit interface
   ```bash
   docs2synth annotate
   ```

6. **[Retriever Training](workflow/retriever-training.md)** - Train custom retrieval models
   ```bash
   docs2synth retriever preprocess
   docs2synth retriever train --mode standard --lr 1e-5 --epochs 10
   docs2synth retriever validate
   ```

7. **[RAG Deployment](workflow/rag-path.md)** - Deploy RAG system
   ```bash
   docs2synth rag ingest
   docs2synth rag run -q "Your question"
   docs2synth rag app
   ```

[**View Complete Workflow Guide →**](workflow/complete-workflow.md){ .md-button .md-button--primary }

This comprehensive guide covers all steps from data preparation to RAG deployment with detailed examples, configuration options, and troubleshooting tips.

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
