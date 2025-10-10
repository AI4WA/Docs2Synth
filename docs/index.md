# Docs2Synth

Welcome to **Docs2Synth** - a Python package designed to help you convert, synthesize, and train retrievers for your document datasets.

## Overview

Docs2Synth provides a complete pipeline for working with document data, from OCR processing to retriever training and RAG (Retrieval-Augmented Generation) deployment.

### Key Features

- **Document Processing**: Convert raw documents using MinerU or other OCR methods
- **Agent-Based QA Generation**: Automatically generate high-quality question-answer pairs with verification
- **Retriever Training**: Train custom retrievers (LayoutLMv3, BERT, etc.)
- **RAG Path**: Out-of-box retriever strategies without training
- **Benchmarking**: Track retrieval performance (Hit@K) and latency metrics

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

```bash
# Clone repository
git clone https://github.com/AI4WA/Docs2Synth.git
cd Docs2Synth

# Create virtual environment (Python ≥3.8)
python -m venv .venv && source .venv/bin/activate

# Install editable package with dev dependencies
pip install -e ".[dev]"
```

### Using Conda

```bash
# Create and activate environment (Python 3.10)
conda env create -f environment.yml
conda activate Docs2Synth

# Verify installation
pytest
```

## Basic Usage

### Generate QA Pairs

```bash
docs2synth generate-qa /path/to/documents /path/to/output.jsonl
```

### Train a Retriever

```bash
docs2synth train-retriever /path/to/qa_pairs.jsonl \
    --output-dir models/retriever \
    --model-name sentence-transformers/all-MiniLM-L6-v2
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
