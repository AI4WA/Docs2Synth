# API Reference

Complete API documentation for Docs2Synth modules.

!!! note "Documentation Status"
    This API reference will be automatically generated from docstrings as modules are implemented. Currently showing manual documentation for implemented modules.

## Preprocess

Document preprocessing and OCR functionality.

### `Docs2Synth.preprocess.ocr`

Functions for OCR processing using various engines (MinerU, Tesseract, Cloud APIs).

**Coming soon** - See [Document Processing](workflow/document-processing.md) for usage examples.

### `Docs2Synth.preprocess.mineru`

MinerU-specific OCR implementation for high-quality document extraction.

**Coming soon** - See [Document Processing](workflow/document-processing.md) for usage examples.

---

## QA Generation

Question-answer pair generation and verification.

### `Docs2Synth.qa.generator`

Core QA generation functionality using LLMs.

**Coming soon** - See [QA Generation](workflow/qa-generation.md) for usage examples.

### `Docs2Synth.qa.verification`

Two-step verification system (meaningfulness + correctness checking).

**Coming soon** - See [QA Generation](workflow/qa-generation.md) for usage examples.

---

## Retriever

Retriever training, inference, and evaluation.

### `Docs2Synth.retriever.train`

Training functionality for custom retrievers.

**Coming soon** - See [Retriever Training](workflow/retriever-training.md) for usage examples.

### `Docs2Synth.retriever.models`

Model architectures (LayoutLMv3, BERT variants, sentence transformers).

**Coming soon** - See [Retriever Training](workflow/retriever-training.md) for usage examples.

### `Docs2Synth.retriever.inference`

Inference and retrieval functionality for trained models.

**Coming soon** - See [Retriever Training](workflow/retriever-training.md) for usage examples.

### `Docs2Synth.retriever.evaluation`

Evaluation metrics and benchmarking tools.

**Coming soon** - See [Retriever Training](workflow/retriever-training.md) for usage examples.

---

## RAG

RAG retrieval strategies and pipelines.

### `Docs2Synth.rag`

Out-of-box retrieval strategies (BM25, dense, hybrid, ColBERT).

**Coming soon** - See [RAG Path](workflow/rag-path.md) for usage examples.

---

## Utils

Utilities for logging, timing, and more.

### `Docs2Synth.utils.logging`

::: Docs2Synth.utils.logging
    options:
      show_root_heading: true
      show_source: true
      members:
        - setup_logging
        - get_logger
        - setup_cli_logging
        - LoggerContext
        - ProgressLogger

### `Docs2Synth.utils.timer`

::: Docs2Synth.utils.timer
    options:
      show_root_heading: true
      show_source: true
      members:
        - timer
        - timeit
        - Timer
        - format_time

---

## CLI

Command-line interface documentation.

### `docs2synth`

Main CLI entry point.

```bash
docs2synth --help
```

**Available Commands:**

#### `generate-qa`

Generate question-answer pairs from documents.

```bash
docs2synth generate-qa INPUT OUTPUT [OPTIONS]
```

**Arguments:**
- `INPUT`: Path to source documents
- `OUTPUT`: Path to save generated QA pairs

**Options:**
- `--num-pairs INTEGER`: Number of QA pairs to generate per document
- `--model TEXT`: LLM model to use (default: gpt-4)
- `--verify`: Enable verification pipeline
- `--help`: Show help message

#### `train-retriever`

Train a retriever model on QA pairs.

```bash
docs2synth train-retriever QA_PATH [OPTIONS]
```

**Arguments:**
- `QA_PATH`: Path to QA pairs dataset

**Options:**
- `--output-dir TEXT`: Where to save trained model (default: models/retriever)
- `--model-name TEXT`: Backbone model (default: sentence-transformers/all-MiniLM-L6-v2)
- `--epochs INTEGER`: Number of training epochs (default: 5)
- `--batch-size INTEGER`: Training batch size (default: 32)
- `--learning-rate FLOAT`: Learning rate (default: 2e-5)
- `--help`: Show help message

---

## Type Definitions

### Document

```python
class Document:
    """Represents a processed document."""

    text: str
    layout: Dict[str, Any]
    metadata: Dict[str, Any]
    document_id: str
```

### QAPair

```python
class QAPair:
    """Represents a question-answer pair."""

    question: str
    answer: str
    context: str
    metadata: Dict[str, Any]
    meaningful_score: float
    correctness_score: float
```

### RetrievalResult

```python
class RetrievalResult:
    """Represents a retrieval result."""

    document_id: str
    score: float
    document: Document
    rank: int
```

---

## Configuration

### Configuration File Format

Docs2Synth supports YAML configuration files:

```yaml
# config.yml
preprocess:
  ocr_engine: mineru
  language: eng
  dpi: 300

qa_generation:
  model: gpt-4
  temperature: 0.7
  num_pairs_per_page: 5
  verification:
    enable_meaningful_check: true
    enable_correctness_check: true

retriever_training:
  model_name: sentence-transformers/all-mpnet-base-v2
  epochs: 5
  batch_size: 32
  learning_rate: 2e-5

rag:
  strategy: hybrid
  sparse_weight: 0.3
  dense_weight: 0.7
```

### Loading Configuration

```python
from Docs2Synth import load_config

config = load_config("config.yml")
```

---

## Exceptions

### `Docs2Synth.exceptions`

```python
class Docs2SynthError(Exception):
    """Base exception for Docs2Synth."""
    pass

class OCRError(Docs2SynthError):
    """Raised when OCR processing fails."""
    pass

class QAGenerationError(Docs2SynthError):
    """Raised when QA generation fails."""
    pass

class TrainingError(Docs2SynthError):
    """Raised when retriever training fails."""
    pass

class RetrievalError(Docs2SynthError):
    """Raised when retrieval fails."""
    pass
```

---

## Examples

For practical examples, see the [workflow documentation](workflow/document-processing.md) and the [examples directory](https://github.com/AI4WA/Docs2Synth/tree/main/examples) in the repository.

---

## Contributing

To contribute to the API documentation:

1. Add docstrings to your code (Google-style format)
2. Update this reference page if adding new modules
3. Run `mkdocs serve` to preview changes
4. Submit a pull request

For more information, see the [contributing guide](https://github.com/AI4WA/Docs2Synth/blob/main/CONTRIBUTING.md).
