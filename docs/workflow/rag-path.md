# RAG Deployment

Deploy retrieval-augmented generation systems.

## Quick Start

```bash
# 1. Ingest documents
docs2synth rag ingest --processed-dir data/processed/

# 2. Query
docs2synth rag run -q "What is the invoice total?"

# 3. Launch demo
docs2synth rag app
```

---

## Strategies

### Naive

Single retrieval + generation pass.

```bash
docs2synth rag run -s naive -q "Your question"
```

**How it works:**
1. Retrieve top-k documents
2. Generate answer with LLM using retrieved context

### Iterative

Multi-step refinement with feedback.

```bash
docs2synth rag run -s iterative -q "Your question" --show-iterations
```

**How it works:**
1. Retrieve documents
2. Generate initial answer
3. Compare with previous iteration
4. Refine until convergence or max iterations

---

## Document Ingestion

Index processed documents into vector store.

```bash
# Basic
docs2synth rag ingest

# With options
docs2synth rag ingest \
  --processed-dir data/processed/ \
  --processor docling \
  --include-context
```

**What's indexed:**
- Each text object embedded separately
- Metadata: source file, object_id, bbox, page, label
- Context stored in metadata (not embedded)

---

## Querying

### CLI Query

```bash
# Basic query
docs2synth rag run -q "What is the total amount?"

# With strategy
docs2synth rag run -s iterative -q "What is the invoice date?"

# Show iteration details
docs2synth rag run -s iterative -q "Question" --show-iterations
```

### Demo App

Interactive Streamlit interface:

```bash
docs2synth rag app
docs2synth rag app --host localhost --port 8501 --no-browser
```

Opens at `http://localhost:8501`

**Features:**
- Interactive query interface
- Document preview with retrieved chunks
- Strategy comparison
- Source attribution with confidence scores

---

## Configuration

**`config.yml`:**
```yaml
rag:
  vector_store:
    type: chroma
    persist_directory: ./data/rag/vector_store
    collection_name: docs2synth_collection

  embedding:
    model: sentence-transformers/all-MiniLM-L6-v2

  strategies:
    naive:
      retriever:
        k: 5  # Top-k documents
      generator:
        provider: openai
        model: gpt-4o-mini
        temperature: 0.7

    iterative:
      retriever:
        k: 5
      generator:
        provider: openai
        model: gpt-4o-mini
        temperature: 0.7
      max_iterations: 3
      similarity_threshold: 0.9
```

---

## Vector Store Management

```bash
# List strategies
docs2synth rag strategies

# Reset vector store (clears all documents)
docs2synth rag reset
```

---

## Custom Strategies

Add custom RAG strategies by extending `RAGStrategy` class:

```python
from docs2synth.rag.strategy import RAGStrategy

class MyCustomStrategy(RAGStrategy):
    def run(self, query: str, state: RAGState) -> RAGResult:
        # Implement custom retrieval + generation logic
        pass
```

Register in config:
```yaml
rag:
  strategies:
    custom:
      class: mymodule.MyCustomStrategy
      retriever:
        k: 10
```

---

## Next Steps

- [Complete Workflow](complete-workflow.md) - Full workflow guide
- [CLI Reference](../cli-reference.md) - All commands
