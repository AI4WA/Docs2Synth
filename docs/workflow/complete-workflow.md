# Complete Workflow Guide

End-to-end workflow: Documents → Preprocess → QA Generation → Verification → Human Annotation → Retriever Training → RAG Deployment

## Prerequisites

1. Install Docs2Synth: `pip install -e ".[dev]"`
2. Copy `config.example.yml` to `config.yml`
3. Add API keys to `config.yml` (add to `.gitignore`)

---

## Stage 1: Data Setup

```bash
mkdir -p data/raw/my_documents
cp /path/to/your/documents/* data/raw/my_documents/
```

Configure `config.yml`:
```yaml
preprocess:
  input_dir: ./data/raw/my_documents/
  output_dir: ./data/processed/
  processor: docling

qa:
  strategies:
    - strategy: semantic
      provider: openai
      model: gpt-4o-mini
      temperature: 0.7

  verifiers:
    - strategy: meaningful
      provider: openai
      model: gpt-4o-mini
    - strategy: correctness
      provider: openai
      model: gpt-4o-mini

agent:
  keys:
    openai_api_key: "sk-..."  # Add your keys directly here
    anthropic_api_key: "sk-ant-..."
    # config.yml is in .gitignore (safe)
```

---

## Stage 2: Preprocess

```bash
docs2synth preprocess data/raw/my_documents/
```

Output: `data/processed/*.json` with text, bboxes, reading order

---

## Stage 3: QA Generation

```bash
# Batch generate QA pairs
docs2synth qa batch

# Single document
docs2synth qa run data/processed/document.json
```

Output: JSON updated with `qa` field containing questions and answers

---

## Stage 4: Verification

```bash
# Verify all documents
docs2synth verify batch

# Single document
docs2synth verify run data/processed/document.json
```

Output: JSON updated with `verification` results (meaningful, correctness)

---

## Stage 5: Human Annotation

```bash
docs2synth annotate
```

Opens Streamlit UI at `http://localhost:8501`:
- Review QA pairs with document images
- Approve/reject pairs
- Add notes
- Track progress

---

## Stage 6: Retriever Training

### Preprocess Data

```bash
docs2synth retriever preprocess \
  --json-dir data/processed/ \
  --image-dir data/raw/my_documents/ \
  --output data/retriever/train.pkl \
  --require-all-verifiers
```

Output: `train.pkl` DataLoader ready for training

### Train Model

```bash
docs2synth retriever train \
  --data-path data/retriever/train.pkl \
  --mode standard \
  --lr 1e-5 \
  --epochs 10 \
  --device cuda
```

Options:
- `--mode`: `standard` | `layout` | `layout-gemini` | `layout-coarse-grained` | `pretrain-layout`
- `--base-model`: `microsoft/layoutlmv3-base` (default) | `microsoft/layoutlmv3-large`
- `--resume`: Checkpoint path to resume training

Output:
- Checkpoints: `models/retriever/checkpoints/checkpoint_epoch_N.pth`
- Final model: `models/retriever/final_model.pth`
- Training curves: `models/retriever/checkpoints/training_curves.png`

### Validate Model

```bash
docs2synth retriever validate \
  --model models/retriever/final_model.pth \
  --data data/retriever/val.pkl
```

Output: ANLS scores, metrics plots, detailed analysis

---

## Stage 7: RAG Deployment

### Ingest Documents

```bash
docs2synth rag ingest \
  --processed-dir data/processed/ \
  --processor docling
```

### Query

```bash
# CLI query
docs2synth rag run -q "What is the invoice total?" -s iterative

# Show iterations
docs2synth rag run -q "Your question" --show-iterations
```

### Launch Demo

```bash
docs2synth rag app
```

Opens at `http://localhost:8501`

---

## Configuration Reference

### Minimal `config.yml`

```yaml
data:
  root_dir: ./data
  processed_dir: ./data/processed

preprocess:
  processor: docling
  input_dir: ./data/raw/my_documents/
  output_dir: ./data/processed/

qa:
  strategies:
    - strategy: semantic
      provider: openai
      model: gpt-4o-mini
      temperature: 0.7
  verifiers:
    - strategy: meaningful
      provider: openai
      model: gpt-4o-mini
    - strategy: correctness
      provider: openai
      model: gpt-4o-mini

retriever:
  preprocessed_data_path: ./data/retriever/train.pkl
  checkpoint_dir: ./models/retriever/checkpoints
  learning_rate: 1e-5
  epochs: 10

rag:
  vector_store:
    type: chroma
    persist_directory: ./data/rag/vector_store
  embedding:
    model: sentence-transformers/all-MiniLM-L6-v2
  strategies:
    naive:
      retriever:
        k: 5
      generator:
        provider: openai
        model: gpt-4o-mini
    iterative:
      retriever:
        k: 5
      generator:
        provider: openai
        model: gpt-4o-mini
      max_iterations: 3

agent:
  keys:
    # Option 1: Direct keys (config.yml is in .gitignore)
    openai_api_key: "sk-..."
    anthropic_api_key: "sk-ant-..."

    # Option 2: Environment variables
    # openai_api_key: "${OPENAI_API_KEY}"
    # anthropic_api_key: "${ANTHROPIC_API_KEY}"
```

---

## Tips & Troubleshooting

### Data Preparation
- Start with 10-20 documents for testing
- Use high-quality images (300 DPI+)
- Test with `docs2synth datasets download docs2synth-dev`

### QA Generation
- Use multiple strategies for diversity
- Monitor API costs (use `gpt-4o-mini` for dev)
- Aim for >80% verification pass rate

### Training
- Minimum 500-1000 annotated QA pairs
- Use 80/20 train/validation split
- Monitor validation ANLS (target >0.7)
- Try different modes: `standard`, `layout`, `layout-gemini`

### Common Issues

**GPU OOM:**
```bash
# Reduce batch size
docs2synth retriever train --batch-size 4

# Or use CPU
docs2synth retriever train --device cpu
```

**Low verification pass rate:**
```bash
# Review failed QA pairs
docs2synth verify batch --verifier-type meaningful

# Adjust QA generation prompts in code
```

**RAG returns irrelevant results:**
```bash
# Increase retrieval candidates
# Edit config.yml: rag.strategies.naive.retriever.k: 10

# Or use re-ranking strategy
docs2synth rag run -q "question" -s rerank
```

---

## Next Steps

- Scale to larger document collections
- Fine-tune with more annotated data
- Customize QA strategies for your domain
- Deploy RAG API for production
- Benchmark on standard datasets (FUNSD, CORD, DocVQA)

---

## Additional Resources

- [Document Processing](document-processing.md) - OCR processor details
- [QA Generation](qa-generation.md) - Strategy customization
- [Retriever Training](retriever-training.md) - Model architecture
- [RAG Deployment](rag-path.md) - RAG strategies
- [CLI Reference](../cli-reference.md) - All commands
- [GitHub Issues](https://github.com/AI4WA/Docs2Synth/issues)
