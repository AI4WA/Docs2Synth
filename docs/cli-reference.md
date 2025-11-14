# CLI Reference

## Global Options

```bash
docs2synth --help                    # Show help
docs2synth --version                 # Show version
docs2synth -v [COMMAND]              # Verbose output
docs2synth -vv [COMMAND]             # Debug output
docs2synth --config PATH [COMMAND]   # Use custom config
```

---

## Dataset Management

```bash
# List datasets
docs2synth datasets list

# Download
docs2synth datasets download funsd
docs2synth datasets download funsd --output-dir ./data
docs2synth datasets download all
```

**Available datasets**: `cord`, `funsd`, `vrd-iu2024-tracka`, `vrd-iu2024-trackb`, `docs2synth-dev`

---

## Document Preprocessing

```bash
# Process documents
docs2synth preprocess document.png
docs2synth preprocess ./images/

# Processor options
docs2synth preprocess doc.png --processor paddleocr  # Default, general OCR
docs2synth preprocess doc.pdf --processor pdfplumber  # Parsed PDFs
docs2synth preprocess doc.png --processor easyocr     # 80+ languages
docs2synth preprocess doc.png --processor docling     # Advanced layout

# Options
docs2synth preprocess ./images/ \
  --processor paddleocr \
  --lang en \
  --output-dir ./processed \
  --device gpu
```

---

## Agent Commands

### vLLM Server

```bash
# Start server
docs2synth agent vllm-server
docs2synth agent vllm-server --model meta-llama/Llama-2-7b-chat-hf
docs2synth agent vllm-server --port 8080 --gpu-memory-utilization 0.8
docs2synth agent vllm-server --trust-remote-code  # For Qwen, Phi, etc.
docs2synth agent vllm-server --tensor-parallel-size 2  # Multi-GPU
```

**Config `config.yml`:**
```yaml
agent:
  vllm:
    model: meta-llama/Llama-2-7b-chat-hf
    base_url: http://localhost:8000/v1
    trust_remote_code: true
    max_model_len: 4096
    gpu_memory_utilization: 0.9
```

### Generate Text

```bash
# Basic
docs2synth agent generate "Explain quantum computing"

# Providers
docs2synth agent generate "Your prompt" --provider openai      # Default
docs2synth agent generate "Your prompt" --provider anthropic
docs2synth agent generate "Your prompt" --provider gemini
docs2synth agent generate "Your prompt" --provider vllm       # Local
docs2synth agent generate "Your prompt" --provider ollama     # Local

# Options
docs2synth agent generate "Your prompt" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --temperature 0.7 \
  --max-tokens 1000

# With images
docs2synth agent generate "What's in this image?" \
  --image photo.jpg \
  --provider openai \
  --model gpt-4o
```

### Chat

```bash
# Basic chat
docs2synth agent chat "What is Python?"

# With history
docs2synth agent chat "Hello" --history-file chat.json
docs2synth agent chat "Tell me more" --history-file chat.json

# With options
docs2synth agent chat "Analyze this code" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --history-file chat.json
```

---

## QA Generation

```bash
# List strategies
docs2synth qa list

# Semantic QA
docs2synth qa semantic "Form contains name field" "John Doe"
docs2synth qa semantic "Context" "Target" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022

# Layout-aware QA
docs2synth qa layout "What is the address?" --image document.png

# Logical-aware QA
docs2synth qa logical "What is the name?" --image document.png

# Single document
docs2synth qa run data/processed/document_docling.json
docs2synth qa run data/images/document.png
docs2synth qa run data/processed/document.json --strategy semantic

# Batch processing
docs2synth qa batch  # Uses config.preprocess.input_dir
docs2synth qa batch data/raw/my_documents/
docs2synth qa batch data/images/ --output-dir data/processed/ --processor docling

# Clean QA pairs
docs2synth qa clean
docs2synth qa clean data/processed/dev
docs2synth qa clean data/processed/document.json
```

---

## QA Verification

```bash
# List verifiers
docs2synth verify list

# Single document
docs2synth verify run data/processed/document.json
docs2synth verify run data/images/document.png
docs2synth verify run data/processed/document.json --verifier-type meaningful

# Batch verification
docs2synth verify batch
docs2synth verify batch data/processed/dev
docs2synth verify batch --verifier-type meaningful
docs2synth verify batch --image-dir data/images

# Clean verification
docs2synth verify clean
docs2synth verify clean data/processed/dev
```

---

## Human Annotation

```bash
# Launch annotation tool
docs2synth annotate
docs2synth annotate data/processed/dev
docs2synth annotate data/processed/dev --image-dir data/images --port 8502
```

Opens at `http://localhost:8501`

---

## Retriever Training

### Preprocess Data

```bash
# Basic
docs2synth retriever preprocess

# Full options
docs2synth retriever preprocess \
  --json-dir data/processed/ \
  --image-dir data/raw/my_documents/ \
  --output data/retriever/preprocessed_train.pkl \
  --processor docling \
  --batch-size 8 \
  --max-length 512 \
  --num-objects 50 \
  --require-all-verifiers
```

### Train Model

```bash
# Basic
docs2synth retriever train --mode standard --lr 1e-5 --epochs 10

# Full options
docs2synth retriever train \
  --data-path data/retriever/preprocessed_train.pkl \
  --val-data-path data/retriever/preprocessed_val.pkl \
  --output-dir models/retriever/checkpoints/ \
  --mode standard \
  --base-model microsoft/layoutlmv3-base \
  --lr 1e-5 \
  --epochs 10 \
  --batch-size 8 \
  --save-every 2 \
  --device cuda

# Resume training
docs2synth retriever train \
  --resume models/retriever/checkpoints/checkpoint_epoch_5.pth \
  --mode standard
```

**Modes**: `standard`, `layout`, `layout-gemini`, `layout-coarse-grained`, `pretrain-layout`

### Validate Model

```bash
# Basic
docs2synth retriever validate

# Full options
docs2synth retriever validate \
  --model models/retriever/final_model.pth \
  --data data/retriever/preprocessed_val.pkl \
  --output models/retriever/validation_reports/ \
  --mode standard \
  --device cuda
```

---

## RAG Deployment

```bash
# List strategies
docs2synth rag strategies

# Ingest documents
docs2synth rag ingest
docs2synth rag ingest \
  --processed-dir data/processed/ \
  --processor docling \
  --include-context

# Query
docs2synth rag run -q "What is the total amount?"
docs2synth rag run -s iterative -q "What is the total?"
docs2synth rag run -s iterative -q "Question" --show-iterations

# Launch demo app
docs2synth rag app
docs2synth rag app --host localhost --port 8501 --no-browser

# Reset vector store
docs2synth rag reset
```

---

## Configuration

### Config File Structure

**`config.yml`:**
```yaml
# Data directories
data:
  root_dir: ./data
  processed_dir: ./data/processed

# Preprocessing
preprocess:
  processor: docling
  input_dir: ./data/raw/my_documents/
  output_dir: ./data/processed/
  lang: en
  device: cuda

# QA generation
qa:
  strategies:
    - strategy: semantic
      provider: openai
      model: gpt-4o-mini
      temperature: 0.7
      max_tokens: 150

  verifiers:
    - strategy: meaningful
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
    - strategy: correctness
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0

# Retriever training
retriever:
  preprocessed_data_path: ./data/retriever/preprocessed_train.pkl
  checkpoint_dir: ./models/retriever/checkpoints
  learning_rate: 1e-5
  epochs: 10
  batch_size: 8

# RAG
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
        temperature: 0.7
    iterative:
      max_iterations: 3
      similarity_threshold: 0.9

# LLM providers
agent:
  provider: openai
  keys:
    openai_api_key: "${OPENAI_API_KEY}"
    anthropic_api_key: "${ANTHROPIC_API_KEY}"
    gemini_api_key: "${GEMINI_API_KEY}"
```

### API Keys

**Option 1: In `config.yml` (recommended):**
```yaml
agent:
  keys:
    openai_api_key: "sk-..."
    anthropic_api_key: "sk-ant-..."
    gemini_api_key: "..."
```

Add `config.yml` to `.gitignore` to keep keys safe.

**Option 2: Environment variables:**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Option 3: `.env` file:**
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Then reference in `config.yml`:
```yaml
agent:
  keys:
    openai_api_key: "${OPENAI_API_KEY}"
```

---

## See Also

- [Complete Workflow Guide](workflow/complete-workflow.md)
- [Document Processing](workflow/document-processing.md)
- [QA Generation](workflow/qa-generation.md)
- [Retriever Training](workflow/retriever-training.md)
- [RAG Deployment](workflow/rag-path.md)
- [API Reference](api-reference.md)
