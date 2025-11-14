# Complete Workflow Guide

This guide walks you through the complete end-to-end workflow for using Docs2Synth to process documents, generate QA pairs, train a retriever model, and deploy a RAG system.

## Overview

The Docs2Synth workflow consists of these stages:

```
Documents → Preprocess → QA Generation → Verification →
Human Annotation → Retriever Training → RAG Deployment
```

Each stage builds on the previous one, progressively enriching your document dataset with structured information suitable for training document understanding models.

---

## Prerequisites

1. **Installation**: Ensure Docs2Synth is installed with all dependencies
2. **Configuration**: Create a `config.yml` file from `config.example.yml`
3. **API Keys**: Set up API keys for LLM providers (OpenAI, Anthropic, etc.) in `config.yml` or `.env`

---

## Stage 1: Data Setup

### Prepare Your Document Collection

Organize your documents in a dedicated directory:

```bash
mkdir -p data/raw/my_documents
cp /path/to/your/documents/* data/raw/my_documents/
```

**Supported formats:**
- Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`
- PDFs: `.pdf`

**Update `config.yml`:**
```yaml
preprocess:
  input_dir: ./data/raw/my_documents/
  output_dir: ./data/processed/
  processor: docling  # or paddleocr, pdfplumber, easyocr
```

**Tip**: Start with the development dataset to test the workflow:
```bash
docs2synth datasets download docs2synth-dev
```

---

## Stage 2: Document Preprocessing

Extract structured text and layout information from your documents.

### Single Document

```bash
docs2synth preprocess data/raw/my_documents/document.png
```

### Batch Processing

```bash
docs2synth preprocess data/raw/my_documents/
```

### With Options

```bash
docs2synth preprocess data/raw/my_documents/ \
  --processor docling \
  --output-dir data/processed/ \
  --lang en
```

**Output**: JSON files with extracted text, bounding boxes, reading order, and page structure:
```
data/processed/
├── document_docling.json
├── invoice_docling.json
└── form_docling.json
```

**What's in the JSON?**
- `objects`: Text objects with `text`, `bbox`, `page`, `label`
- `context`: Full document text in reading order
- `reading_order_ids`: Sequential list of object IDs
- `process_metadata`: Processor info, timestamp, version
- `document_metadata`: Original filename, dimensions

---

## Stage 3: QA Pair Generation

Generate question-answer pairs for each text object using configured LLM strategies.

### Configuration

Ensure your `config.yml` has QA strategies configured:

```yaml
qa:
  strategies:
    - strategy: semantic
      provider: openai
      model: gpt-4o-mini
      temperature: 0.7
      max_tokens: 150

    - strategy: layout_aware
      provider: anthropic
      model: claude-3-5-sonnet-20241022
      temperature: 0.5
      max_tokens: 150
```

### Generate QA Pairs

**Single document:**
```bash
docs2synth qa run data/processed/document_docling.json
```

**Batch processing:**
```bash
docs2synth qa batch
```

With explicit paths:
```bash
docs2synth qa batch data/raw/my_documents/ \
  --output-dir data/processed/ \
  --processor docling
```

**Output**: JSON files updated with `qa` field for each object:
```json
{
  "objects": {
    "obj_0": {
      "text": "Invoice Number: INV-2024-001",
      "bbox": [100, 200, 300, 220],
      "qa": [
        {
          "strategy": "semantic",
          "provider": "openai",
          "model": "gpt-4o-mini",
          "question": "What is the invoice number?",
          "answer": "INV-2024-001"
        }
      ]
    }
  }
}
```

---

## Stage 4: QA Verification

Automatically verify generated QA pairs using configured verifiers (meaningful, correctness).

### Configuration

Add verifiers to `config.yml`:

```yaml
qa:
  verifiers:
    - strategy: meaningful
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0

    - strategy: correctness
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
```

### Verify QA Pairs

**Single document:**
```bash
docs2synth verify run data/processed/document_docling.json
```

**Batch verification:**
```bash
docs2synth verify batch
```

With options:
```bash
docs2synth verify batch data/processed/ \
  --verifier-type meaningful \
  --image-dir data/raw/my_documents/
```

**Output**: JSON files updated with verification results:
```json
{
  "qa": [
    {
      "question": "What is the invoice number?",
      "answer": "INV-2024-001",
      "verification": {
        "meaningful": {
          "response": "Yes",
          "explanation": "Question is clear and answerable"
        },
        "correctness": {
          "response": "Yes",
          "explanation": "Answer matches the text exactly"
        }
      }
    }
  ]
}
```

**View results:**
```bash
# Shows pass rate and statistics
# Example output:
# Done! Processed 10 files, 234 objects, verified 187 QA pairs
# Pass rate: 156/187 (83.4%) passed all verifiers
```

### Clean Verification Results

If you need to re-run verification:
```bash
docs2synth verify clean data/processed/
```

---

## Stage 5: Human Annotation

Manually review and annotate QA pairs using the interactive Streamlit interface.

### Launch Annotation Tool

```bash
docs2synth annotate
```

With custom settings:
```bash
docs2synth annotate data/processed/ \
  --image-dir data/raw/my_documents/ \
  --port 8502
```

### Annotation Interface

The tool provides:

1. **Document viewer**: Shows document image with bounding boxes
2. **QA pair display**: Lists all generated questions and answers
3. **Verifier results**: Shows automatic verification status
4. **Manual controls**:
   - Approve/Reject buttons
   - Add explanation notes
   - Navigate between documents and QA pairs
5. **Progress tracking**: Shows annotation statistics

**Workflow:**
1. Review each QA pair
2. Check if question is clear and answerable
3. Verify answer correctness
4. Approve good pairs, reject poor ones
5. Add notes for edge cases
6. Move to next document

**Output**: JSON files updated with `human_annotation` field:
```json
{
  "qa": [
    {
      "question": "What is the invoice number?",
      "answer": "INV-2024-001",
      "verification": {...},
      "human_annotation": {
        "approved": true,
        "timestamp": "2024-01-15T10:30:00Z",
        "annotator": "user@example.com",
        "notes": "Clear and accurate"
      }
    }
  ]
}
```

---

## Stage 6: Retriever Training

Train a custom document retriever model using the annotated QA pairs.

### Step 6.1: Preprocess Training Data

Convert annotated JSON files into training format:

```bash
docs2synth retriever preprocess
```

With options:
```bash
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

**Parameters:**
- `--require-all-verifiers`: Only include QA pairs that passed all verifiers (recommended)
- `--batch-size`: Training batch size (default: 8)
- `--max-length`: Maximum sequence length (default: 512)
- `--num-objects`: Max objects per document (default: 50)

**Output**: Preprocessed DataLoader pickle file ready for training

### Step 6.2: Train the Model

```bash
docs2synth retriever train \
  --mode standard \
  --lr 1e-5 \
  --epochs 10
```

**Training modes:**
- `standard`: Standard span-based QA training
- `layout`: Layout-aware training with grid representations
- `layout-gemini`: Gemini variant with grid representations
- `layout-coarse-grained`: Coarse-grained training
- `pretrain-layout`: Layout pretraining

**Full options:**
```bash
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
```

**Resume from checkpoint:**
```bash
docs2synth retriever train \
  --resume models/retriever/checkpoints/checkpoint_epoch_5.pth \
  --mode standard
```

**Output:**
- Checkpoints: `models/retriever/checkpoints/checkpoint_epoch_N.pth`
- Final model: `models/retriever/final_model.pth`
- Training curves: `models/retriever/checkpoints/training_curves.png`
- Training history: `models/retriever/checkpoints/training_history.json`

**Monitor training:**
```
Epoch 1/10
  Train ANLS: 0.7234 | Train Loss: 0.4567
  Val ANLS: 0.6891 | Val Loss: 0.5123
  Saved checkpoint: models/retriever/checkpoints/checkpoint_epoch_1.pth

Epoch 2/10
  Train ANLS: 0.7891 | Train Loss: 0.3456
  Val ANLS: 0.7234 | Val Loss: 0.4567
  ...
```

### Step 6.3: Validate the Model

Evaluate model performance on validation data:

```bash
docs2synth retriever validate
```

With options:
```bash
docs2synth retriever validate \
  --model models/retriever/final_model.pth \
  --data data/retriever/preprocessed_val.pkl \
  --output models/retriever/validation_reports/ \
  --mode standard \
  --device cuda
```

**Output:**
- Detailed analysis: `validation_reports/detailed_analysis.txt`
- Metrics plot: `validation_reports/validation_metrics.png`
- Console summary:
```
VALIDATION RESULTS
======================================================================

ANLS Score:
  Mean:            0.8234
  Std:             0.1567
  Median:          0.8456
  Range:           [0.0000, 1.0000]
  Perfect matches: 45 (23.4%)
  Zero matches:    12 (6.2%)

Prediction Length:
  Predicted mean:  3.45 words
  Ground truth:    3.67 words

SANITY CHECKS
======================================================================
✅ Empty predictions: 3 (1.6%) - Good
✅ Prediction diversity: 156 unique (81.2%) - Good

WORST PREDICTIONS (Top 5)
======================================================================
1. ANLS: 0.0234
   Predicted:     invoice date
   Ground Truth:  01/15/2024
```

---

## Stage 7: RAG Deployment

Deploy your trained retriever in a RAG (Retrieval-Augmented Generation) system.

### Step 7.1: Ingest Documents

Index your processed documents into the vector store:

```bash
docs2synth rag ingest
```

With options:
```bash
docs2synth rag ingest \
  --processed-dir data/processed/ \
  --processor docling \
  --include-context
```

**What happens:**
- Extracts text from each object in processed JSON files
- Embeds text using configured embedding model
- Stores in vector database with metadata (source, object_id, bbox, page, label)
- Context stored in metadata (not embedded) for reference

**Output:**
```
Ingested 1,234 document chunks into the vector store.
```

### Step 7.2: Query the RAG System

Ask questions using the command-line interface:

```bash
docs2synth rag run -q "What is the total amount on invoice INV-2024-001?"
```

With strategy selection:
```bash
docs2synth rag run \
  -s iterative \
  -q "What is the total amount on invoice INV-2024-001?" \
  --show-iterations
```

**Available strategies:**
- `naive`: Single retrieval + generation pass
- `iterative`: Multi-step refinement with feedback
- `rag_fusion`: Query reformulation and fusion
- `rerank`: Retrieval with re-ranking

**Output:**
```
Final answer:
The total amount on invoice INV-2024-001 is $1,234.56

Iterations:
Step 1
  Similarity vs. previous: N/A
  Answer:
    The total amount is $1,234.56
  Retrieved context:
    - score=0.891 source=data/processed/invoice_docling.json object_id=obj_23
    - score=0.823 source=data/processed/invoice_docling.json object_id=obj_45
```

### Step 7.3: Launch Interactive Demo

Start the Streamlit web interface:

```bash
docs2synth rag app
```

With options:
```bash
docs2synth rag app \
  --host localhost \
  --port 8501 \
  --no-browser
```

**Access at:** `http://localhost:8501`

**Features:**
- Interactive query interface
- Document preview with highlighted retrieved chunks
- Strategy comparison
- Query history
- Confidence scores
- Source attribution

### Step 7.4: Manage Vector Store

**List configured strategies:**
```bash
docs2synth rag strategies
```

**Reset vector store:**
```bash
docs2synth rag reset
```

---

## Configuration Reference

### Complete `config.yml` Example

```yaml
# Data directories
data:
  root_dir: ./data
  datasets_dir: ./data/datasets
  processed_dir: ./data/processed
  qa_pairs_dir: ./data/qa_pairs
  models_dir: ./models
  logs_dir: ./logs

# Document preprocessing
preprocess:
  processor: docling
  lang: en
  device: cuda
  input_dir: ./data/raw/my_documents/
  output_dir: ./data/processed/

# QA generation
qa:
  strategies:
    - strategy: semantic
      provider: openai
      model: gpt-4o-mini
      temperature: 0.7
      max_tokens: 150

    - strategy: layout_aware
      provider: anthropic
      model: claude-3-5-sonnet-20241022
      temperature: 0.5
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
  run_id: experiment_001
  preprocessed_data_path: ./data/retriever/{run_id}/preprocessed_train.pkl
  validation_data_path: ./data/retriever/{run_id}/preprocessed_val.pkl
  checkpoint_dir: ./models/retriever/{run_id}/checkpoints
  model_path: ./models/retriever/{run_id}/final_model.pth
  learning_rate: 1e-5
  epochs: 10
  save_every: 2

# RAG configuration
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
        k: 5
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

# LLM agent/provider keys
agent:
  provider: openai
  keys:
    openai_api_key: "sk-proj-..."
    anthropic_api_key: "sk-ant-..."
```

---

## Tips & Best Practices

### Data Preparation
- Start with 10-20 documents to test the workflow
- Use high-quality images (300 DPI or higher for scanned docs)
- Ensure consistent document types in each batch

### QA Generation
- Use multiple strategies to generate diverse questions
- Semantic strategy works best for factual extraction
- Layout-aware strategy good for form-like documents
- Monitor API costs (use cheaper models for dev/testing)

### Verification
- Always run verification before annotation
- Aim for >80% pass rate before human review
- If pass rate is low, adjust QA generation prompts

### Human Annotation
- Annotate at least 100-200 QA pairs for initial training
- Focus on high-quality annotations over quantity
- Document annotation guidelines for consistency
- Regular breaks to maintain annotation quality

### Retriever Training
- Start with 500-1000 annotated QA pairs minimum
- Use 80/20 train/validation split
- Monitor validation ANLS score (target >0.7)
- Early stopping if validation stops improving
- Experiment with different base models (LayoutLMv3, BERT)

### RAG Deployment
- Test with diverse queries before production
- Monitor retrieval quality (relevance of returned chunks)
- Tune retrieval parameters (k, similarity threshold)
- Consider query reformulation for better results
- Cache frequent queries to reduce API costs

---

## Troubleshooting

### Preprocessing Issues

**Problem**: OCR quality is poor
- Solution: Try different processors (docling, paddleocr, easyocr)
- Ensure input images are high resolution
- Check image preprocessing (rotation, contrast)

### QA Generation Issues

**Problem**: QA pairs are low quality
- Check LLM provider API keys and quotas
- Adjust temperature (lower = more deterministic)
- Review and improve generation prompts
- Use better models (gpt-4o instead of gpt-4o-mini)

**Problem**: Rate limits or API errors
- Add retry logic in config
- Use multiple providers (fallback strategy)
- Reduce batch size
- Implement request throttling

### Verification Issues

**Problem**: Low verification pass rate
- Review QA generation strategy configuration
- Check verifier prompts and thresholds
- Manually inspect failed examples
- Adjust verifier models or parameters

### Training Issues

**Problem**: Model not converging
- Reduce learning rate (try 1e-6 or 5e-6)
- Increase training data size
- Check for data quality issues
- Try different base models
- Monitor gradient norms for exploding gradients

**Problem**: Out of memory (OOM)
- Reduce batch size
- Reduce max_length parameter
- Use smaller base model
- Enable gradient checkpointing
- Use CPU training (slower but works)

**Problem**: Validation ANLS not improving
- Check for overfitting (train/val gap)
- Add more diverse training data
- Try different training mode (layout vs standard)
- Adjust hyperparameters (learning rate, epochs)

### RAG Issues

**Problem**: Retrieval returns irrelevant chunks
- Check embedding model quality
- Increase k (retrieve more candidates)
- Review chunking strategy (object-level vs paragraph-level)
- Use re-ranking strategy
- Fine-tune embedding model on domain data

**Problem**: Generated answers are incorrect
- Verify retrieved chunks contain necessary info
- Adjust generation prompt
- Use better LLM model
- Implement answer verification step
- Add context window management

---

## Next Steps

After completing this workflow, you can:

1. **Scale Up**: Process larger document collections
2. **Fine-tune**: Improve models with more annotated data
3. **Customize**: Adapt QA strategies for your specific domain
4. **Integrate**: Deploy RAG API for applications
5. **Benchmark**: Evaluate against standard datasets (FUNSD, CORD, DocVQA)
6. **Research**: Experiment with new strategies and models

---

## Additional Resources

- [Document Processing Guide](document-processing.md)
- [QA Generation Guide](qa-generation.md)
- [Retriever Training Guide](retriever-training.md)
- [RAG Deployment Guide](rag-path.md)
- [CLI Reference](../cli-reference.md)
- [API Reference](../api-reference.md)
- [GitHub Issues](https://github.com/AI4WA/Docs2Synth/issues)
