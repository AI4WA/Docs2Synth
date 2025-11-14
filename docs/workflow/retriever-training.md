# Retriever Training

Train custom document retrieval models on annotated QA pairs.

## Quick Start

```bash
# 1. Preprocess data
docs2synth retriever preprocess \
  --json-dir data/processed/ \
  --image-dir data/raw/my_documents/ \
  --require-all-verifiers

# 2. Train model
docs2synth retriever train \
  --mode standard \
  --lr 1e-5 \
  --epochs 10 \
  --device cuda

# 3. Validate
docs2synth retriever validate
```

---

## Training Modes

### Standard

Span-based QA training for text retrieval.

```bash
docs2synth retriever train --mode standard --lr 1e-5 --epochs 10
```

### Layout

Layout-aware training with grid representations.

```bash
docs2synth retriever train --mode layout --lr 1e-5 --epochs 10
```

### Layout-Gemini

Gemini variant with grid representations.

```bash
docs2synth retriever train --mode layout-gemini --lr 5e-6 --epochs 10
```

### Layout-Coarse-Grained

Coarse-grained layout training.

```bash
docs2synth retriever train --mode layout-coarse-grained --lr 1e-5 --epochs 10
```

### Pretrain-Layout

Layout pretraining using grid embeddings.

```bash
docs2synth retriever train --mode pretrain-layout --lr 1e-4 --epochs 5
```

---

## Data Preprocessing

Convert annotated JSON to training format.

```bash
docs2synth retriever preprocess \
  --json-dir data/processed/ \
  --image-dir data/raw/my_documents/ \
  --output data/retriever/train.pkl \
  --processor docling \
  --batch-size 8 \
  --max-length 512 \
  --num-objects 50 \
  --require-all-verifiers
```

**Parameters:**
- `--require-all-verifiers`: Only include QA pairs passing all verifiers
- `--batch-size`: Training batch size (default: 8)
- `--max-length`: Max sequence length (default: 512)
- `--num-objects`: Max objects per document (default: 50)

---

## Training

```bash
# Basic
docs2synth retriever train --mode standard --lr 1e-5 --epochs 10

# Full options
docs2synth retriever train \
  --data-path data/retriever/train.pkl \
  --val-data-path data/retriever/val.pkl \
  --output-dir models/retriever/checkpoints/ \
  --mode standard \
  --base-model microsoft/layoutlmv3-base \
  --lr 1e-5 \
  --epochs 10 \
  --batch-size 8 \
  --save-every 2 \
  --device cuda

# Resume from checkpoint
docs2synth retriever train \
  --resume models/retriever/checkpoints/checkpoint_epoch_5.pth \
  --mode standard
```

**Base models:**
- `microsoft/layoutlmv3-base` (default, 133M params)
- `microsoft/layoutlmv3-large` (368M params)

---

## Validation

Evaluate model performance on validation set.

```bash
docs2synth retriever validate

# With options
docs2synth retriever validate \
  --model models/retriever/final_model.pth \
  --data data/retriever/val.pkl \
  --output models/retriever/validation_reports/
```

**Metrics:**
- **ANLS Score**: Average Normalized Levenshtein Similarity
- **Loss**: Validation loss
- **Perfect matches**: Percentage of exact predictions
- **Prediction diversity**: Unique predictions ratio

---

## Output Files

Training outputs:
- `models/retriever/checkpoints/checkpoint_epoch_N.pth` - Epoch checkpoints
- `models/retriever/final_model.pth` - Final trained model
- `models/retriever/checkpoints/training_curves.png` - Training plots
- `models/retriever/checkpoints/training_history.json` - Metrics history

Validation outputs:
- `validation_reports/detailed_analysis.txt` - Full analysis
- `validation_reports/validation_metrics.png` - Metrics visualization

---

## Configuration

**`config.yml`:**
```yaml
retriever:
  run_id: experiment_001
  preprocessed_data_path: ./data/retriever/{run_id}/train.pkl
  validation_data_path: ./data/retriever/{run_id}/val.pkl
  checkpoint_dir: ./models/retriever/{run_id}/checkpoints
  model_path: ./models/retriever/{run_id}/final_model.pth
  learning_rate: 1e-5
  epochs: 10
  batch_size: 8
  save_every: 2
  base_model: microsoft/layoutlmv3-base
```

---

## Next Steps

- [RAG Deployment](rag-path.md) - Deploy trained retriever
- [Complete Workflow](complete-workflow.md) - Full workflow guide
