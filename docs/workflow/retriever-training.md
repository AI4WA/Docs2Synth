# Retriever Training

Train custom retrieval models on your QA dataset to build domain-specific document retrievers.

## Overview

Docs2Synth supports training various retriever architectures, from lightweight sentence transformers to specialized document understanding models like LayoutLMv3.

## Supported Models

### Sentence Transformers

Fast, efficient retrievers based on BERT and similar architectures:

- `sentence-transformers/all-MiniLM-L6-v2` (Lightweight, fast)
- `sentence-transformers/all-mpnet-base-v2` (Best quality/speed tradeoff)
- `sentence-transformers/multi-qa-mpnet-base-dot-v1` (Optimized for QA)

### LayoutLMv3

Layout-aware retriever that understands document structure:

```python
from Docs2Synth.retriever import train

# Train LayoutLMv3 retriever
model = train.train_retriever(
    qa_pairs="qa_pairs.jsonl",
    model_name="microsoft/layoutlmv3-base",
    output_dir="models/layoutlmv3-retriever"
)
```

### BERT Variants

Standard BERT-based models:

- `bert-base-uncased`
- `roberta-base`
- `distilbert-base-uncased` (Faster, smaller)

## Basic Training

### CLI Usage

```bash
# Train with default settings
docs2synth train-retriever qa_pairs.jsonl \
    --output-dir models/retriever

# Custom model
docs2synth train-retriever qa_pairs.jsonl \
    --output-dir models/retriever \
    --model-name sentence-transformers/all-mpnet-base-v2 \
    --epochs 5 \
    --batch-size 32
```

### Python API

```python
from Docs2Synth.retriever import train

# Load QA pairs
qa_pairs = train.load_qa_pairs("qa_pairs.jsonl")

# Train retriever
model = train.train_retriever(
    qa_pairs=qa_pairs,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    output_dir="models/retriever",
    epochs=5,
    batch_size=32,
    learning_rate=2e-5
)

print(f"Model saved to {model.output_dir}")
```

## Training Configuration

### Basic Configuration

```python
training_config = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "evaluation_steps": 1000,
    "save_steps": 1000,
}

model = train.train_retriever(
    qa_pairs=qa_pairs,
    **training_config
)
```

### Advanced Configuration

```python
advanced_config = {
    # Model settings
    "model_name": "microsoft/layoutlmv3-base",
    "max_seq_length": 512,
    "pooling_mode": "mean",

    # Training settings
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,

    # Loss function
    "loss": "MultipleNegativesRankingLoss",
    "scale": 20.0,

    # Evaluation
    "evaluation_steps": 500,
    "eval_batch_size": 64,

    # Hardware
    "use_amp": True,  # Automatic Mixed Precision
    "num_workers": 4,
    "device": "cuda",
}

model = train.train_retriever(
    qa_pairs=qa_pairs,
    **advanced_config
)
```

## Loss Functions

### Multiple Negatives Ranking Loss

Best for most use cases:

```python
from Docs2Synth.retriever.losses import MultipleNegativesRankingLoss

loss = MultipleNegativesRankingLoss(
    model=model,
    scale=20.0,  # Temperature for softmax
    similarity_fct="cosine"
)
```

### Contrastive Loss

For harder negative mining:

```python
from Docs2Synth.retriever.losses import ContrastiveLoss

loss = ContrastiveLoss(
    model=model,
    margin=0.5,
    distance_metric="euclidean"
)
```

## Data Preparation

### From QA Pairs

```python
from Docs2Synth.retriever.dataloaders import QADataLoader

# Load and prepare data
dataloader = QADataLoader(
    qa_pairs="qa_pairs.jsonl",
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)

train_data = dataloader.get_train_data()
val_data = dataloader.get_val_data()
test_data = dataloader.get_test_data()
```

### Hard Negative Mining

Improve retriever quality with hard negatives:

```python
from Docs2Synth.retriever import hard_negatives

# Mine hard negatives
enhanced_data = hard_negatives.mine_hard_negatives(
    qa_pairs=qa_pairs,
    corpus=document_corpus,
    num_negatives=5,
    strategy="bm25"  # or "random", "semi-hard"
)
```

## Evaluation

### During Training

```python
from Docs2Synth.retriever.evaluation import Evaluator

evaluator = Evaluator(
    val_data=val_data,
    metrics=["hit@1", "hit@5", "hit@10", "mrr", "ndcg@10"]
)

# Evaluate during training
model = train.train_retriever(
    qa_pairs=qa_pairs,
    evaluator=evaluator,
    evaluation_steps=500
)
```

### Post-Training Evaluation

```python
from Docs2Synth.retriever.evaluation import evaluate_retriever

# Evaluate trained model
results = evaluate_retriever(
    model=model,
    test_data=test_data,
    metrics=["hit@1", "hit@5", "hit@10", "mrr"]
)

print(results)
# {
#   "hit@1": 0.75,
#   "hit@5": 0.92,
#   "hit@10": 0.96,
#   "mrr": 0.82
# }
```

## Inference

### Retrieve Documents

```python
from Docs2Synth.retriever.inference import Retriever

# Load trained model
retriever = Retriever.from_pretrained("models/retriever")

# Index documents
retriever.index_documents(document_corpus)

# Retrieve for a query
results = retriever.retrieve(
    query="What is the main purpose of the system?",
    top_k=5
)

for rank, result in enumerate(results, 1):
    print(f"{rank}. {result['document_id']} (score: {result['score']:.3f})")
```

### Batch Retrieval

```python
# Retrieve for multiple queries
queries = ["Query 1", "Query 2", "Query 3"]

results = retriever.batch_retrieve(
    queries=queries,
    top_k=10,
    batch_size=32
)
```

## Optimization

### Speed Optimization

```python
# Use ONNX for faster inference
from Docs2Synth.retriever.optimization import optimize_for_inference

optimized_model = optimize_for_inference(
    model_path="models/retriever",
    format="onnx",
    quantization="dynamic"
)

# 2-3x faster inference
```

### Index Optimization

```python
# Use FAISS for large-scale retrieval
from Docs2Synth.retriever.indexing import FAISSIndex

index = FAISSIndex(
    model=retriever,
    index_type="IVF",  # Inverted file index
    nlist=100  # Number of clusters
)

index.add_documents(document_corpus)
results = index.search(query, top_k=10)
```

## Best Practices

1. **Start small**: Begin with a lightweight model (MiniLM) for prototyping
2. **Use hard negatives**: Significantly improves retrieval quality
3. **Monitor validation metrics**: Watch for overfitting
4. **Tune hyperparameters**: Learning rate and batch size matter
5. **Evaluate on held-out test set**: Don't overfit to validation set
6. **Consider domain adaptation**: Fine-tune on your specific document type

## Benchmarking

Track retrieval performance:

```python
from Docs2Synth.retriever import benchmark

# Run comprehensive benchmark
results = benchmark.evaluate_retriever(
    model=model,
    test_queries=test_queries,
    document_corpus=corpus,
    metrics=["hit@k", "mrr", "ndcg", "latency"]
)

# Generate report
benchmark.generate_report(results, output="benchmark_report.html")
```

## Next Steps

After training your retriever, you can:

- Deploy for [RAG applications](rag-path.md)
- Integrate into production pipelines
- Continue monitoring and improving performance

## API Reference

For detailed API documentation, see the [API Reference](../api-reference.md#retriever).
