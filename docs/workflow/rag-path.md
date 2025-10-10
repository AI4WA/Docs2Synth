# RAG Path

Deploy document retrieval systems using out-of-box strategies without training custom models.

## Overview

The RAG (Retrieval-Augmented Generation) path provides pre-built retrieval strategies that work immediately without requiring custom model training. Ideal for rapid prototyping, smaller datasets, or when training resources are limited.

## Available Strategies

### BM25 (Sparse Retrieval)

Classic keyword-based retrieval using term frequency and inverse document frequency:

```python
from Docs2Synth.rag import BM25Retriever

# Initialize BM25 retriever
retriever = BM25Retriever()

# Index documents
retriever.index_documents(document_corpus)

# Retrieve
results = retriever.retrieve(
    query="What is document processing?",
    top_k=5
)
```

**Pros:**
- No training required
- Fast and efficient
- Works well for keyword-based queries
- Low memory footprint

**Cons:**
- Doesn't understand semantic meaning
- Sensitive to exact keyword matches
- Poor performance on paraphrased queries

### Dense Retrieval (Pre-trained)

Use pre-trained sentence transformers:

```python
from Docs2Synth.rag import DenseRetriever

# Initialize with pre-trained model
retriever = DenseRetriever(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Index documents
retriever.index_documents(document_corpus)

# Retrieve
results = retriever.retrieve(
    query="What is document processing?",
    top_k=5
)
```

**Available Models:**
- `all-mpnet-base-v2`: Best general-purpose model
- `all-MiniLM-L6-v2`: Fast and lightweight
- `multi-qa-mpnet-base-dot-v1`: Optimized for Q&A
- `msmarco-distilbert-base-v4`: Trained on MS MARCO dataset

**Pros:**
- Understands semantic similarity
- Handles paraphrased queries
- No training required

**Cons:**
- May not be optimal for specific domains
- Slower than BM25
- Higher memory usage

### Hybrid Retrieval

Combine sparse and dense retrieval:

```python
from Docs2Synth.rag import HybridRetriever

# Initialize hybrid retriever
retriever = HybridRetriever(
    sparse_weight=0.3,  # BM25 weight
    dense_weight=0.7,   # Dense retriever weight
    dense_model="sentence-transformers/all-mpnet-base-v2"
)

# Index documents
retriever.index_documents(document_corpus)

# Retrieve
results = retriever.retrieve(
    query="What is document processing?",
    top_k=5
)
```

**Pros:**
- Best of both worlds
- Robust to different query types
- Better overall performance

**Cons:**
- More complex setup
- Slower than individual methods

### ColBERT (Late Interaction)

Token-level matching for fine-grained retrieval:

```python
from Docs2Synth.rag import ColBERTRetriever

# Initialize ColBERT
retriever = ColBERTRetriever(
    model_name="colbert-ir/colbertv2.0"
)

# Index documents (pre-compute token embeddings)
retriever.index_documents(document_corpus)

# Retrieve
results = retriever.retrieve(
    query="What is document processing?",
    top_k=5
)
```

**Pros:**
- High accuracy
- Captures fine-grained matches
- Good for complex documents

**Cons:**
- High storage requirements
- Slower indexing
- More memory intensive

## Quick Start Examples

### Minimal Setup

```python
from Docs2Synth.rag import QuickRetriever

# Automatic strategy selection
retriever = QuickRetriever.auto(
    document_corpus=documents,
    strategy="best"  # or "fast", "balanced"
)

# Query
results = retriever.retrieve("your query here", top_k=5)
```

### With Configuration

```python
from Docs2Synth.rag import QuickRetriever

config = {
    "strategy": "hybrid",
    "sparse_weight": 0.3,
    "dense_weight": 0.7,
    "index_type": "faiss",
    "normalize_scores": True
}

retriever = QuickRetriever(config)
retriever.index_documents(documents)
```

## Indexing Strategies

### In-Memory Index

Fast, but limited by RAM:

```python
retriever = DenseRetriever(index_type="memory")
retriever.index_documents(documents)
```

### FAISS Index

For large-scale retrieval:

```python
from Docs2Synth.rag.indexing import FAISSIndex

retriever = DenseRetriever(
    index_type="faiss",
    index_config={
        "type": "IVF",  # Inverted file index
        "nlist": 100,   # Number of clusters
        "nprobe": 10    # Number of clusters to search
    }
)

retriever.index_documents(documents)
```

### Persistent Index

Save and load indexes:

```python
# Index and save
retriever.index_documents(documents)
retriever.save_index("my_index")

# Load later
retriever = DenseRetriever.load_index("my_index")
```

## Re-ranking

Improve retrieval quality with re-ranking:

```python
from Docs2Synth.rag import ReRanker

# Initial retrieval
retriever = DenseRetriever()
initial_results = retriever.retrieve(query, top_k=100)

# Re-rank top results
reranker = ReRanker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

final_results = reranker.rerank(
    query=query,
    documents=initial_results,
    top_k=5
)
```

## Integration with LLMs

### RAG Pipeline

```python
from Docs2Synth.rag import RAGPipeline

# Complete RAG setup
rag = RAGPipeline(
    retriever_strategy="hybrid",
    llm="gpt-4",
    context_window=5
)

# Query and generate
response = rag.query(
    "What are the main features of the system?",
    return_sources=True
)

print(response.answer)
print(f"Sources: {response.sources}")
```

### Custom Prompts

```python
from Docs2Synth.rag import RAGPipeline

prompt_template = """
Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:
"""

rag = RAGPipeline(
    retriever_strategy="dense",
    llm="gpt-4",
    prompt_template=prompt_template
)

response = rag.query("your question")
```

## Evaluation

### Retrieval Metrics

```python
from Docs2Synth.rag.evaluation import evaluate_retrieval

# Evaluate retriever
results = evaluate_retrieval(
    retriever=retriever,
    test_queries=test_queries,
    ground_truth=ground_truth,
    metrics=["hit@1", "hit@5", "mrr", "ndcg@10"]
)

print(results)
```

### End-to-End Evaluation

```python
from Docs2Synth.rag.evaluation import evaluate_rag

# Evaluate complete RAG pipeline
results = evaluate_rag(
    rag_pipeline=rag,
    test_questions=questions,
    ground_truth_answers=answers,
    metrics=["accuracy", "faithfulness", "relevance"]
)
```

## Performance Optimization

### Batch Processing

```python
# Process multiple queries efficiently
queries = ["query 1", "query 2", "query 3"]

results = retriever.batch_retrieve(
    queries=queries,
    top_k=5,
    batch_size=32
)
```

### Caching

```python
from Docs2Synth.rag import CachedRetriever

# Enable caching for repeated queries
retriever = CachedRetriever(
    base_retriever=DenseRetriever(),
    cache_size=1000
)
```

## Best Practices

1. **Start with hybrid retrieval**: Best balance of performance and simplicity
2. **Use re-ranking for top results**: Significantly improves precision
3. **Monitor latency**: Track retrieval time in production
4. **Cache frequent queries**: Reduce repeated computation
5. **Tune top_k**: Usually 5-10 is sufficient for most applications
6. **Index optimization**: Use FAISS for large document collections

## Benchmarking

Compare different strategies:

```python
from Docs2Synth.rag import benchmark_strategies

# Compare multiple strategies
results = benchmark_strategies(
    document_corpus=corpus,
    test_queries=queries,
    strategies=["bm25", "dense", "hybrid", "colbert"],
    metrics=["hit@5", "mrr", "latency"]
)

# Generate comparison report
benchmark.generate_report(results, "strategy_comparison.html")
```

## When to Use RAG vs Training

**Use RAG (No Training) when:**
- Quick prototyping needed
- Small to medium datasets (<100K documents)
- Limited computational resources
- General-purpose retrieval is sufficient

**Train Custom Retriever when:**
- Large-scale datasets (>100K documents)
- Highly specialized domain
- Need optimal performance
- Have computational resources

## Next Steps

- Combine with [trained retrievers](retriever-training.md) for hybrid approaches
- Deploy to production with monitoring
- Iterate on prompt engineering for better LLM responses

## API Reference

For detailed API documentation, see the [API Reference](../api-reference.md#rag).
