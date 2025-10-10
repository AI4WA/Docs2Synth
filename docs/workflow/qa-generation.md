# QA Generation

Generate high-quality question-answer pairs from processed documents using agent-based approaches with built-in verification.

## Overview

The QA generation workflow uses LLM-based agents to automatically create question-answer pairs from document content, with a two-step verification process to ensure quality.

## Architecture

```
Document Content
      ↓
QA Pair Generation (LLM)
      ↓
Meaningful Verifier (Agent 1)
      ↓
Correctness Checker (Agent 2)
      ↓
Human Judgement (Optional)
      ↓
Final QA Dataset
```

## Basic Usage

### Simple Generation

```python
from Docs2Synth.qa import generator

# Generate QA pairs from a document
qa_pairs = generator.generate_qa_pairs(
    document_content="Your document text here",
    num_pairs=10
)

# Output format
# [
#   {
#     "question": "What is...",
#     "answer": "The answer is...",
#     "context": "Relevant excerpt from document",
#     "metadata": {...}
#   }
# ]
```

### CLI Usage

```bash
# Generate QA pairs from processed documents
docs2synth generate-qa /path/to/documents /path/to/output.jsonl

# With custom settings
docs2synth generate-qa \
    /path/to/documents \
    /path/to/output.jsonl \
    --num-pairs 50 \
    --model gpt-4 \
    --verify
```

## QA Generation Strategy

### Single-Page Context

Generate QA pairs from individual pages:

```python
from Docs2Synth.qa import generator

# Single-page generation
qa_pairs = generator.generate_from_page(
    page_content=page_text,
    page_number=1,
    num_pairs=5
)
```

### Multi-Page Context (Future)

For complex questions spanning multiple pages:

```python
# Coming soon: multi-page context
qa_pairs = generator.generate_from_pages(
    pages=[page1, page2, page3],
    num_pairs=10,
    context_window=3
)
```

## Two-Step Verification

### 1. Meaningful Verifier

Checks if the question is meaningful and answerable:

```python
from Docs2Synth.qa.verification import meaningful_verifier

# Verify meaningfulness
is_meaningful = meaningful_verifier.verify(
    question="What is the capital of France?",
    context="France is a country in Europe..."
)
# Returns: {
#   "is_meaningful": True,
#   "reason": "Question is clear and answerable",
#   "score": 0.95
# }
```

**Criteria checked:**
- Question clarity and specificity
- Answerability from context
- Relevance to document content
- Avoiding ambiguity

### 2. Correctness Checker

Verifies that the answer is accurate given the context:

```python
from Docs2Synth.qa.verification import correctness_checker

# Check correctness
is_correct = correctness_checker.verify(
    question="What is the capital of France?",
    answer="Paris",
    context="The capital of France is Paris..."
)
# Returns: {
#   "is_correct": True,
#   "confidence": 0.98,
#   "issues": []
# }
```

**Criteria checked:**
- Factual accuracy
- Answer completeness
- Consistency with context
- No hallucinations

## Human Judgement

Quick human review for final quality control:

```python
from Docs2Synth.qa import human_review

# Review interface
reviewed = human_review.annotate(
    qa_pairs=qa_pairs,
    output_file="reviewed_qa.jsonl"
)

# Human annotator sees:
# Question: "What is...?"
# Answer: "The answer is..."
# [Keep] [Discard] [Edit]
```

### CLI Review Tool

```bash
# Launch interactive review interface
docs2synth review-qa qa_pairs.jsonl --output reviewed_qa.jsonl
```

## Configuration

Configure QA generation in `config.yml`:

```yaml
qa_generation:
  model: gpt-4
  temperature: 0.7
  num_pairs_per_page: 5

  verification:
    enable_meaningful_check: true
    enable_correctness_check: true
    meaningful_threshold: 0.8
    correctness_threshold: 0.85

  human_review:
    enable: true
    sample_rate: 0.1  # Review 10% of pairs
```

## Advanced Features

### Custom Prompts

Customize the generation prompts:

```python
from Docs2Synth.qa import generator

custom_prompt = """
Given the following document excerpt, generate a question-answer pair
focusing on technical details and implementation specifics.

Document: {context}

Generate:
"""

qa_pairs = generator.generate_qa_pairs(
    document_content=text,
    prompt_template=custom_prompt
)
```

### Filtering and Post-processing

```python
from Docs2Synth.qa import filters

# Filter by quality scores
high_quality = filters.filter_by_score(
    qa_pairs,
    min_meaningful_score=0.9,
    min_correctness_score=0.9
)

# Remove duplicates
unique_pairs = filters.remove_duplicates(qa_pairs)

# Balance question types
balanced = filters.balance_question_types(qa_pairs)
```

## Quality Metrics

Track generation quality:

```python
from Docs2Synth.qa import metrics

# Evaluate generated QA pairs
evaluation = metrics.evaluate_qa_dataset(qa_pairs)

print(evaluation)
# {
#   "total_pairs": 100,
#   "avg_meaningful_score": 0.92,
#   "avg_correctness_score": 0.89,
#   "question_type_distribution": {...},
#   "avg_answer_length": 45
# }
```

## Best Practices

1. **Start with high-quality documents**: Better OCR → Better QA
2. **Use verification**: Don't skip the two-step verification
3. **Human review sample**: Review at least 10% manually
4. **Diverse question types**: Ensure mix of factual, reasoning, and analytical questions
5. **Context preservation**: Keep context snippets for debugging

## Data Format

Generated QA pairs are saved in JSONL format:

```json
{
  "question": "What is the main purpose of the system?",
  "answer": "The system is designed to process documents efficiently.",
  "context": "The document processing system...",
  "metadata": {
    "document_id": "doc_001",
    "page_number": 5,
    "meaningful_score": 0.95,
    "correctness_score": 0.92,
    "generated_at": "2025-01-15T10:30:00Z"
  }
}
```

## Next Steps

After generating QA pairs, proceed to [Retriever Training](retriever-training.md) to train custom retrieval models.

## API Reference

For detailed API documentation, see the [API Reference](../api-reference.md#qa).
