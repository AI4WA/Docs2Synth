# QA Generation

Generate question-answer pairs from documents using LLMs.

## Strategies

### Semantic

Generate questions from document context and target answer.

```bash
docs2synth qa semantic "Form contains name field" "John Doe"
docs2synth qa semantic "Context" "Target" --provider anthropic
```

### Layout-Aware

Transform questions to be spatially aware (position-based).

```bash
docs2synth qa layout "What is the address?" --image document.png
```

Example output:
```
Original: What is the address?
Layout-aware: What text is in the upper right section?
```

### Logical-Aware

Transform questions to be structure-aware (section-based).

```bash
docs2synth qa logical "What is the name?" --image document.png
```

Example output:
```
Original: What is the name?
Logical-aware: What value is in the 'Personal Information' section for the 'Name' field?
```

---

## Batch Processing

### Single Document

```bash
docs2synth qa run data/processed/document.json
docs2synth qa run data/images/document.png
docs2synth qa run data/processed/document.json --strategy semantic
```

### Batch Generation

```bash
# Uses config.preprocess.input_dir
docs2synth qa batch

# Explicit paths
docs2synth qa batch data/raw/my_documents/
docs2synth qa batch data/images/ --output-dir data/processed/ --processor docling
```

---

## Configuration

**`config.yml`:**
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

    - strategy: logical_aware
      provider: gemini
      model: gemini-1.5-flash
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
```

---

## Verification

Automatically verify QA quality with built-in verifiers.

### Meaningful Verifier

Checks if question is clear and answerable.

```bash
docs2synth verify run data/processed/document.json --verifier-type meaningful
```

### Correctness Checker

Validates answer matches document content.

```bash
docs2synth verify run data/processed/document.json --verifier-type correctness
```

### Batch Verification

```bash
docs2synth verify batch
docs2synth verify batch data/processed/dev
docs2synth verify batch --image-dir data/images
```

---

## Output Format

JSON updated with `qa` field:

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
          "answer": "INV-2024-001",
          "verification": {
            "meaningful": {
              "response": "Yes",
              "explanation": "Question is clear and answerable"
            },
            "correctness": {
              "response": "Yes",
              "explanation": "Answer matches text"
            }
          }
        }
      ]
    }
  }
}
```

---

## Supported Providers

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-opus`, `claude-3-sonnet`
- **Google**: `gemini-1.5-pro`, `gemini-1.5-flash`
- **Doubao**: ByteDance models
- **Ollama**: Local models (llama3, mistral, etc.)
- **vLLM**: High-performance local inference

---

## Next Steps

- [Verification](complete-workflow.md#stage-4-verification) - Verify QA quality
- [Human Annotation](complete-workflow.md#stage-5-human-annotation) - Manual review
- [Retriever Training](retriever-training.md) - Train models on QA pairs

For complete workflow: [Complete Workflow Guide](complete-workflow.md)
