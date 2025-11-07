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

### Python API

=== "Simple QA Generation (explicit provider)"
    ```python
    from docs2synth.agent import QAGenerator

    # Explicit provider/model
    generator = QAGenerator(provider="openai", model="gpt-3.5-turbo")

    content = "Python is a high-level programming language..."
    qa_pair = generator.generate_qa_pair(content)

    print(f"Question: {qa_pair['question']}")
    print(f"Answer: {qa_pair['answer']}")
    ```

=== "Batch Generation"
    ```python
    from docs2synth.agent import QAGenerator

    generator = QAGenerator(provider="openai", model="gpt-3.5-turbo")

    # Generate multiple QA pairs
    contents = ["Content 1...", "Content 2...", "Content 3..."]
    qa_pairs = generator.generate_qa_pairs(contents)

    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
    ```

=== "With Verification"
    ```python
    from docs2synth.agent import QAGenerator

    generator = QAGenerator(provider="openai", model="gpt-3.5-turbo")

    # Generate with two-step verification
    qa_pair = generator.generate_with_verification(
        content,
        meaningful_check=True,
        correctness_check=True
    )

    if qa_pair:
        print("✓ QA pair passed verification")
    ```

=== "Switch Providers"
    ```python
    from docs2synth.agent import AgentWrapper, QAGenerator

    # Use OpenAI
    generator = QAGenerator(provider="openai", model="gpt-4")

    # Switch to Ollama (local)
    generator = QAGenerator(provider="ollama", model="llama2")

    # Switch to Anthropic Claude
    generator = QAGenerator(provider="anthropic", model="claude-3-5-sonnet-20241022")
    ```

## Supported Providers

The agent wrapper supports multiple LLM providers:

### Cloud APIs
- **OpenAI** (`openai`): GPT-4, GPT-3.5-turbo, etc.
- **Anthropic** (`anthropic` or `claude`): Claude 3.5 Sonnet, Claude 3 Opus, etc.
- **Google Gemini** (`gemini` or `google`): gemini-pro, gemini-pro-vision
- **豆包/Doubao** (`doubao`): doubao-pro-32k, doubao-lite-4k

### Local Models
- **Ollama** (`ollama`): llama2, mistral, codellama, etc. (requires local Ollama server)
- **Hugging Face** (`huggingface` or `hf`): Any Hugging Face model (requires local GPU/CPU)

## Configuration

Provider-driven configuration in `config.yml` (recommended):

```yaml
agent:
  # Choose active provider here: openai | anthropic | gemini | doubao | ollama | huggingface
  provider: openai

  # Centralized API keys (used to backfill per-provider configs)
  keys:
    openai_api_key: "sk-proj-..."
    anthropic_api_key: "sk-ant-..."
    google_api_key: "..."
    doubao_api_key: "..."
    huggingface_token: "hf_..."  # for gated models

  # Provider-specific blocks. Switch by changing agent.provider.
  openai:
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 1000

  anthropic:
    model: claude-3-5-sonnet-20241022
    temperature: 0.7
    max_tokens: 1000

  gemini:
    model: gemini-1.5-pro
    temperature: 0.7
    max_output_tokens: 1000

  doubao:
    model: doubao-pro-32k
    base_url: https://ark.cn-beijing.volces.com/api/v3
    temperature: 0.7
    max_tokens: 1000

  ollama:
    model: llama3
    host: http://localhost:11434
    temperature: 0.7

  huggingface:
    model: meta-llama/Llama-2-7b-chat-hf
    device: auto
    max_new_tokens: 512
    temperature: 0.7
```

Loading behavior:
- By default, code does NOT auto-read `config.yml`.
- To load config automatically, either set `DOCS2SYNTH_CONFIG=/path/to/config.yml` or pass `config_path` when constructing the agent.

Examples:

```python
# Auto-read via environment variable
import os
os.environ["DOCS2SYNTH_CONFIG"] = "./config.yml"
from docs2synth.agent import AgentWrapper
agent = AgentWrapper()

# Or pass config_path explicitly
from docs2synth.agent import AgentWrapper
agent = AgentWrapper(config_path="./config.yml")
```

## Advanced Usage

### Direct Agent Usage

```python
from docs2synth.agent import AgentWrapper

# Initialize agent
agent = AgentWrapper(provider="openai", model="gpt-4")

# Generate text
response = agent.generate(
    prompt="Explain machine learning",
    system_prompt="You are a helpful AI assistant.",
    temperature=0.7,
    max_tokens=500
)

print(response.content)
print(f"Token usage: {response.usage}")

# Chat interface
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What's the weather?"}
]
response = agent.chat(messages)
print(response.content)
```

### Custom Prompts

```python
from docs2synth.agent import QAGenerator

custom_prompt = """Generate a question-answer pair from this content:

{content}

Format as JSON with 'question' and 'answer' keys."""

generator = QAGenerator(
    provider="openai",
    model="gpt-4",
    prompt_template=custom_prompt
)
```