# CLI Reference

Docs2Synth provides a comprehensive command-line interface for document processing, QA generation, and agent interactions.

## Global Options

These options are available for all commands:

```bash
docs2synth --help                    # Show help message
docs2synth --version                 # Show version
docs2synth -v [COMMAND]              # Verbose output
docs2synth -vv [COMMAND]             # Very verbose output
docs2synth --config CONFIG [COMMAND] # Use custom config file
```

---

## Dataset Management

### List Available Datasets

List all datasets that can be downloaded:

```bash
docs2synth datasets list
```

**Output Example:**
```
Available datasets:
  - cord
  - funsd
  - vrd-iu2024-tracka
  - vrd-iu2024-trackb
```

### Download a Dataset

Download a specific dataset:

```bash
docs2synth datasets download vrd-iu2024-tracka
```

With custom output directory:

```bash
docs2synth datasets download vrd-iu2024-tracka --output-dir ./data
```

### Download All Datasets

Download all available datasets at once:

```bash
docs2synth datasets download all
```

With custom output directory:

```bash
docs2synth datasets download all --output-dir ./data
```

---

## Document Preprocessing

Process documents with OCR to extract text and structure.

### Basic Preprocessing

Process a single image:

```bash
docs2synth preprocess document.png
```

Process all images in a directory:

```bash
docs2synth preprocess ./images/
```

### Processor Options

#### PaddleOCR (Default - General Purpose)

```bash
docs2synth preprocess document.png --processor paddleocr
```

#### PDFPlumber (For Parsed PDFs)

```bash
docs2synth preprocess document.pdf --processor pdfplumber
```

#### EasyOCR (80+ Languages)

```bash
docs2synth preprocess document.png --processor easyocr
```

### Advanced Options

Specify OCR language:

```bash
docs2synth preprocess document.png --lang en
```

Custom output directory:

```bash
docs2synth preprocess document.png --output-dir ./processed
```

Specify device (CPU/GPU):

```bash
docs2synth preprocess document.png --device gpu
docs2synth preprocess document.png --device cpu
```

Complete example with all options:

```bash
docs2synth preprocess ./images/ \
  --processor paddleocr \
  --lang en \
  --output-dir ./processed \
  --device gpu
```

---

## Agent Commands

Interact with LLM agents for text generation and chat.

### Start vLLM Server

Start a vLLM OpenAI-compatible API server for high-performance local inference:

```bash
# Start with config.yml settings
docs2synth agent vllm-server

# Override model
docs2synth agent vllm-server --model meta-llama/Llama-2-7b-chat-hf

# Custom port and GPU settings
docs2synth agent vllm-server --port 8080 --gpu-memory-utilization 0.8

# Enable trust_remote_code for models like Qwen
docs2synth agent vllm-server --trust-remote-code

# Multi-GPU setup (use 2 GPUs)
docs2synth agent vllm-server --tensor-parallel-size 2
```

**Server Options:**
- `--config-path`: Path to config.yml (default: ./config.yml)
- `--model`: Model to load (overrides config)
- `--host`: Host to bind (default: 0.0.0.0)
- `--port`: Port to bind (default: 8000)
- `--trust-remote-code`: Enable for custom models (Qwen, Phi, etc.)
- `--max-model-len`: Maximum context length
- `--gpu-memory-utilization`: GPU memory usage (0.0-1.0)
- `--tensor-parallel-size`: Number of GPUs for parallelism

**After starting the server:**
1. Keep the terminal open (press Ctrl+C to stop)
2. Test: `curl http://localhost:8000/health`
3. Use the vLLM provider in another terminal:
   ```bash
   docs2synth agent generate "Your prompt" --provider vllm
   ```

### Generate Text

Basic text generation:

```bash
docs2synth agent generate "Explain quantum computing"
```

#### Provider Options

Use different LLM providers:

```bash
# OpenAI (default)
docs2synth agent generate "Your prompt" --provider openai

# Anthropic Claude
docs2synth agent generate "Your prompt" --provider anthropic

# Google Gemini
docs2synth agent generate "Your prompt" --provider gemini

# ByteDance Doubao
docs2synth agent generate "Your prompt" --provider doubao

# Ollama (local models)
docs2synth agent generate "Your prompt" --provider ollama

# Hugging Face
docs2synth agent generate "Your prompt" --provider huggingface

# vLLM (high-performance local inference)
# Fastest option for local deployment, best with server mode

# Server mode (recommended): Start server then use provider
# 1. Start vLLM server (easy way):
docs2synth agent vllm-server

# 2. Then use the provider (in another terminal):
docs2synth agent generate "Your prompt" --provider vllm

# With images (if model supports vision):
docs2synth agent generate "What's in this image?" \
  --image photo.jpg \
  --provider vllm
```

#### Model Selection

Specify a specific model:

```bash
docs2synth agent generate "Your prompt" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022
```

```bash
docs2synth agent generate "Your prompt" \
  --provider openai \
  --model gpt-4o
```

#### Generation Parameters

Control output with generation parameters:

```bash
docs2synth agent generate "Your prompt" \
  --temperature 0.7 \
  --max-tokens 1000
```

#### System Prompts

Add system instructions:

```bash
docs2synth agent generate "What is Python?" \
  --system-prompt "You are a helpful programming tutor"
```

#### JSON Mode

Get structured JSON output:

```bash
docs2synth agent generate "List 3 programming languages" \
  --response-format json
```

#### Vision Models

Analyze images with vision-capable models:

```bash
docs2synth agent generate "What's in this image?" \
  --image photo.jpg \
  --provider openai \
  --model gpt-4o
```

#### Custom Configuration

Use a custom config file:

```bash
docs2synth agent generate "Your prompt" \
  --config-path ./custom-config.yml
```

### Chat with Message History

Basic chat:

```bash
docs2synth agent chat "What is Python?"
```

#### Provider and Model Selection

```bash
docs2synth agent chat "Explain AI" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022
```

#### Chat with Images

```bash
docs2synth agent chat "What's in this image?" \
  --image photo.jpg \
  --provider openai \
  --model gpt-4o
```

#### Persistent Chat History

Save and load chat history from a JSON file:

```bash
# First message (creates history.json)
docs2synth agent chat "Hello" --history-file chat.json

# Continue conversation (loads and updates history.json)
docs2synth agent chat "Tell me more" --history-file chat.json
```

#### Complete Chat Example

```bash
docs2synth agent chat "Analyze this code" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --temperature 0.5 \
  --max-tokens 2000 \
  --history-file chat.json
```

---

## QA Generation Commands

Generate question-answer pairs for document understanding.

### List Available Strategies

```bash
docs2synth qa list
```

**Output:**
```
Available QA generation strategies:
  - semantic
  - layout_aware
  - logical_aware
```

### Semantic QA Generation

Generate questions from document context and target answer:

```bash
docs2synth qa semantic "Form contains name, address fields" "John Doe"
```

With provider and model:

```bash
docs2synth qa semantic "Document context" "Target answer" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022
```

With vision model and image:

```bash
docs2synth qa semantic "Context here" "Target" \
  --provider openai \
  --model gpt-4o \
  --image document.png
```

With generation parameters:

```bash
docs2synth qa semantic "Context" "Target" \
  --temperature 0.7 \
  --max-tokens 500
```

### Layout-Aware QA

Transform questions to be spatially aware (position-based):

```bash
docs2synth qa layout "What is the name?"
```

With image for better spatial understanding:

```bash
docs2synth qa layout "What is the address?" \
  --provider anthropic \
  --image document.png
```

**Example Output:**
```
Original Question:
What is the address?

Layout-Aware Question:
What is the text in the upper right section of the form?
```

### Logical-Aware QA

Transform questions to be structure-aware (section-based):

```bash
docs2synth qa logical "What is the address?"
```

With image:

```bash
docs2synth qa logical "What is the name?" \
  --provider anthropic \
  --image document.png
```

**Example Output:**
```
Original Question:
What is the name?

Logical-Aware Question:
What is the value in the 'Personal Information' section for the 'Name' field?
```

### Batch QA Generation

Generate QA pairs using all strategies configured in `config.yml`:

```bash
docs2synth qa generate "Form contains name, address fields" "John Doe"
```

With image:

```bash
docs2synth qa generate "Context here" "Target" --image document.png
```

Use a specific strategy only:

```bash
docs2synth qa generate "Context" "Target" --strategy semantic
```

Custom config file:

```bash
docs2synth qa generate "Context" "Target" \
  --config-path ./custom-config.yml
```

**Example Output:**
```
Generating QA questions using 3 strategy(ies) from config...

[semantic] Using openai/gpt-4o...
  Question: What name appears in the personal information section?

[layout_aware] Using anthropic/claude-3-5-sonnet-20241022...
  Question: What text is located in the top-left corner of the form?

[logical_aware] Using gemini/gemini-1.5-flash...
  Question: In the 'Personal Details' section, what value is provided for the name field?

============================================================
Summary:

[semantic] openai/gpt-4o
  Question: What name appears in the personal information section?

[layout_aware] anthropic/claude-3-5-sonnet-20241022
  Original: What name appears in the personal information section?
  Question: What text is located in the top-left corner of the form?

[logical_aware] gemini/gemini-1.5-flash
  Original: What name appears in the personal information section?
  Question: In the 'Personal Details' section, what value is provided for the name field?
```

### Single Document QA Generation

Generate QA pairs for a single document (auto-pairs JSON with image):

```bash
# Using JSON file
docs2synth qa run data/processed/dev/document_docling.json

# Using image file (finds corresponding JSON)
docs2synth qa run data/images/document.png

# Use specific strategy only
docs2synth qa run data/processed/document.json --strategy semantic
```

### Batch QA Processing

Generate QA pairs for multiple documents using strategies from `config.yml`:

```bash
# Process all images from config.preprocess.input_dir
docs2synth qa batch

# Process specific directory
docs2synth qa batch data/raw/my_documents/

# Process single image
docs2synth qa batch data/images/document.png

# Use specific output directory and processor
docs2synth qa batch data/images/ \
  --output-dir data/processed/ \
  --processor docling
```

**What it does:**
1. Reads image files from input path
2. Finds corresponding preprocessed JSON files
3. Generates QA pairs for each text object using ALL configured strategies
4. Saves results back to JSON files with "qa" field added

**Example Output:**
```
Input: data/raw/my_documents/
Output dir: data/processed/
Processor: docling
Strategies from config.yml: semantic, layout_aware

Processing document_001.png...
  Generated 12 QA pairs

Processing document_002.png...
  Generated 8 QA pairs

Done! Processed 2 files, 45 objects, generated 20 questions
```

### Clean QA Pairs

Remove generated QA pairs from JSON outputs:

```bash
# Clean all files from config.preprocess.output_dir
docs2synth qa clean

# Clean specific directory
docs2synth qa clean data/processed/dev

# Clean specific JSON file
docs2synth qa clean data/processed/dev/document_docling.json

# Clean by image path
docs2synth qa clean data/images/document.png
```

---

## QA Verification Commands

Automatically verify the quality of generated QA pairs.

### List Available Verifiers

```bash
docs2synth verify list
```

**Output:**
```
Available verification strategies:
  - meaningful
  - correctness
```

### Verify Single Document

Verify QA pairs in a single JSON file:

```bash
# Using JSON file
docs2synth verify run data/processed/dev/document_docling.json

# Using image file
docs2synth verify run data/images/document.png

# Use specific verifier only
docs2synth verify run data/processed/document.json --verifier-type meaningful
```

**Output:**
```
Done! Processed 23 objects, verified 18 QA pairs, 15 passed all verifiers
```

### Batch Verification

Verify QA pairs in multiple JSON files:

```bash
# Verify all files from config.preprocess.output_dir
docs2synth verify batch

# Verify specific directory
docs2synth verify batch data/processed/dev

# Verify single file
docs2synth verify batch data/processed/dev/document.json

# Use specific verifier only
docs2synth verify batch --verifier-type meaningful

# Specify image search directories
docs2synth verify batch --image-dir data/images --image-dir data/raw
```

**Output:**
```
Input: data/processed/dev
Verifiers: meaningful, correctness
Image search dirs: data/raw/my_documents/

Processing document_001_docling.json...
Processing document_002_docling.json...

Done! Processed 10 files, 234 objects, verified 187 QA pairs
Pass rate: 156/187 (83.4%) passed all verifiers
```

### Clean Verification Results

Remove verification results from JSON outputs:

```bash
# Clean all files
docs2synth verify clean

# Clean specific directory
docs2synth verify clean data/processed/dev

# Clean specific file
docs2synth verify clean data/processed/dev/document_docling.json
```

---

## Human Annotation Commands

Launch interactive interface for manual QA pair annotation.

### Launch Annotation Tool

```bash
# Launch with default settings (from config)
docs2synth annotate

# Launch with specific directory
docs2synth annotate data/processed/dev

# Specify custom image directory
docs2synth annotate data/processed/dev --image-dir data/images

# Use custom port
docs2synth annotate --port 8502
```

**Access:** Opens at `http://localhost:8501`

**Features:**
- Visual document display with bounding boxes
- Review automatic verifier results
- Manual approve/reject QA pairs
- Add explanation notes
- Progress tracking and statistics
- Navigate between documents and QA pairs

---

## Retriever Training Commands

Train custom document retriever models on annotated QA pairs.

### Preprocess Training Data

Convert annotated JSON files into training format:

```bash
# Use defaults from config
docs2synth retriever preprocess

# Specify all parameters
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
- `--json-dir`: Directory with processed JSON files (default: config.preprocess.output_dir)
- `--image-dir`: Directory with images (default: config.preprocess.input_dir)
- `--output`: Output pickle file path
- `--processor`: Processor name to filter JSON files (default: docling)
- `--batch-size`: Training batch size (default: 8)
- `--max-length`: Maximum sequence length (default: 512)
- `--num-objects`: Max objects per document (default: 50)
- `--require-all-verifiers`: Only include QA pairs passing all verifiers (default: true)

**Output:**
```
Preprocessing JSON QA pairs → DataLoader pickle...
  JSON directory: data/processed/
  Image directory: data/raw/my_documents/
  Output: data/retriever/preprocessed_train.pkl
  Processor: docling
  Batch size: 8
  Max sequence length: 512
  Max objects: 50
  Require all verifiers: True

✓ Preprocessing complete!
  Saved: data/retriever/preprocessed_train.pkl
  QA pairs: 1,234
  Batches: 155

  Use with:
    docs2synth retriever train --data-path data/retriever/preprocessed_train.pkl
```

### Train Retriever Model

```bash
# Basic training (uses config defaults)
docs2synth retriever train --mode standard --lr 1e-5 --epochs 10

# With all options
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

# Resume from checkpoint
docs2synth retriever train \
  --resume models/retriever/checkpoints/checkpoint_epoch_5.pth \
  --mode standard

# Use different base model
docs2synth retriever train \
  --base-model microsoft/layoutlmv3-large \
  --mode layout \
  --lr 1e-5
```

**Training Modes:**
- `standard`: Standard span-based QA training
- `layout`: Layout-aware training with grid representations
- `layout-gemini`: Gemini variant with grid representations
- `layout-coarse-grained`: Coarse-grained training
- `pretrain-layout`: Layout pretraining using grid embeddings

**Output:**
```
Training retriever model in 'standard' mode...
  Base model: microsoft/layoutlmv3-base
  Data: data/retriever/preprocessed_train.pkl
  Output: models/retriever/checkpoints/
  Learning rate: 1e-5
  Epochs: 10
  Device: cuda:0 (auto-detected: NVIDIA GeForce RTX 3090)

Loading model...
  Creating custom QA model from HuggingFace: microsoft/layoutlmv3-base

Epoch 1/10
  Train ANLS: 0.7234 | Train Loss: 0.4567
  Val ANLS: 0.6891 | Val Loss: 0.5123
  Saved checkpoint: models/retriever/checkpoints/checkpoint_epoch_1.pth

...

✓ Training complete!
  Final model saved: models/retriever/final_model.pth
  Best ANLS: 0.8456
  Checkpoints saved in: models/retriever/checkpoints/
```

### Validate Retriever Model

Evaluate trained model performance:

```bash
# Auto-discover model and data from config
docs2synth retriever validate

# Specify model and data explicitly
docs2synth retriever validate \
  --model models/retriever/final_model.pth \
  --data data/retriever/preprocessed_val.pkl

# With custom output directory
docs2synth retriever validate \
  --model models/retriever/final_model.pth \
  --data data/retriever/preprocessed_val.pkl \
  --output models/retriever/validation_reports/ \
  --mode standard \
  --device cuda
```

**Output:**
```
Validating retriever training results...
  Model: models/retriever/final_model.pth
  Data:  data/retriever/preprocessed_val.pkl
  Mode:  standard

✓ Evaluation complete
    ANLS: 0.8234 | Loss: 0.3456

======================================================================
VALIDATION RESULTS
======================================================================

ANLS Score:
  Mean:            0.8234
  Std:             0.1567
  Median:          0.8456
  Perfect matches: 45 (23.4%)
  Zero matches:    12 (6.2%)

✅ Empty predictions: 3 (1.6%) - Good
✅ Prediction diversity: 156 unique (81.2%) - Good

OUTPUT FILES
  Detailed analysis: models/retriever/validation_reports/detailed_analysis.txt
  Metrics plot:      models/retriever/validation_reports/validation_metrics.png
```

---

## RAG Deployment Commands

Deploy retrieval-augmented generation systems.

### List RAG Strategies

```bash
docs2synth rag strategies
```

**Output:**
```
Available RAG strategies:
- naive
- iterative
```

### Ingest Documents

Index documents into the vector store:

```bash
# Ingest from config.preprocess.output_dir
docs2synth rag ingest

# Specify directory and processor
docs2synth rag ingest \
  --processed-dir data/processed/ \
  --processor docling \
  --include-context
```

**Output:**
```
Ingested 1,234 document chunks into the vector store.
```

### Query RAG System

Ask questions using the RAG system:

```bash
# Basic query
docs2synth rag run -q "What is the total amount on invoice INV-2024-001?"

# Use specific strategy
docs2synth rag run \
  -s iterative \
  -q "What is the total amount?"

# Show iteration details
docs2synth rag run \
  -s iterative \
  -q "What is the invoice date?" \
  --show-iterations
```

**Output:**
```
Final answer:
The total amount on invoice INV-2024-001 is $1,234.56

Iterations:
Step 1
  Answer: The total amount is $1,234.56
  Retrieved context:
    - score=0.891 source=data/processed/invoice_docling.json object_id=obj_23
```

### Launch RAG Demo App

Start interactive Streamlit interface:

```bash
# Launch with defaults
docs2synth rag app

# Custom host and port
docs2synth rag app --host localhost --port 8501

# Headless mode (no browser)
docs2synth rag app --no-browser
```

**Access:** Opens at `http://localhost:8501`

### Reset Vector Store

Clear all indexed documents:

```bash
docs2synth rag reset
```

**Note:** This requires confirmation prompt.

---

## Common Patterns

### Using Configuration Files

Most commands support `--config-path` to use custom configuration:

```bash
# Create a config.yml in your working directory
# Commands will automatically use ./config.yml if it exists

# Or specify a custom path
docs2synth --config ./my-config.yml [COMMAND]
```

### Verbose Output

Get detailed logs for debugging:

```bash
# Basic verbosity
docs2synth -v preprocess document.png

# Very verbose (debug level)
docs2synth -vv preprocess document.png
```

### Combining Options

You can combine multiple options for complex workflows:

```bash
docs2synth -vv --config ./config.yml agent generate "Your prompt" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --temperature 0.8 \
  --max-tokens 2000 \
  --response-format json \
  --image document.png
```

---

## Environment Variables

Docs2Synth respects the following environment variables:

- `DOCS2SYNTH_CONFIG`: Default path to config file
- Provider-specific API keys (see [Configuration](#configuration) below)

```bash
# Set default config path
export DOCS2SYNTH_CONFIG=/path/to/config.yml

# Then run commands without --config-path
docs2synth agent generate "Your prompt"
```

---

## Configuration

### Config File Structure

Create a `config.yml` file in your working directory:

```yaml
# LLM Provider Configuration
agent:
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
      default_model: gpt-4o
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-3-5-sonnet-20241022
    gemini:
      api_key: ${GEMINI_API_KEY}
      default_model: gemini-1.5-flash

# QA Generation Strategies
qa:
  strategies:
    - strategy: semantic
      provider: openai
      model: gpt-4o
      temperature: 0.7
    - strategy: layout_aware
      provider: anthropic
      model: claude-3-5-sonnet-20241022
      temperature: 0.5
    - strategy: logical_aware
      provider: gemini
      model: gemini-1.5-flash
      temperature: 0.7

# Data Directories
data:
  raw_dir: ./data/raw
  processed_dir: ./data/processed
  datasets_dir: ./data/datasets

# Logging
logging:
  level: INFO
  file: ./logs/docs2synth.log
```

### API Keys

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export DOUBAO_API_KEY="..."
export HUGGINGFACE_TOKEN="hf_..."  # For gated HF models
```

Or use a `.env` file:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
HUGGINGFACE_TOKEN=hf_...
```

### vLLM Setup

vLLM is a high-performance inference engine for local LLM deployment. It uses a server mode with an OpenAI-compatible API.

#### Server Mode Setup

1. **Install vLLM** (requires CUDA-capable GPU):
```bash
pip install vllm
```

2. **Start the vLLM server**:
```bash
# Easy way: Use the built-in command
docs2synth agent vllm-server

# Or manually with python
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000

# With advanced options
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

3. **Configure in config.yml**:
```yaml
agent:
  vllm:
    model: meta-llama/Llama-2-7b-chat-hf
    base_url: http://localhost:8000/v1

    # Server startup parameters (used by 'docs2synth agent vllm-server')
    trust_remote_code: true  # Required for some models
    max_model_len: 4096
    gpu_memory_utilization: 0.9
```

4. **Use the provider**:
```bash
docs2synth agent generate "Your prompt" --provider vllm
```

#### Vision-Language Models with vLLM

For vision models like Qwen-VL:

```yaml
agent:
  vllm:
    model: Qwen/Qwen-VL-Chat
    base_url: http://localhost:8000/v1
    trust_remote_code: true
    max_model_len: 2048
    gpu_memory_utilization: 0.9
```

Start the server with vision model support:
```bash
docs2synth agent vllm-server --trust-remote-code
```

Then use with images:
```bash
docs2synth agent generate "Describe this image" \
  --image document.png \
  --provider vllm
```

#### Troubleshooting vLLM

**GPU Out of Memory:**
- Reduce `max_model_len` (try 4096, 2048, or 1024)
- Reduce `gpu_memory_utilization` (try 0.8 or 0.7)
- Use a smaller model or quantization

**Model Loading Errors:**
- Set `trust_remote_code: true` for custom models
- Check vLLM version: `pip install --upgrade vllm`
- Ensure your GPU has sufficient VRAM for the model

**Server Connection Issues:**
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall settings
- Ensure `base_url` matches server address in config.yml

---

## Examples Gallery

### Example 1: Complete Document Processing Pipeline

```bash
# 1. Download dataset
docs2synth datasets download funsd --output-dir ./data

# 2. Preprocess documents
docs2synth preprocess ./data/funsd/images/ \
  --processor paddleocr \
  --output-dir ./data/processed \
  --device gpu

# 3. Generate QA pairs
docs2synth qa generate "Form contains date field" "2024-01-15" \
  --image ./data/funsd/images/form1.png
```

### Example 2: Multi-Provider QA Generation

```bash
# Generate with OpenAI
docs2synth qa semantic "Document text" "Answer" --provider openai

# Generate with Anthropic
docs2synth qa semantic "Document text" "Answer" --provider anthropic

# Generate with Gemini
docs2synth qa semantic "Document text" "Answer" --provider gemini
```

### Example 3: Interactive Chat Session

```bash
# Start conversation
docs2synth agent chat "Help me understand Python decorators" \
  --provider anthropic \
  --history-file python-help.json

# Continue conversation
docs2synth agent chat "Can you show me an example?" \
  --history-file python-help.json

# Ask follow-up
docs2synth agent chat "How do I use @property?" \
  --history-file python-help.json
```

### Example 4: Vision-Based Document Analysis

```bash
# Analyze document structure
docs2synth agent generate "Describe the layout and structure of this document" \
  --image invoice.png \
  --provider openai \
  --model gpt-4o

# Generate layout-aware questions
docs2synth qa layout "What is the total amount?" \
  --image invoice.png \
  --provider openai \
  --model gpt-4o
```

---

## Tips and Best Practices

### 1. Use Config Files for Repeated Tasks

Instead of passing options every time:

```bash
# Bad: Repetitive
docs2synth agent generate "prompt1" --provider anthropic --model claude-3-5-sonnet-20241022
docs2synth agent generate "prompt2" --provider anthropic --model claude-3-5-sonnet-20241022

# Good: Use config
# Set default in config.yml, then just:
docs2synth agent generate "prompt1"
docs2synth agent generate "prompt2"
```

### 2. Use History Files for Long Conversations

```bash
# Maintain context across multiple commands
docs2synth agent chat "Question 1" --history-file session.json
docs2synth agent chat "Question 2" --history-file session.json
docs2synth agent chat "Question 3" --history-file session.json
```

### 3. Vision Models for Better QA

When working with documents, always use vision models with images for better results:

```bash
# Good: Provides visual context
docs2synth qa semantic "Form context" "Answer" \
  --image form.png \
  --provider openai \
  --model gpt-4o

# Less accurate: Text-only
docs2synth qa semantic "Form context" "Answer"
```

### 4. Batch Processing

Process multiple files with shell loops:

```bash
# Process all images in a directory
for img in ./images/*.png; do
  docs2synth preprocess "$img" --output-dir ./processed
done
```

### 5. Debug with Verbose Flags

Use verbose output when troubleshooting:

```bash
docs2synth -vv preprocess document.png
```

---

## Troubleshooting

### Command Not Found

```bash
# Make sure package is installed
pip install -e .

# Or use full module path
python -m docs2synth.cli [COMMAND]
```

### API Key Errors

```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Or verify config file
cat config.yml
```

### Config File Not Found

```bash
# Check for config.yml in current directory
ls -la config.yml

# Or specify path explicitly
docs2synth --config ./path/to/config.yml [COMMAND]
```

### GPU Not Available

```bash
# Force CPU mode
docs2synth preprocess document.png --device cpu
```

---

## See Also

- [Complete Workflow Guide](workflow/complete-workflow.md) - End-to-end workflow from documents to RAG deployment
- [Document Processing Workflow](workflow/document-processing.md)
- [QA Generation Workflow](workflow/qa-generation.md)
- [Retriever Training Workflow](workflow/retriever-training.md)
- [RAG Path Workflow](workflow/rag-path.md)
- [API Reference](api-reference.md)
- [MCP Integration](mcp-integration.md)
