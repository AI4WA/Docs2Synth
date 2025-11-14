# API Reference

Complete API documentation for Docs2Synth modules. This reference covers the public Python APIs for programmatic usage.

!!! tip "Getting Started"
    For practical usage examples and complete workflow instructions, see:

    - [Complete Workflow Guide](workflow/complete-workflow.md) - End-to-end workflow from documents to RAG deployment
    - [CLI Reference](cli-reference.md) - Command-line interface documentation with examples

## Overview

Docs2Synth provides both a CLI and programmatic Python API. This reference documents the Python API for users who want to:

- Build custom workflows
- Integrate Docs2Synth into existing applications
- Extend functionality with custom components
- Automate document processing pipelines

---

## Core Package

### Installation & Import

```python
import docs2synth
from docs2synth import __version__
```

### Package-Level Exports

#### `__version__`
**Type:** `str`

The installed version of Docs2Synth.

```python
print(docs2synth.__version__)  # "0.1.0"
```

### Configuration & Logging

The package auto-initializes global configuration on import:

```python
from docs2synth.utils.config import get_config
from docs2synth.utils.logging import get_logger

config = get_config()
logger = get_logger(__name__)
```

---

## Configuration (`docs2synth.utils.config`)

### Classes

#### `Config`

Configuration manager supporting dot notation and YAML serialization.

**Constructor:**
```python
Config(config_dict: Dict[str, Any] | None = None)
```

**Class Methods:**
```python
@classmethod
def from_yaml(cls, yaml_path: str | Path) -> Config:
    """Load configuration from YAML file."""
```

**Methods:**
```python
def get(self, key: str, default: Any = None) -> Any:
    """Get config value using dot notation (e.g., 'data.root_dir')."""

def set(self, key: str, value: Any) -> None:
    """Set config value using dot notation."""

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary."""

def save(self, yaml_path: str | Path) -> None:
    """Save configuration to YAML file."""
```

**Example:**
```python
from docs2synth.utils.config import Config

# Load from YAML
config = Config.from_yaml("config.yml")

# Access with dot notation
root_dir = config.get("data.root_dir")
model_name = config.get("agent.model", default="gpt-4")

# Update values
config.set("agent.temperature", 0.8)
config.save("config.yml")
```

### Functions

#### `get_config()`
```python
def get_config() -> Config:
    """Get global config instance."""
```

#### `set_config()`
```python
def set_config(config: Config) -> None:
    """Set global config."""
```

#### `load_config()`
```python
def load_config(yaml_path: str | Path) -> Config:
    """Load and set global config."""
```

---

## Logging (`docs2synth.utils.logging`)

### Constants

#### `LOG_FORMAT`
**Type:** `str`

Default log format with filename and line numbers:
```
%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s
```

### Functions

#### `setup_logging()`
```python
def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path = "./logs/docs2synth.log"
) -> None:
    """Setup logging with file and console handlers."""
```

#### `setup_logging_from_config()`
```python
def setup_logging_from_config(config: Any = None) -> None:
    """Setup logging from Config object."""
```

#### `setup_cli_logging()`
```python
def setup_cli_logging(verbose: int = 0, config: Any = None) -> None:
    """Setup logging for CLI with verbosity levels (0-2)."""
```

#### `get_logger()`
```python
def get_logger(name: str) -> logging.Logger:
    """Get logger instance by name."""
```

#### `configure_third_party_loggers()`
```python
def configure_third_party_loggers(level: int = logging.WARNING) -> None:
    """Configure third-party library loggers to reduce noise."""
```

#### `log_function_call()` (Decorator)
```python
def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator to log function calls with arguments."""
```

**Example:**
```python
from docs2synth.utils.logging import get_logger, log_function_call

logger = get_logger(__name__)

@log_function_call(logger)
def process_document(path: str) -> dict:
    logger.info(f"Processing {path}")
    return {"status": "success"}
```

### Classes

#### `LoggerContext`

Context manager for temporary log level changes.

```python
with LoggerContext(logger, logging.DEBUG):
    # Temporarily enable debug logging
    logger.debug("Detailed debug info")
```

#### `ProgressLogger`

Progress tracking with periodic logging.

```python
progress = ProgressLogger("Processing", total=100)
for i in range(100):
    progress.update(i)
progress.complete()
```

**Methods:**
```python
def __init__(
    self,
    name: str,
    total: int,
    logger: logging.Logger | None = None,
    log_interval: int = 10
)

def update(self, current: int | None = None) -> None:
    """Update progress."""

def complete(self) -> None:
    """Mark as complete."""
```

---

## Timing Utilities (`docs2synth.utils.timer`)

### Functions

#### `timer()` (Context Manager)
```python
with timer("operation_name"):
    # Code to time
    process_data()
```

#### `timeit()` (Decorator)
```python
@timeit(log_level=logging.INFO)
def expensive_function():
    # Automatically logs execution time
    pass
```

#### `format_time()`
```python
def format_time(seconds: float) -> str:
    """Format seconds to human-readable string (e.g., '1h 23m 45s')."""
```

### Classes

#### `Timer`

Reusable timer for multiple operations.

```python
timer = Timer()
timer.start("preprocessing")
# ... do work ...
timer.stop("preprocessing")
print(f"Took {timer.elapsed('preprocessing')} seconds")
```

---

## Document Processing Schema (`docs2synth.preprocess.schema`)

### Type Aliases

```python
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Polygon = List[Tuple[float, float]]  # [(x, y), ...]
```

### Enums

#### `LabelType`

Semantic granularity labels for document objects.

```python
class LabelType(Enum):
    TEXT = "text"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    DOCUMENT = "document"
    OTHER = "other"
```

### Dataclasses

#### `QAPair`

Question-answer pair with verification metadata.

**Attributes:**
```python
question: str
answer: Optional[str]
strategy: Optional[str]
verification: Optional[Dict[str, Dict[str, Any]]]
extra: Dict[str, Any]
```

**Methods:**
```python
def to_dict() -> Dict[str, Any]

@staticmethod
def from_dict(data: Mapping[str, Any]) -> QAPair
```

#### `DocumentObject`

Extracted document object with layout information.

**Attributes:**
```python
object_id: int
text: str
bbox: BBox
label: LabelType
polygon: Optional[Polygon]
page: Optional[int]
score: Optional[float]
qa: List[QAPair]
extra: Dict[str, Any]
```

#### `DocumentProcessResult`

Complete document processing result.

**Attributes:**
```python
objects: Dict[int, DocumentObject]
object_list: List[DocumentObject]
bbox_list: List[BBox]
context: str
reading_order_ids: List[int]
process_metadata: ProcessMetadata
document_metadata: DocumentMetadata
qa_metadata: Optional[RunMetadata]
verify_metadata: Optional[RunMetadata]
```

**Methods:**
```python
def to_dict() -> Dict[str, Any]
def to_json(*, indent: Optional[int] = None) -> str

@staticmethod
def from_dict(data: Mapping[str, Any]) -> DocumentProcessResult
```

**Example:**
```python
from docs2synth.preprocess.schema import DocumentProcessResult

# Load processed document
with open("output/doc.json") as f:
    result = DocumentProcessResult.from_dict(json.load(f))

# Access objects
for obj in result.object_list:
    print(f"Object {obj.object_id}: {obj.text}")
    print(f"  BBox: {obj.bbox}")
    print(f"  QA Pairs: {len(obj.qa)}")
```

---

## Preprocessing (`docs2synth.preprocess`)

### Functions

#### `run_preprocess()`

Process documents to extract layout and text.

```python
def run_preprocess(
    path: str | Path,
    *,
    processor: str = "paddleocr",
    output_dir: Optional[str | Path] = None,
    lang: Optional[str] = None,
    device: Optional[str] = None,
    config: Optional[Config] = None
) -> Tuple[int, int, List[Path]]:
    """
    Process documents with specified processor.

    Args:
        path: Path to document or directory
        processor: Processor name ("paddleocr", "docling", "pdfplumber")
        output_dir: Output directory for results
        lang: Language code (e.g., "en")
        device: Device for processing ("cpu", "cuda")
        config: Configuration object

    Returns:
        Tuple of (processed_count, failed_count, output_paths)
    """
```

**Example:**
```python
from docs2synth.preprocess.runner import run_preprocess

processed, failed, outputs = run_preprocess(
    "data/raw/documents/",
    processor="docling",
    output_dir="data/processed/",
    device="cuda"
)

print(f"Processed {processed} documents, {failed} failed")
for path in outputs:
    print(f"  -> {path}")
```

---

## QA Generation (`docs2synth.qa`)

### Configuration Classes

#### `QAStrategyConfig`

Configuration for a single QA generation strategy.

```python
def __init__(
    self,
    strategy: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    prompt_template: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    group_id: Optional[str] = None,
    **kwargs: Any
)
```

#### `QAGenerationConfig`

Container for multiple QA strategies.

```python
# Load from config
qa_config = QAGenerationConfig.from_yaml("config.yml")

# Get specific strategy
semantic_config = qa_config.get_strategy_config("semantic")

# List all strategies
strategies = qa_config.list_strategies()
```

#### `QAVerificationConfig`

Configuration for QA verification.

```python
# Load from config
verify_config = QAVerificationConfig.from_yaml("config.yml")

# Get specific verifier
meaningful_config = verify_config.get_verifier_config("meaningful")

# List all verifiers
verifiers = verify_config.list_verifiers()
```

### Generators

#### `BaseQAGenerator` (Abstract)

Base class for QA generators.

```python
class BaseQAGenerator(ABC):
    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any
    )

    @abstractmethod
    def generate(
        self,
        context: str,
        target: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Generate QA pair from context."""
```

#### Concrete Generators

**`SemanticQAGenerator`**
Generates semantic questions from document context.

**`LayoutAwareQAGenerator`**
Generates questions that require layout understanding.

**`LogicalAwareQAGenerator`**
Generates questions that require logical reasoning.

#### Factory

```python
from docs2synth.qa.generators import QAGeneratorFactory

generator = QAGeneratorFactory.create_from_config(
    strategy_config,
    config_path="config.yml"
)
result = generator.generate(context="...", target="...")
```

### Verifiers

#### `BaseQAVerifier` (Abstract)

Base class for QA verifiers.

```python
@abstractmethod
def verify(
    self,
    question: str,
    answer: str,
    context: Optional[str] = None,
    image: Optional[Any] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Verify QA pair quality."""
```

#### Concrete Verifiers

**`MeaningfulVerifier`**
Verifies if the question is meaningful given the context.

**`CorrectnessVerifier`**
Verifies if the answer is correct for the question.

### Batch Processing

#### `process_batch()`

Process multiple documents for QA generation.

```python
from docs2synth.qa.qa_batch import process_batch
from docs2synth.qa.config import QAGenerationConfig

qa_config = QAGenerationConfig.from_yaml("config.yml")
processed, generated, failed = process_batch(
    input_path=Path("data/processed/"),
    output_dir=Path("data/qa/"),
    qa_config=qa_config,
    processor_name="docling"
)
```

#### `process_batch_verification()`

Verify QA pairs in batch.

```python
from docs2synth.qa.verify_batch import process_batch_verification
from docs2synth.qa.config import QAVerificationConfig

verify_config = QAVerificationConfig.from_yaml("config.yml")
processed, verified, rejected, failed = process_batch_verification(
    input_path=Path("data/qa/"),
    verification_config=verify_config
)
```

---

## Agent System (`docs2synth.agent`)

### AgentWrapper

High-level unified interface for LLM providers (OpenAI, Anthropic, Gemini, vLLM).

```python
from docs2synth.agent import AgentWrapper

# Create agent from config
agent = AgentWrapper(provider="openai", model="gpt-4")

# Generate response
response = agent.generate(
    prompt="Extract key information from this text",
    system_prompt="You are a helpful assistant",
    temperature=0.7,
    max_tokens=500
)
print(response.content)

# Chat with messages
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the weather?"}
]
response = agent.chat(messages)

# Switch provider dynamically
agent.switch_provider("anthropic", model="claude-3-sonnet")
```

**Methods:**
```python
def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    response_format: Optional[str] = None,
    image: Optional[Any] = None,
    **kwargs: Any
) -> LLMResponse

def chat(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    response_format: Optional[str] = None,
    image: Optional[Any] = None,
    **kwargs: Any
) -> LLMResponse

def switch_provider(
    self,
    provider: str,
    model: Optional[str] = None,
    **kwargs: Any
) -> None
```

**Properties:**
```python
@property
def model(self) -> str:
    """Current model name."""

@property
def provider_type(self) -> str:
    """Current provider type."""
```

### LLMResponse

```python
@dataclass
class LLMResponse:
    content: str                    # Generated text
    model: str                      # Model used
    usage: Dict[str, Any]          # Token usage info
    metadata: Dict[str, Any]       # Additional metadata
```

---

## Retriever Training (`docs2synth.retriever`)

### Dataset Functions

#### `load_verified_qa_pairs()`

Load verified QA pairs for training.

```python
from docs2synth.retriever.dataset import load_verified_qa_pairs

qa_pairs = load_verified_qa_pairs(
    data_dir="data/qa/",
    processor_name="docling",
    verifier_types=["meaningful", "correctness"]
)

for question, answer, metadata in qa_pairs:
    print(f"Q: {question}")
    print(f"A: {answer}")
```

### Model Creation

#### `create_model_for_qa()`

Create LayoutLM model for document QA.

```python
from docs2synth.retriever.model import create_model_for_qa

model, tokenizer = create_model_for_qa(
    model_name="microsoft/layoutlmv3-base",
    num_labels=2
)
```

### Training Functions

#### `train()`

Standard training function.

```python
from docs2synth.retriever.training import train

train(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    output_dir="models/retriever",
    learning_rate=1e-5,
    num_epochs=10,
    batch_size=8
)
```

### Metrics

#### `calculate_anls()`

Calculate Average Normalized Levenshtein Similarity.

```python
from docs2synth.retriever.metrics import calculate_anls

score = calculate_anls(
    prediction="The answer is 42",
    ground_truth="The answer is 42.",
    threshold=0.5
)
```

---

## RAG System (`docs2synth.rag`)

### Types

#### `DocumentChunk`

Document chunk with metadata.

```python
@dataclass
class DocumentChunk:
    text: str
    metadata: Optional[Mapping[str, object]]
    chunk_id: str
    embedding: Optional[np.ndarray]
```

#### `RetrievedDocument`

Retrieved document with similarity score.

```python
@dataclass
class RetrievedDocument:
    chunk: DocumentChunk
    score: float
```

#### `RAGResult`

Complete RAG generation result.

```python
@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_documents: Sequence[RetrievedDocument]
    iterations: Sequence[IterationResult]
    metadata: Mapping[str, object]
```

### Embedding Model

```python
from docs2synth.rag.embeddings import EmbeddingModel

embedding_model = EmbeddingModel(
    model_name="BAAI/bge-base-en-v1.5",
    device="cuda"
)

# Embed query
query_embedding = embedding_model.embed_query("What is RAG?")

# Embed batch
doc_embeddings = embedding_model.embed_texts([
    "Document 1 text",
    "Document 2 text"
])

print(f"Embedding dimension: {embedding_model.dimension}")
```

### Vector Store

#### `FaissVectorStore`

FAISS-based vector storage.

```python
from docs2synth.rag.vector_store import FaissVectorStore
from docs2synth.rag.types import DocumentChunk

# Create store
vector_store = FaissVectorStore(dimension=768)

# Add embeddings
chunks = [
    DocumentChunk(text="Document 1", metadata={"source": "file1.pdf"}, chunk_id="1"),
    DocumentChunk(text="Document 2", metadata={"source": "file2.pdf"}, chunk_id="2")
]
vector_store.add_embeddings(embeddings, chunks)

# Search
query_embedding = embedding_model.embed_query("search query")
results = vector_store.search(query_embedding, top_k=5)

for doc in results:
    print(f"Score: {doc.score} - {doc.chunk.text}")

# Persist
vector_store.save("vector_store.faiss")
loaded_store = FaissVectorStore.load("vector_store.faiss")
```

### RAG Pipeline

#### `RAGPipeline`

Main orchestration class for RAG workflows.

```python
from docs2synth.rag.pipeline import RAGPipeline
from docs2synth.utils.config import Config

# Load from config
config = Config.from_yaml("config.yml")
pipeline = RAGPipeline.from_config(config)

# Add documents
documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "Machine learning is a subset of AI."
]
chunks = pipeline.add_documents(documents)

# Run query with different strategies
result = pipeline.run(
    query="What is the capital of France?",
    strategy_name="naive"
)

print(f"Answer: {result.answer}")
print(f"Sources: {[doc.chunk.metadata for doc in result.retrieved_documents]}")

# List available strategies
print(f"Available strategies: {pipeline.strategies}")

# Reset
pipeline.reset()
```

### RAG Strategies

#### `NaiveRAGStrategy`

Simple retrieval + generation strategy.

```python
from docs2synth.rag.strategies import NaiveRAGStrategy

strategy = NaiveRAGStrategy(
    embedding_model=embedding_model,
    vector_store=vector_store,
    agent=agent,
    top_k=5
)

result = strategy.generate("What is machine learning?")
```

#### `EnhancedIterativeRAGStrategy`

Iterative refinement with similarity checks.

```python
from docs2synth.rag.strategies import EnhancedIterativeRAGStrategy

strategy = EnhancedIterativeRAGStrategy(
    embedding_model=embedding_model,
    vector_store=vector_store,
    agent=agent,
    top_k=5,
    max_iterations=3,
    similarity_threshold=0.7
)

result = strategy.generate("Complex query requiring iteration")
```

---

## Utility Functions

### PDF Image Processing (`docs2synth.utils.pdf_images`)

```python
from docs2synth.utils.pdf_images import (
    convert_pdf_to_images,
    get_pdf_images,
    convert_pdfs_in_directory
)

# Convert single PDF
images = convert_pdf_to_images("document.pdf", dpi=300)

# Get cached images
cached_images = get_pdf_images("document.pdf")

# Batch convert directory
count = convert_pdfs_in_directory("data/pdfs/", dpi=300)
```

### Text Processing (`docs2synth.utils.text`)

```python
from docs2synth.utils.text import truncate_context, calculate_max_context_length

# Truncate text to fit model context
truncated, was_truncated = truncate_context(
    context=long_text,
    provider="openai",
    model="gpt-4",
    max_tokens=4000
)

# Calculate max context length
max_length = calculate_max_context_length(
    provider="openai",
    model="gpt-4",
    prompt_template_overhead=300
)
```

---

## Configuration Options

### Key Configuration Paths

**Data paths:**
```yaml
data:
  root_dir: "./data"
  datasets_dir: "./data/datasets"
  processed_dir: "./data/processed"
  qa_pairs_dir: "./data/qa"
  models_dir: "./models"
  logs_dir: "./logs"
```

**Logging:**
```yaml
logging:
  level: "INFO"
  file:
    path: "./logs/docs2synth.log"
    max_bytes: 10485760  # 10MB
    backup_count: 5
  third_party:
    level: "WARNING"
```

**Agent/LLM:**
```yaml
agent:
  provider: "openai"
  model: "gpt-4"
  openai:
    api_key: "sk-..."
    model: "gpt-4"
  anthropic:
    api_key: "..."
    model: "claude-3-sonnet"
```

**QA Generation:**
```yaml
qa:
  - strategy: "semantic"
    provider: "openai"
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 500
  - strategy: "layout_aware"
    provider: "anthropic"
    model: "claude-3-sonnet"
```

**RAG:**
```yaml
rag:
  embedding:
    model_name: "BAAI/bge-base-en-v1.5"
  vector_store:
    type: "faiss"
    persist_path: "./data/vector_store"
  strategies:
    naive:
      type: "naive"
      top_k: 5
    enhanced:
      type: "enhanced"
      top_k: 5
      max_iterations: 3
      similarity_threshold: 0.7
```

---

## Complete Example

Here's a complete example using the Python API:

```python
from pathlib import Path
from docs2synth.utils.config import Config
from docs2synth.utils.logging import setup_logging, get_logger
from docs2synth.preprocess.runner import run_preprocess
from docs2synth.qa.config import QAGenerationConfig, QAVerificationConfig
from docs2synth.qa.qa_batch import process_batch
from docs2synth.qa.verify_batch import process_batch_verification
from docs2synth.rag.pipeline import RAGPipeline

# Setup
setup_logging()
logger = get_logger(__name__)
config = Config.from_yaml("config.yml")

# 1. Preprocess documents
logger.info("Preprocessing documents...")
processed, failed, outputs = run_preprocess(
    "data/raw/documents/",
    processor="docling",
    output_dir="data/processed/",
    config=config
)

# 2. Generate QA pairs
logger.info("Generating QA pairs...")
qa_config = QAGenerationConfig.from_config(config)
processed, generated, failed = process_batch(
    input_path=Path("data/processed/"),
    output_dir=Path("data/qa/"),
    qa_config=qa_config,
    config=config
)

# 3. Verify QA pairs
logger.info("Verifying QA pairs...")
verify_config = QAVerificationConfig.from_config(config)
processed, verified, rejected, failed = process_batch_verification(
    input_path=Path("data/qa/"),
    verification_config=verify_config
)

# 4. Setup RAG pipeline
logger.info("Setting up RAG pipeline...")
pipeline = RAGPipeline.from_config(config)

# Load documents into RAG
documents = []
for json_file in Path("data/qa/").glob("*.json"):
    with open(json_file) as f:
        result = DocumentProcessResult.from_dict(json.load(f))
        documents.append(result.context)

pipeline.add_documents(documents)

# 5. Query the RAG system
query = "What are the key features of the product?"
result = pipeline.run(query, strategy_name="enhanced")

logger.info(f"Query: {query}")
logger.info(f"Answer: {result.answer}")
logger.info(f"Retrieved {len(result.retrieved_documents)} documents")
```

---

## Next Steps

- [Complete Workflow Guide](workflow/complete-workflow.md) - End-to-end workflow
- [CLI Reference](cli-reference.md) - Command-line interface
- [Development Guide](development/dependency-management.md) - Contributing to Docs2Synth
