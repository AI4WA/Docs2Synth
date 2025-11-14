"""Tests for RAG pipeline orchestration."""

from typing import Sequence

import numpy as np
import pytest

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.rag.base import PromptConfig
from docs2synth.rag.pipeline import (
    PipelineComponents,
    RAGPipeline,
    StrategyNotFoundError,
    _create_vector_store,
    _resolve_prompt_config,
)
from docs2synth.rag.strategies import NaiveRAGStrategy
from docs2synth.rag.types import DocumentChunk, RAGState
from docs2synth.rag.vector_store import FaissVectorStore


class DummyEmbedder:
    """Simple embedder for testing."""

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return dummy embeddings."""
        n = len(texts)
        if n == 0:
            return np.empty((0, self.dimension), dtype="float32")
        return np.random.rand(n, self.dimension).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Return dummy query embedding."""
        return np.random.rand(self.dimension).astype("float32")


class DummyLLM(BaseLLMProvider):
    """Simple LLM for testing."""

    def __init__(self, response: str = "Test response") -> None:
        super().__init__(model="dummy")
        self.response = response

    def generate(self, *args, **kwargs) -> LLMResponse:  # pragma: no cover
        return LLMResponse(content=self.response, model=self.model)

    def chat(self, messages, **kwargs) -> LLMResponse:
        return LLMResponse(content=self.response, model=self.model)


def test_resolve_prompt_config_with_full_config() -> None:
    """Test resolving prompt config from strategy config."""
    strategy_cfg = {
        "prompt": {
            "system": "Custom system prompt",
            "user": "Custom user: {query}",
        }
    }
    prompt = _resolve_prompt_config(strategy_cfg)
    assert prompt.system == "Custom system prompt"
    assert prompt.user == "Custom user: {query}"


def test_resolve_prompt_config_with_defaults() -> None:
    """Test resolving prompt config uses defaults when not specified."""
    strategy_cfg = {}
    prompt = _resolve_prompt_config(strategy_cfg)
    # Should use default prompts
    assert "helpful assistant" in prompt.system
    assert "{query}" in prompt.user
    assert "{context}" in prompt.user


def test_resolve_prompt_config_partial() -> None:
    """Test resolving prompt config with partial specification."""
    strategy_cfg = {"prompt": {"system": "Only system prompt"}}
    prompt = _resolve_prompt_config(strategy_cfg)
    assert prompt.system == "Only system prompt"
    # User prompt should use default
    assert "{query}" in prompt.user


def test_create_vector_store_faiss() -> None:
    """Test creating FAISS vector store."""
    config = {"type": "faiss", "dimension": 128, "normalize": True}
    store = _create_vector_store(config, dimension=None)
    assert isinstance(store, FaissVectorStore)
    assert store.dimension == 128


def test_create_vector_store_uses_dimension_param() -> None:
    """Test creating vector store uses dimension parameter when not in config."""
    config = {"type": "faiss", "normalize": False}
    store = _create_vector_store(config, dimension=256)
    assert isinstance(store, FaissVectorStore)
    assert store.dimension == 256


def test_create_vector_store_unsupported_type() -> None:
    """Test creating vector store with unsupported type raises error."""
    config = {"type": "unsupported_store"}
    with pytest.raises(ValueError, match="Unsupported vector store type"):
        _create_vector_store(config, dimension=128)


def test_pipeline_components_dataclass() -> None:
    """Test PipelineComponents dataclass."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)

    assert components.embedder == embedder
    assert components.vector_store == store
    assert components.llm == llm


def test_rag_pipeline_initialization() -> None:
    """Test RAGPipeline initialization."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()
    prompt = PromptConfig(system="System", user="User: {query}")

    strategy = NaiveRAGStrategy(
        name="naive",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=5,
    )

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {"naive": strategy})

    assert pipeline.embedder == embedder
    assert pipeline.vector_store == store
    assert pipeline.llm == llm
    assert "naive" in pipeline.strategies


def test_rag_pipeline_strategies_property() -> None:
    """Test RAGPipeline strategies property returns strategy names."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()
    prompt = PromptConfig(system="System", user="User: {query}")

    strategy1 = NaiveRAGStrategy(
        name="strategy1",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
    )
    strategy2 = NaiveRAGStrategy(
        name="strategy2",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
    )

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {"strategy1": strategy1, "strategy2": strategy2})

    strategies = pipeline.strategies
    assert len(strategies) == 2
    assert "strategy1" in strategies
    assert "strategy2" in strategies


def test_pipeline_add_documents() -> None:
    """Test adding documents to pipeline."""
    embedder = DummyEmbedder(dimension=3)
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {})

    texts = ["Document 1", "Document 2", "Document 3"]
    docs = pipeline.add_documents(texts)

    assert len(docs) == 3
    assert all(isinstance(doc, DocumentChunk) for doc in docs)
    assert docs[0].text == "Document 1"
    assert docs[1].text == "Document 2"
    assert docs[2].text == "Document 3"
    # Documents should be added to vector store
    assert len(store) == 3


def test_pipeline_add_documents_with_metadata() -> None:
    """Test adding documents with metadata to pipeline."""
    embedder = DummyEmbedder(dimension=3)
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {})

    texts = ["Doc 1", "Doc 2"]
    metadatas = [{"source": "file1.txt"}, {"source": "file2.txt"}]
    docs = pipeline.add_documents(texts, metadatas=metadatas)

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "file1.txt"
    assert docs[1].metadata["source"] == "file2.txt"


def test_pipeline_add_documents_metadata_mismatch() -> None:
    """Test adding documents with mismatched metadata length raises error."""
    embedder = DummyEmbedder(dimension=3)
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {})

    texts = ["Doc 1", "Doc 2"]
    metadatas = [{"source": "file1.txt"}]  # Only one metadata for two texts

    with pytest.raises(ValueError, match="Length of metadatas must match texts"):
        pipeline.add_documents(texts, metadatas=metadatas)


def test_pipeline_run_strategy() -> None:
    """Test running a strategy through pipeline."""
    embedder = DummyEmbedder(dimension=3)
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM(response="Pipeline answer")
    prompt = PromptConfig(system="System", user="Q: {query}\nA:")

    strategy = NaiveRAGStrategy(
        name="test_strategy",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=2,
    )

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {"test_strategy": strategy})

    # Add some documents
    pipeline.add_documents(["Document about Python", "Document about Java"])

    # Run the strategy
    result = pipeline.run("What is Python?", strategy_name="test_strategy")

    assert result.final_answer == "Pipeline answer"
    assert len(result.iterations) == 1


def test_pipeline_run_with_state() -> None:
    """Test running a strategy with existing state."""
    embedder = DummyEmbedder(dimension=3)
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM(response="Answer")
    prompt = PromptConfig(system="System", user="Q: {query}\nA:")

    strategy = NaiveRAGStrategy(
        name="strategy",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
    )

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {"strategy": strategy})
    pipeline.add_documents(["Doc 1"])

    # Create a state
    state = RAGState()
    result = pipeline.run("Query", strategy_name="strategy", state=state)

    # State should be updated
    assert len(state.iterations) == 1
    assert result.final_answer == "Answer"


def test_pipeline_run_unknown_strategy() -> None:
    """Test running unknown strategy raises StrategyNotFoundError."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {})

    with pytest.raises(StrategyNotFoundError):
        pipeline.run("Query", strategy_name="nonexistent")


def test_pipeline_reset() -> None:
    """Test pipeline reset clears vector store."""
    embedder = DummyEmbedder(dimension=3)
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()

    components = PipelineComponents(embedder=embedder, vector_store=store, llm=llm)
    pipeline = RAGPipeline(components, {})

    # Add documents
    pipeline.add_documents(["Doc 1", "Doc 2", "Doc 3"])
    assert len(store) == 3

    # Reset
    pipeline.reset()
    assert len(store) == 0


def test_strategy_not_found_error_is_key_error() -> None:
    """Test StrategyNotFoundError is a subclass of KeyError."""
    assert issubclass(StrategyNotFoundError, KeyError)
    error = StrategyNotFoundError("test_strategy")
    assert isinstance(error, KeyError)
