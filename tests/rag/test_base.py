"""Tests for RAG base classes and utilities."""

from typing import Optional, Sequence

import numpy as np

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.rag.base import PromptConfig, RAGStrategy
from docs2synth.rag.types import (
    DocumentChunk,
    RAGResult,
    RAGState,
    RetrievedDocument,
)
from docs2synth.rag.vector_store import FaissVectorStore


class DummyEmbedder:
    """Simple embedder for testing."""

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return dummy embeddings."""
        return np.random.rand(len(texts), self.dimension).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Return dummy query embedding."""
        return np.random.rand(self.dimension).astype("float32")


class DummyLLM(BaseLLMProvider):
    """Simple LLM for testing."""

    def __init__(self, response: str = "Test response") -> None:
        super().__init__(model="dummy")
        self.response = response
        self.call_count = 0
        self.last_messages = None

    def generate(self, *args, **kwargs) -> LLMResponse:  # pragma: no cover
        return LLMResponse(content=self.response, model=self.model)

    def chat(self, messages, **kwargs) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        return LLMResponse(content=self.response, model=self.model)


class ConcreteRAGStrategy(RAGStrategy):
    """Concrete implementation for testing base class."""

    def generate(self, query: str, state: Optional[RAGState] = None) -> RAGResult:
        """Simple implementation for testing."""
        documents = self._retrieve(query)
        context = self._build_context(documents)
        answer = self._invoke_llm(query, context)
        iteration = self._build_iteration(1, query, documents, answer)
        result = RAGResult(final_answer=answer)
        result.add_iteration(iteration)
        return result


def test_prompt_config_creation() -> None:
    """Test PromptConfig dataclass."""
    prompt = PromptConfig(
        system="You are a helpful assistant.",
        user="Question: {query}\nContext: {context}\nAnswer:",
    )
    assert "helpful assistant" in prompt.system
    assert "{query}" in prompt.user
    assert "{context}" in prompt.user


def test_rag_strategy_initialization() -> None:
    """Test RAGStrategy base class initialization."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()
    prompt = PromptConfig(system="System", user="User: {query}")

    strategy = ConcreteRAGStrategy(
        name="test_strategy",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=5,
    )

    assert strategy.name == "test_strategy"
    assert strategy.vector_store == store
    assert strategy.embedder == embedder
    assert strategy.llm == llm
    assert strategy.prompt == prompt
    assert strategy.top_k == 5


def test_retrieve_method() -> None:
    """Test _retrieve method returns documents from vector store."""
    embedder = DummyEmbedder(dimension=3)
    store = FaissVectorStore(dimension=3, normalize=False)

    # Add some documents
    docs = [
        DocumentChunk(id="1", text="Apple pie recipe"),
        DocumentChunk(id="2", text="Banana bread recipe"),
    ]
    embeddings = embedder.embed_texts([doc.text for doc in docs])
    store.add_embeddings(embeddings, docs)

    llm = DummyLLM()
    prompt = PromptConfig(system="System", user="User: {query}")

    strategy = ConcreteRAGStrategy(
        name="test",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=2,
    )

    results = strategy._retrieve("apple")
    assert len(results) <= 2
    assert all(isinstance(doc, RetrievedDocument) for doc in results)


def test_build_context_formats_documents() -> None:
    """Test _build_context method formats retrieved documents."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()
    prompt = PromptConfig(system="System", user="User: {query}")

    strategy = ConcreteRAGStrategy(
        name="test",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
    )

    documents = [
        RetrievedDocument(id="1", text="First document", score=0.9),
        RetrievedDocument(id="2", text="Second document", score=0.8),
    ]

    context = strategy._build_context(documents)
    assert "First document" in context
    assert "Second document" in context
    assert "0.900" in context  # score formatting
    assert "0.800" in context


def test_build_context_with_metadata() -> None:
    """Test _build_context includes metadata in output."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()
    prompt = PromptConfig(system="System", user="User: {query}")

    strategy = ConcreteRAGStrategy(
        name="test",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
    )

    documents = [
        RetrievedDocument(
            id="1",
            text="Document with metadata",
            score=0.95,
            metadata={"source": "test.pdf", "page": 1},
        ),
    ]

    context = strategy._build_context(documents)
    assert "Document with metadata" in context
    assert "source=test.pdf" in context
    assert "page=1" in context


def test_build_context_empty_documents() -> None:
    """Test _build_context with empty document list."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()
    prompt = PromptConfig(system="System", user="User: {query}")

    strategy = ConcreteRAGStrategy(
        name="test",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
    )

    context = strategy._build_context([])
    assert context == ""


def test_invoke_llm_calls_chat() -> None:
    """Test _invoke_llm properly calls the LLM with formatted prompts."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM(response="This is the LLM answer")
    prompt = PromptConfig(
        system="You are helpful.",
        user="Q: {query}\nContext: {context}\nA:",
    )

    strategy = ConcreteRAGStrategy(
        name="test",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
    )

    answer = strategy._invoke_llm("What is AI?", "AI is artificial intelligence.")
    assert answer == "This is the LLM answer"
    assert llm.call_count == 1
    assert llm.last_messages is not None
    assert len(llm.last_messages) == 2
    assert llm.last_messages[0]["role"] == "system"
    assert llm.last_messages[0]["content"] == "You are helpful."
    assert llm.last_messages[1]["role"] == "user"
    assert "What is AI?" in llm.last_messages[1]["content"]
    assert "AI is artificial intelligence." in llm.last_messages[1]["content"]


def test_build_iteration_creates_correct_result() -> None:
    """Test _build_iteration creates IterationResult correctly."""
    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=3, normalize=False)
    llm = DummyLLM()
    prompt = PromptConfig(system="System", user="User: {query}")

    strategy = ConcreteRAGStrategy(
        name="test",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
    )

    documents = [
        RetrievedDocument(id="1", text="Doc 1", score=0.9),
        RetrievedDocument(id="2", text="Doc 2", score=0.8),
    ]

    iteration = strategy._build_iteration(
        step=1,
        query="test query",
        documents=documents,
        answer="test answer",
        similarity=0.95,
    )

    assert iteration.step == 1
    assert iteration.query == "test query"
    assert iteration.answer == "test answer"
    assert iteration.similarity == 0.95
    assert len(iteration.retrieved) == 2
    assert iteration.retrieved[0].id == "1"
    assert iteration.retrieved[1].id == "2"


def test_concrete_strategy_generate_flow() -> None:
    """Test end-to-end generate flow with concrete strategy."""
    embedder = DummyEmbedder(dimension=3)
    store = FaissVectorStore(dimension=3, normalize=False)

    # Add documents
    docs = [
        DocumentChunk(id="1", text="Python is a programming language"),
        DocumentChunk(id="2", text="JavaScript is also a language"),
    ]
    embeddings = embedder.embed_texts([doc.text for doc in docs])
    store.add_embeddings(embeddings, docs)

    llm = DummyLLM(response="Python is great for beginners")
    prompt = PromptConfig(
        system="You are a coding expert.",
        user="Q: {query}\nContext: {context}\nA:",
    )

    strategy = ConcreteRAGStrategy(
        name="test",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=2,
    )

    result = strategy.generate("What is Python?")
    assert result.final_answer == "Python is great for beginners"
    assert len(result.iterations) == 1
    assert result.iterations[0].query == "What is Python?"
    assert len(result.iterations[0].retrieved) <= 2
