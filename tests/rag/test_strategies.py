from typing import Sequence

import numpy as np

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.rag.base import PromptConfig
from docs2synth.rag.strategies import EnhancedIterativeRAGStrategy, NaiveRAGStrategy
from docs2synth.rag.types import DocumentChunk
from docs2synth.rag.vector_store import FaissVectorStore


class DummyEmbedder:
    """Deterministic toy embedder for tests."""

    def __init__(self) -> None:
        self.vocabulary = ["apple", "banana", "carrot", "recipe"]
        self.dimension = len(self.vocabulary)

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype="float32")
        tokens = text.lower().split()
        for token in tokens:
            if token in self.vocabulary:
                vec[self.vocabulary.index(token)] += 1.0
        return vec

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._encode(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode(text)


class SequencedLLMProvider(BaseLLMProvider):
    """LLM provider that returns pre-seeded responses."""

    def __init__(self, outputs: Sequence[str]) -> None:
        super().__init__(model="sequence-llm")
        self.outputs = list(outputs)
        self.calls = 0

    def _next_content(self) -> str:
        if self.calls < len(self.outputs):
            content = self.outputs[self.calls]
        else:
            content = self.outputs[-1] if self.outputs else ""
        self.calls += 1
        return content

    def generate(self, *args, **kwargs) -> LLMResponse:  # pragma: no cover - not used
        return LLMResponse(content=self._next_content(), model=self.model)

    def chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content=self._next_content(), model=self.model)


def _setup_store(embedder: DummyEmbedder, texts: Sequence[str]) -> FaissVectorStore:
    store = FaissVectorStore(dimension=embedder.dimension, normalize=False)
    documents = [DocumentChunk(id=str(i), text=text) for i, text in enumerate(texts)]
    store.add_embeddings(embedder.embed_texts(texts), documents)
    return store


def test_naive_strategy_returns_single_iteration() -> None:
    embedder = DummyEmbedder()
    store = _setup_store(embedder, ["Apple pie recipe", "Banana smoothie ideas"])
    llm = SequencedLLMProvider(["Use the apple pie recipe from the context."])
    prompt = PromptConfig(
        system="Test system prompt.",
        user="Question: {query}\nContext: {context}\nAnswer:",
    )

    strategy = NaiveRAGStrategy(
        name="naive",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=2,
    )

    result = strategy.generate("How do I bake an apple pie?")
    assert result.final_answer.startswith("Use the apple pie recipe")
    assert len(result.iterations) == 1
    assert result.iterations[0].retrieved  # context returned


def test_enhanced_strategy_converges_when_answers_stabilize() -> None:
    embedder = DummyEmbedder()
    store = _setup_store(
        embedder,
        [
            "Apple pie recipe with cinnamon",
            "Banana bread recipe for breakfast",
            "Carrot cake instructions",
        ],
    )
    outputs = [
        "Initial draft answer about apple pie.",
        "Refined answer about apple pie with cinnamon.",
        "Refined answer about apple pie with cinnamon.",  # identical to trigger convergence
    ]
    llm = SequencedLLMProvider(outputs)
    prompt = PromptConfig(
        system="Iterative system prompt.",
        user="Question: {query}\nContext: {context}\nAnswer:",
    )

    strategy = EnhancedIterativeRAGStrategy(
        name="enhanced",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=2,
        max_iterations=5,
        similarity_threshold=0.95,
    )

    result = strategy.generate("How do I bake an apple pie?")
    assert result.final_answer.endswith("cinnamon.")
    # Should require three iterations: first draft, second refinement, third matches second
    assert len(result.iterations) == 3
    assert result.iterations[-1].similarity == 1.0


def test_enhanced_strategy_max_iterations() -> None:
    """Test that enhanced strategy respects max_iterations limit."""
    embedder = DummyEmbedder()
    store = _setup_store(embedder, ["Document 1", "Document 2"])
    # Different answers each time to prevent convergence
    outputs = ["Answer 1", "Answer 2", "Answer 3", "Answer 4", "Answer 5"]
    llm = SequencedLLMProvider(outputs)
    prompt = PromptConfig(
        system="System",
        user="Q: {query}\nContext: {context}\nA:",
    )

    strategy = EnhancedIterativeRAGStrategy(
        name="enhanced",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=2,
        max_iterations=3,
        similarity_threshold=0.99,  # High threshold so it won't converge
    )

    result = strategy.generate("Test query")
    # Should stop at max_iterations even without convergence
    assert len(result.iterations) == 3


def test_enhanced_strategy_augments_query() -> None:
    """Test that enhanced strategy augments query with previous context."""
    embedder = DummyEmbedder()
    store = _setup_store(embedder, ["Doc 1", "Doc 2"])
    outputs = ["First answer", "Second answer"]
    llm = SequencedLLMProvider(outputs)
    prompt = PromptConfig(
        system="System",
        user="Q: {query}\nContext: {context}\nA:",
    )

    strategy = EnhancedIterativeRAGStrategy(
        name="enhanced",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=1,
        max_iterations=2,
    )

    result = strategy.generate("What is AI?")
    # Second iteration should have augmented query
    assert len(result.iterations) == 2
    assert "What is AI?" in result.iterations[0].query
    # Second query should include previous context
    assert "What is AI?" in result.iterations[1].query


def test_enhanced_strategy_compare_answers() -> None:
    """Test the _compare_answers method."""
    from docs2synth.rag.strategies import EnhancedIterativeRAGStrategy

    # Identical strings
    assert EnhancedIterativeRAGStrategy._compare_answers("hello", "hello") == 1.0

    # Completely different strings
    similarity = EnhancedIterativeRAGStrategy._compare_answers("apple", "orange banana")
    assert 0.0 <= similarity < 0.5

    # Similar strings
    similarity = EnhancedIterativeRAGStrategy._compare_answers(
        "The quick brown fox", "The quick brown dog"
    )
    assert 0.5 < similarity < 1.0

    # Empty strings
    assert EnhancedIterativeRAGStrategy._compare_answers("", "") == 0.0
    assert EnhancedIterativeRAGStrategy._compare_answers("hello", "") == 0.0


def test_our_retriever_strategy_retrieve_bypasses_vector_store() -> None:
    """Test OurRetrieverRAGStrategy retrieves by computing direct similarities."""
    from docs2synth.rag.strategies import OurRetrieverRAGStrategy

    embedder = DummyEmbedder()
    store = _setup_store(embedder, ["Apple pie recipe", "Banana bread", "Carrot soup"])
    llm = SequencedLLMProvider(["Using custom retriever for answer"])
    prompt = PromptConfig(
        system="System",
        user="Q: {query}\nContext: {context}\nA:",
    )

    strategy = OurRetrieverRAGStrategy(
        name="our_retriever",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=2,
        max_iterations=1,
    )

    result = strategy.generate("apple")
    # Should still return results
    assert result.final_answer == "Using custom retriever for answer"
    assert len(result.iterations) == 1
    assert len(result.iterations[0].retrieved) <= 2


def test_our_retriever_strategy_computes_similarity_directly() -> None:
    """Test OurRetrieverRAGStrategy computes similarities with embedder."""
    from docs2synth.rag.strategies import OurRetrieverRAGStrategy

    embedder = DummyEmbedder()
    store = _setup_store(embedder, ["Document 1", "Document 2", "Document 3"])
    llm = SequencedLLMProvider(["Answer"])
    prompt = PromptConfig(system="System", user="Q: {query}\nA:")

    strategy = OurRetrieverRAGStrategy(
        name="our_retriever",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=3,
    )

    result = strategy.generate("test query")
    # Should retrieve all 3 documents
    assert len(result.iterations[0].retrieved) == 3
    # All documents should have scores
    for doc in result.iterations[0].retrieved:
        assert hasattr(doc, "score")
        assert isinstance(doc.score, float)


def test_our_retriever_strategy_empty_store() -> None:
    """Test OurRetrieverRAGStrategy with empty vector store."""
    from docs2synth.rag.strategies import OurRetrieverRAGStrategy

    embedder = DummyEmbedder()
    store = FaissVectorStore(dimension=embedder.dimension, normalize=False)
    llm = SequencedLLMProvider(["No documents available"])
    prompt = PromptConfig(system="System", user="Q: {query}\nA:")

    strategy = OurRetrieverRAGStrategy(
        name="our_retriever",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=5,
    )

    result = strategy.generate("test query")
    # Should return empty retrieval
    assert len(result.iterations[0].retrieved) == 0


def test_our_retriever_strategy_respects_top_k() -> None:
    """Test OurRetrieverRAGStrategy respects top_k parameter."""
    from docs2synth.rag.strategies import OurRetrieverRAGStrategy

    embedder = DummyEmbedder()
    # Add many documents
    documents = [f"Document {i}" for i in range(10)]
    store = _setup_store(embedder, documents)
    llm = SequencedLLMProvider(["Answer"])
    prompt = PromptConfig(system="System", user="Q: {query}\nA:")

    strategy = OurRetrieverRAGStrategy(
        name="our_retriever",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=3,
    )

    result = strategy.generate("test")
    # Should only retrieve top_k=3 documents
    assert len(result.iterations[0].retrieved) == 3


def test_our_retriever_strategy_iterative_behavior() -> None:
    """Test OurRetrieverRAGStrategy performs iterative refinement."""
    from docs2synth.rag.strategies import OurRetrieverRAGStrategy

    embedder = DummyEmbedder()
    store = _setup_store(embedder, ["Doc 1", "Doc 2"])
    outputs = ["First answer", "Refined answer", "Refined answer"]  # Converges on 3rd
    llm = SequencedLLMProvider(outputs)
    prompt = PromptConfig(system="System", user="Q: {query}\nA:")

    strategy = OurRetrieverRAGStrategy(
        name="our_retriever",
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompt=prompt,
        top_k=2,
        max_iterations=5,
        similarity_threshold=0.95,
    )

    result = strategy.generate("test")
    # Should perform multiple iterations until convergence
    assert len(result.iterations) >= 2
    assert result.final_answer == "Refined answer"
