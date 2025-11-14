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
