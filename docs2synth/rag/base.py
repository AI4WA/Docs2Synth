"""Base abstractions for Retrieval-Augmented Generation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

from docs2synth.agent.base import BaseLLMProvider
from docs2synth.rag.embeddings import EmbeddingModel
from docs2synth.rag.types import (
    IterationResult,
    RAGResult,
    RAGState,
    RetrievedDocument,
)
from docs2synth.rag.vector_store import VectorStore
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptConfig:
    """Prompt template configuration."""

    system: str
    user: str


class RAGStrategy(ABC):
    """Abstract base class for RAG strategies."""

    def __init__(
        self,
        name: str,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
        llm: BaseLLMProvider,
        prompt: PromptConfig,
        top_k: int = 5,
    ) -> None:
        self.name = name
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.prompt = prompt
        self.top_k = top_k
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate(self, query: str, state: Optional[RAGState] = None) -> RAGResult:
        """Execute the retrieval-augmented generation flow."""

    def _retrieve(self, query: str) -> Sequence[RetrievedDocument]:
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, top_k=self.top_k)

    def _build_context(self, documents: Sequence[RetrievedDocument]) -> str:
        lines: List[str] = []
        for doc in documents:
            header = f"[score={doc.score:.3f}]"
            if doc.metadata:
                metadata_repr = ", ".join(
                    f"{key}={value}" for key, value in doc.metadata.items()
                )
                header += f" ({metadata_repr})"
            lines.append(header)
            lines.append(doc.text.strip())
            lines.append("")  # spacer
        return "\n".join(lines).strip()

    def _invoke_llm(self, query: str, context: str) -> str:
        """Render the prompt and invoke the LLM provider."""
        user_prompt = self.prompt.user.format(query=query, context=context)
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm.chat(messages)
        self.logger.debug("LLM response metadata: %s", response.metadata)
        return response.content.strip()

    def _build_iteration(
        self,
        step: int,
        query: str,
        documents: Sequence[RetrievedDocument],
        answer: str,
        similarity: Optional[float] = None,
    ) -> IterationResult:
        return IterationResult(
            step=step,
            query=query,
            retrieved=list(documents),
            answer=answer,
            similarity=similarity,
        )
