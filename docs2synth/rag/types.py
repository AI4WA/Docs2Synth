"""Typed containers used across Docs2Synth RAG modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DocumentChunk:
    """Represents an indexed document chunk in the vector store."""

    id: str
    text: str
    metadata: Dict[str, Any] | None = None


@dataclass
class RetrievedDocument(DocumentChunk):
    """Document chunk returned from similarity search."""

    score: float = 0.0


@dataclass
class IterationResult:
    """Stores data for a single retrieval + generation iteration."""

    query: str
    retrieved: Sequence[RetrievedDocument]
    answer: str
    similarity: Optional[float] = None
    step: int = 0


@dataclass
class RAGResult:
    """Result returned by a RAG strategy."""

    final_answer: str
    iterations: List[IterationResult] = field(default_factory=list)

    def add_iteration(self, iteration: IterationResult) -> None:
        """Append an iteration to the result history."""
        self.iterations.append(iteration)


@dataclass
class RAGState:
    """Conversation state shared across strategy runs."""

    iterations: List[IterationResult] = field(default_factory=list)

    @property
    def last_iteration(self) -> Optional[IterationResult]:
        """Return the latest iteration result."""
        if not self.iterations:
            return None
        return self.iterations[-1]

    def push(self, iteration: IterationResult) -> None:
        """Append a new iteration result to the history."""
        self.iterations.append(iteration)

    def reset(self) -> None:
        """Clear history."""
        self.iterations.clear()
