"""Retrieval-Augmented Generation utilities for Docs2Synth."""

from .embeddings import EmbeddingModel
from .pipeline import RAGPipeline
from .strategies import (
    EnhancedIterativeRAGStrategy,
    NaiveRAGStrategy,
    OurRetrieverRAGStrategy,
)
from .types import (
    DocumentChunk,
    IterationResult,
    RAGResult,
    RAGState,
    RetrievedDocument,
)
from .vector_store import FaissVectorStore

__all__ = [
    "EmbeddingModel",
    "FaissVectorStore",
    "RAGPipeline",
    "NaiveRAGStrategy",
    "EnhancedIterativeRAGStrategy",
    "OurRetrieverRAGStrategy",
    "DocumentChunk",
    "RetrievedDocument",
    "IterationResult",
    "RAGResult",
    "RAGState",
]
