"""Vector store abstractions for Docs2Synth RAG."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence

import numpy as np

from docs2synth.rag.types import DocumentChunk, RetrievedDocument
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract base vector store."""

    @abstractmethod
    def add_embeddings(
        self, embeddings: np.ndarray, documents: Sequence[DocumentChunk]
    ) -> None:
        """Add embedded documents to the store."""

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[RetrievedDocument]:
        """Retrieve similar documents for a query embedding."""

    @abstractmethod
    def get_all_documents(self) -> List[DocumentChunk]:
        """Get all documents stored in the vector store."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of stored vectors."""

    @abstractmethod
    def reset(self) -> None:
        """Remove all stored vectors and metadata."""


class FaissVectorStore(VectorStore):
    """FAISS-backed vector store with optional persistence."""

    def __init__(
        self,
        dimension: int | None = None,
        persist_path: str | Path | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise ImportError(
                "faiss-cpu is required for the FAISS vector store. "
                "Install with `pip install faiss-cpu`."
            ) from exc

        self._dimension = dimension
        self._normalize = normalize
        self._documents: List[DocumentChunk] = []
        self._index = None
        self._persist_path = Path(persist_path) if persist_path else None
        self._meta_path = (
            self._persist_path.with_suffix(self._persist_path.suffix + ".meta.json")
            if self._persist_path
            else None
        )

        if self._persist_path and self._persist_path.exists():
            self._load_from_disk()

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _ensure_index(self, embeddings: np.ndarray) -> None:
        import faiss

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array [n, dim]")

        if self._index is None:
            dim = embeddings.shape[1]
            if self._dimension is not None and self._dimension != dim:
                raise ValueError(
                    f"Expected embeddings of dimension {self._dimension}, got {dim}"
                )
            self._dimension = dim
            self._index = faiss.IndexFlatIP(dim)
        elif embeddings.shape[1] != self._dimension:
            # Dimension mismatch: reset the index and documents
            logger.warning(
                f"Embedding dimension mismatch: store={self._dimension}, data={embeddings.shape[1]}. "
                f"Resetting vector store to use new dimension."
            )
            # Clear existing index and documents
            self._index = None
            self._documents = []
            # Clean up persisted files if they exist
            if self._persist_path and self._persist_path.exists():
                self._persist_path.unlink()
            if self._meta_path and self._meta_path.exists():
                self._meta_path.unlink()
            # Create new index with correct dimension
            self._dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)

    def add_embeddings(
        self, embeddings: np.ndarray, documents: Sequence[DocumentChunk]
    ) -> None:
        import faiss

        if not len(documents):
            return
        if embeddings.shape[0] != len(documents):
            raise ValueError("Embeddings and documents must be aligned in length")

        self._ensure_index(embeddings)
        data = np.asarray(embeddings, dtype="float32")
        if self._normalize:
            faiss.normalize_L2(data)
        self._index.add(data)
        self._documents.extend(documents)

        if self._persist_path:
            self._save_to_disk()

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[RetrievedDocument]:
        import faiss

        if self._index is None or len(self._documents) == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension mismatch: store={self._dimension}, query={query.shape[1]}. "
                f"This usually means the vector store was created with a different embedding model. "
                f"Please reset the vector store (docs2synth rag reset) and re-index your documents "
                f"(docs2synth rag ingest) with the current embedding model."
            )
        if self._normalize:
            faiss.normalize_L2(query)

        scores, indices = self._index.search(query, top_k)
        retrieved: List[RetrievedDocument] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            retrieved.append(
                RetrievedDocument(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata or {},
                    score=float(score),
                )
            )
        return retrieved

    def get_all_documents(self) -> List[DocumentChunk]:
        """Get all documents stored in the vector store."""
        return list(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    def reset(self) -> None:
        """Clear the index and remove persisted files if configured."""
        self._index = None
        self._documents = []
        if self._persist_path and self._persist_path.exists():
            self._persist_path.unlink()
        if self._meta_path and self._meta_path.exists():
            self._meta_path.unlink()

    def _save_to_disk(self) -> None:
        import faiss

        if not self._persist_path:
            return
        if self._index is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._persist_path))
        if self._meta_path:
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"id": doc.id, "text": doc.text, "metadata": doc.metadata or {}}
                        for doc in self._documents
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        logger.info("Persisted FAISS index to %s", self._persist_path)

    def _load_from_disk(self) -> None:
        import faiss

        logger.info("Loading FAISS index from %s", self._persist_path)
        self._index = faiss.read_index(str(self._persist_path))
        self._dimension = self._index.d
        if self._meta_path and self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._documents = [
                DocumentChunk(
                    id=item["id"], text=item["text"], metadata=item.get("metadata")
                )
                for item in data
            ]
        else:
            raise FileNotFoundError(
                f"Missing metadata file for persisted index: {self._meta_path}"
            )
