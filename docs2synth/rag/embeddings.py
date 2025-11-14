"""Embedding utilities for RAG pipelines."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _load_sentence_transformer(
    model_name: str, device: str | None
) -> "SentenceTransformer":
    """Load and cache a sentence-transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - handled during runtime
        raise ImportError(
            "sentence-transformers is required for RAG embeddings. "
            "Install with `pip install sentence-transformers`."
        ) from exc

    logger.info("Loading embedding model %s (device=%s)", model_name, device or "auto")
    return SentenceTransformer(model_name, device=device or None)


class EmbeddingModel:
    """Wrapper around SentenceTransformer with normalization helpers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = _load_sentence_transformer(model_name, device)
        self._dimension: int | None = None

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of texts."""
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query and return a 1D vector."""
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        """Return the dimensionality of embeddings produced by the model."""
        if self._dimension is None:
            test = self.embed_texts(["dimension-probe"])
            self._dimension = int(test.shape[1]) if test.size else 0
        return self._dimension

    def __repr__(self) -> str:
        return f"EmbeddingModel(model_name={self.model_name}, device={self.device})"
