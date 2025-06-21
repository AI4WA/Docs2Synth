"""Retriever inference utilities."""

from __future__ import annotations

from typing import Sequence

__all__ = ["retrieve"]

def retrieve(query: str, corpus: Sequence[str], model_path: str) -> list[int]:  # pragma: no cover
    """Return indices of top documents for `query` from `corpus`."""
    raise NotImplementedError 