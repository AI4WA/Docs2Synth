"""Data loading utilities for retriever training/evaluation."""

from __future__ import annotations

from typing import Iterable, Mapping, Any

__all__ = [
    "load_qa_pairs",
]

def load_qa_pairs(path: str) -> Iterable[Mapping[str, Any]]:  # pragma: no cover
    """Load QA pairs for retriever training from `path`."""
    raise NotImplementedError 