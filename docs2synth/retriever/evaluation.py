"""Evaluation utilities for retriever models."""

from __future__ import annotations

from typing import Iterable, Mapping

__all__ = ["evaluate_retriever"]

def evaluate_retriever(
    qa_pairs: Iterable[Mapping[str, str]],
    model_path: str,
    top_k: int = 5,
) -> float:  # pragma: no cover
    """Return Mean Reciprocal Rank (MRR) @ `top_k`."""
    raise NotImplementedError 