"""Retriever model wrappers.

This module provides convenience wrappers around Hugging Face Transformers
(or other backends) for training and inference. Replace the stubs with actual
implementations as needed.
"""

from __future__ import annotations

from typing import Protocol

class Retriever(Protocol):
    """Simple retriever protocol for typing purposes."""

    def encode(self, texts: list[str]) -> "numpy.ndarray":
        ...

    def similarity(self, queries: list[str], documents: list[str]) -> "numpy.ndarray":
        ...

__all__ = ["Retriever"] 