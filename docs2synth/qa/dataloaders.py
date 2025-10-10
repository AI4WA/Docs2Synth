"""Data loading utilities for QA pair generation.

Functions in this module should load raw documents or pre‐processed datasets
and yield the formats expected by the QA generation pipeline.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = [
    "load_documents",
]


def load_documents(path: str) -> Iterable[Mapping[str, Any]]:  # pragma: no cover
    """Load documents from a given `path`.

    At the moment this is a stub; replace with your own logic to load whatever
    data format you are working with (e.g. JSONL, CSV, Parquet).
    """
    raise NotImplementedError
