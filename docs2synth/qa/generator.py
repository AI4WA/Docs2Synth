"""LLM‐based question‐answer pair generation utilities."""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "generate_qa_pairs",
]

def generate_qa_pairs(documents: Iterable[Mapping[str, str]]) -> Iterable[dict[str, str]]:  # pragma: no cover
    """Generate QA pairs from an iterable of `documents`.

    Parameters
    ----------
    documents: iterable of mappings
        Each mapping should at least contain the raw text under the key
        ``"text"``. You may add more metadata fields as needed.

    Yields
    ------
    dict
        A mapping with (at minimum) ``"question"`` and ``"answer"`` keys.
    """

    for doc in documents:
        text = doc.get("text", "")
        logger.debug("Generating QA pair for doc length=%d", len(text))

        # TODO: integrate with your favourite LLM provider (e.g. OpenAI, HF Hub)
        # Here we simply yield a placeholder.
        yield {
            "question": "<generated question>",
            "answer": "<generated answer>",
        } 