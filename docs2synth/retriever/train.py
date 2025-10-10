"""Training loop for retriever models."""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

__all__ = ["train_retriever"]


def train_retriever(
    qa_pairs: Iterable[Mapping[str, str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "models/retriever",
) -> str:  # pragma: no cover
    """Train a retriever on `qa_pairs` and save to `output_dir`.

    Returns the path to the saved model directory.
    """
    logger.info("Starting training with %d QA pairs", len(list(qa_pairs)))

    # TODO: implement actual training; integrate with sentence-transformers Trainer.

    logger.warning("Retriever training not yet implemented; returning output_dir")
    return output_dir
