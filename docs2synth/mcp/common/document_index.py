"""Lightweight document index for processed documents.

Shared by tools and resources modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docs2synth.utils import get_config, get_logger

logger = get_logger(__name__)


class DocumentIndex:
    """Document index for searching processed documents."""

    def __init__(self, data_dir: str | Path | None = None):
        if data_dir is None:
            config = get_config()
            data_dir = config.get("data.processed_dir", "./data/processed")

        self.data_dir = Path(data_dir)
        self.documents: dict[str, dict[str, Any]] = {}
        self._load_documents()

    def _load_documents(self) -> None:
        """Load all processed documents from the data directory."""
        processed_dir = self.data_dir / "images"
        if not processed_dir.exists():
            logger.warning(f"Processed documents directory not found: {processed_dir}")
            return

        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                text_content: list[str] = []
                if "objects" in data:
                    for obj in data["objects"].values():
                        if "text" in obj and str(obj["text"]).strip():
                            text_content.append(str(obj["text"]).strip())

                doc_id = json_file.stem
                self.documents[doc_id] = {
                    "id": doc_id,
                    "title": f"Document {doc_id}",
                    "text": " ".join(text_content),
                    "url": f"file://{json_file.absolute()}",
                    "metadata": {
                        "source_file": str(json_file),
                        "object_count": len(data.get("objects", {})),
                        "processor": (
                            "PaddleOCR" if "_easyocr" not in doc_id else "EasyOCR"
                        ),
                    },
                }
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(f"Failed to load document {json_file}: {e}")

        logger.info(f"Loaded {len(self.documents)} documents from {processed_dir}")

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for documents containing the query text."""
        if not str(query).strip():
            return []

        query_lower = str(query).lower()
        results: list[dict[str, Any]] = []

        for doc in self.documents.values():
            if (
                query_lower in doc["text"].lower()
                or query_lower in doc["title"].lower()
            ):
                results.append(doc)

        def relevance_score(d: dict[str, Any]) -> int:
            text_lower = d["text"].lower()
            title_lower = d["title"].lower()
            return text_lower.count(query_lower) + title_lower.count(query_lower)

        results.sort(key=relevance_score, reverse=True)
        return results[:limit]

    def fetch(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a specific document by ID."""
        return self.documents.get(str(doc_id))


_document_index: DocumentIndex | None = None


def get_document_index() -> DocumentIndex:
    """Get or create the global document index."""
    global _document_index
    if _document_index is None:
        _document_index = DocumentIndex()
    return _document_index
