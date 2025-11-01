"""Document fetch tool."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent

from ..common.document_index import get_document_index


async def handle_fetch(args: dict[str, Any]) -> list[TextContent]:
    """Fetch a specific document by ID."""
    doc_id = args.get("doc_id")
    doc_index = get_document_index()
    doc = doc_index.fetch(doc_id)
    if doc is None:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Document not found", "id": doc_id}),
            )
        ]
    return [TextContent(type="text", text=json.dumps(doc, indent=2))]


TOOL_SPEC = {
    "name": "fetch",
    "description": "Fetch the full content of a specific document by its unique identifier.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "Unique identifier for the document",
            }
        },
        "required": ["doc_id"],
    },
}
