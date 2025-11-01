"""Document search tool."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent

from ..common.document_index import get_document_index


async def handle_search(args: dict[str, Any]) -> list[TextContent]:
    """Search through processed documents."""
    query = args.get("query", "")
    doc_index = get_document_index()
    results = doc_index.search(query, limit=10)
    formatted = [{"id": d["id"], "title": d["title"], "url": d["url"]} for d in results]
    return [TextContent(type="text", text=json.dumps({"results": formatted}, indent=2))]


TOOL_SPEC = {
    "name": "search",
    "description": "Search through processed documents for relevant content based on a query string.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            }
        },
        "required": ["query"],
    },
}
