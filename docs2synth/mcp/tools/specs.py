from __future__ import annotations

from typing import Any

# Central registry for tool specifications used by the MCP server
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_datasets",
        "description": "List dataset names registered for Docs2Synth downloads.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "dataset_info",
        "description": "Describe the download URL and suggested command for a dataset.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Dataset name"}},
            "required": ["name"],
        },
    },
    {
        "name": "active_config",
        "description": "Return Docs2Synth default configuration values and active overrides.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
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
    },
    {
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
    },
]
