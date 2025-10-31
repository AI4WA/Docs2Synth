from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from mcp.server import Server
from mcp.types import TextContent, Tool

from docs2synth.utils import get_config, get_logger

from ..common.document_index import get_document_index
from .specs import TOOL_SPECS as BASE_TOOL_SPECS

logger = get_logger(__name__)


def register_tools(server: Server) -> None:
    """Register tool metadata and handlers."""

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [Tool(**spec) for spec in BASE_TOOL_SPECS]

    async def handle_list_datasets(_args: Dict[str, Any]) -> List[TextContent]:
        from docs2synth.datasets.downloader import DATASETS

        names = sorted(DATASETS.keys())
        return [
            TextContent(type="text", text=json.dumps({"datasets": names}, indent=2))
        ]

    async def handle_dataset_info(args: Dict[str, Any]) -> List[TextContent]:
        from docs2synth.datasets.downloader import DATASETS

        dataset_name = args.get("name")
        datasets = {key.lower(): key for key in DATASETS.keys()}
        key = (
            datasets.get(str(dataset_name).lower())
            if dataset_name is not None
            else None
        )

        if key is None:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Unknown dataset '{dataset_name}'. Use list_datasets to see available values.",
                        }
                    ),
                )
            ]

        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "name": key,
                        "download_url": DATASETS[key],
                        "cli_example": f"docs2synth datasets download {key}",
                    },
                    indent=2,
                ),
            )
        ]

    async def handle_active_config(_args: Dict[str, Any]) -> List[TextContent]:
        config = get_config()
        return [TextContent(type="text", text=json.dumps(config.to_dict(), indent=2))]

    async def handle_search(args: Dict[str, Any]) -> List[TextContent]:
        query = args.get("query", "")
        doc_index = get_document_index()
        results = doc_index.search(query, limit=10)
        formatted = [
            {"id": d["id"], "title": d["title"], "url": d["url"]} for d in results
        ]
        return [
            TextContent(type="text", text=json.dumps({"results": formatted}, indent=2))
        ]

    async def handle_fetch(args: Dict[str, Any]) -> List[TextContent]:
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

    handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "list_datasets": handle_list_datasets,
        "dataset_info": handle_dataset_info,
        "active_config": handle_active_config,
        "search": handle_search,
        "fetch": handle_fetch,
    }

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> List[TextContent]:
        logger.debug(f"Tool called: {name}")
        try:
            handler = handlers.get(name)
            if not handler:
                return [
                    TextContent(
                        type="text", text=json.dumps({"error": f"Unknown tool: {name}"})
                    )
                ]
            return await handler(arguments or {})
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Error executing tool {name}: {e}", exc_info=True)
            return [
                TextContent(
                    type="text", text=json.dumps({"error": str(e), "tool": name})
                )
            ]
