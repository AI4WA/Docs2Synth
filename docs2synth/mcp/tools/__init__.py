"""MCP tools registration and integration."""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from docs2synth.utils import get_logger

from . import active_config, dataset_info, fetch, list_datasets, search

logger = get_logger(__name__)

# Collect all tool modules
TOOL_MODULES = [
    list_datasets,
    dataset_info,
    active_config,
    search,
    fetch,
]

# Collect tool specifications and handlers
TOOL_SPECS = [module.TOOL_SPEC for module in TOOL_MODULES]
TOOL_HANDLERS = {
    module.TOOL_SPEC["name"]: getattr(module, f"handle_{module.TOOL_SPEC['name']}")
    for module in TOOL_MODULES
}


def register_tools(server: Server) -> None:
    """Register all MCP tools with the server."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available tools."""
        return [Tool(**spec) for spec in TOOL_SPECS]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Handle tool invocation by routing to the appropriate handler."""
        logger.debug(f"Tool called: {name}")
        try:
            handler = TOOL_HANDLERS.get(name)
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
