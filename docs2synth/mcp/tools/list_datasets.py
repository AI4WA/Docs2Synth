"""List available datasets tool."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent


async def handle_list_datasets(_args: dict[str, Any]) -> list[TextContent]:
    """List all available datasets for download."""
    from docs2synth.datasets.downloader import DATASETS

    names = sorted(DATASETS.keys())
    return [TextContent(type="text", text=json.dumps({"datasets": names}, indent=2))]


TOOL_SPEC = {
    "name": "list_datasets",
    "description": "List dataset names registered for Docs2Synth downloads.",
    "inputSchema": {"type": "object", "properties": {}},
}
