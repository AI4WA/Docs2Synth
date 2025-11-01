"""Active configuration tool."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent

from docs2synth.utils import get_config


async def handle_active_config(_args: dict[str, Any]) -> list[TextContent]:
    """Return current Docs2Synth configuration."""
    config = get_config()
    return [TextContent(type="text", text=json.dumps(config.to_dict(), indent=2))]


TOOL_SPEC = {
    "name": "active_config",
    "description": "Return Docs2Synth default configuration values and active overrides.",
    "inputSchema": {"type": "object", "properties": {}},
}
