"""Dataset information tool."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent


async def handle_dataset_info(args: dict[str, Any]) -> list[TextContent]:
    """Get detailed information about a specific dataset."""
    from docs2synth.datasets.downloader import DATASETS

    dataset_name = args.get("name")
    datasets = {key.lower(): key for key in DATASETS.keys()}
    key = datasets.get(str(dataset_name).lower()) if dataset_name is not None else None

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


TOOL_SPEC = {
    "name": "dataset_info",
    "description": "Describe the download URL and suggested command for a dataset.",
    "inputSchema": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Dataset name"}},
        "required": ["name"],
    },
}
