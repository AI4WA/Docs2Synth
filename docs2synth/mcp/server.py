"""MCP server definition for Docs2Synth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from docs2synth import __version__
from docs2synth.utils import get_config, get_logger

SERVER_NAME = "Docs2Synth MCP"
SERVER_INSTRUCTIONS = (
    "Exposes Docs2Synth utilities for dataset discovery and configuration. "
    "Use the available tools to list dataset registries or inspect default "
    "paths before orchestrating downstream pipelines."
)

logger = get_logger(__name__)


def build_server() -> FastMCP:
    """Create and configure the FastMCP server instance."""
    server = FastMCP(
        name=SERVER_NAME,
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
        include_fastmcp_meta=False,
    )

    @server.tool(description="List dataset names registered for Docs2Synth downloads.")
    def list_datasets() -> list[str]:
        from docs2synth.datasets.downloader import DATASETS

        names = sorted(DATASETS.keys())
        logger.info("MCP tool list_datasets invoked (%d datasets)", len(names))
        return names

    @server.tool(
        description="Describe the download URL and suggested command for a dataset.",
    )
    def dataset_info(name: str) -> dict[str, Any]:
        from docs2synth.datasets.downloader import DATASETS

        datasets = {key.lower(): key for key in DATASETS.keys()}
        key = datasets.get(name.lower())
        if key is None:
            raise ValueError(
                f"Unknown dataset '{name}'. Use list_datasets to see available values."
            )

        download_url = DATASETS[key]
        logger.info("MCP tool dataset_info invoked for dataset '%s'", key)
        return {
            "name": key,
            "download_url": download_url,
            "cli_example": f"docs2synth datasets download {key}",
        }

    @server.tool(
        description="Return Docs2Synth default configuration values and active overrides."
    )
    def active_config() -> dict[str, Any]:
        # Config returns nested dict that is JSON serialisable
        logger.info("MCP tool active_config invoked")
        config = get_config()
        return config.to_dict()

    example_config_path = Path(__file__).resolve().parents[2] / "config.example.yml"
    if example_config_path.exists():

        @server.resource(
            "resource://docs2synth/config-example",
            description="Example configuration YAML bundled with Docs2Synth.",
            mime_type="text/yaml",
        )
        def config_example() -> str:
            logger.info("MCP resource config-example requested")
            return example_config_path.read_text(encoding="utf-8")

    return server
