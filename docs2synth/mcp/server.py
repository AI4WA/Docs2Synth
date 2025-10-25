"""MCP server definition for Docs2Synth."""

from __future__ import annotations

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

    # Add simple mock resources using decorator approach
    @server.resource(
        "resource://docs2synth/info",
        description="Simple info resource for testing.",
        mime_type="text/plain",
    )
    def info_resource() -> str:
        logger.info("MCP resource info requested")
        return "This is a simple info resource for testing Docs2Synth MCP server."

    @server.resource(
        "resource://docs2synth/status",
        description="Server status resource for testing.",
        mime_type="application/json",
    )
    def status_resource() -> str:
        logger.info("MCP resource status requested")
        return '{"status": "running", "version": "0.1.0", "server": "Docs2Synth MCP"}'

    # Add simple mock prompts
    @server.prompt(
        "hello",
        description="Simple hello prompt for testing.",
    )
    def hello_prompt() -> str:
        logger.info("MCP prompt hello invoked")
        return "Hello! This is a simple test prompt from Docs2Synth MCP server."

    @server.prompt(
        "help",
        description="Help prompt with basic server information.",
    )
    def help_prompt() -> str:
        logger.info("MCP prompt help invoked")
        return """# Docs2Synth MCP Help

This server provides:
- **Tools**: list_datasets, dataset_info, active_config
- **Resources**: info, status
- **Prompts**: hello, help

Use the tools to interact with datasets and configuration."""

    return server
