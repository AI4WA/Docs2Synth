"""Command line interface for running Docs2Synth MCP server."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import click
import uvicorn

from docs2synth.mcp.config import MCPConfig
from docs2synth.mcp.server import build_server
from docs2synth.utils import setup_logging_from_config
from docs2synth.utils.logging import LOG_FORMAT


def _is_broken_pipe(exc: BaseException) -> bool:
    """Check if exception is a BrokenPipeError (including nested exceptions)."""
    if isinstance(exc, BrokenPipeError):
        return True
    if nested := getattr(exc, "exceptions", None):
        return any(_is_broken_pipe(child) for child in nested)
    return False


def _setup_logging(log_level: str = "info") -> None:
    """Configure logging for MCP server."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    try:
        setup_logging_from_config()
    except (OSError, Exception):
        pass  # Use default configuration


@click.group()
def main() -> None:
    """Run Docs2Synth as an MCP provider."""


@main.command()
@click.option(
    "--host", default=None, help="Bind address (default: from config or 0.0.0.0)"
)
@click.option(
    "--port", default=None, type=int, help="Port to bind (default: from config or 8009)"
)
@click.option("--log-level", default="info", help="Uvicorn log level")
@click.option(
    "--config", default=None, help="Path to config.mcp.yml (default: ./config.mcp.yml)"
)
def sse(host: str | None, port: int | None, log_level: str, config: str | None) -> None:
    """Start MCP server with StreamableHTTP transport.

    Configuration priority: CLI arguments > environment variables > config file > defaults
    """
    _setup_logging(log_level)

    # Load configuration
    config_path = Path(config) if config else None
    mcp_config = MCPConfig.load(config_path)

    # Override with CLI arguments
    if host:
        mcp_config.server.host = host
    if port:
        mcp_config.server.port = port

    # Build server
    app, http_transport, mcp_server = build_server(mcp_config)

    # Create lifespan context for transport initialization
    @asynccontextmanager
    async def lifespan(_app):
        """Initialize StreamableHTTP transport and run MCP server."""
        async with http_transport.connect() as (read_stream, write_stream):
            # Start MCP server in background
            async def run_mcp_server():
                try:
                    await mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp_server.create_initialization_options(),
                    )
                except Exception as e:
                    logging.error(f"MCP server error: {e}", exc_info=True)

            task = asyncio.create_task(run_mcp_server())

            yield

            # Cleanup
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.router.lifespan_context = lifespan

    # Display configuration (concise)
    click.echo(
        f"MCP server: {mcp_config.server.base_url} — StreamableHTTP (MCP 2025-03-26)"
    )
    click.echo("Endpoints: /mcp (OAuth) · /health")
    click.echo(
        f"OAuth: discovery={mcp_config.oauth.discovery_url} · validation={'Introspection' if mcp_config.oauth.use_introspection else 'JWT'} · client_id={mcp_config.oauth.client_id}"
    )
    click.echo(
        "Connect via MCP Inspector with OAuth 2.0; authorize and use Bearer token."
    )
    click.echo("")

    # Run server
    uvicorn.run(
        app,
        host=mcp_config.server.host,
        port=mcp_config.server.port,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
