"""Command line helpers for running Docs2Synth MCP servers."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from docs2synth.mcp.server import build_server
from docs2synth.utils import setup_logging_from_config
from docs2synth.utils.logging import LOG_FORMAT

# Fix sys.excepthook to use the original Python excepthook
# The exceptiongroup library replaces it but can fail in certain contexts
# while the mcp stdio transport uses sys.stderr for logging, the http transport uses sys.stdout
# so we need to restore the original excepthook to avoid logging to the wrong stream
_original_excepthook = sys.__excepthook__
sys.excepthook = _original_excepthook


def _run(coro: asyncio.Future) -> None:
    """
    Run the MCP server asynchronously and handle exceptions.

    Args:
        coro: The asynchronous coroutine to run.

    Raises:
        KeyboardInterrupt: If the user interrupts the server.
        BaseException: If the server terminates unexpectedly.

    Returns:
        None

    Notes:
        - The exceptiongroup library replaces the original excepthook but can fail in certain contexts
        - The mcp stdio transport uses sys.stderr for logging, the http transport uses sys.stdout
        - so we need to restore the original excepthook to avoid logging to the wrong stream
    """
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        click.echo("Received interrupt. Shutting down MCP server.")
    except BaseException as exc:  # noqa: BLE001 - we want to inspect nested groups
        if _is_broken_pipe(exc):
            # Client disconnected; exit quietly.
            return
        import traceback

        click.echo("Error: MCP server terminated unexpectedly", err=True)
        traceback.print_exception(exc, file=sys.stderr)
        sys.exit(1)


def _is_broken_pipe(exc: BaseException) -> bool:
    """Check if exception is a BrokenPipeError (including nested exceptions)."""
    if isinstance(exc, BrokenPipeError):
        return True
    # Check nested exception groups
    if nested := getattr(exc, "exceptions", None):
        return any(_is_broken_pipe(child) for child in nested)
    return False


def _setup_logging() -> None:
    """Configure logging to stderr for MCP server."""
    # MCP servers should log to stderr (stdout is reserved for JSON-RPC)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Try to load additional config, but don't fail if it doesn't exist
    try:
        setup_logging_from_config()
    except (OSError, Exception):
        pass  # Use default configuration above


@click.group()
def main() -> None:
    """Run Docs2Synth as an MCP provider."""


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address.")
@click.option("--port", default=8000, show_default=True, help="Port to bind.", type=int)
@click.option("--log-level", default="info", help="Uvicorn log level.")
def sse(host: str, port: int, log_level: str) -> None:
    """Expose the MCP server over SSE (Server-Sent Events) transport for ChatGPT integration."""
    _setup_logging()
    server = build_server()
    _run(
        server.run_http_async(
            log_level=log_level,
            transport="sse",
            host=host,
            port=port,
        )
    )


@main.command()
def stdio() -> None:
    """Expose the MCP server using stdio transport."""
    _setup_logging()
    server = build_server()
    _run(server.run_stdio_async(log_level="warning"))


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    main()
