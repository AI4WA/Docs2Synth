"""Main CLI entry point for Docs2Synth.

This module provides the main CLI command group and registers
all subcommands from the commands package.
"""

from __future__ import annotations

from pathlib import Path

import click

# Import all command modules
from docs2synth.cli.commands import (
    agent_group,
    annotate_command,
    datasets,
    preprocess,
    qa_group,
    rag_group,
    retriever_group,
    verify_group,
)
from docs2synth.utils import get_config, load_config, setup_cli_logging


@click.group()
@click.version_option(version="0.1.0", prog_name="docs2synth")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be repeated: -v, -vv)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.pass_context
def cli(ctx: click.Context, verbose: int, config: str | None) -> None:
    """Docs2Synth - Document processing and retriever training toolkit.

    A Python package for converting, synthesizing, and training retrievers
    for document datasets.
    """
    # Ensure ctx.obj exists
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Load configuration first (needed for logging setup)
    if config:
        cfg = load_config(config)
        ctx.obj["config"] = cfg
        ctx.obj["config_path"] = Path(config).resolve()
    else:
        ctx.obj["config"] = get_config()
        default_config_path = Path("config.yml")
        ctx.obj["config_path"] = (
            default_config_path.resolve() if default_config_path.exists() else None
        )

    # Initialize logging as early as possible so later logs use our handlers
    setup_cli_logging(verbose=verbose, config=ctx.obj["config"])


# Register all commands
cli.add_command(datasets)
cli.add_command(preprocess)
cli.add_command(agent_group)
cli.add_command(qa_group)
cli.add_command(rag_group)
cli.add_command(retriever_group)
cli.add_command(verify_group)
cli.add_command(annotate_command)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """Entry point for CLI."""
    import sys

    cli(args=argv or sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
