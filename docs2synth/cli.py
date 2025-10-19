"""Command-line interface for Docs2Synth.

This module provides CLI commands for document processing, QA generation,
and retriever training using Click framework with proper error handling
and logging integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from docs2synth.utils import get_logger, load_config, setup_cli_logging

logger = get_logger(__name__)


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
    else:
        from docs2synth.utils import get_config

        ctx.obj["config"] = get_config()

    # Initialize logging as early as possible so later logs use our handlers
    setup_cli_logging(verbose=verbose, config=ctx.obj["config"])


@cli.command("datasets")
@click.argument("action", type=click.Choice(["download", "list"]))
@click.argument("name", required=False)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Directory to save datasets (default: from config)",
)
@click.pass_context
def datasets(
    ctx: click.Context, action: str, name: str | None, output_dir: str | None
) -> None:
    """Manage datasets.

    ACTION: 'download' or 'list'
    NAME: Dataset name (required for download, use 'all' to download all)

    Examples:
        docs2synth datasets list
        docs2synth datasets download vrd-iu2024-tracka
        docs2synth datasets download all
    """
    from docs2synth.datasets.downloader import DATASETS, download_dataset

    try:
        if action == "list":
            click.echo(click.style("Available datasets:", fg="green", bold=True))
            for dataset_name in DATASETS.keys():
                click.echo(f"  - {dataset_name}")

        elif action == "download":
            if name is None:
                click.echo(
                    click.style("✗ Error: NAME required for download", fg="red"),
                    err=True,
                )
                sys.exit(1)

            if name == "all":
                if output_dir:
                    click.echo(
                        click.style(
                            f"Downloading all datasets to {output_dir}...", fg="blue"
                        )
                    )
                else:
                    click.echo(click.style("Downloading all datasets...", fg="blue"))
                for dataset_name in DATASETS.keys():
                    click.echo(
                        click.style(f"\nDownloading {dataset_name}...", fg="cyan")
                    )
                    dataset_path = download_dataset(dataset_name, output_dir)
                    click.echo(
                        click.style(
                            f"✓ {dataset_name} saved to {dataset_path}", fg="green"
                        )
                    )
                click.echo(click.style("\n✓ All datasets downloaded!", fg="green"))
            else:
                if output_dir:
                    click.echo(
                        click.style(f"Downloading {name} to {output_dir}...", fg="blue")
                    )
                else:
                    click.echo(click.style(f"Downloading {name}...", fg="blue"))
                dataset_path = download_dataset(name, output_dir)
                click.echo(
                    click.style(f"✓ Dataset saved to {dataset_path}", fg="green")
                )

    except ValueError as e:
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Dataset operation failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command("preprocess")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--processor",
    "processor_name",
    type=click.Choice(["paddleocr", "pdfplumber", "easyocr"], case_sensitive=False),
    default="paddleocr",
    show_default=True,
    help="Name of the processor to use (paddleocr: general OCR, pdfplumber: parsed PDFs, easyocr: 80+ languages OCR).",
)
@click.option(
    "--lang",
    type=str,
    default=None,
    help="Optional OCR language override (e.g., en)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write processed outputs (defaults to config data.processed_dir)",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "gpu", "cuda"], case_sensitive=False),
    default=None,
    help="Device for OCR inference. If omitted, auto-select GPU when available.",
)
@click.pass_context
def preprocess(
    ctx: click.Context,
    path: Path,
    processor_name: str,
    lang: str | None,
    output_dir: Path | None,
    device: str | None,
) -> None:
    """Preprocess an image file or all images in a directory.

    PATH can be a file or a directory. If a directory is provided, all files in
    that directory are processed. Results are written as JSON into the
    configured output directory (data.processed_dir).
    """

    from docs2synth.preprocess.runner import run_preprocess

    cfg = ctx.obj.get("config")
    try:
        num_success, num_failed, _ = run_preprocess(
            path,
            processor=processor_name,
            output_dir=output_dir,
            lang=lang,
            device=device,
            config=cfg,
        )
        click.echo(
            click.style(
                f"Done. Success: {num_success}, Failed: {num_failed}", fg="blue"
            )
        )
    except Exception as e:
        logger.exception("Preprocess command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """Entry point for the console script."""
    cli(args=argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
