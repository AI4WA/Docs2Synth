"""Preprocessing commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from docs2synth.utils import get_logger

logger = get_logger(__name__)


@click.command("preprocess")
@click.argument("path", type=click.Path(path_type=Path), required=False)
@click.option(
    "--processor",
    "processor_name",
    type=click.Choice(["paddleocr", "pdfplumber", "easyocr"], case_sensitive=False),
    default=None,
    help="Name of the processor to use (paddleocr: general OCR, pdfplumber: parsed PDFs, easyocr: 80+ languages OCR). Defaults to config.preprocess.processor.",
)
@click.option(
    "--lang",
    type=str,
    default=None,
    help="Optional OCR language override (e.g., en). Defaults to config.preprocess.lang.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write processed outputs (defaults to config.preprocess.output_dir or config.data.processed_dir)",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "gpu", "cuda"], case_sensitive=False),
    default=None,
    help="Device for OCR inference. Defaults to config.preprocess.device (or auto-select GPU when available).",
)
@click.pass_context
def preprocess(
    ctx: click.Context,
    path: Path | None,
    processor_name: str | None,
    lang: str | None,
    output_dir: Path | None,
    device: str | None,
) -> None:
    """Preprocess an image file or all images in a directory.

    PATH can be a file or a directory. If a directory is provided, all files in
    that directory are processed. Results are written as JSON into the
    configured output directory (data.processed_dir).

    If PATH is not provided, uses config.preprocess.input_dir. If neither is set,
    an error is returned.

    Default values for processor, lang, and device are read from config.preprocess
    if not specified via command-line options.
    """

    from docs2synth.preprocess.runner import run_preprocess

    cfg = ctx.obj.get("config")

    # Get input path: CLI argument > config.preprocess.input_dir > error
    if path is None:
        input_dir = cfg.get("preprocess.input_dir")
        if input_dir is None:
            click.echo(
                click.style(
                    "✗ Error: PATH argument is required, or set config.preprocess.input_dir",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)
        path = Path(input_dir)

    # Validate that the path exists
    if not path.exists():
        click.echo(
            click.style(
                f"✗ Error: Input path does not exist: {path}",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    # Get defaults from config if not provided via CLI
    if processor_name is None:
        processor_name = cfg.get("preprocess.processor", "paddleocr")

    if lang is None:
        lang = cfg.get("preprocess.lang")

    if device is None:
        device = cfg.get("preprocess.device")

    # Get output_dir from config if not provided via CLI
    if output_dir is None:
        output_dir = cfg.get("preprocess.output_dir")
        if output_dir is not None:
            output_dir = Path(output_dir)

    try:
        logger.info(
            f"Running preprocessing with: processor={processor_name}, lang={lang}, device={device}, output_dir={output_dir}"
        )
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
