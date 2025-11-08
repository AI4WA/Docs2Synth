"""Datasets management commands."""

from __future__ import annotations

import sys

import click

from docs2synth.utils import get_logger

logger = get_logger(__name__)


@click.command("datasets")
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
