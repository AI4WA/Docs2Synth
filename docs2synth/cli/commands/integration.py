"""CLI helpers for orchestrating multi-stage Docs2Synth runs."""

from __future__ import annotations

import click


@click.command("run")
@click.pass_context
def run_command(ctx: click.Context) -> None:
    """Execute preprocess → QA → verify → retriever training pipeline."""
    # Import here to avoid circular import
    from docs2synth.integration.pipeline import run_pipeline

    click.echo(
        click.style(
            "Starting Docs2Synth automation pipeline (annotation skipped)...",
            fg="blue",
            bold=True,
        )
    )
    run_pipeline(ctx, include_validation=True)
