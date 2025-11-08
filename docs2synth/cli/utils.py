"""Shared utility functions for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


def resolve_config_path(config_path: str | None) -> str | None:
    """Resolve default config path to ./config.yml when available."""
    if not config_path and Path("./config.yml").exists():
        return "./config.yml"
    return config_path


def build_gen_kwargs(
    temperature: float | None,
    max_tokens: int | None,
    response_format: str | None = None,
) -> dict[str, Any]:
    """Build generation kwargs from CLI options."""
    params: dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if response_format:
        params["response_format"] = response_format
    return params


def load_history_file(history_file: str | None) -> list[dict[str, str]]:
    """Load chat history from JSON file."""
    if not history_file:
        return []

    try:
        with open(history_file, "r") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise ValueError("History file must contain a list of messages")
        return messages
    except Exception as e:
        click.echo(
            click.style(f"✗ Error loading history file: {e}", fg="yellow"),
            err=True,
        )
        click.echo("Starting with empty history...", err=True)
        return []


def save_history_file(history_file: str | None, messages: list[dict[str, str]]) -> None:
    """Save chat history to JSON file."""
    if not history_file:
        return

    try:
        with open(history_file, "w") as f:
            json.dump(messages, f, indent=2)
        click.echo(
            click.style(f"\n✓ History saved to {history_file}", fg="green", dim=True)
        )
    except Exception as e:
        click.echo(
            click.style(f"\n⚠ Could not save history: {e}", fg="yellow", dim=True),
            err=True,
        )
