"""Agent commands for text generation and chat."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from docs2synth.cli.utils import (
    build_gen_kwargs,
    load_history_file,
    resolve_config_path,
    save_history_file,
)
from docs2synth.utils import get_logger

logger = get_logger(__name__)


@click.group("agent")
@click.pass_context
def agent_group(ctx: click.Context) -> None:
    """LLM agent commands for text generation and chat."""
    pass


@agent_group.command("generate")
@click.argument("prompt", type=str)
@click.option(
    "--provider",
    type=str,
    default="openai",
    show_default=True,
    help="Provider name (openai, anthropic, gemini, doubao, ollama, huggingface)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (optional, uses provider default if not specified)",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses DOCS2SYNTH_CONFIG env var or ./config.yml if set)",
)
@click.option(
    "--system-prompt",
    type=str,
    default=None,
    help="System prompt for chat models",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum tokens to generate",
)
@click.option(
    "--response-format",
    type=click.Choice(["json", "text"]),
    default=None,
    help="Response format (json for JSON mode)",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to image file (optional, for vision models)",
)
@click.pass_context
def agent_generate(
    ctx: click.Context,
    prompt: str,
    provider: str,
    model: str | None,
    config_path: str | None,
    system_prompt: str | None,
    temperature: float | None,
    max_tokens: int | None,
    response_format: str | None,
    image: str | None,
) -> None:
    """Generate text from a prompt using LLM agents.

    PROMPT: The text prompt to generate from.

    Examples:
        docs2synth agent generate "Explain quantum computing"
        docs2synth agent generate "List 3 items" --provider anthropic --response-format json
        docs2synth agent generate "What's in this image?" --image photo.jpg --provider openai --model gpt-4o
    """
    from PIL import Image as PILImage

    from docs2synth.agent import AgentWrapper

    try:
        # Build kwargs for AgentWrapper
        agent_kwargs: dict[str, Any] = {}
        # Resolve config_path default to ./config.yml if present
        if not config_path and Path("./config.yml").exists():
            config_path = "./config.yml"

        if model:
            agent_kwargs["model"] = model
        if config_path:
            agent_kwargs["config_path"] = config_path

        agent = AgentWrapper(provider=provider, **agent_kwargs)

        # Build generation kwargs
        gen_kwargs: dict[str, Any] = {}
        if system_prompt:
            gen_kwargs["system_prompt"] = system_prompt
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        if response_format:
            gen_kwargs["response_format"] = response_format
        if image:
            image_obj = PILImage.open(image)
            gen_kwargs["image"] = image_obj

        click.echo(click.style(f"Generating with {provider}...", fg="blue"))
        response = agent.generate(prompt, **gen_kwargs)

        click.echo(click.style("\nResponse:", fg="green", bold=True))
        click.echo(response.content)

        if response.usage:
            click.echo(
                click.style(f"\nToken usage: {response.usage}", fg="cyan", dim=True)
            )

    except Exception as e:
        logger.exception("Agent generate command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@agent_group.command("chat")
@click.argument("message", type=str)
@click.option(
    "--provider",
    type=str,
    default="openai",
    show_default=True,
    help="Provider name (openai, anthropic, gemini, doubao, ollama, huggingface)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (optional, uses provider default if not specified)",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses DOCS2SYNTH_CONFIG env var or ./config.yml if set)",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum tokens to generate",
)
@click.option(
    "--response-format",
    type=click.Choice(["json", "text"]),
    default=None,
    help="Response format (json for JSON mode)",
)
@click.option(
    "--history-file",
    type=click.Path(),
    default=None,
    help="Path to JSON file with chat history (optional)",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to image file (optional, for vision models)",
)
@click.pass_context
def agent_chat(
    ctx: click.Context,
    message: str,
    provider: str,
    model: str | None,
    config_path: str | None,
    temperature: float | None,
    max_tokens: int | None,
    response_format: str | None,
    history_file: str | None,
    image: str | None,
) -> None:
    """Chat with LLM agents using message history.

    MESSAGE: The user message to send.

    Examples:
        docs2synth agent chat "What is Python?"
        docs2synth agent chat "Explain AI" --provider anthropic --model claude-3-5-sonnet-20241022
        docs2synth agent chat "What's in this image?" --image photo.jpg --provider openai --model gpt-4o
        docs2synth agent chat "Hello" --history-file chat.json
    """
    from PIL import Image as PILImage

    from docs2synth.agent import AgentWrapper

    try:
        config_path = resolve_config_path(config_path)

        agent_kwargs: dict[str, Any] = {}
        if model:
            agent_kwargs["model"] = model
        if config_path:
            agent_kwargs["config_path"] = config_path
        agent = AgentWrapper(provider=provider, **agent_kwargs)

        messages = load_history_file(history_file)
        messages.append({"role": "user", "content": message})

        gen_kwargs = build_gen_kwargs(temperature, max_tokens, response_format)
        if image:
            image_obj = PILImage.open(image)
            gen_kwargs["image"] = image_obj

        click.echo(click.style(f"Chatting with {provider}...", fg="blue"))
        response = agent.chat(messages, **gen_kwargs)

        click.echo(click.style("\nResponse:", fg="green", bold=True))
        click.echo(response.content)

        if response.usage:
            click.echo(
                click.style(f"\nToken usage: {response.usage}", fg="cyan", dim=True)
            )

        messages.append({"role": "assistant", "content": response.content})
        save_history_file(history_file, messages)

    except Exception as e:
        logger.exception("Agent chat command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)
