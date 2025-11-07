"""Command-line interface for Docs2Synth.

This module provides CLI commands for document processing, QA generation,
and retriever training using Click framework with proper error handling
and logging integration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from docs2synth.utils import get_logger, load_config, setup_cli_logging

logger = get_logger(__name__)


def _resolve_config_path(config_path: str | None) -> str | None:
    """Resolve default config path to ./config.yml when available."""
    if not config_path and Path("./config.yml").exists():
        return "./config.yml"
    return config_path


def _build_gen_kwargs(
    temperature: float | None,
    max_tokens: int | None,
    response_format: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if response_format:
        params["response_format"] = response_format
    return params


def _load_history_file(history_file: str | None) -> list[dict[str, str]]:
    if not history_file:
        return []
    import json

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


def _save_history_file(
    history_file: str | None, messages: list[dict[str, str]]
) -> None:
    if not history_file:
        return
    import json

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


@cli.group("agent")
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
        config_path = _resolve_config_path(config_path)

        agent_kwargs: dict[str, Any] = {}
        if model:
            agent_kwargs["model"] = model
        if config_path:
            agent_kwargs["config_path"] = config_path
        agent = AgentWrapper(provider=provider, **agent_kwargs)

        messages = _load_history_file(history_file)
        messages.append({"role": "user", "content": message})

        gen_kwargs = _build_gen_kwargs(temperature, max_tokens, response_format)
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
        _save_history_file(history_file, messages)

    except Exception as e:
        logger.exception("Agent chat command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


# QA command group
@cli.group("qa")
@click.pass_context
def qa_group(ctx: click.Context) -> None:
    """QA generation commands for different strategies."""
    pass


@qa_group.command("semantic")
@click.argument("context", type=str)
@click.argument("target", type=str)
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
    default=0.7,
    show_default=True,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum tokens to generate",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to document image (optional, for vision models)",
)
@click.pass_context
def qa_semantic(
    ctx: click.Context,
    context: str,
    target: str,
    provider: str,
    model: str | None,
    config_path: str | None,
    temperature: float,
    max_tokens: int | None,
    image: str | None,
) -> None:
    """Generate semantic question from context and target.

    CONTEXT: Document context (e.g., OCR text from document)
    TARGET: Target answer or object to generate question for

    Examples:
        docs2synth qa semantic "Form contains name, address fields" "John Doe"
        docs2synth qa semantic "Context here" "Target" --provider anthropic --image doc.png
    """
    from PIL import Image as PILImage

    from docs2synth.qa import QAGeneratorFactory

    try:
        # Resolve config_path
        if not config_path and Path("./config.yml").exists():
            config_path = "./config.yml"

        gen_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens

        # Load image if provided
        if image:
            img = PILImage.open(image)
            gen_kwargs["image"] = img

        click.echo(
            click.style(f"Generating semantic question with {provider}...", fg="blue")
        )

        generator = QAGeneratorFactory.create(
            strategy="semantic",
            provider=provider,
            model=model,
            config_path=config_path,
        )

        question = generator.generate(context=context, target=target, **gen_kwargs)

        click.echo(click.style("\nGenerated Question:", fg="green", bold=True))
        click.echo(question)

    except Exception as e:
        logger.exception("QA semantic generation failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@qa_group.command("layout")
@click.argument("question", type=str)
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
    default=0.7,
    show_default=True,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum tokens to generate",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to document image (recommended for understanding spatial layout)",
)
@click.pass_context
def qa_layout(
    ctx: click.Context,
    question: str,
    provider: str,
    model: str | None,
    config_path: str | None,
    temperature: float,
    max_tokens: int | None,
    image: str | None,
) -> None:
    """Transform question to layout-aware (spatial position) question.

    QUESTION: Original question to transform

    Examples:
        docs2synth qa layout "What is the name?"
        docs2synth qa layout "What is the address?" --provider anthropic --image doc.png
    """
    from PIL import Image as PILImage

    from docs2synth.qa import QAGeneratorFactory

    try:
        # Resolve config_path
        if not config_path and Path("./config.yml").exists():
            config_path = "./config.yml"

        gen_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens

        # Load image if provided
        if image:
            img = PILImage.open(image)
            gen_kwargs["image"] = img

        click.echo(
            click.style(
                f"Generating layout-aware question with {provider}...", fg="blue"
            )
        )

        generator = QAGeneratorFactory.create(
            strategy="layout_aware",
            provider=provider,
            model=model,
            config_path=config_path,
        )

        layout_question = generator.generate(question=question, **gen_kwargs)

        click.echo(click.style("\nOriginal Question:", fg="yellow", bold=True))
        click.echo(question)
        click.echo(click.style("\nLayout-Aware Question:", fg="green", bold=True))
        click.echo(layout_question)

    except Exception as e:
        logger.exception("QA layout generation failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@qa_group.command("logical")
@click.argument("question", type=str)
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
    default=0.7,
    show_default=True,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum tokens to generate",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to document image (optional, can help understand document structure)",
)
@click.pass_context
def qa_logical(
    ctx: click.Context,
    question: str,
    provider: str,
    model: str | None,
    config_path: str | None,
    temperature: float,
    max_tokens: int | None,
    image: str | None,
) -> None:
    """Transform question to logical-aware (document sections) question.

    QUESTION: Original question to transform

    Examples:
        docs2synth qa logical "What is the address?"
        docs2synth qa logical "What is the name?" --provider anthropic --image doc.png
    """
    from PIL import Image as PILImage

    from docs2synth.qa import QAGeneratorFactory

    try:
        # Resolve config_path
        if not config_path and Path("./config.yml").exists():
            config_path = "./config.yml"

        gen_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens

        # Load image if provided
        if image:
            img = PILImage.open(image)
            gen_kwargs["image"] = img

        click.echo(
            click.style(
                f"Generating logical-aware question with {provider}...", fg="blue"
            )
        )

        generator = QAGeneratorFactory.create(
            strategy="logical_aware",
            provider=provider,
            model=model,
            config_path=config_path,
        )

        logical_question = generator.generate(question=question, **gen_kwargs)

        click.echo(click.style("\nOriginal Question:", fg="yellow", bold=True))
        click.echo(question)
        click.echo(click.style("\nLogical-Aware Question:", fg="green", bold=True))
        click.echo(logical_question)

    except Exception as e:
        logger.exception("QA logical generation failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


def _load_qa_config(config_path: str | None) -> tuple[Any, list[Any]]:
    """Load and validate QA configuration from config file."""
    from docs2synth.qa import QAGenerationConfig
    from docs2synth.utils.config import Config

    if not config_path and Path("./config.yml").exists():
        config_path = "./config.yml"

    if not config_path:
        click.echo(
            click.style(
                "✗ Error: config.yml not found. Please specify --config-path or create config.yml",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    config = Config.from_yaml(config_path)
    qa_config = QAGenerationConfig.from_config(config)

    if qa_config is None or not qa_config.strategies:
        click.echo(
            click.style(
                "✗ Error: No QA strategies found in config.yml. Please configure 'qa' section.",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    return qa_config, config_path


def _filter_strategies(qa_config: Any, strategy: str | None) -> list[Any]:
    """Filter strategies by name if specified."""
    if strategy:
        strategy_config = qa_config.get_strategy_config(strategy)
        if strategy_config is None:
            available = ", ".join(qa_config.list_strategies())
            click.echo(
                click.style(
                    f"✗ Error: Strategy '{strategy}' not found in config. Available: {available}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)
        return [strategy_config]
    return qa_config.strategies


def _build_qa_gen_kwargs(strategy_config: Any, image_obj: Any | None) -> dict[str, Any]:
    """Build generation kwargs from strategy config and image."""
    gen_kwargs: dict[str, Any] = {}
    if image_obj:
        gen_kwargs["image"] = image_obj
    if strategy_config.temperature is not None:
        gen_kwargs["temperature"] = strategy_config.temperature
    if strategy_config.max_tokens is not None:
        gen_kwargs["max_tokens"] = strategy_config.max_tokens
    return gen_kwargs


def _run_semantic_strategies(
    semantic_strategies: list[Any],
    context: str,
    target: str,
    image_obj: Any | None,
    config_path: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Run semantic QA generation strategies."""
    from docs2synth.qa import QAGeneratorFactory

    results = []
    semantic_question = None

    for strategy_config in semantic_strategies:
        try:
            model_display = (
                strategy_config.model if strategy_config.model else "default"
            )
            click.echo(
                click.style(
                    f"\n[{strategy_config.strategy}] Using {strategy_config.provider}/{model_display}...",
                    fg="cyan",
                )
            )

            generator = QAGeneratorFactory.create_from_config(
                strategy_config, config_path=config_path
            )
            gen_kwargs = _build_qa_gen_kwargs(strategy_config, image_obj)

            question = generator.generate(context=context, target=target, **gen_kwargs)
            semantic_question = question
            results.append(
                {
                    "strategy": strategy_config.strategy,
                    "provider": strategy_config.provider,
                    "model": strategy_config.model,
                    "question": question,
                }
            )
            click.echo(click.style(f"  Question: {question}", fg="green"))

        except Exception as e:
            logger.exception(
                f"Failed to generate with strategy '{strategy_config.strategy}'"
            )
            click.echo(
                click.style(
                    f"  ✗ Error with {strategy_config.strategy}: {e}", fg="red"
                ),
                err=True,
            )
            continue

    return results, semantic_question


def _run_transform_strategies(
    transform_strategies: list[Any],
    semantic_question: str | None,
    image_obj: Any | None,
    config_path: str | None,
) -> list[dict[str, Any]]:
    """Run layout-aware and logical-aware QA generation strategies."""
    from docs2synth.qa import QAGeneratorFactory

    results = []

    for strategy_config in transform_strategies:
        try:
            if semantic_question is None:
                click.echo(
                    click.style(
                        f"\n[{strategy_config.strategy}] ⚠ Skipping (no semantic question available)",
                        fg="yellow",
                    )
                )
                continue

            model_display = (
                strategy_config.model if strategy_config.model else "default"
            )
            click.echo(
                click.style(
                    f"\n[{strategy_config.strategy}] Using {strategy_config.provider}/{model_display}...",
                    fg="cyan",
                )
            )

            generator = QAGeneratorFactory.create_from_config(
                strategy_config, config_path=config_path
            )
            gen_kwargs = _build_qa_gen_kwargs(strategy_config, image_obj)

            transformed_question = generator.generate(
                question=semantic_question, **gen_kwargs
            )
            results.append(
                {
                    "strategy": strategy_config.strategy,
                    "provider": strategy_config.provider,
                    "model": strategy_config.model,
                    "question": transformed_question,
                    "original_question": semantic_question,
                }
            )
            click.echo(click.style(f"  Question: {transformed_question}", fg="green"))

        except Exception as e:
            logger.exception(
                f"Failed to generate with strategy '{strategy_config.strategy}'"
            )
            click.echo(
                click.style(
                    f"  ✗ Error with {strategy_config.strategy}: {e}", fg="red"
                ),
                err=True,
            )
            continue

    return results


def _print_qa_summary(results: list[dict[str, Any]]) -> None:
    """Print summary of generated QA questions."""
    if results:
        click.echo(click.style("\n" + "=" * 60, fg="blue", bold=True))
        click.echo(click.style("Summary:", fg="blue", bold=True))
        for result in results:
            model_display = result.get("model") if result.get("model") else "default"
            click.echo(
                click.style(
                    f"\n[{result['strategy']}] {result['provider']}/{model_display}",
                    fg="cyan",
                )
            )
            if "original_question" in result:
                click.echo(
                    click.style(
                        f"  Original: {result['original_question']}", fg="yellow"
                    )
                )
            click.echo(click.style(f"  Question: {result['question']}", fg="green"))
    else:
        click.echo(click.style("\n⚠ No questions generated", fg="yellow"), err=True)


@qa_group.command("generate")
@click.argument("context", type=str)
@click.argument("target", type=str)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses DOCS2SYNTH_CONFIG env var or ./config.yml if set)",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to document image (optional, for vision models)",
)
@click.option(
    "--strategy",
    type=str,
    default=None,
    help="Specific strategy to use (optional, uses all configured strategies if not specified)",
)
@click.pass_context
def qa_generate(
    ctx: click.Context,
    context: str,
    target: str,
    config_path: str | None,
    image: str | None,
    strategy: str | None,
) -> None:
    """Generate QA questions using strategies configured in config.yml.

    CONTEXT: Document context (e.g., OCR text from document)
    TARGET: Target answer or object to generate question for

    This command reads QA generation strategies from config.yml and generates
    questions using all configured strategies (or a specific one if --strategy is provided).

    Examples:
        docs2synth qa generate "Form contains name, address fields" "John Doe"
        docs2synth qa generate "Context here" "Target" --image doc.png
        docs2synth qa generate "Context" "Target" --strategy semantic
    """
    from PIL import Image as PILImage

    try:
        # Load configuration
        qa_config, config_path = _load_qa_config(config_path)
        strategies_to_run = _filter_strategies(qa_config, strategy)

        # Load image if provided
        image_obj = None
        if image:
            image_obj = PILImage.open(image)

        click.echo(
            click.style(
                f"Generating QA questions using {len(strategies_to_run)} strategy(ies) from config...",
                fg="blue",
                bold=True,
            )
        )

        # First pass: Generate semantic questions
        semantic_strategies = [s for s in strategies_to_run if s.strategy == "semantic"]
        results, semantic_question = _run_semantic_strategies(
            semantic_strategies, context, target, image_obj, config_path
        )

        # Second pass: Transform questions (layout_aware, logical_aware)
        transform_strategies = [
            s
            for s in strategies_to_run
            if s.strategy in ["layout_aware", "logical_aware"]
        ]
        transform_results = _run_transform_strategies(
            transform_strategies, semantic_question, image_obj, config_path
        )
        results.extend(transform_results)

        # Print summary
        _print_qa_summary(results)

    except Exception as e:
        logger.exception("QA generate command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@qa_group.command("list")
@click.pass_context
def qa_list(ctx: click.Context) -> None:
    """List all available QA generation strategies."""
    from docs2synth.qa import QAGeneratorFactory

    try:
        strategies = QAGeneratorFactory.list_strategies()
        click.echo(
            click.style("Available QA generation strategies:", fg="blue", bold=True)
        )
        for strategy in strategies:
            click.echo(f"  - {strategy}")
    except Exception as e:
        logger.exception("Failed to list QA strategies")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """Entry point for the console script."""
    cli(args=argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
