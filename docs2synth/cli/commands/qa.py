"""QA generation command-line interface commands.

This module provides CLI commands for QA generation using different strategies
(semantic, layout-aware, logical-aware) and batch processing of documents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
from PIL import Image as PILImage

from docs2synth.utils import get_logger

logger = get_logger(__name__)


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


@click.group("qa")
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
    help="Provider name (openai, anthropic, gemini, doubao, ollama, huggingface, vllm)",
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
    help="Provider name (openai, anthropic, gemini, doubao, ollama, huggingface, vllm)",
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
    help="Provider name (openai, anthropic, gemini, doubao, ollama, huggingface, vllm)",
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


@qa_group.command("batch")
@click.argument("input_path", type=click.Path(path_type=Path), required=False)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing preprocessed JSON files (defaults to config.preprocess.output_dir)",
)
@click.option(
    "--processor",
    type=str,
    default=None,
    help="Processor name used for JSON files (defaults to config.preprocess.processor)",
)
@click.pass_context
def qa_batch(
    ctx: click.Context,
    input_path: Path | None,
    output_dir: Path | None,
    processor: str | None,
) -> None:
    """Generate QA pairs for images using config.yml strategies.

    INPUT_PATH can be a single image file or a directory of images.
    If not provided, uses config.preprocess.input_dir.

    This command:
    1. Reads image files from INPUT_PATH (single file or directory)
    2. Finds corresponding preprocessed JSON files in output_dir
    3. Generates QA pairs for each text object using ALL strategies from config.yml
    4. Saves results back to JSON files with "qa" field added to each object

    Examples:
        # Process all images from config.preprocess.input_dir
        docs2synth qa batch

        # Process specific image directory
        docs2synth qa batch data/images

        # Process a single image
        docs2synth qa batch data/images/document.png
    """
    from docs2synth.qa.config import QAGenerationConfig
    from docs2synth.qa.qa_batch import process_batch

    cfg = ctx.obj.get("config")

    # Get input path: CLI argument > config.preprocess.input_dir
    if input_path is None:
        input_dir = cfg.get("preprocess.input_dir")
        if input_dir is None:
            click.echo(
                click.style(
                    "✗ Error: INPUT_PATH argument is required, or set config.preprocess.input_dir",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)
        input_path = Path(input_dir)

    # Validate path exists
    if not input_path.exists():
        click.echo(
            click.style(
                f"✗ Error: Input path does not exist: {input_path}",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    # Get output_dir: CLI option > config.preprocess.output_dir > config.data.processed_dir
    if output_dir is None:
        output_dir_str = cfg.get("preprocess.output_dir") or cfg.get(
            "data.processed_dir"
        )
        if output_dir_str is None:
            click.echo(
                click.style(
                    "✗ Error: --output-dir is required, or set config.preprocess.output_dir",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)
        output_dir = Path(output_dir_str)

    # Validate output_dir exists
    if not output_dir.exists():
        click.echo(
            click.style(
                f"✗ Error: Output directory does not exist: {output_dir}",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    # Get processor name
    if processor is None:
        processor = cfg.get("preprocess.processor", "paddleocr")

    # Load QA configuration from config.yml
    try:
        qa_config = QAGenerationConfig.from_config(cfg)
        strategies = qa_config.list_strategies()

        if not strategies:
            click.echo(
                click.style(
                    "✗ Error: No QA strategies configured in config.yml",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        click.echo(click.style(f"Input: {input_path}", fg="blue"))
        click.echo(click.style(f"Output dir: {output_dir}", fg="blue"))
        click.echo(click.style(f"Processor: {processor}", fg="blue"))
        click.echo(
            click.style(
                f"Strategies from config.yml: {', '.join(strategies)}", fg="blue"
            )
        )

        num_files, num_objects, num_questions = process_batch(
            input_path=input_path,
            output_dir=output_dir,
            qa_config=qa_config,
            processor_name=processor,
        )

        click.echo(
            click.style(
                f"\nDone! Processed {num_files} files, {num_objects} objects, generated {num_questions} questions",
                fg="green",
                bold=True,
            )
        )
    except Exception as e:
        logger.exception("Batch QA command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)
