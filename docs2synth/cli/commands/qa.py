"""QA generation command-line interface commands.

This module provides CLI commands for QA generation using different strategies
(semantic, layout-aware, logical-aware) and batch processing of documents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click

from docs2synth.qa.qa_batch import IMAGE_EXTENSIONS, clean_batch_qa, find_json_for_image
from docs2synth.utils import get_logger
from docs2synth.utils.text import truncate_context

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

    # Truncate context if too long (use first strategy config for model info)
    strategy_config_for_truncation = (
        semantic_strategies[0] if semantic_strategies else None
    )
    truncated_context, was_truncated = truncate_context(
        context,
        max_tokens=(
            strategy_config_for_truncation.max_tokens
            if strategy_config_for_truncation
            else None
        ),
        provider=(
            strategy_config_for_truncation.provider
            if strategy_config_for_truncation
            else None
        ),
        model=(
            strategy_config_for_truncation.model
            if strategy_config_for_truncation
            else None
        ),
    )
    if was_truncated:
        click.echo(
            click.style(
                f"⚠ Warning: Context length ({len(context)} chars) exceeds maximum. "
                f"Truncated to {len(truncated_context)} chars.",
                fg="yellow",
            ),
            err=True,
        )

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

            question = generator.generate(
                context=truncated_context, target=target, **gen_kwargs
            )
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


@qa_group.command("run")
@click.argument("input_path", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses ./config.yml if present)",
)
@click.option(
    "--strategy",
    type=str,
    default=None,
    help="Only run the specified strategy (defaults to all configured strategies)",
)
@click.pass_context
def qa_run(  # noqa: C901
    ctx: click.Context,
    input_path: Path,
    config_path: str | None,
    strategy: str | None,
) -> None:
    """Run QA generation for a single document (image or JSON).

    INPUT_PATH can be either a preprocessed JSON file with OCR results or the
    corresponding document image. The command will locate the paired file,
    generate QA pairs using configured strategies, and write the results back to
    the JSON file.

    Examples:
        docs2synth qa run data/processed/dev/document_docling.json
        docs2synth qa run data/datasets/docs2synth-dev/docs2synth-dev/images/document.png
        docs2synth qa run data/.../document.png --strategy semantic
    """
    from docs2synth.qa import QAGenerationConfig
    from docs2synth.qa.qa_batch import find_json_for_image, process_document
    from docs2synth.qa.verify_batch import find_image_for_json
    from docs2synth.utils.config import Config

    try:
        # Resolve configuration
        resolved_config_path = None
        if config_path:
            cfg = Config.from_yaml(config_path)
            resolved_config_path = config_path
        else:
            cfg = ctx.obj.get("config")
            if cfg is None:
                default_cfg = Path("./config.yml")
                if default_cfg.exists():
                    cfg = Config.from_yaml(default_cfg)
                    resolved_config_path = str(default_cfg)
                else:
                    click.echo(
                        click.style(
                            "✗ Error: config.yml not found. Please specify --config-path or create config.yml",
                            fg="red",
                        ),
                        err=True,
                    )
                    sys.exit(1)
            else:
                default_cfg = Path("./config.yml")
                if default_cfg.exists():
                    resolved_config_path = str(default_cfg)

        qa_config = QAGenerationConfig.from_config(cfg)
        if qa_config is None or not qa_config.strategies:
            click.echo(
                click.style(
                    "✗ Error: No QA strategies found in config.yml. Please configure 'qa' section.",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        strategies_to_run = _filter_strategies(qa_config, strategy)
        filtered_qa_config = QAGenerationConfig(strategies=strategies_to_run)

        # Resolve paired image/JSON paths
        input_path = Path(input_path)
        processor_name = cfg.get("preprocess.processor", "docling")
        preprocess_input_dir = cfg.get("preprocess.input_dir")
        preprocess_output_dir = cfg.get("preprocess.output_dir")
        data_processed_dir = cfg.get("data.processed_dir")

        json_path: Optional[Path] = None
        image_path: Optional[Path] = None

        if input_path.suffix.lower() == ".json":
            json_path = input_path
            image_dirs = [json_path.parent]
            if preprocess_input_dir:
                image_dirs.append(Path(preprocess_input_dir))
            image_path = find_image_for_json(json_path, image_dirs)
            if not image_path:
                click.echo(
                    click.style(
                        f"✗ Error: Image not found for {json_path.name}",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
        else:
            image_path = input_path
            candidate_dirs = []
            if preprocess_output_dir:
                candidate_dirs.append(Path(preprocess_output_dir))
            if data_processed_dir:
                candidate_dirs.append(Path(data_processed_dir))
            candidate_dirs.append(image_path.parent)

            json_path = None
            for candidate_dir in candidate_dirs:
                candidate = find_json_for_image(
                    image_path, candidate_dir, processor_name
                )
                if candidate:
                    json_path = candidate
                    break

            if json_path is None:
                click.echo(
                    click.style(
                        f"✗ Error: JSON not found for {image_path.name}",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)

        if json_path is None or image_path is None:
            click.echo(
                click.style(
                    "✗ Error: Unable to resolve both JSON and image paths for input",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        click.echo(
            click.style(
                f"Generating QA pairs using {len(strategies_to_run)} strategy(ies) from config...",
                fg="blue",
                bold=True,
            )
        )

        # Run generation
        num_objects, num_questions = process_document(
            image_path=image_path,
            json_path=json_path,
            qa_config=filtered_qa_config,
            config=cfg,
            config_path=resolved_config_path,
        )

        click.echo(
            click.style(
                f"Done! Processed {json_path.name}: {num_objects} objects, {num_questions} questions",
                fg="green",
                bold=True,
            )
        )

    except Exception as e:
        logger.exception("QA run command failed")
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
            config=cfg,
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


@qa_group.command("clean")
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
def qa_clean(  # noqa: C901
    ctx: click.Context,
    input_path: Path | None,
    output_dir: Path | None,
    processor: str | None,
) -> None:
    """Remove QA pairs from JSON outputs.

    INPUT_PATH can be a JSON file, an image file, or a directory. When omitted,
    all JSON files in config.preprocess.output_dir are cleaned.

    Examples:
        docs2synth qa clean data/processed/dev/document_docling.json
        docs2synth qa clean data/images/document.png
        docs2synth qa clean data/processed/dev
        docs2synth qa clean
    """
    cfg = ctx.obj.get("config")
    if cfg is None:
        click.echo(
            click.style(
                "✗ Error: Configuration not loaded. Run command via docs2synth CLI.",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    if processor is None:
        processor = cfg.get("preprocess.processor", "docling")

    # Determine output directory when needed
    default_output_dir = cfg.get("preprocess.output_dir") or cfg.get(
        "data.processed_dir"
    )
    if output_dir is None and default_output_dir is not None:
        output_dir = Path(default_output_dir)
    elif output_dir is not None:
        output_dir = Path(output_dir)

    json_files: set[Path] = set()

    def collect_from_image(image_path: Path) -> None:
        if output_dir is None:
            click.echo(
                click.style(
                    "✗ Error: --output-dir is required when cleaning from image paths",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)
        json_path = find_json_for_image(image_path, output_dir, processor)
        if json_path:
            json_files.add(json_path)
        else:
            click.echo(
                click.style(
                    f"⚠ Warning: JSON not found for image {image_path.name}",
                    fg="yellow",
                ),
                err=True,
            )

    def collect_from_directory(directory: Path) -> None:
        found_json = False
        for json_path in directory.glob("*.json"):
            json_files.add(json_path)
            found_json = True
        if found_json:
            return
        for ext in IMAGE_EXTENSIONS:
            for image_path in directory.glob(f"*{ext}"):
                collect_from_image(image_path)

    if input_path is None:
        if output_dir is None or not output_dir.exists():
            click.echo(
                click.style(
                    "✗ Error: No OUTPUT_DIR configured. Provide --output-dir or set config.preprocess.output_dir",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)
        collect_from_directory(output_dir)
    else:
        input_path = Path(input_path)
        if input_path.is_file():
            if input_path.suffix.lower() == ".json":
                json_files.add(input_path)
            elif input_path.suffix.lower() in IMAGE_EXTENSIONS:
                collect_from_image(input_path)
            else:
                click.echo(
                    click.style(
                        f"✗ Error: Unsupported file type: {input_path}",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
        elif input_path.is_dir():
            collect_from_directory(input_path)
        else:
            click.echo(
                click.style(
                    f"✗ Error: Input path does not exist: {input_path}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

    if not json_files:
        click.echo(click.style("⚠ No JSON files found to clean", fg="yellow"), err=True)
        return

    files_processed, objects_modified, qa_removed = clean_batch_qa(sorted(json_files))

    click.echo(
        click.style(
            f"Cleaned {files_processed} file(s): removed {qa_removed} QA pairs from {objects_modified} objects",
            fg="green",
            bold=True,
        )
    )
