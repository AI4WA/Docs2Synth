"""Verification command-line interface commands.

This module provides CLI commands for verifying QA pairs using different
verification strategies (meaningful, correctness, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from docs2synth.qa import QAVerificationConfig
from docs2synth.qa.qa_batch import IMAGE_EXTENSIONS, find_json_for_image
from docs2synth.qa.verify_batch import (
    clean_batch_verification,
    find_image_for_json,
    process_document_verification,
)
from docs2synth.utils import get_logger
from docs2synth.utils.config import Config

logger = get_logger(__name__)


def _prepare_verifier_kwargs(
    verifier_type: str,
    question: str,
    answer: str | None,
    context: str | None,
    image_obj: Any,
    temperature: float | None,
    max_tokens: int | None,
) -> tuple[dict[str, Any], bool]:
    """Prepare verification kwargs for a specific verifier type.

    Args:
        verifier_type: Type of verifier (meaningful, correctness, etc.)
        question: Generated question
        answer: Target answer
        context: Document context
        image_obj: Document image object
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        Tuple of (verify_kwargs dict, should_skip bool)
    """
    verify_kwargs: dict[str, Any] = {}
    if temperature is not None:
        verify_kwargs["temperature"] = temperature
    if max_tokens is not None:
        verify_kwargs["max_tokens"] = max_tokens
    verify_kwargs["image"] = image_obj

    if verifier_type == "meaningful":
        if context is None and image_obj is None:
            return {}, True
        if context is not None:
            verify_kwargs["context"] = context
    elif verifier_type == "correctness":
        if answer is None:
            return {}, True

    return verify_kwargs, False


def _load_verification_config(config_path: str | None) -> tuple[Any, str]:
    """Load verification configuration from config file.

    Args:
        config_path: Path to config file (or None to auto-detect)

    Returns:
        Tuple of (verification_config, resolved_config_path)

    Raises:
        SystemExit: If config file not found or invalid
    """
    from docs2synth.qa import QAVerificationConfig
    from docs2synth.utils.config import Config

    # Resolve config_path
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

    # Load verification config
    config = Config.from_yaml(config_path)
    verification_config = QAVerificationConfig.from_config(config)

    if verification_config is None or not verification_config.verifiers:
        click.echo(
            click.style(
                "✗ Error: No verifiers found in config.yml. Please configure 'verifiers' section.",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    return verification_config, config_path


def _display_verification_result(result: dict[str, Any]) -> None:
    """Display verification result to console.

    Args:
        result: Result dictionary from verifier
    """
    response = result.get("response", result.get("Response", "Unknown"))
    explanation = result.get("explanation", result.get("Explanation", ""))

    if response.lower() == "yes":
        color = "green"
        symbol = "✓"
    elif response.lower() == "no":
        color = "red"
        symbol = "✗"
    else:
        color = "yellow"
        symbol = "?"

    click.echo(click.style(f"\n{symbol} Response: {response}", fg=color, bold=True))
    if explanation:
        click.echo(click.style(f"Explanation: {explanation}", fg=color))


def _run_single_verifier(
    verifier_config: Any,
    question: str,
    answer: str | None,
    context: str | None,
    image_obj: Any,
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any] | None:
    """Run a single verifier and return results.

    Args:
        verifier_config: Verifier configuration
        question: Question to verify
        answer: Target answer (optional, required for correctness)
        context: Document context (optional, required for meaningful)
        image_obj: Document image object (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Verification result dictionary or None if skipped
    """
    from docs2synth.qa.verifiers import create_verifier

    verifier_type = verifier_config.strategy
    click.echo(
        click.style(f"\nRunning {verifier_type} verifier...", fg="blue", bold=True)
    )

    # Prepare kwargs for this verifier
    verify_kwargs, should_skip = _prepare_verifier_kwargs(
        verifier_type, question, answer, context, image_obj, temperature, max_tokens
    )

    if should_skip:
        click.echo(
            click.style(
                f"⊘ Skipping {verifier_type} verifier (missing required inputs)",
                fg="yellow",
            )
        )
        return None

    # Create verifier and run verification
    try:
        verifier = create_verifier(verifier_config)
        result = verifier.verify(question=question, answer=answer, **verify_kwargs)
        return result
    except Exception as e:
        logger.exception(f"{verifier_type} verification failed")
        click.echo(
            click.style(f"✗ Error in {verifier_type} verifier: {e}", fg="red"),
            err=True,
        )
        return None


@click.group("verify")
@click.pass_context
def verify_group(ctx: click.Context) -> None:
    """Verification commands for QA pairs and document content."""
    pass


@verify_group.command("run")
@click.argument("input_path", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses ./config.yml if present)",
)
@click.option(
    "--verifier-type",
    type=str,
    default=None,
    help="Only run the specified verifier type (defaults to all configured verifiers)",
)
@click.pass_context
def verify_run(  # noqa: C901
    ctx: click.Context,
    input_path: Path,
    config_path: str | None,
    verifier_type: str | None,
) -> None:
    """Verify QA pairs in a single JSON file (minimal API).

    INPUT_PATH: Path to a JSON file with QA pairs or the corresponding document image

    The command will:
      - Load verifiers from config.yml
      - Resolve the matching JSON/Image pair
      - Verify all QA pairs in the JSON and write results back

    Example:
        docs2synth verify run data/processed/dev/document_docling.json
        docs2synth verify run data/images/document.png
    """
    try:
        # Resolve config
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

        verification_config = QAVerificationConfig.from_yaml(str(config_path))
        if verification_config is None:
            click.echo(
                click.style(
                    "✗ Error: No verifiers configured in config.yml (qa.verifiers).",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        if verifier_type:
            verifier_config = verification_config.get_verifier_config(verifier_type)
            if verifier_config is None:
                available = ", ".join(verification_config.list_verifiers())
                click.echo(
                    click.style(
                        f"✗ Error: Verifier type '{verifier_type}' not found in config. Available: {available}",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
            verification_config = QAVerificationConfig(verifiers=[verifier_config])

        # Load config for locating paired files
        image_dirs: list[Path] = []
        json_dirs: list[Path] = []
        processor_name = "docling"
        try:
            full_cfg = Config.from_yaml(str(config_path))
            preprocess_input_dir = full_cfg.get("preprocess.input_dir")
            preprocess_output_dir = full_cfg.get("preprocess.output_dir")
            data_processed_dir = full_cfg.get("data.processed_dir")
            processor_name = full_cfg.get("preprocess.processor", processor_name)
            if preprocess_input_dir:
                image_dirs.append(Path(preprocess_input_dir))
            if preprocess_output_dir:
                json_dirs.append(Path(preprocess_output_dir))
            if data_processed_dir:
                json_dirs.append(Path(data_processed_dir))
        except Exception:
            pass

        # Resolve JSON and image paths based on input
        if input_path.suffix.lower() == ".json":
            json_path = input_path
            image_dirs.insert(0, json_path.parent)
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
            json_dirs.insert(0, image_path.parent)
            json_path = None
            for candidate_dir in json_dirs:
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

        # Execute verification
        num_objects, num_verified, num_passed = process_document_verification(
            json_path=json_path,
            image_path=image_path,
            verification_config=verification_config,
            config_path=str(config_path),
        )

        # Report
        click.echo(
            click.style(
                f"Done! Processed {num_objects} objects, verified {num_verified} QA pairs, "
                f"{num_passed} passed all verifiers",
                fg="green",
                bold=True,
            )
        )

    except Exception as e:
        logger.exception("Verification command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@verify_group.command("list")
@click.pass_context
def verify_list(ctx: click.Context) -> None:
    """List available verification strategies.

    Examples:
        docs2synth verify list
    """
    from docs2synth.qa.verifiers import VERIFIER_REGISTRY

    click.echo(click.style("Available verification strategies:", fg="green", bold=True))
    for strategy_name in VERIFIER_REGISTRY.keys():
        click.echo(f"  - {strategy_name}")


@verify_group.command("batch")
@click.argument("input_path", type=click.Path(path_type=Path), required=False)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses ./config.yml if present)",
)
@click.option(
    "--verifier-type",
    type=str,
    default=None,
    help="Only run specific verifier type (meaningful, correctness)",
)
@click.option(
    "--image-dir",
    type=click.Path(path_type=Path),
    multiple=True,
    help="Additional directories to search for images (can be specified multiple times, defaults to config.preprocess.input_dir)",
)
@click.pass_context
def verify_batch(
    ctx: click.Context,
    input_path: Path | None,
    config_path: str | None,
    verifier_type: str | None,
    image_dir: tuple[Path, ...],
) -> None:
    """Verify QA pairs in JSON files using configured verifiers.

    INPUT_PATH can be a single JSON file or a directory of JSON files.
    If not provided, uses config.preprocess.output_dir.

    This command:
    1. Reads JSON files containing QA pairs (generated by 'docs2synth qa batch')
    2. Finds corresponding image files
    3. Runs configured verifiers on each QA pair
    4. Adds verification results to the JSON files

    Examples:
        # Verify all JSON files from config.preprocess.output_dir
        docs2synth verify batch

        # Verify all JSON files in a directory
        docs2synth verify batch data/processed/dev

        # Verify a single JSON file
        docs2synth verify batch data/processed/dev/document.json

        # Use specific config file
        docs2synth verify batch --config-path config.yml

        # Only run meaningful verifier
        docs2synth verify batch --verifier-type meaningful

        # Specify additional image search directories
        docs2synth verify batch --image-dir data/images --image-dir data/raw
    """
    from docs2synth.qa.verify_batch import process_batch_verification

    cfg = ctx.obj.get("config")

    try:
        # Get input path: CLI argument > config.preprocess.output_dir
        if input_path is None:
            output_dir = cfg.get("preprocess.output_dir")
            if output_dir is None:
                click.echo(
                    click.style(
                        "✗ Error: INPUT_PATH argument is required, or set config.preprocess.output_dir",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
            input_path = Path(output_dir)

        # Validate input path exists
        if not input_path.exists():
            click.echo(
                click.style(
                    f"✗ Error: Input path does not exist: {input_path}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        # Load verification config
        verification_config, resolved_config_path = _load_verification_config(
            config_path
        )

        # Filter verifiers if verifier_type specified
        if verifier_type:
            verifier_config = verification_config.get_verifier_config(verifier_type)
            if verifier_config is None:
                available = ", ".join(verification_config.list_verifiers())
                click.echo(
                    click.style(
                        f"✗ Error: Verifier type '{verifier_type}' not found in config. "
                        f"Available: {available}",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
            # Create a new config with only the specified verifier
            from docs2synth.qa.config import QAVerificationConfig

            verification_config = QAVerificationConfig(verifiers=[verifier_config])

        # Prepare image search directories
        # Priority: CLI option > config.preprocess.input_dir > same directory as JSON files
        image_dirs = list(image_dir) if image_dir else None
        if image_dirs is None:
            # Try to get from config
            input_dir = cfg.get("preprocess.input_dir")
            if input_dir:
                image_dirs = [Path(input_dir)]
            else:
                # Fallback: search in same directory as JSON files
                if input_path.is_file():
                    image_dirs = [input_path.parent]
                else:
                    image_dirs = [input_path]

        # Display configuration
        click.echo(click.style(f"Input: {input_path}", fg="blue"))
        click.echo(
            click.style(
                f"Verifiers: {', '.join(verification_config.list_verifiers())}",
                fg="blue",
            )
        )
        click.echo(
            click.style(
                f"Image search dirs: {', '.join(str(d) for d in image_dirs)}", fg="blue"
            )
        )

        # Process batch verification
        num_files, num_objects, num_verified, num_passed = process_batch_verification(
            input_path=input_path,
            verification_config=verification_config,
            image_dirs=image_dirs,
            config_path=resolved_config_path,
        )

        # Display results
        if num_verified > 0:
            pass_rate = (num_passed / num_verified) * 100
            click.echo(
                click.style(
                    f"\nDone! Processed {num_files} files, {num_objects} objects, "
                    f"verified {num_verified} QA pairs",
                    fg="green",
                    bold=True,
                )
            )
            click.echo(
                click.style(
                    f"Pass rate: {num_passed}/{num_verified} ({pass_rate:.1f}%) passed all verifiers",
                    fg="cyan",
                )
            )
        else:
            click.echo(
                click.style(
                    "\n⚠ No QA pairs were verified. Make sure JSON files contain QA pairs.",
                    fg="yellow",
                )
            )

    except Exception as e:
        logger.exception("Batch verification command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@verify_group.command("clean")
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
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses ./config.yml if present)",
)
@click.pass_context
def verify_clean(  # noqa: C901
    ctx: click.Context,
    input_path: Path | None,
    output_dir: Path | None,
    processor: str | None,
    config_path: str | None,
) -> None:
    """Remove verification results from JSON outputs.

    INPUT_PATH can be a JSON file, an image file, or a directory. When omitted,
    all JSON files in config.preprocess.output_dir are cleaned.

    Examples:
        docs2synth verify clean data/processed/dev/document_docling.json
        docs2synth verify clean data/images/document.png
        docs2synth verify clean data/processed/dev
        docs2synth verify clean --config-path config.yml
    """
    cfg: Config | None = None
    if config_path:
        cfg = Config.from_yaml(config_path)
    else:
        cfg = ctx.obj.get("config")
        if cfg is None:
            default_cfg = Path("./config.yml")
            if default_cfg.exists():
                cfg = Config.from_yaml(default_cfg)
            else:
                click.echo(
                    click.style(
                        "✗ Error: config.yml not found. Please specify --config-path or run from docs2synth CLI.",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)

    if processor is None:
        processor = cfg.get("preprocess.processor", "docling")

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

    files_processed, pairs_modified, entries_removed = clean_batch_verification(
        sorted(json_files)
    )

    click.echo(
        click.style(
            f"Cleaned {files_processed} file(s): cleared verification results from {pairs_modified} QA pairs (removed {entries_removed} verifier responses)",
            fg="green",
            bold=True,
        )
    )
