"""Verification command-line interface commands.

This module provides CLI commands for verifying QA pairs using different
verification strategies (meaningful, correctness, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
from PIL import Image as PILImage

from docs2synth.utils import get_logger

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
@click.argument("question", type=str)
@click.option(
    "--answer",
    type=str,
    default=None,
    help="Target answer (required for correctness verification)",
)
@click.option(
    "--context",
    type=str,
    default=None,
    help="Document context (required for meaningful verification)",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to document image (optional)",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses ./config.yml if present)",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature for verifier (overrides config)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum tokens for verifier (overrides config)",
)
@click.pass_context
def verify_run(
    ctx: click.Context,
    question: str,
    answer: str | None,
    context: str | None,
    image: str | None,
    config_path: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> None:
    """Verify a question using configured verifiers.

    QUESTION: The question to verify

    Examples:
        docs2synth verify run "What is the total?" --answer "42" --context "Invoice total: 42"
        docs2synth verify run "What is shown?" --context "Document text" --image doc.png
    """
    try:
        # Load verification config
        verification_config, _ = _load_verification_config(config_path)

        # Load image if provided
        image_obj = None
        if image:
            image_obj = PILImage.open(image)
            click.echo(click.style(f"Loaded image: {image}", fg="blue"))

        # Display what we're verifying
        click.echo(click.style("\nVerifying Question:", fg="cyan", bold=True))
        click.echo(f"Question: {question}")
        if answer:
            click.echo(f"Answer: {answer}")
        if context:
            click.echo(f"Context: {context[:100]}...")
        if image:
            click.echo(f"Image: {image}")

        # Run all configured verifiers
        results = {}
        for verifier_config in verification_config.verifiers:
            result = _run_single_verifier(
                verifier_config,
                question,
                answer,
                context,
                image_obj,
                temperature,
                max_tokens,
            )
            if result:
                verifier_type = verifier_config.strategy
                results[verifier_type] = result
                _display_verification_result(result)

        # Summary
        if results:
            click.echo(click.style("\n" + "=" * 50, fg="cyan"))
            click.echo(click.style("Verification Summary:", fg="cyan", bold=True))
            for verifier_type, result in results.items():
                response = result.get("response", result.get("Response", "Unknown"))
                if response.lower() == "yes":
                    symbol, color = "✓", "green"
                elif response.lower() == "no":
                    symbol, color = "✗", "red"
                else:
                    symbol, color = "?", "yellow"
                click.echo(
                    click.style(
                        f"  {symbol} {verifier_type}: {response}", fg=color, bold=True
                    )
                )
        else:
            click.echo(
                click.style(
                    "\n⚠ No verifiers were run (missing required inputs)",
                    fg="yellow",
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
