"""RAG command group for Docs2Synth CLI."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import click

from docs2synth.rag.pipeline import RAGPipeline, StrategyNotFoundError
from docs2synth.rag.types import RAGResult, RAGState
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


def _get_pipeline(ctx: click.Context) -> RAGPipeline:
    if "config" not in ctx.obj:
        raise click.ClickException(
            "Configuration not loaded. Use top-level CLI options."
        )
    if "rag_pipeline" not in ctx.obj:
        ctx.obj["rag_pipeline"] = RAGPipeline.from_config(
            ctx.obj["config"], config_path=ctx.obj.get("config_path")
        )
    return ctx.obj["rag_pipeline"]


def _ingest_processed_file(
    path: Path,
    include_context: bool = True,
) -> List[Tuple[str, dict]]:
    """Extract individual object texts from processed JSON.

    Each object's text is embedded separately. Context is stored in metadata
    but not embedded, allowing precise retrieval of specific objects.

    Returns:
        List of (text, metadata) tuples, one per object with non-empty text.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load processed JSON %s: %s", path, exc)
        return []

    process_meta = data.get("process_metadata", {}) or {}
    document_meta = data.get("document_metadata", {}) or {}
    context = data.get("context") if include_context else None

    chunks: List[Tuple[str, dict]] = []
    objects = data.get("objects", {})

    if isinstance(objects, dict):
        for obj_id, obj in objects.items():
            if not isinstance(obj, dict):
                continue
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            # Each object's text is embedded separately
            metadata = {
                "source": str(path),
                "object_id": str(obj_id),
                "processor": process_meta.get("processor_name"),
                "original_document": document_meta.get("filename"),
            }

            # Include context in metadata for reference, but don't embed it
            if context and isinstance(context, str) and context.strip():
                metadata["context"] = context.strip()

            # Include object-level metadata if available
            if "bbox" in obj:
                metadata["bbox"] = obj["bbox"]
            if "page" in obj:
                metadata["page"] = obj["page"]
            if "label" in obj:
                metadata["label"] = obj["label"]

            chunks.append((text.strip(), metadata))

    return chunks


@click.group(name="rag")
@click.pass_context
def rag_group(ctx: click.Context) -> None:
    """RAG experimentation commands."""
    ctx.ensure_object(dict)


@rag_group.command("strategies")
@click.pass_context
def list_strategies(ctx: click.Context) -> None:
    """List configured RAG strategies."""
    pipeline = _get_pipeline(ctx)
    click.echo("Available RAG strategies:")
    for name in pipeline.strategies:
        click.echo(f"- {name}")


@rag_group.command("ingest")
@click.option(
    "--processed-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing preprocessed JSON outputs (defaults to preprocess.output_dir).",
)
@click.option(
    "--processor",
    "processor_filter",
    type=click.Choice(
        ["docling", "paddleocr", "pdfplumber", "easyocr"], case_sensitive=False
    ),
    default=None,
    help="Filter processed JSON files by processor suffix "
    "(defaults to preprocess.processor in config.yml).",
)
@click.option(
    "--include-context/--exclude-context",
    default=True,
    show_default=True,
    help="Include the 'context' field from processed JSON when present.",
)
@click.pass_context
def ingest_documents(
    ctx: click.Context,
    processed_dir: Optional[Path],
    processor_filter: Optional[str],
    include_context: bool,
) -> None:
    """Index documents from preprocessed JSON outputs into the vector store."""
    pipeline = _get_pipeline(ctx)
    texts: List[str] = []
    metadatas: List[dict] = []

    cfg = ctx.obj.get("config")

    # Process JSON outputs
    if processed_dir is None:
        default_dir = None
        if cfg is not None:
            default_path = cfg.get("preprocess.output_dir")
            if default_path:
                default_dir = Path(default_path)
        processed_dir = default_dir

    effective_processor = processor_filter
    if effective_processor is None and cfg is not None:
        cfg_processor = cfg.get("preprocess.processor")
        if isinstance(cfg_processor, str) and cfg_processor.strip():
            effective_processor = cfg_processor.strip()

    if processed_dir and processed_dir.exists():
        suffixes = {
            "docling": "_docling.json",
            "paddleocr": "_paddleocr.json",
            "pdfplumber": "_pdfplumber.json",
            "easyocr": "_easyocr.json",
        }
        for json_path in sorted(processed_dir.glob("*.json")):
            if effective_processor:
                processor_filter_lower = effective_processor.lower()
                expected_suffix = suffixes.get(processor_filter_lower)
                if expected_suffix:
                    if not json_path.name.endswith(expected_suffix):
                        continue
                else:
                    logger.warning(
                        "Unknown processor '%s' specified; ingesting all files.",
                        effective_processor,
                    )
                    effective_processor = None
            chunks = _ingest_processed_file(
                json_path,
                include_context=include_context,
            )
            for text, metadata in chunks:
                texts.append(text)
                metadatas.append(metadata)

    if not texts:
        raise click.ClickException("No text content found to ingest.")

    pipeline.add_documents(texts, metadatas)
    click.echo(f"Ingested {len(texts)} document chunks into the vector store.")


def _render_result(result: RAGResult, show_iterations: bool) -> None:
    click.secho("Final answer:", bold=True)
    click.echo(result.final_answer or "<empty>")

    if show_iterations:
        click.echo()
        click.secho("Iterations:", bold=True)
        for iteration in result.iterations:
            click.secho(f"Step {iteration.step}", fg="cyan")
            if iteration.similarity is not None:
                click.echo(f"  Similarity vs. previous: {iteration.similarity:.3f}")
            click.echo("  Answer:")
            click.echo("    " + iteration.answer.replace("\n", "\n    "))
            click.echo("  Retrieved context:")
            for doc in iteration.retrieved:
                meta = doc.metadata
                source = meta.get("source", "N/A")
                obj_id = meta.get("object_id")
                if obj_id:
                    click.echo(
                        f"    - score={doc.score:.3f} source={source} object_id={obj_id}"
                    )
                else:
                    click.echo(f"    - score={doc.score:.3f} source={source}")


@rag_group.command("run")
@click.option(
    "-s",
    "--strategy",
    default="naive",
    show_default=True,
    help="Name of the RAG strategy to execute.",
)
@click.option(
    "-q",
    "--query",
    required=True,
    help="Question to ask the RAG pipeline.",
)
@click.option(
    "--show-iterations/--hide-iterations",
    default=False,
    show_default=True,
    help="Display intermediate iterations from the strategy.",
)
@click.pass_context
def run_query(
    ctx: click.Context,
    strategy: str,
    query: str,
    show_iterations: bool,
) -> None:
    """Execute a query against the configured RAG strategy."""
    pipeline = _get_pipeline(ctx)
    if len(pipeline.vector_store) == 0:
        raise click.ClickException(
            "Vector store is empty. Ingest documents first using `docs2synth rag ingest`."
        )

    logger.info("Running RAG query with strategy '%s'", strategy)
    state = RAGState()
    try:
        result = pipeline.run(query, strategy_name=strategy, state=state)
    except StrategyNotFoundError as exc:
        available = ", ".join(pipeline.strategies)
        raise click.ClickException(
            f"Unknown strategy '{strategy}'. Available strategies: {available}"
        ) from exc

    _render_result(result, show_iterations=show_iterations)


@rag_group.command("reset")
@click.confirmation_option(
    prompt="This will clear the persisted vector store. Continue?"
)
@click.pass_context
def reset_store(ctx: click.Context) -> None:
    """Clear the vector store contents."""
    pipeline = _get_pipeline(ctx)
    pipeline.reset()
    click.echo("Vector store cleared.")


@rag_group.command("app")
@click.option(
    "--host",
    default="localhost",
    show_default=True,
    help="Host interface for the Streamlit server.",
)
@click.option(
    "--port",
    default=8501,
    show_default=True,
    help="Port for the Streamlit server.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Run Streamlit in headless mode without opening a browser.",
)
@click.pass_context
def launch_app(ctx: click.Context, host: str, port: int, no_browser: bool) -> None:
    """Launch the Streamlit RAG playground."""
    try:
        from docs2synth.rag import streamlit_app  # noqa: F401
    except ImportError as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Failed to import streamlit app module: {exc}"
        ) from exc

    script_path = Path(streamlit_app.__file__).resolve()
    project_root = script_path.parents[2]

    cmd = [
        "streamlit",
        "run",
        str(script_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    env = os.environ.copy()

    config_path = ctx.obj.get("config_path")
    if config_path:
        env["DOCS2SYNTH_CONFIG"] = str(config_path)

    if no_browser:
        env["STREAMLIT_SERVER_HEADLESS"] = "true"

    click.echo(f"Launching Streamlit app at http://{host}:{port}")
    try:
        subprocess.run(cmd, env=env, cwd=str(project_root), check=True)
    except FileNotFoundError as exc:
        raise click.ClickException(
            "Streamlit executable not found. Install with `pip install streamlit`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"Streamlit exited with code {exc.returncode}"
        ) from exc
