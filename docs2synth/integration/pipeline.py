"""Helpers for orchestrating the full Docs2Synth pipeline without annotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import click

from docs2synth.cli.commands.preprocess import preprocess as preprocess_command
from docs2synth.cli.commands.qa import qa_batch as qa_batch_command
from docs2synth.cli.commands.rag import _get_pipeline
from docs2synth.cli.commands.rag import ingest_documents as rag_ingest_command
from docs2synth.cli.commands.retriever import (
    retriever_preprocess as retriever_preprocess_command,
)
from docs2synth.cli.commands.retriever import retriever_train as retriever_train_command
from docs2synth.cli.commands.retriever import (
    retriever_validate as retriever_validate_command,
)
from docs2synth.cli.commands.verify import verify_batch as verify_batch_command


@dataclass(frozen=True)
class PipelineStep:
    """Represents a single pipeline step."""

    name: str
    runner: Callable[[click.Context], None]

    def run(self, ctx: click.Context) -> None:
        """Execute the step runner."""
        self.runner(ctx)


def _run_preprocess(ctx: click.Context) -> None:
    ctx.invoke(
        preprocess_command,
        path=None,
        processor_name=None,
        lang=None,
        output_dir=None,
        device=None,
    )


def _run_qa_batch(ctx: click.Context) -> None:
    ctx.invoke(
        qa_batch_command,
        input_path=None,
        output_dir=None,
        processor=None,
    )


def _run_verify_batch(ctx: click.Context) -> None:
    ctx.invoke(
        verify_batch_command,
        input_path=None,
        config_path=None,
        verifier_type=None,
        image_dir=(),
    )


def _run_retriever_preprocess(ctx: click.Context) -> None:
    ctx.invoke(
        retriever_preprocess_command,
        json_dir=None,
        image_dir=None,
        output=None,
        processor=None,
        batch_size=None,
        max_length=None,
        num_objects=None,
        require_all_verifiers=True,
    )


def _run_retriever_train(ctx: click.Context) -> None:
    ctx.invoke(
        retriever_train_command,
        model_path=None,
        base_model="microsoft/layoutlmv3-base",
        data_path=None,
        val_data_path=None,
        output_dir=None,
        mode="standard",
        lr=None,
        epochs=None,
        batch_size=None,
        save_every=None,
        resume=None,
        device=None,
    )


def _run_retriever_validate(ctx: click.Context) -> None:
    ctx.invoke(
        retriever_validate_command,
        model=None,
        data=None,
        output=None,
        mode="standard",
        device=None,
    )


def _run_rag_reset(ctx: click.Context) -> None:
    """Reset the RAG vector store (bypasses confirmation for automation)."""
    pipeline = _get_pipeline(ctx)
    pipeline.reset()
    click.echo("Vector store cleared.")


def _run_rag_ingest(ctx: click.Context) -> None:
    ctx.invoke(
        rag_ingest_command,
        processed_dir=None,
        processor=None,
        include_context=True,
    )


def build_pipeline_steps(include_validation: bool = True) -> List[PipelineStep]:
    """Create the default end-to-end pipeline step list.

    Args:
        include_validation: Whether to append the retriever validation phase.

    Returns:
        Ordered list of PipelineStep definitions.
    """

    steps: List[PipelineStep] = [
        PipelineStep("Preprocess documents", _run_preprocess),
        PipelineStep("Generate QA pairs", _run_qa_batch),
        PipelineStep("Verify QA pairs", _run_verify_batch),
        PipelineStep("Prepare retriever dataset", _run_retriever_preprocess),
        PipelineStep("Train retriever", _run_retriever_train),
    ]

    if include_validation:
        steps.append(PipelineStep("Validate retriever", _run_retriever_validate))

    steps.extend(
        [
            PipelineStep("Reset RAG vector store", _run_rag_reset),
            PipelineStep("Ingest documents", _run_rag_ingest),
        ]
    )

    return steps


def run_pipeline(ctx: click.Context, include_validation: bool = True) -> None:
    """Run the Docs2Synth automation pipeline via existing CLI commands.

    The run intentionally skips the human annotation phase by chaining:
    preprocess → QA batch → verification → retriever preprocess → retriever train
    → retriever validate (optional) → RAG reset → RAG ingest.
    """

    steps = build_pipeline_steps(include_validation=include_validation)

    total = len(steps)
    for index, step in enumerate(steps, start=1):
        click.echo(
            click.style(
                f"\n[{index}/{total}] {step.name}",
                fg="cyan",
                bold=True,
            )
        )
        step.run(ctx)

    click.echo(
        click.style(
            "\n✓ End-to-end pipeline complete",
            fg="green",
            bold=True,
        )
    )
