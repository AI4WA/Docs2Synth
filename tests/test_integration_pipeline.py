"""Tests for the Docs2Synth integration pipeline helpers."""

from docs2synth.integration.pipeline import build_pipeline_steps


def test_build_pipeline_steps_default_includes_validation() -> None:
    steps = build_pipeline_steps()
    names = [step.name for step in steps]

    assert names == [
        "Preprocess documents",
        "Generate QA pairs",
        "Verify QA pairs",
        "Prepare retriever dataset",
        "Train retriever",
        "Validate retriever",
        "Reset RAG vector store",
        "Ingest documents",
    ]

    runner_names = [step.runner.__name__ for step in steps]
    assert runner_names == [
        "_run_preprocess",
        "_run_qa_batch",
        "_run_verify_batch",
        "_run_retriever_preprocess",
        "_run_retriever_train",
        "_run_retriever_validate",
        "_run_rag_reset",
        "_run_rag_ingest",
    ]


def test_build_pipeline_steps_without_validation() -> None:
    steps = build_pipeline_steps(include_validation=False)
    names = [step.name for step in steps]

    assert "Validate retriever" not in names
    assert names == [
        "Preprocess documents",
        "Generate QA pairs",
        "Verify QA pairs",
        "Prepare retriever dataset",
        "Train retriever",
        "Reset RAG vector store",
        "Ingest documents",
    ]
    assert len(steps) == 7
