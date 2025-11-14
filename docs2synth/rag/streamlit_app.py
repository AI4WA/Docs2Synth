"""Streamlit interface to compare RAG strategies side-by-side."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import streamlit as st

from docs2synth.rag.pipeline import RAGPipeline
from docs2synth.rag.types import RAGResult, RAGState
from docs2synth.utils.config import get_config


def _initialize_pipeline() -> RAGPipeline:
    config = get_config()
    config_path = os.getenv("DOCS2SYNTH_CONFIG")
    return RAGPipeline.from_config(
        config, config_path=Path(config_path) if config_path else None
    )


def _ensure_session_state() -> None:
    if "rag_pipeline" not in st.session_state:
        st.session_state["rag_pipeline"] = _initialize_pipeline()
    if "rag_documents" not in st.session_state:
        st.session_state["rag_documents"] = []


def _split_documents(raw_text: str) -> List[str]:
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    return blocks


def _render_result(label: str, result: RAGResult) -> None:
    st.markdown(f"### {label}")
    st.write(result.final_answer or "_No answer generated._")

    with st.expander("Iterations", expanded=True):
        for iteration in result.iterations:
            st.markdown(f"**Step {iteration.step}**")
            if iteration.similarity is not None:
                st.caption(f"Similarity to previous answer: {iteration.similarity:.3f}")
            st.markdown("**Answer**")
            st.write(iteration.answer)
            st.markdown("**Retrieved Context**")
            for doc in iteration.retrieved:
                st.markdown(f"*Score:* {doc.score:.3f}")
                if doc.metadata:
                    st.code(str(doc.metadata))
                st.write(doc.text)
            st.divider()


def _render_sidebar(pipeline: RAGPipeline) -> None:
    """Render the sidebar with corpus status and reset button."""
    st.header("Corpus")
    if len(pipeline.vector_store) == 0:
        st.warning(
            "The vector store is empty. Use `docs2synth rag ingest` to index documents."
        )
    else:
        st.success("Documents already indexed via CLI ingest.")

    if st.button("Reset Pipeline"):
        st.session_state["rag_pipeline"] = _initialize_pipeline()
        st.session_state["rag_documents"] = []
        st.info("Pipeline reset. Re-run CLI ingest to rebuild the index.")


def _select_strategies(strategies: List[str]) -> tuple[List[str], List[str]]:
    """Select and order strategies for display.

    Returns:
        Tuple of (selected_strategies, selected_labels)
    """
    strategy_map = {
        "naive": "Naive RAG",
        "enhanced": "Enhanced RAG",
        "our_retriever": "Our Retriever RAG",
    }

    # Prefer these three strategies in order
    preferred_strategies = ["naive", "enhanced", "our_retriever"]
    selected_strategies = []
    selected_labels = []

    for strategy in preferred_strategies:
        if strategy in strategies:
            selected_strategies.append(strategy)
            selected_labels.append(strategy_map.get(strategy, strategy.title()))

    # If we don't have all three, add any remaining strategies
    for strategy in strategies:
        if strategy not in selected_strategies:
            selected_strategies.append(strategy)
            selected_labels.append(strategy_map.get(strategy, strategy.title()))

    return selected_strategies, selected_labels


def _run_strategies(
    pipeline: RAGPipeline, query: str, selected_strategies: List[str]
) -> dict[str, RAGResult | None]:
    """Run all selected strategies and return results.

    Returns:
        Dictionary mapping strategy names to results (or None if failed)
    """
    states = {strategy: RAGState() for strategy in selected_strategies}
    results = {}

    with st.spinner("Generating answers..."):
        for strategy in selected_strategies:
            try:
                results[strategy] = pipeline.run(
                    query, strategy, state=states[strategy]
                )
            except Exception as e:
                st.error(f"Error running {strategy}: {e}")
                results[strategy] = None

    return results


def _display_results(
    selected_strategies: List[str],
    selected_labels: List[str],
    results: dict[str, RAGResult | None],
) -> None:
    """Display results in columns."""
    num_cols = len(selected_strategies)
    if num_cols == 1:
        cols = [st.container()]
    elif num_cols == 2:
        cols = st.columns(2)
    else:
        cols = st.columns(3)

    for i, (strategy, label) in enumerate(zip(selected_strategies, selected_labels)):
        with cols[i]:
            if results.get(strategy) is not None:
                _render_result(label, results[strategy])
            else:
                st.error(f"Failed to generate result for {label}")


def main() -> None:
    st.set_page_config(page_title="Docs2Synth RAG Playground", layout="wide")
    st.title("RAG Strategy Playground")
    st.caption(
        "Compare different RAG strategies: Naive RAG, Enhanced Iterative RAG, and Our Retriever RAG."
    )

    _ensure_session_state()
    pipeline: RAGPipeline = st.session_state["rag_pipeline"]

    with st.sidebar:
        _render_sidebar(pipeline)

    query = st.text_input("Ask a question", "")
    strategies = list(pipeline.strategies)
    selected_strategies, selected_labels = _select_strategies(strategies)

    if len(selected_strategies) == 0:
        st.warning("No strategies configured. Update config to proceed.")
        return

    run_clicked = st.button("Run Query")

    if run_clicked:
        if not query.strip():
            st.warning("Enter a question before running the query.")
            return
        if len(pipeline.vector_store) == 0:
            st.warning("Index documents first before querying the pipeline.")
            return

        results = _run_strategies(pipeline, query, selected_strategies)
        _display_results(selected_strategies, selected_labels, results)


if __name__ == "__main__":
    main()
