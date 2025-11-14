"""Tests for RAG type definitions and dataclasses."""

from docs2synth.rag.types import (
    DocumentChunk,
    IterationResult,
    RAGResult,
    RAGState,
    RetrievedDocument,
)


def test_document_chunk_creation() -> None:
    """Test basic DocumentChunk creation."""
    doc = DocumentChunk(id="1", text="Hello world")
    assert doc.id == "1"
    assert doc.text == "Hello world"
    assert doc.metadata is None


def test_document_chunk_with_metadata() -> None:
    """Test DocumentChunk with metadata."""
    metadata = {"source": "test.pdf", "page": 1}
    doc = DocumentChunk(id="2", text="Test content", metadata=metadata)
    assert doc.id == "2"
    assert doc.text == "Test content"
    assert doc.metadata == metadata
    assert doc.metadata["source"] == "test.pdf"
    assert doc.metadata["page"] == 1


def test_retrieved_document_inherits_from_document_chunk() -> None:
    """Test RetrievedDocument is a DocumentChunk with score."""
    doc = RetrievedDocument(id="3", text="Retrieved text", score=0.95)
    assert isinstance(doc, DocumentChunk)
    assert doc.id == "3"
    assert doc.text == "Retrieved text"
    assert doc.score == 0.95


def test_retrieved_document_default_score() -> None:
    """Test RetrievedDocument has default score of 0.0."""
    doc = RetrievedDocument(id="4", text="Text")
    assert doc.score == 0.0


def test_retrieved_document_with_metadata_and_score() -> None:
    """Test RetrievedDocument with both metadata and score."""
    metadata = {"source": "doc.txt"}
    doc = RetrievedDocument(id="5", text="Content", metadata=metadata, score=0.87)
    assert doc.id == "5"
    assert doc.text == "Content"
    assert doc.metadata == metadata
    assert doc.score == 0.87


def test_iteration_result_creation() -> None:
    """Test IterationResult creation."""
    retrieved = [
        RetrievedDocument(id="1", text="Doc 1", score=0.9),
        RetrievedDocument(id="2", text="Doc 2", score=0.8),
    ]
    iteration = IterationResult(
        query="test query",
        retrieved=retrieved,
        answer="test answer",
        similarity=0.95,
        step=1,
    )
    assert iteration.query == "test query"
    assert len(iteration.retrieved) == 2
    assert iteration.answer == "test answer"
    assert iteration.similarity == 0.95
    assert iteration.step == 1


def test_iteration_result_optional_fields() -> None:
    """Test IterationResult with optional fields as None."""
    iteration = IterationResult(
        query="query",
        retrieved=[],
        answer="answer",
    )
    assert iteration.similarity is None
    assert iteration.step == 0


def test_rag_result_creation() -> None:
    """Test RAGResult creation."""
    result = RAGResult(final_answer="Final answer")
    assert result.final_answer == "Final answer"
    assert result.iterations == []


def test_rag_result_add_iteration() -> None:
    """Test adding iterations to RAGResult."""
    result = RAGResult(final_answer="Answer")
    iteration1 = IterationResult(query="q1", retrieved=[], answer="a1", step=1)
    iteration2 = IterationResult(query="q2", retrieved=[], answer="a2", step=2)

    result.add_iteration(iteration1)
    assert len(result.iterations) == 1
    assert result.iterations[0] == iteration1

    result.add_iteration(iteration2)
    assert len(result.iterations) == 2
    assert result.iterations[1] == iteration2


def test_rag_state_empty() -> None:
    """Test RAGState starts empty."""
    state = RAGState()
    assert state.iterations == []
    assert state.last_iteration is None


def test_rag_state_push() -> None:
    """Test pushing iterations to RAGState."""
    state = RAGState()
    iteration1 = IterationResult(query="q1", retrieved=[], answer="a1", step=1)
    iteration2 = IterationResult(query="q2", retrieved=[], answer="a2", step=2)

    state.push(iteration1)
    assert len(state.iterations) == 1
    assert state.last_iteration == iteration1

    state.push(iteration2)
    assert len(state.iterations) == 2
    assert state.last_iteration == iteration2


def test_rag_state_last_iteration() -> None:
    """Test RAGState.last_iteration property."""
    state = RAGState()
    assert state.last_iteration is None

    iteration1 = IterationResult(query="q1", retrieved=[], answer="a1", step=1)
    state.push(iteration1)
    assert state.last_iteration == iteration1

    iteration2 = IterationResult(query="q2", retrieved=[], answer="a2", step=2)
    state.push(iteration2)
    assert state.last_iteration == iteration2
    assert state.last_iteration != iteration1


def test_rag_state_reset() -> None:
    """Test RAGState.reset() clears history."""
    state = RAGState()
    iteration1 = IterationResult(query="q1", retrieved=[], answer="a1", step=1)
    iteration2 = IterationResult(query="q2", retrieved=[], answer="a2", step=2)

    state.push(iteration1)
    state.push(iteration2)
    assert len(state.iterations) == 2
    assert state.last_iteration is not None

    state.reset()
    assert state.iterations == []
    assert state.last_iteration is None
