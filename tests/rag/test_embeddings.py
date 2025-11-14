"""Tests for embedding utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docs2synth.rag.embeddings import EmbeddingModel


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing."""
    mock = MagicMock()
    mock.encode.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32"
    )
    return mock


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_embedding_model_initialization(mock_load) -> None:
    """Test EmbeddingModel initialization."""
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model", device="cpu", normalize=True)

    assert embedder.model_name == "test-model"
    assert embedder.device == "cpu"
    assert embedder.normalize is True
    mock_load.assert_called_once_with("test-model", "cpu")


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_embed_texts(mock_load) -> None:
    """Test embedding multiple texts."""
    mock_model = MagicMock()
    # Return different embeddings for each text
    mock_model.encode.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32"
    )
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model", normalize=False)
    texts = ["Hello world", "Goodbye world"]
    embeddings = embedder.embed_texts(texts)

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    mock_model.encode.assert_called_once()


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_embed_query(mock_load) -> None:
    """Test embedding a single query."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]], dtype="float32")
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model", normalize=False)
    query_embedding = embedder.embed_query("test query")

    assert query_embedding.shape == (3,)
    assert query_embedding.dtype == np.float32
    assert np.allclose(query_embedding, np.array([0.7, 0.8, 0.9]))


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_embed_empty_texts(mock_load) -> None:
    """Test embedding empty list of texts."""
    mock_model = MagicMock()
    # For dimension probe
    mock_model.encode.side_effect = [
        np.array([[0.1, 0.2, 0.3]], dtype="float32"),  # dimension probe
        np.empty((0, 3), dtype="float32"),  # actual empty call
    ]
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model", normalize=False)
    embeddings = embedder.embed_texts([])

    assert embeddings.shape == (0, 3)
    assert embeddings.dtype == np.float32


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_dimension_property(mock_load) -> None:
    """Test dimension property returns correct embedding dimension."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5]], dtype="float32"
    )
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model")

    # First call should probe dimension
    dim = embedder.dimension
    assert dim == 5

    # Second call should use cached value
    dim2 = embedder.dimension
    assert dim2 == 5

    # Encode should only be called once for dimension probe
    assert mock_model.encode.call_count == 1


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_normalization_enabled(mock_load) -> None:
    """Test that normalization flag is passed to encode."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype="float32")
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model", normalize=True)
    embedder.embed_texts(["test"])

    # Check that normalize_embeddings=True is passed
    call_kwargs = mock_model.encode.call_args[1]
    assert call_kwargs["normalize_embeddings"] is True


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_normalization_disabled(mock_load) -> None:
    """Test that normalization can be disabled."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype="float32")
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model", normalize=False)
    embedder.embed_texts(["test"])

    # Check that normalize_embeddings=False is passed
    call_kwargs = mock_model.encode.call_args[1]
    assert call_kwargs["normalize_embeddings"] is False


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_dtype_conversion(mock_load) -> None:
    """Test that embeddings are converted to float32 if needed."""
    mock_model = MagicMock()
    # Return float64 embeddings
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype="float64")
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model", normalize=False)
    embeddings = embedder.embed_texts(["test"])

    # Should be converted to float32
    assert embeddings.dtype == np.float32


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_repr(mock_load) -> None:
    """Test __repr__ method."""
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="my-model", device="cuda:0", normalize=True)
    repr_str = repr(embedder)

    assert "EmbeddingModel" in repr_str
    assert "my-model" in repr_str
    assert "cuda:0" in repr_str


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_default_model_name(mock_load) -> None:
    """Test that default model name is used when not specified."""
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    embedder = EmbeddingModel()

    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    mock_load.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", None)


@patch("docs2synth.rag.embeddings._load_sentence_transformer")
def test_encode_parameters(mock_load) -> None:
    """Test that correct parameters are passed to encode."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype="float32")
    mock_load.return_value = mock_model

    embedder = EmbeddingModel(model_name="test-model", normalize=True)
    embedder.embed_texts(["test text"])

    # Check all expected parameters
    call_kwargs = mock_model.encode.call_args[1]
    assert call_kwargs["show_progress_bar"] is False
    assert call_kwargs["convert_to_numpy"] is True
    assert call_kwargs["normalize_embeddings"] is True
