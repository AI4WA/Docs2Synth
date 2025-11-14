import numpy as np

from docs2synth.rag.types import DocumentChunk
from docs2synth.rag.vector_store import FaissVectorStore


def test_faiss_vector_store_add_and_search() -> None:
    store = FaissVectorStore(dimension=3, normalize=False)
    documents = [
        DocumentChunk(id="1", text="apple and banana"),
        DocumentChunk(id="2", text="carrots and celery"),
        DocumentChunk(id="3", text="banana smoothie recipe"),
    ]
    embeddings = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.85, 0.05, 0.1],
        ],
        dtype="float32",
    )

    store.add_embeddings(embeddings, documents)
    query = np.array([1.0, 0.0, 0.0], dtype="float32")

    results = store.search(query, top_k=2)
    assert len(results) == 2
    assert results[0].id in {"1", "3"}
    assert all(result.score > 0 for result in results)


def test_faiss_vector_store_persistence(tmp_path) -> None:
    index_path = tmp_path / "faiss.index"
    store = FaissVectorStore(dimension=2, persist_path=index_path, normalize=False)
    docs = [
        DocumentChunk(id="a", text="lorem ipsum"),
        DocumentChunk(id="b", text="dolor sit amet"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    reloaded = FaissVectorStore(persist_path=index_path, normalize=False)
    query = np.array([0.9, 0.1], dtype="float32")
    results = reloaded.search(query, top_k=1)
    assert results[0].id == "a"


def test_faiss_vector_store_empty_search() -> None:
    """Test searching an empty vector store returns empty results."""
    store = FaissVectorStore(dimension=3, normalize=False)
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    results = store.search(query, top_k=5)
    assert results == []


def test_faiss_vector_store_with_metadata() -> None:
    """Test storing and retrieving documents with metadata."""
    store = FaissVectorStore(dimension=2, normalize=False)
    docs = [
        DocumentChunk(id="1", text="doc1", metadata={"source": "file1.txt", "page": 1}),
        DocumentChunk(id="2", text="doc2", metadata={"source": "file2.txt", "page": 2}),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    query = np.array([1.0, 0.0], dtype="float32")
    results = store.search(query, top_k=1)
    assert results[0].metadata["source"] == "file1.txt"
    assert results[0].metadata["page"] == 1


def test_faiss_vector_store_top_k_larger_than_store() -> None:
    """Test that top_k larger than store size returns all documents."""
    store = FaissVectorStore(dimension=2, normalize=False)
    docs = [
        DocumentChunk(id="1", text="doc1"),
        DocumentChunk(id="2", text="doc2"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    query = np.array([0.5, 0.5], dtype="float32")
    results = store.search(query, top_k=10)
    # Should return only 2 results (all available)
    assert len(results) == 2


def test_faiss_vector_store_dimension_property() -> None:
    """Test dimension property returns correct dimension."""
    store = FaissVectorStore(dimension=128, normalize=False)
    assert store.dimension == 128


def test_faiss_vector_store_len() -> None:
    """Test __len__ returns correct number of documents."""
    store = FaissVectorStore(dimension=3, normalize=False)
    assert len(store) == 0

    docs = [
        DocumentChunk(id="1", text="doc1"),
        DocumentChunk(id="2", text="doc2"),
        DocumentChunk(id="3", text="doc3"),
    ]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32"
    )
    store.add_embeddings(embeddings, docs)
    assert len(store) == 3


def test_faiss_vector_store_reset() -> None:
    """Test reset clears the store."""
    store = FaissVectorStore(dimension=2, normalize=False)
    docs = [DocumentChunk(id="1", text="doc1")]
    embeddings = np.array([[1.0, 0.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    assert len(store) == 1
    store.reset()
    assert len(store) == 0
    # Searching empty store should return empty list
    results = store.search(np.array([1.0, 0.0], dtype="float32"), top_k=1)
    assert results == []


def test_faiss_vector_store_reset_with_persistence(tmp_path) -> None:
    """Test reset removes persisted files."""
    index_path = tmp_path / "test.index"
    meta_path = tmp_path / "test.index.meta.json"

    store = FaissVectorStore(dimension=2, persist_path=index_path, normalize=False)
    docs = [DocumentChunk(id="1", text="doc1")]
    embeddings = np.array([[1.0, 0.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    # Files should exist after adding
    assert index_path.exists()
    assert meta_path.exists()

    store.reset()
    # Files should be deleted after reset
    assert not index_path.exists()
    assert not meta_path.exists()


def test_faiss_vector_store_get_all_documents() -> None:
    """Test get_all_documents returns all stored documents."""
    store = FaissVectorStore(dimension=2, normalize=False)
    docs = [
        DocumentChunk(id="1", text="doc1"),
        DocumentChunk(id="2", text="doc2"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    all_docs = store.get_all_documents()
    assert len(all_docs) == 2
    assert all_docs[0].id == "1"
    assert all_docs[1].id == "2"


def test_faiss_vector_store_add_empty_documents() -> None:
    """Test adding empty list of documents does nothing."""
    store = FaissVectorStore(dimension=3, normalize=False)
    embeddings = np.array([], dtype="float32").reshape(0, 3)
    docs = []
    store.add_embeddings(embeddings, docs)
    assert len(store) == 0


def test_faiss_vector_store_dimension_mismatch_raises() -> None:
    """Test adding embeddings with wrong dimension raises error."""
    store = FaissVectorStore(dimension=3, normalize=False)
    # Add documents with correct dimension
    docs = [DocumentChunk(id="1", text="doc1")]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    # Try to add documents with wrong dimension (resets the store)
    docs2 = [DocumentChunk(id="2", text="doc2")]
    embeddings2 = np.array([[1.0, 0.0]], dtype="float32")  # 2D instead of 3D
    store.add_embeddings(embeddings2, docs2)

    # Store should have been reset and now contains only the new document
    assert len(store) == 1
    assert store.dimension == 2


def test_faiss_vector_store_search_dimension_mismatch() -> None:
    """Test searching with wrong dimension raises error."""
    store = FaissVectorStore(dimension=3, normalize=False)
    docs = [DocumentChunk(id="1", text="doc1")]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    # Try to search with wrong dimension
    query = np.array([1.0, 0.0], dtype="float32")  # 2D instead of 3D
    try:
        store.search(query, top_k=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Query dimension mismatch" in str(e)


def test_faiss_vector_store_query_1d_conversion() -> None:
    """Test that 1D query vectors are automatically reshaped to 2D."""
    store = FaissVectorStore(dimension=3, normalize=False)
    docs = [DocumentChunk(id="1", text="doc1")]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    # Query with 1D array
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    results = store.search(query, top_k=1)
    assert len(results) == 1
    assert results[0].id == "1"


def test_faiss_vector_store_embeddings_not_aligned() -> None:
    """Test adding embeddings and documents with mismatched lengths raises error."""
    store = FaissVectorStore(dimension=3, normalize=False)
    docs = [DocumentChunk(id="1", text="doc1")]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"
    )  # 2 embeddings

    try:
        store.add_embeddings(embeddings, docs)  # 1 document
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be aligned in length" in str(e)
