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
