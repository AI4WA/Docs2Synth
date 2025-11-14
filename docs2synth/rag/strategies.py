"""Concrete RAG strategy implementations."""

from __future__ import annotations

import difflib
from typing import List, Optional, Sequence

from docs2synth.rag.base import PromptConfig, RAGStrategy
from docs2synth.rag.types import (
    IterationResult,
    RAGResult,
    RAGState,
    RetrievedDocument,
)


class NaiveRAGStrategy(RAGStrategy):
    """Single-pass retrieval + generation strategy."""

    def generate(self, query: str, state: Optional[RAGState] = None) -> RAGResult:
        state = state or RAGState()
        documents = self._retrieve(query)
        context = self._build_context(documents)
        answer = self._invoke_llm(query, context)

        iteration = self._build_iteration(
            step=len(state.iterations) + 1,
            query=query,
            documents=documents,
            answer=answer,
        )

        result = RAGResult(final_answer=answer)
        result.add_iteration(iteration)
        state.push(iteration)
        return result


class EnhancedIterativeRAGStrategy(RAGStrategy):
    """Iterative strategy that feeds back prior results until convergence."""

    def __init__(
        self,
        name: str,
        vector_store,
        embedder,
        llm,
        prompt: PromptConfig,
        top_k: int = 5,
        max_iterations: int = 3,
        similarity_threshold: float = 0.9,
    ) -> None:
        super().__init__(name, vector_store, embedder, llm, prompt, top_k=top_k)
        self.max_iterations = max(1, max_iterations)
        self.similarity_threshold = similarity_threshold

    def generate(self, query: str, state: Optional[RAGState] = None) -> RAGResult:
        state = state or RAGState()
        result = RAGResult(final_answer="")

        previous_iteration = state.last_iteration

        for step in range(1, self.max_iterations + 1):
            augmented_query = self._augment_query(query, previous_iteration)
            documents = self._retrieve(augmented_query)
            context = self._build_context(documents)
            answer = self._invoke_llm(query, context)

            similarity = None
            if previous_iteration is not None:
                similarity = self._compare_answers(previous_iteration.answer, answer)

            iteration = self._build_iteration(
                step=len(state.iterations) + step,
                query=augmented_query,
                documents=documents,
                answer=answer,
                similarity=similarity,
            )
            result.add_iteration(iteration)
            state.push(iteration)
            previous_iteration = iteration
            result.final_answer = answer

            if similarity is not None and similarity >= self.similarity_threshold:
                break

        return result

    def _augment_query(self, query: str, previous: Optional[IterationResult]) -> str:
        """Compose a query using previous retrieval/answer context."""
        if previous is None:
            return query

        doc_context = "\n".join(doc.text for doc in previous.retrieved)
        parts = [
            f"Current question:\n{query}",
            "Previous retrieval context:\n" + doc_context if doc_context else "",
            "Previous answer:\n" + previous.answer if previous.answer else "",
        ]
        return "\n\n".join(part for part in parts if part).strip()

    @staticmethod
    def _compare_answers(a: str, b: str) -> float:
        """Return similarity ratio between two answers."""
        if not a or not b:
            return 0.0
        matcher = difflib.SequenceMatcher(a=a, b=b)
        return matcher.ratio()


class OurRetrieverRAGStrategy(EnhancedIterativeRAGStrategy):
    """RAG strategy using custom trained retriever model with iterative refinement.

    This strategy uses a custom retriever model (e.g., trained LayoutLM) for retrieval
    and applies iterative refinement similar to EnhancedIterativeRAGStrategy.
    
    Unlike other strategies, this one bypasses vector store search and directly
    computes similarities using the retriever model's embeddings.
    """

    def _retrieve(self, query: str) -> Sequence[RetrievedDocument]:
        """Retrieve documents using direct embedding similarity with retriever model.
        
        This bypasses the vector store search and directly computes similarities
        between the query and all documents using the retriever model embeddings.
        """
        import numpy as np
        
        # Get all documents from vector store
        all_documents = self.vector_store.get_all_documents()
        if not all_documents:
            return []
        
        # Compute query embedding using retriever model
        query_embedding = self.embedder.embed_query(query)
        query_emb = np.asarray(query_embedding, dtype="float32")
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        
        # Normalize query embedding if needed
        if hasattr(self.embedder, 'normalize') and self.embedder.normalize:
            norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
            query_emb = query_emb / (norm + 1e-8)
        
        # Get all document texts
        doc_texts = [doc.text for doc in all_documents]
        
        # Compute embeddings for all documents using retriever model
        doc_embeddings = self.embedder.embed_texts(doc_texts)
        doc_emb = np.asarray(doc_embeddings, dtype="float32")
        
        # Normalize document embeddings if needed
        if hasattr(self.embedder, 'normalize') and self.embedder.normalize:
            norm = np.linalg.norm(doc_emb, axis=1, keepdims=True)
            doc_emb = doc_emb / (norm + 1e-8)
        
        # Compute cosine similarity (dot product for normalized vectors)
        similarities = np.dot(doc_emb, query_emb.T).flatten()
        
        # Get top_k indices
        top_k = min(self.top_k, len(all_documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build retrieved documents
        retrieved: List[RetrievedDocument] = []
        for idx in top_indices:
            doc = all_documents[idx]
            retrieved.append(
                RetrievedDocument(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata or {},
                    score=float(similarities[idx]),
                )
            )
        
        return retrieved
