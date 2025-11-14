"""Pipeline utilities to orchestrate Docs2Synth RAG strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from uuid import uuid4

from docs2synth.agent.base import BaseLLMProvider
from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.rag.base import PromptConfig, RAGStrategy
from docs2synth.rag.embeddings import EmbeddingModel
from docs2synth.rag.strategies import (
    EnhancedIterativeRAGStrategy,
    NaiveRAGStrategy,
    OurRetrieverRAGStrategy,
)
from docs2synth.rag.types import DocumentChunk, RAGResult, RAGState
from docs2synth.rag.vector_store import FaissVectorStore, VectorStore
from docs2synth.utils.config import Config
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the retrieved context to answer the question."
)
DEFAULT_USER_PROMPT = "Question:\n{query}\n\nRelevant context:\n{context}\n\nAnswer in complete sentences."


class StrategyNotFoundError(KeyError):
    """Raised when a requested strategy is not registered in the pipeline."""


def _resolve_prompt_config(strategy_cfg: Mapping[str, object]) -> PromptConfig:
    prompt_cfg = (
        strategy_cfg.get("prompt", {}) if isinstance(strategy_cfg, Mapping) else {}
    )
    system_prompt = prompt_cfg.get("system", DEFAULT_SYSTEM_PROMPT)  # type: ignore[arg-type]
    user_prompt = prompt_cfg.get("user", DEFAULT_USER_PROMPT)  # type: ignore[arg-type]
    return PromptConfig(system=system_prompt, user=user_prompt)


def _create_vector_store(
    config: Mapping[str, object], dimension: Optional[int]
) -> VectorStore:
    store_type = str(config.get("type", "faiss")).lower()
    if store_type != "faiss":
        raise ValueError(f"Unsupported vector store type: {store_type}")
    persist_path = config.get("persist_path")
    normalize = bool(config.get("normalize", True))
    dim_override = config.get("dimension")
    dim = int(dim_override) if dim_override is not None else dimension
    return FaissVectorStore(
        dimension=dim, persist_path=persist_path, normalize=normalize
    )


def _resolve_llm_provider(
    config: Config,
    rag_section: Mapping[str, object],
    config_path: Optional[Path] = None,
) -> BaseLLMProvider:
    agent_cfg = config.get("agent") or {}
    rag_llm_cfg = rag_section.get("llm", {}) if isinstance(rag_section, Mapping) else {}

    provider_name = (
        rag_llm_cfg.get("provider") if isinstance(rag_llm_cfg, Mapping) else None
    )
    if not provider_name:
        provider_name = agent_cfg.get("provider")
    if not provider_name:
        raise ValueError(
            "No LLM provider configured. Set agent.provider or rag.llm.provider."
        )

    provider_name = str(provider_name)
    model_override: Optional[str] = None
    provider_kwargs: Dict[str, Any] = {}

    if isinstance(rag_llm_cfg, Mapping):
        for key, value in rag_llm_cfg.items():
            if key == "provider":
                continue
            if key == "model":
                model_override = value  # type: ignore[assignment]
            else:
                provider_kwargs[key] = value

    logger.info("Initializing LLM provider '%s' for RAG", provider_name)
    agent = AgentWrapper(
        provider=provider_name,
        model=model_override,
        config_path=str(config_path) if config_path else None,
        **provider_kwargs,
    )
    return agent.provider


@dataclass
class PipelineComponents:
    """Resolved components for a RAG pipeline."""

    embedder: EmbeddingModel
    vector_store: VectorStore
    llm: BaseLLMProvider


class RAGPipeline:
    """Coordinates vector store, embedding, and strategies."""

    def __init__(
        self,
        components: PipelineComponents,
        strategies: Dict[str, RAGStrategy],
    ) -> None:
        self.embedder = components.embedder
        self.vector_store = components.vector_store
        self.llm = components.llm
        self._strategies = strategies

    @classmethod
    def from_config(
        cls, config: Config, config_path: Optional[Path] = None
    ) -> "RAGPipeline":
        rag_cfg = config.get("rag") or {}

        embedding_cfg = (
            rag_cfg.get("embedding", {}) if isinstance(rag_cfg, Mapping) else {}
        )

        # Check if custom retriever model is specified
        # First try rag.embedding.retriever_model_path, then fall back to retriever.model_path
        retriever_model_path = embedding_cfg.get("retriever_model_path")
        if not retriever_model_path:
            # Auto-read from retriever.model_path if available (default behavior)
            retriever_cfg = config.get("retriever") or {}
            if isinstance(retriever_cfg, Mapping):
                model_path_str = retriever_cfg.get("model_path")
                if model_path_str:
                    # Resolve {run_id} placeholder if present
                    run_id = retriever_cfg.get("run_id")
                    if isinstance(model_path_str, str) and "{run_id}" in model_path_str:
                        if run_id:
                            retriever_model_path = model_path_str.format(run_id=run_id)
                        else:
                            retriever_model_path = model_path_str.format(
                                run_id="default"
                            )
                    else:
                        retriever_model_path = model_path_str
                    logger.info(
                        "Auto-resolved retriever_model_path from retriever.model_path: %s",
                        retriever_model_path,
                    )

        # Use OurRetrieverEmbeddingModel - retriever model is required
        if not retriever_model_path:
            raise ValueError(
                "Retriever model is required. Either set rag.embedding.retriever_model_path "
                "or configure retriever.model_path in config.yml"
            )

        from docs2synth.rag.our_retriever import OurRetrieverEmbeddingModel

        embedder = OurRetrieverEmbeddingModel(
            model_path=retriever_model_path,
            device=embedding_cfg.get("device"),
            normalize=bool(embedding_cfg.get("normalize", True)),
        )
        logger.info(
            "Using custom retriever model for embeddings: %s", retriever_model_path
        )

        vector_store_cfg = (
            rag_cfg.get("vector_store", {}) if isinstance(rag_cfg, Mapping) else {}
        )
        vector_store = _create_vector_store(vector_store_cfg, embedder.dimension)

        llm = _resolve_llm_provider(config, rag_cfg, config_path=config_path)

        strategies_cfg = (
            rag_cfg.get("strategies", {}) if isinstance(rag_cfg, Mapping) else {}
        )
        strategies: Dict[str, RAGStrategy] = {}

        for name, strategy_cfg in strategies_cfg.items():
            if not isinstance(strategy_cfg, Mapping):
                raise ValueError(f"Strategy config for '{name}' must be a mapping")

            strategy_type = str(strategy_cfg.get("type", "naive")).lower()
            prompt_config = _resolve_prompt_config(strategy_cfg)
            top_k = int(strategy_cfg.get("top_k", rag_cfg.get("top_k", 5)))

            if strategy_type == "naive":
                strategy = NaiveRAGStrategy(
                    name=name,
                    vector_store=vector_store,
                    embedder=embedder,
                    llm=llm,
                    prompt=prompt_config,
                    top_k=top_k,
                )
            elif strategy_type == "our_retriever":
                strategy = OurRetrieverRAGStrategy(
                    name=name,
                    vector_store=vector_store,
                    embedder=embedder,
                    llm=llm,
                    prompt=prompt_config,
                    top_k=top_k,
                    max_iterations=int(strategy_cfg.get("max_iterations", 3)),
                    similarity_threshold=float(
                        strategy_cfg.get("similarity_threshold", 0.9)
                    ),
                )
            elif strategy_type in {"enhanced", "iterative", "retriever_enhanced"}:
                strategy = EnhancedIterativeRAGStrategy(
                    name=name,
                    vector_store=vector_store,
                    embedder=embedder,
                    llm=llm,
                    prompt=prompt_config,
                    top_k=top_k,
                    max_iterations=int(strategy_cfg.get("max_iterations", 3)),
                    similarity_threshold=float(
                        strategy_cfg.get("similarity_threshold", 0.9)
                    ),
                )
            else:
                raise ValueError(f"Unsupported RAG strategy type: {strategy_type}")

            strategies[name] = strategy

        components = PipelineComponents(
            embedder=embedder, vector_store=vector_store, llm=llm
        )
        return cls(components, strategies)

    @property
    def strategies(self) -> Sequence[str]:
        return tuple(self._strategies.keys())

    def add_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Optional[Mapping[str, object]]]] = None,
    ) -> Sequence[DocumentChunk]:
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match texts.")

        documents: List[DocumentChunk] = []
        for i, text in enumerate(texts):
            metadata_value = metadatas[i] if metadatas else None
            metadata_dict = dict(metadata_value) if metadata_value else {}
            documents.append(
                DocumentChunk(
                    id=str(uuid4()),
                    text=text,
                    metadata=metadata_dict,
                )
            )

        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_embeddings(embeddings, documents)
        return documents

    def run(
        self, query: str, strategy_name: str, state: Optional[RAGState] = None
    ) -> RAGResult:
        if strategy_name not in self._strategies:
            raise StrategyNotFoundError(strategy_name)
        strategy = self._strategies[strategy_name]
        return strategy.generate(query, state=state)

    def reset(self) -> None:
        """Reset the underlying vector store and strategy state."""
        self.vector_store.reset()
