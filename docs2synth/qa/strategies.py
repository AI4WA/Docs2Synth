"""Strategy registry and factory for QA generators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.qa.config import QAGenerationConfig, QAStrategyConfig
from docs2synth.qa.generators.base import BaseQAGenerator
from docs2synth.qa.generators.layout_aware import LayoutAwareQAGenerator
from docs2synth.qa.generators.logical_aware import LogicalAwareQAGenerator
from docs2synth.qa.generators.semantic import SemanticQAGenerator
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)

# Strategy registry
STRATEGY_REGISTRY: Dict[str, Type[BaseQAGenerator]] = {
    "semantic": SemanticQAGenerator,
    "layout_aware": LayoutAwareQAGenerator,
    "logical_aware": LogicalAwareQAGenerator,
}


class QAGeneratorFactory:
    """Factory for creating QA generators based on strategy name."""

    @staticmethod
    def create(
        strategy: str,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseQAGenerator:
        """Create a QA generator instance.

        Args:
            strategy: Strategy name (semantic, layout_aware, logical_aware)
            agent: Pre-configured AgentWrapper instance (optional)
            provider: Provider name if creating new agent
            model: Model name if creating new agent
            **kwargs: Additional arguments passed to generator and AgentWrapper

        Returns:
            QA generator instance

        Raises:
            ValueError: If strategy is not registered

        Example:
            >>> # Using existing agent
            >>> agent = AgentWrapper(provider="openai", model="gpt-4")
            >>> generator = QAGeneratorFactory.create(
            ...     strategy="semantic",
            ...     agent=agent
            ... )
            >>> question = generator.generate(context="...", target="...")
            >>>
            >>> # Or create agent automatically
            >>> generator = QAGeneratorFactory.create(
            ...     strategy="semantic",
            ...     provider="openai",
            ...     model="gpt-4"
            ... )
        """
        strategy_lower = strategy.lower().strip()

        if strategy_lower not in STRATEGY_REGISTRY:
            available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
            raise ValueError(
                f"Unknown QA generation strategy: {strategy}. "
                f"Available strategies: {available}"
            )

        generator_class = STRATEGY_REGISTRY[strategy_lower]
        logger.info(f"Creating {generator_class.__name__} generator")

        return generator_class(
            agent=agent,
            provider=provider,
            model=model,
            **kwargs,
        )

    @staticmethod
    def register(strategy: str, generator_class: Type[BaseQAGenerator]) -> None:
        """Register a new QA generation strategy.

        Args:
            strategy: Strategy name
            generator_class: Generator class (must inherit from BaseQAGenerator)

        Example:
            >>> class CustomGenerator(BaseQAGenerator):
            ...     def generate(self, ...):
            ...         ...
            >>> QAGeneratorFactory.register("custom", CustomGenerator)
        """
        if not issubclass(generator_class, BaseQAGenerator):
            raise TypeError(
                f"Generator class must inherit from BaseQAGenerator, "
                f"got {generator_class}"
            )

        strategy_lower = strategy.lower().strip()
        STRATEGY_REGISTRY[strategy_lower] = generator_class
        logger.info(f"Registered QA generation strategy: {strategy_lower}")

    @staticmethod
    def list_strategies() -> list[str]:
        """List all registered strategies.

        Returns:
            List of strategy names
        """
        return sorted(STRATEGY_REGISTRY.keys())

    @staticmethod
    def create_from_config(
        strategy_config: QAStrategyConfig,
        config_path: Optional[str] = None,
    ) -> BaseQAGenerator:
        """Create a QA generator from configuration.

        Args:
            strategy_config: QAStrategyConfig instance
            config_path: Path to config.yml (optional, for AgentWrapper)

        Returns:
            QA generator instance

        Example:
            >>> from docs2synth.qa.config import QAStrategyConfig
            >>> config = QAStrategyConfig(
            ...     strategy="semantic",
            ...     provider="openai",
            ...     model="gpt-4",
            ...     prompt_template="Custom template..."
            ... )
            >>> generator = QAGeneratorFactory.create_from_config(config)
        """
        # Build kwargs for generator initialization
        generator_kwargs: Dict[str, Any] = {}
        if strategy_config.prompt_template:
            generator_kwargs["prompt_template"] = strategy_config.prompt_template

        # Build kwargs for AgentWrapper
        agent_kwargs: Dict[str, Any] = {}
        if config_path:
            agent_kwargs["config_path"] = config_path
        if strategy_config.temperature is not None:
            agent_kwargs["temperature"] = strategy_config.temperature
        if strategy_config.max_tokens is not None:
            agent_kwargs["max_tokens"] = strategy_config.max_tokens
        agent_kwargs.update(strategy_config.extra_kwargs)

        return QAGeneratorFactory.create(
            strategy=strategy_config.strategy,
            provider=strategy_config.provider,
            model=strategy_config.model,
            **generator_kwargs,
            **agent_kwargs,
        )

    @staticmethod
    def create_all_from_config(
        qa_config: QAGenerationConfig,
        config_path: Optional[str] = None,
    ) -> List[BaseQAGenerator]:
        """Create all QA generators from configuration.

        Args:
            qa_config: QAGenerationConfig instance
            config_path: Path to config.yml (optional, for AgentWrapper)

        Returns:
            List of QA generator instances

        Example:
            >>> from docs2synth.qa.config import QAGenerationConfig
            >>> qa_config = QAGenerationConfig.from_yaml("config.yml")
            >>> generators = QAGeneratorFactory.create_all_from_config(qa_config)
            >>> for gen in generators:
            ...     print(gen)
        """
        generators = []
        for strategy_config in qa_config.strategies:
            try:
                generator = QAGeneratorFactory.create_from_config(
                    strategy_config, config_path=config_path
                )
                generators.append(generator)
            except Exception as e:
                logger.error(
                    f"Failed to create generator for strategy '{strategy_config.strategy}': {e}"
                )
                continue
        return generators
