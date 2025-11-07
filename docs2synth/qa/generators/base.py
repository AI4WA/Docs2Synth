"""Base class for QA generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class BaseQAGenerator(ABC):
    """Abstract base class for QA generation strategies.

    Each QA generation strategy should inherit from this class and implement
    the generate method to produce questions based on different approaches.
    All strategies use AgentWrapper for model interaction.
    """

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize QA generator.

        Args:
            agent: Pre-configured AgentWrapper instance (optional)
            provider: Provider name if creating new agent (e.g., 'openai', 'qwen2vl')
            model: Model name if creating new agent
            **kwargs: Additional arguments for AgentWrapper initialization

        Raises:
            ValueError: If neither agent nor provider is specified
        """
        if agent is None and provider is None:
            raise ValueError("Either 'agent' or 'provider' must be specified")

        if agent is None:
            self.agent = AgentWrapper(provider=provider, model=model, **kwargs)
        else:
            self.agent = agent

        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate(
        self,
        context: str,
        target: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a question based on the strategy.

        Args:
            context: Document context (e.g., OCR text from document)
            target: Target answer or object to generate question for
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments specific to the strategy

        Returns:
            Generated question string

        Raises:
            NotImplementedError: If strategy doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement generate method")

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(provider={self.agent.provider_type}, model={self.agent.model})"
