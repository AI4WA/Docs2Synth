"""Base class for QA verifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class BaseQAVerifier(ABC):
    """Abstract base class for QA verification strategies.

    Each QA verification strategy should inherit from this class and implement
    the verify method to validate QA pairs based on different approaches.
    All strategies use AgentWrapper for model interaction.
    """

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize QA verifier.

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
    def verify(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
        image: Optional[Any] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Verify a QA pair based on the strategy.

        Args:
            question: Generated question
            answer: Target answer
            context: Document context (e.g., OCR text from document) - optional
            image: Document image (optional, for vision models)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments specific to the strategy

        Returns:
            Dictionary with verification results, typically including:
            - 'response': 'Yes' or 'No'
            - 'explanation': Optional explanation string
            - Additional strategy-specific fields

        Raises:
            NotImplementedError: If strategy doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement verify method")

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(provider={self.agent.provider_type}, model={self.agent.model})"
