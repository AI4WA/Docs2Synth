"""Base classes for LLM agent providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class LLMResponse:
    """Response from LLM provider."""

    def __init__(
        self,
        content: str,
        model: str,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize LLM response.

        Args:
            content: Generated text content
            model: Model name used
            usage: Token usage information (prompt_tokens, completion_tokens, total_tokens)
            metadata: Additional metadata from the provider
        """
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.metadata = metadata or {}

    def __str__(self) -> str:
        """String representation."""
        return self.content

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"LLMResponse(content={self.content[:50]}..., model={self.model}, usage={self.usage})"


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface to ensure seamless switching
    between different models (cloud APIs, local models, etc.).
    """

    def __init__(self, model: str, **kwargs: Any):
        """Initialize LLM provider.

        Args:
            model: Model identifier/name
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.config = kwargs
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text from prompt.

        Args:
            prompt: User prompt/text
            system_prompt: Optional system prompt for chat models
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" for JSON mode, None for text)
            **kwargs: Provider-specific parameters. May include:
                - image: PIL.Image, image path, base64 string, or image URL (for vision models)
                - images: List of images (for multimodal models)
                - Other provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Note:
            For vision-language models, pass images via kwargs['image'] or kwargs['images'].
            Supported formats depend on the provider (PIL.Image, file path, base64, URL).
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                For vision models, content can be a list with text and image items.
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" for JSON mode, None for text)
            **kwargs: Provider-specific parameters. May include:
                - image: PIL.Image, image path, base64 string, or image URL (for vision models)
                - images: List of images (for multimodal models)
                - Other provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Note:
            For vision-language models, images can be passed:
            1. Via kwargs['image'] or kwargs['images']
            2. Or embedded in messages content (provider-specific format)
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model={self.model})"
