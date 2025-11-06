"""Factory for creating LLM provider instances."""

from __future__ import annotations

from typing import Any, Dict, Optional

from docs2synth.agent.base import BaseLLMProvider
from docs2synth.agent.providers import PROVIDER_REGISTRY
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    @staticmethod
    def create_provider(
        provider: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'ollama')
            model: Model name (optional, uses provider default if not specified)
            **kwargs: Provider-specific configuration

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider is not supported

        Example:
            >>> # Create OpenAI provider (api_key/model are auto-read from config.yml)
            >>> provider = LLMProviderFactory.create_provider(
            ...     "openai",
            ... )
            >>>
            >>> # Create Ollama provider
            >>> provider = LLMProviderFactory.create_provider(
            ...     "ollama",
            ...     model="llama2",
            ...     base_url="http://localhost:11434",
            ... )
        """
        provider_lower = provider.lower().strip()

        if provider_lower not in PROVIDER_REGISTRY:
            available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported provider: {provider}. Available providers: {available}"
            )

        provider_class = PROVIDER_REGISTRY[provider_lower]
        logger.info(f"Creating {provider_class.__name__} with model={model}")

        # Remove model from kwargs if it's None, let provider use default
        if model is not None:
            kwargs["model"] = model

        return provider_class(**kwargs)

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseLLMProvider:
        """Create provider from configuration dictionary.

        Args:
            config: Configuration dict with 'provider' key and provider-specific settings

        Returns:
            LLM provider instance

        Example:
            >>> config = {
            ...     "provider": "openai",
            ...     "model": "gpt-4",
            ...     "api_key": "sk-...",
            ... }
            >>> provider = LLMProviderFactory.create_from_config(config)
        """
        if "provider" not in config:
            raise ValueError("Configuration must include 'provider' key")

        provider = config.pop("provider")
        return LLMProviderFactory.create_provider(provider, **config)
