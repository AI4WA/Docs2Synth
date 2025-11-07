"""High-level wrapper for LLM agents with seamless provider switching."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from docs2synth.agent.base import LLMResponse
from docs2synth.agent.factory import LLMProviderFactory
from docs2synth.utils.config import Config
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class AgentWrapper:
    """High-level wrapper for LLM agents.

    Provides a unified interface for interacting with different LLM providers,
    allowing seamless switching between cloud APIs and local models.

    Example:
        >>> # Initialize with OpenAI
        >>> agent = AgentWrapper(provider="openai", model="gpt-4", api_key="sk-...")
        >>> response = agent.generate("What is AI?")
        >>>
        >>> # Switch to Ollama (same interface!)
        >>> agent = AgentWrapper(provider="ollama", model="llama2")
        >>> response = agent.generate("What is AI?")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize agent wrapper.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'ollama')
                     If None, loads from config file
            model: Model name (optional, uses provider default if not specified)
            config_path: Path to config file (optional, uses default if None)
            **kwargs: Provider-specific configuration

        Example:
            >>> # From config file
            >>> agent = AgentWrapper()
            >>>
            >>> # Explicit provider
            >>> agent = AgentWrapper(provider="openai", model="gpt-4", api_key="sk-...")
        """
        self.logger = get_logger(__name__)

        # Load configs
        agent_config, providers_config, keys_config = self._load_agent_configs(
            config_path
        )

        # Resolve provider name
        provider = self._resolve_provider_name(provider, kwargs, agent_config)

        # Build provider kwargs and final model
        kwargs, model = self._build_provider_kwargs_and_model(
            provider, model, kwargs, agent_config, providers_config, keys_config
        )

        self.provider_name = provider
        self.provider = LLMProviderFactory.create_provider(provider, model, **kwargs)
        self.logger.info(
            f"Initialized {self.provider_name} agent with model={self.provider.model}"
        )

    def _load_agent_configs(
        self, config_path: Optional[str]
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        agent_config: Dict[str, Any] = {}
        providers_config: Dict[str, Any] = {}
        keys_config: Dict[str, Any] = {}
        try:
            cfg = None
            if config_path:
                cfg = Config.from_yaml(config_path)
            else:
                env_path = os.getenv("DOCS2SYNTH_CONFIG")
                if env_path:
                    cfg = Config.from_yaml(env_path)
            if cfg is not None:
                agent_ns = cfg.get("agent", {}) or {}
                agent_config = agent_ns
                providers_config = cfg.get("providers", {}) or {}
                keys_config = cfg.get("keys", {}) or {}
                nested_keys = (
                    agent_ns.get("keys") if isinstance(agent_ns, dict) else None
                )
                if isinstance(nested_keys, dict) and nested_keys:
                    keys_config = nested_keys
        except Exception:
            agent_config, providers_config, keys_config = {}, {}, {}
        return agent_config, providers_config, keys_config

    def _resolve_provider_name(
        self,
        provider: Optional[str],
        kwargs: Dict[str, Any],
        agent_config: Dict[str, Any],
    ) -> str:
        resolved = (
            provider or kwargs.pop("provider", None) or agent_config.get("provider")
        )
        if resolved is None:
            raise ValueError("Provider must be specified")
        return resolved

    def _build_provider_kwargs_and_model(
        self,
        provider: str,
        model: Optional[str],
        kwargs: Dict[str, Any],
        agent_config: Dict[str, Any],
        providers_config: Dict[str, Any],
        keys_config: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional[str]]:
        # Determine provider block
        provider_block: Dict[str, Any] = {}
        if isinstance(agent_config, dict) and provider:
            maybe_block = agent_config.get(provider)
            if isinstance(maybe_block, dict):
                provider_block = maybe_block
        if not provider_block and providers_config:
            provider_block = providers_config.get(provider, {}) or {}

        merged_kwargs: Dict[str, Any] = {}
        if provider_block:
            for k, v in provider_block.items():
                if k != "model":
                    merged_kwargs[k] = v

        if agent_config and isinstance(agent_config.get("config", {}), dict):
            merged_kwargs.update(agent_config.get("config", {}) or {})

        merged_kwargs.update(kwargs)

        # Resolve model
        final_model = (
            model
            or (agent_config.get("model") if isinstance(agent_config, dict) else None)
            or provider_block.get("model")
        )

        # Backfill keys
        if keys_config:
            provider_to_key = {
                "openai": ("api_key", "openai_api_key"),
                "anthropic": ("api_key", "anthropic_api_key"),
                "gemini": ("api_key", "google_api_key"),
                "google": ("api_key", "google_api_key"),
                "doubao": ("api_key", "doubao_api_key"),
                "huggingface": ("hf_token", "huggingface_token"),
            }
            mapping = provider_to_key.get(provider)
            if mapping:
                target_kw, key_name = mapping
                if target_kw not in merged_kwargs and key_name in keys_config:
                    merged_kwargs[target_kw] = keys_config[key_name]

        return merged_kwargs, final_model

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
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Example:
            >>> response = agent.generate("Explain quantum computing")
            >>> print(response.content)
            >>> # JSON mode
            >>> response = agent.generate("List 3 items", response_format="json")
        """
        return self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )

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
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" for JSON mode, None for text)
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ...     {"role": "user", "content": "What's the weather?"},
            ... ]
            >>> response = agent.chat(messages)
            >>> # JSON mode
            >>> response = agent.chat(messages, response_format="json")
        """
        return self.provider.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )

    def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Switch to a different provider.

        Args:
            provider: New provider name
            model: New model name (optional)
            **kwargs: Provider-specific configuration

        Example:
            >>> agent = AgentWrapper(provider="openai", model="gpt-4")
            >>> agent.switch_provider("ollama", model="llama2")
        """
        self.provider = LLMProviderFactory.create_provider(provider, model, **kwargs)
        self.provider_name = provider
        self.logger.info(
            f"Switched to {provider} provider with model={self.provider.model}"
        )

    @property
    def model(self) -> str:
        """Get current model name."""
        return self.provider.model

    @property
    def provider_type(self) -> str:
        """Get current provider name."""
        return self.provider_name

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AgentWrapper(provider={self.provider_name}, model={self.provider.model})"
        )
