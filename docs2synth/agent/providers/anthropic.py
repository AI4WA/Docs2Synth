"""Anthropic (Claude) provider implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            model: Model name (e.g., 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229')
            api_key: Anthropic API key (from config or ANTHROPIC_API_KEY env var)
            **kwargs: Additional Anthropic client parameters
        """
        if Anthropic is None:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        # Persist full provider config (generation defaults)
        super().__init__(model, **kwargs)

        # Build Anthropic client with ONLY supported client options
        allowed_client_keys = {
            "api_key",
            "base_url",
            "timeout",
            "max_retries",
            "http_client",
        }
        client_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k in allowed_client_keys
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        # If api_key is None, Anthropic() falls back to ANTHROPIC_API_KEY env var

        self.client = Anthropic(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using Anthropic API."""
        # Anthropic Messages API expects content blocks
        user_content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": user_content}]

        # Anthropic supports JSON mode via response_format parameter
        if response_format == "json":
            # Add JSON schema requirement to system prompt or use response_format
            # Note: Anthropic uses a different approach - we'll add JSON instruction to prompt
            if system_prompt:
                system_prompt = f"{system_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text."
            else:
                system_prompt = (
                    "You must respond with valid JSON only, no additional text."
                )
            # Anthropic API doesn't have direct JSON mode, so we rely on prompt engineering

        # Merge default generation kwargs from provider config
        default_keys = {"temperature", "max_tokens", "top_p", "stop_sequences"}
        merged_kwargs: Dict[str, Any] = {
            k: v for k, v in self.config.items() if k in default_keys
        }
        if temperature is not None:
            merged_kwargs["temperature"] = temperature
        merged_kwargs["max_tokens"] = (
            max_tokens or merged_kwargs.get("max_tokens") or 4096
        )
        merged_kwargs.update(kwargs)

        # Anthropic newer models don't allow both temperature and top_p
        # Prefer temperature if both are present
        if "temperature" in merged_kwargs and "top_p" in merged_kwargs:
            merged_kwargs.pop("top_p")

        system_blocks = (
            [{"type": "text", "text": system_prompt}] if system_prompt else None
        )

        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **merged_kwargs,
        }
        if system_blocks is not None:
            create_kwargs["system"] = system_blocks

        response = self.client.messages.create(**create_kwargs)

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage=usage,
            metadata={"stop_reason": response.stop_reason},
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with message history."""
        # Extract system prompt if present
        system_prompt = None
        formatted_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                formatted_messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}],
                    }
                )

        # Anthropic supports JSON mode via prompt engineering
        if response_format == "json":
            if system_prompt:
                system_prompt = f"{system_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text."
            else:
                system_prompt = (
                    "You must respond with valid JSON only, no additional text."
                )

        default_keys = {"temperature", "max_tokens", "top_p", "stop_sequences"}
        merged_kwargs: Dict[str, Any] = {
            k: v for k, v in self.config.items() if k in default_keys
        }
        if temperature is not None:
            merged_kwargs["temperature"] = temperature
        merged_kwargs["max_tokens"] = (
            max_tokens or merged_kwargs.get("max_tokens") or 4096
        )
        merged_kwargs.update(kwargs)

        # Anthropic newer models don't allow both temperature and top_p
        # Prefer temperature if both are present
        if "temperature" in merged_kwargs and "top_p" in merged_kwargs:
            merged_kwargs.pop("top_p")

        system_blocks = (
            [{"type": "text", "text": system_prompt}] if system_prompt else None
        )

        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            **merged_kwargs,
        }
        if system_blocks is not None:
            create_kwargs["system"] = system_blocks

        response = self.client.messages.create(**create_kwargs)

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage=usage,
            metadata={"stop_reason": response.stop_reason},
        )
