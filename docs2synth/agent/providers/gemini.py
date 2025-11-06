"""Google Gemini provider implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def __init__(
        self, model: str = "gemini-pro", api_key: Optional[str] = None, **kwargs: Any
    ):
        """Initialize Gemini provider.

        Args:
            model: Model name (e.g., 'gemini-pro', 'gemini-pro-vision')
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            **kwargs: Additional Gemini configuration
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package is required. Install with: pip install google-generativeai"
            )

        # Persist generation defaults in provider config
        super().__init__(model, **kwargs)
        # Try to get API key from parameter (from config) first, then fallback to env var
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set it in config.yml (agent.config.api_key) or set GOOGLE_API_KEY env var."
            )

        genai.configure(api_key=api_key)
        # Do not pass generation kwargs into model constructor; store them in self.config
        self.client = genai.GenerativeModel(model_name=self.model)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using Gemini API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Gemini JSON mode via prompt engineering
        if response_format == "json":
            full_prompt = f"{full_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

        default_keys = {"temperature", "max_output_tokens", "top_p", "top_k"}
        generation_config = {k: v for k, v in self.config.items() if k in default_keys}
        if temperature is not None:
            generation_config["temperature"] = temperature
        generation_config["max_output_tokens"] = max_tokens or generation_config.get(
            "max_output_tokens"
        )
        generation_config.update(kwargs)

        response = self.client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(**generation_config),
        )

        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
                "total_tokens": getattr(
                    response.usage_metadata, "total_token_count", 0
                ),
            }

        return LLMResponse(
            content=response.text,
            model=self.model,
            usage=usage,
            metadata={
                "finish_reason": getattr(response.candidates[0], "finish_reason", None)
            },
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
        # Gemini uses a chat history format
        chat = self.client.start_chat(history=[])

        # Build conversation history
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat.send_message(msg["content"])
            elif msg["role"] == "assistant":
                # Gemini doesn't support explicit assistant messages in history
                # We'll skip them or handle them differently
                pass

        # Send the last message
        last_message = messages[-1]["content"]
        # Gemini JSON mode via prompt engineering
        if response_format == "json":
            last_message = f"{last_message}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

        default_keys = {"temperature", "max_output_tokens", "top_p", "top_k"}
        generation_config = {k: v for k, v in self.config.items() if k in default_keys}
        if temperature is not None:
            generation_config["temperature"] = temperature
        generation_config["max_output_tokens"] = max_tokens or generation_config.get(
            "max_output_tokens"
        )
        generation_config.update(kwargs)

        response = chat.send_message(
            last_message,
            generation_config=genai.types.GenerationConfig(**generation_config),
        )

        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
                "total_tokens": getattr(
                    response.usage_metadata, "total_token_count", 0
                ),
            }

        return LLMResponse(
            content=response.text,
            model=self.model,
            usage=usage,
            metadata={
                "finish_reason": getattr(response.candidates[0], "finish_reason", None)
            },
        )
