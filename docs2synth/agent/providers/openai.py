"""OpenAI provider implementation."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(
        self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs: Any
    ):
        """Initialize OpenAI provider.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key (from config or OPENAI_API_KEY env var)
            **kwargs: Additional OpenAI client parameters
        """
        if OpenAI is None:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        # Persist full provider config (defaults for generation)
        super().__init__(model, **kwargs)

        # Build OpenAI client with ONLY supported client options
        allowed_client_keys = {
            "api_key",
            "base_url",
            "organization",
            "project",
            "timeout",
            "http_client",
        }
        client_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k in allowed_client_keys
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            logger.info(
                "OpenAI api_key not provided via config; falling back to OPENAI_API_KEY env var if set"
            )

        self.client = OpenAI(**client_kwargs)

    def _encode_image(self, image: Any) -> str:
        """Encode PIL Image to base64 string."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError(
                "PIL (Pillow) is required for image support. Install with: pip install Pillow"
            )

        if not isinstance(image, PILImage.Image):
            # If it's already a file path or URL, return as-is
            if isinstance(image, (str, bytes)):
                return str(image)
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert PIL Image to base64
        buffer = io.BytesIO()
        # Save as JPEG (most compatible format)
        if image.mode in ("RGBA", "LA", "P"):
            # Convert RGBA/LA/P to RGB for JPEG
            rgb_image = PILImage.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            rgb_image.paste(
                image, mask=image.split()[-1] if image.mode == "RGBA" else None
            )
            image = rgb_image
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    def _build_message_content(self, text: str, image: Optional[Any] = None) -> Any:
        """Build message content with optional image."""
        if image is None:
            return text

        # For vision models, content must be a list
        image_url = self._encode_image(image)
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using OpenAI API."""
        # Extract image from kwargs
        image = kwargs.pop("image", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = self._build_message_content(prompt, image)
        messages.append({"role": "user", "content": user_content})

        # Merge default generation kwargs from provider config
        default_keys = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "logit_bias",
            "seed",
        }
        merged_kwargs: Dict[str, Any] = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }
        # Call-time args override defaults
        if temperature is not None:
            merged_kwargs["temperature"] = temperature
        if max_tokens is not None:
            merged_kwargs["max_tokens"] = max_tokens
        # Filter out None values from kwargs
        merged_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        # Support JSON mode
        if response_format == "json":
            merged_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **merged_kwargs,
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": (
                response.usage.completion_tokens if response.usage else 0
            ),
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            usage=usage,
            metadata={"finish_reason": response.choices[0].finish_reason},
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
        # Extract image from kwargs
        image = kwargs.pop("image", None)

        # Convert message format if needed
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # If this is the last user message and we have an image, add image to content
            if (
                role == "user"
                and image is not None
                and msg == messages[-1]
                and isinstance(content, str)
            ):
                content = self._build_message_content(content, image)
            elif isinstance(content, list):
                # Content is already in the correct format (e.g., from a previous call)
                content = content
            else:
                # Regular text content
                content = content

            formatted_messages.append({"role": role, "content": content})

        # Merge default generation kwargs from provider config
        default_keys = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "logit_bias",
            "seed",
        }
        merged_kwargs: Dict[str, Any] = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }
        if temperature is not None:
            merged_kwargs["temperature"] = temperature
        if max_tokens is not None:
            merged_kwargs["max_tokens"] = max_tokens
        # Filter out None values from kwargs
        merged_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        # Support JSON mode
        if response_format == "json":
            merged_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            **merged_kwargs,
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": (
                response.usage.completion_tokens if response.usage else 0
            ),
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            usage=usage,
            metadata={"finish_reason": response.choices[0].finish_reason},
        )
