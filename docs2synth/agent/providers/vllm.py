"""vLLM provider implementation for local high-performance LLM inference."""

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


class VLLMProvider(BaseLLMProvider):
    """vLLM provider for local high-performance LLM inference.

    Connects to a running vLLM OpenAI-compatible API server.

    Server mode:
        Start vLLM server: docs2synth agent vllm-server
        Or manually: python -m vllm.entrypoints.openai.api_server --model <model_name>
        Default endpoint: http://localhost:8000/v1

    See https://github.com/vllm-project/vllm for more information.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize vLLM provider in server mode.

        Args:
            model: Model name/identifier
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            api_key: API key (optional, vLLM server typically doesn't require this)
            **kwargs: Additional parameters for OpenAI client
        """
        # Persist full provider config (defaults for generation)
        super().__init__(model, **kwargs)

        # Server mode: use OpenAI-compatible HTTP API
        if OpenAI is None:
            raise ImportError(
                "openai package is required for vLLM provider. Install with: pip install openai"
            )

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
        # Set base_url (vLLM default endpoint)
        if "base_url" not in client_kwargs:
            client_kwargs["base_url"] = base_url
        # vLLM typically doesn't require API keys, but OpenAI client requires the parameter
        # Use provided api_key or a dummy value (vLLM server will ignore it)
        if "api_key" not in client_kwargs:
            client_kwargs["api_key"] = api_key if api_key else "EMPTY"

        logger.info(f"Connecting to vLLM server at {client_kwargs['base_url']}")
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
        """Generate text using vLLM server mode."""
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
        """Chat completion with message history using vLLM server mode."""
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
