"""Ollama provider implementation for local models."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM models.

    Requires Ollama to be running locally. Default endpoint: http://localhost:11434
    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        """Initialize Ollama provider.

        Args:
            model: Model name (e.g., 'llama2', 'mistral', 'codellama')
            base_url: Ollama server URL
            **kwargs: Additional configuration
        """
        if requests is None:
            raise ImportError(
                "requests package is required. Install with: pip install requests"
            )

        # Persist generation defaults; filter out client-only args
        client_keys = {"base_url", "host", "timeout"}
        client_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys()) if k in client_keys
        }
        if "host" in client_kwargs and "base_url" not in client_kwargs:
            client_kwargs["base_url"] = client_kwargs.pop("host")
        if client_kwargs.get("base_url"):
            base_url = client_kwargs["base_url"]
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")

    def _encode_image(self, image: Any) -> str:
        """Encode image to base64 string for Ollama API.

        Args:
            image: PIL.Image, file path, or bytes

        Returns:
            Base64 encoded string (without data URI prefix)
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError(
                "PIL (Pillow) is required for image support. Install with: pip install Pillow"
            )

        # If it's a PIL Image, convert to base64
        if isinstance(image, PILImage.Image):
            buffer = io.BytesIO()
            # Save as JPEG for compatibility
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
            return base64.b64encode(image_bytes).decode("utf-8")
        # If it's a file path, load and encode
        elif isinstance(image, str):
            # Check if it's a URL (Ollama doesn't support URLs directly)
            if image.startswith(("http://", "https://")):
                raise ValueError(
                    "Ollama API does not support image URLs directly. "
                    "Please download the image first or pass a PIL Image."
                )
            # It's a file path
            with open(image, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("utf-8")
        # If it's already bytes
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using Ollama API."""
        # Extract image from kwargs
        image = kwargs.pop("image", None)

        # Ollama JSON mode via prompt engineering
        if response_format == "json":
            prompt = f"{prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

        # Merge defaults from self.config
        default_keys = {
            "temperature",
            "top_p",
            "top_k",
            "num_predict",
            "repeat_penalty",
            "stop",
        }
        merged_options = {k: v for k, v in self.config.items() if k in default_keys}
        if temperature is not None:
            merged_options["temperature"] = temperature
        if max_tokens is not None:
            merged_options["num_predict"] = max_tokens
        elif "num_predict" not in merged_options:
            merged_options["num_predict"] = None
        merged_options.update(kwargs)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": merged_options,
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Add image if provided (Ollama expects base64-encoded images in "images" array)
        if image is not None:
            payload["images"] = [self._encode_image(image)]

        # Extract timeout from options if present
        timeout = merged_options.pop("timeout", 300)
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0)
            + data.get("eval_count", 0),
        }

        return LLMResponse(
            content=data.get("response", ""),
            model=self.model,
            usage=usage,
            metadata={"done": data.get("done", False)},
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

        # Ollama JSON mode via prompt engineering
        if response_format == "json" and messages:
            # Add JSON instruction to the last user message
            last_message = messages[-1]
            if last_message.get("role") == "user":
                messages = messages.copy()
                messages[-1] = {
                    "role": "user",
                    "content": f"{last_message['content']}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown.",
                }

        default_keys = {
            "temperature",
            "top_p",
            "top_k",
            "num_predict",
            "repeat_penalty",
            "stop",
        }
        merged_options = {k: v for k, v in self.config.items() if k in default_keys}
        if temperature is not None:
            merged_options["temperature"] = temperature
        if max_tokens is not None:
            merged_options["num_predict"] = max_tokens
        elif "num_predict" not in merged_options:
            merged_options["num_predict"] = None
        merged_options.update(kwargs)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": merged_options,
        }

        # Add image if provided (Ollama expects base64-encoded images in "images" array)
        if image is not None:
            payload["images"] = [self._encode_image(image)]

        # Extract timeout from options if present
        timeout = merged_options.pop("timeout", 300)
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0)
            + data.get("eval_count", 0),
        }

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self.model,
            usage=usage,
            metadata={"done": data.get("done", False)},
        )
