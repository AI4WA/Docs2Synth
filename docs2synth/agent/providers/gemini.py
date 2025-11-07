"""Google Gemini provider implementation."""

from __future__ import annotations

import io
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

    def _prepare_image(self, image: Any) -> Any:
        """Prepare image for Gemini API.

        Args:
            image: Image in various formats (PIL Image, file path, bytes)

        Returns:
            PIL Image object ready for Gemini API

        Raises:
            ImportError: If PIL is not installed
            ValueError: If image type is unsupported
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError(
                "PIL (Pillow) is required for image support. Install with: pip install Pillow"
            )

        if isinstance(image, PILImage.Image):
            return image
        elif isinstance(image, str):
            return PILImage.open(image)
        elif isinstance(image, bytes):
            return PILImage.open(io.BytesIO(image))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _build_generation_config(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs: Any,
    ) -> Optional[genai.types.GenerationConfig]:
        """Build generation config for Gemini API.

        Args:
            temperature: Temperature setting
            max_tokens: Maximum output tokens
            **kwargs: Additional generation parameters

        Returns:
            GenerationConfig object or None
        """
        default_keys = {"temperature", "max_output_tokens", "top_p", "top_k"}
        generation_config = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }
        if temperature is not None:
            generation_config["temperature"] = temperature
        generation_config["max_output_tokens"] = max_tokens or generation_config.get(
            "max_output_tokens"
        )
        # Filter out None values from kwargs
        generation_config.update({k: v for k, v in kwargs.items() if v is not None})

        return (
            genai.types.GenerationConfig(**generation_config)
            if generation_config
            else None
        )

    def _extract_usage_metadata(self, response: Any) -> Dict[str, int]:
        """Extract usage metadata from Gemini response.

        Args:
            response: Gemini API response object

        Returns:
            Dictionary with token usage information
        """
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
        return usage

    def _extract_response_content(self, response: Any) -> tuple[str, Optional[int]]:
        """Extract content and finish reason from Gemini response.

        Args:
            response: Gemini API response object

        Returns:
            Tuple of (content, finish_reason)
        """
        finish_reason = None
        content = ""
        if not response.candidates:
            return content, finish_reason

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)

        # finish_reason 2 = SAFETY (content filtered)
        # finish_reason 3 = RECITATION (content recitation)
        # finish_reason 4 = OTHER (other reasons)
        if finish_reason in (2, 3, 4):
            content = self._handle_filtered_content(response, candidate, finish_reason)
        else:
            content = self._extract_text_content(response)

        return content, finish_reason

    def _handle_filtered_content(
        self, response: Any, candidate: Any, finish_reason: int
    ) -> str:
        """Handle content that was filtered by safety filters.

        Args:
            response: Gemini API response object
            candidate: Response candidate object
            finish_reason: Finish reason code

        Returns:
            Content string (may be filtered message)
        """
        try:
            return response.text
        except ValueError:
            # Content was filtered, provide a helpful error message
            safety_ratings = getattr(candidate, "safety_ratings", [])
            safety_info = (
                ", ".join(
                    [
                        f"{getattr(r, 'category', 'UNKNOWN')}: {getattr(r, 'probability', 'UNKNOWN')}"
                        for r in safety_ratings
                    ]
                )
                if safety_ratings
                else "unknown"
            )
            content = (
                f"[Content filtered by safety filters. "
                f"Finish reason: {finish_reason}, Safety ratings: {safety_info}]"
            )
            logger.warning(
                f"Gemini response was filtered. Finish reason: {finish_reason}, "
                f"Safety ratings: {safety_info}"
            )
            return content

    def _extract_text_content(self, response: Any) -> str:
        """Extract text content from normal response.

        Args:
            response: Gemini API response object

        Returns:
            Content string
        """
        try:
            return response.text
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to extract text from Gemini response: {e}")
            return "[Failed to extract response content]"

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
        # Extract image from kwargs
        image = kwargs.pop("image", None)

        # Build content list for Gemini API
        # Gemini supports multimodal input: [text, image, text, ...]
        content_parts = []

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Gemini JSON mode via prompt engineering
        if response_format == "json":
            full_prompt = f"{full_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

        content_parts.append(full_prompt)

        # Add image if provided
        if image is not None:
            content_parts.append(self._prepare_image(image))

        generation_config = self._build_generation_config(
            temperature, max_tokens, **kwargs
        )

        response = self.client.generate_content(
            content_parts,
            generation_config=generation_config,
        )

        usage = self._extract_usage_metadata(response)
        content, finish_reason = self._extract_response_content(response)

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            metadata={"finish_reason": finish_reason},
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

        # Extract image from kwargs
        image = kwargs.pop("image", None)

        # Build content parts for Gemini API
        content_parts = [last_message]
        if image is not None:
            content_parts.append(self._prepare_image(image))

        generation_config = self._build_generation_config(
            temperature, max_tokens, **kwargs
        )

        response = chat.send_message(
            content_parts,
            generation_config=generation_config,
        )

        usage = self._extract_usage_metadata(response)
        content, finish_reason = self._extract_response_content(response)

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            metadata={"finish_reason": finish_reason},
        )
