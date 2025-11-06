"""Ollama provider implementation for local models."""

from __future__ import annotations

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
