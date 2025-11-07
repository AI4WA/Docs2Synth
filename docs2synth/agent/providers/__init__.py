"""LLM provider implementations."""

from __future__ import annotations

from docs2synth.agent.providers.anthropic import AnthropicProvider
from docs2synth.agent.providers.doubao import DoubaoProvider
from docs2synth.agent.providers.gemini import GeminiProvider
from docs2synth.agent.providers.huggingface import HuggingFaceProvider
from docs2synth.agent.providers.ollama import OllamaProvider
from docs2synth.agent.providers.openai import OpenAIProvider

__all__ = [
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "DoubaoProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
]

PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "doubao": DoubaoProvider,
    "ollama": OllamaProvider,
    "huggingface": HuggingFaceProvider,
}
