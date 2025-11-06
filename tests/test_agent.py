"""Tests for agent module."""

from __future__ import annotations

import pytest

from docs2synth.agent import (
    AgentWrapper,
    BaseLLMProvider,
    LLMProviderFactory,
    LLMResponse,
    QAGenerator,
)


def test_imports():
    """Test that all agent modules can be imported."""
    assert AgentWrapper is not None
    assert BaseLLMProvider is not None
    assert LLMProviderFactory is not None
    assert LLMResponse is not None
    assert QAGenerator is not None


def test_llm_response():
    """Test LLMResponse class."""
    response = LLMResponse(
        content="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    assert response.content == "Test response"
    assert response.model == "test-model"
    assert response.usage["total_tokens"] == 15
    assert str(response) == "Test response"


def test_llm_provider_factory():
    """Test LLMProviderFactory."""
    # Test that factory can list available providers
    from docs2synth.agent.providers import PROVIDER_REGISTRY

    assert len(PROVIDER_REGISTRY) > 0
    assert "openai" in PROVIDER_REGISTRY
    assert "anthropic" in PROVIDER_REGISTRY
    assert "ollama" in PROVIDER_REGISTRY

    # Test that invalid provider raises error
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMProviderFactory.create_provider("invalid_provider")


def test_agent_wrapper_requires_provider():
    """Test that AgentWrapper requires provider."""
    # Should raise error if no provider specified and no config
    with pytest.raises(ValueError, match="Provider must be specified"):
        AgentWrapper()


def test_qa_generator_requires_provider():
    """Test that QAGenerator requires provider."""
    # Should raise error if no provider specified
    with pytest.raises(
        ValueError, match="Either 'agent' or 'provider' must be specified"
    ):
        QAGenerator()


@pytest.mark.skip(reason="Requires actual API keys or local models")
def test_openai_provider_integration():
    """Integration test for OpenAI provider (requires API key)."""
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    provider = LLMProviderFactory.create_provider(
        "openai", model="gpt-3.5-turbo", api_key=api_key
    )

    response = provider.generate("Say 'Hello, World!'")
    assert isinstance(response, LLMResponse)
    assert len(response.content) > 0
    assert "Hello" in response.content or "hello" in response.content.lower()


@pytest.mark.skip(reason="Requires Ollama running locally")
def test_ollama_provider_integration():
    """Integration test for Ollama provider (requires local Ollama)."""
    try:
        provider = LLMProviderFactory.create_provider("ollama", model="llama2")
        response = provider.generate("Say 'Hello'")
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
    except Exception as e:
        if "Connection" in str(e) or "refused" in str(e).lower():
            pytest.skip("Ollama not running locally")
        raise
