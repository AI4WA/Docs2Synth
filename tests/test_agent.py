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


# ============================================================================
# Provider initialization tests
# ============================================================================


def test_openai_provider_initialization():
    """Test OpenAI provider initialization without API calls."""
    from docs2synth.agent.providers.openai import OpenAIProvider

    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="test-key")
    assert provider.model == "gpt-3.5-turbo"
    # api_key is not stored in config, it's passed to client
    assert hasattr(provider, "client")


def test_anthropic_provider_initialization():
    """Test Anthropic provider initialization without API calls."""
    from docs2synth.agent.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="test-key")
    assert provider.model == "claude-3-5-sonnet-20241022"
    assert hasattr(provider, "client")


def test_ollama_provider_initialization():
    """Test Ollama provider initialization."""
    from docs2synth.agent.providers.ollama import OllamaProvider

    # Ollama might have different initialization
    try:
        provider = OllamaProvider(model="llama2", base_url="http://localhost:11434")
        assert provider.model == "llama2"
        assert hasattr(provider, "client")
    except Exception:
        # Skip if ollama has different requirements
        pytest.skip("Ollama provider has different initialization requirements")


def test_gemini_provider_initialization():
    """Test Gemini provider initialization."""
    from docs2synth.agent.providers.gemini import GeminiProvider

    provider = GeminiProvider(model="gemini-pro", api_key="test-key")
    assert provider.model == "gemini-pro"


def test_doubao_provider_initialization():
    """Test Doubao provider initialization."""
    from docs2synth.agent.providers.doubao import DoubaoProvider

    provider = DoubaoProvider(model="doubao-pro", api_key="test-key")
    assert provider.model == "doubao-pro"


def test_huggingface_provider_initialization():
    """Test HuggingFace provider initialization."""
    from docs2synth.agent.providers.huggingface import HuggingFaceProvider

    # HuggingFace requires specific setup, skip if not available
    try:
        provider = HuggingFaceProvider(
            model="gpt2", hf_token="test-token", use_inference_api=False
        )
        assert provider.model == "gpt2"
    except (ImportError, OSError, Exception):
        pytest.skip("HuggingFace provider requires specific dependencies")


# ============================================================================
# Provider mock tests
# ============================================================================


def test_openai_provider_generate_with_mock(monkeypatch):
    """Test OpenAI generate with mocked response."""
    from unittest.mock import MagicMock, Mock

    from docs2synth.agent.providers.openai import OpenAIProvider

    # Create mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Hello, World!"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    # Create provider and mock client
    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    provider.client = mock_client

    # Test generate
    response = provider.generate("Test prompt")
    assert response.content == "Hello, World!"
    assert response.model == "gpt-3.5-turbo"
    assert response.usage["total_tokens"] == 15


def test_openai_provider_chat_with_mock(monkeypatch):
    """Test OpenAI chat with mocked response."""
    from unittest.mock import MagicMock, Mock

    from docs2synth.agent.providers.openai import OpenAIProvider

    # Create mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Chat response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 30

    # Create provider and mock client
    provider = OpenAIProvider(model="gpt-4", api_key="test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    provider.client = mock_client

    # Test chat
    messages = [{"role": "user", "content": "Hello"}]
    response = provider.chat(messages)
    assert response.content == "Chat response"
    assert response.usage["total_tokens"] == 30


def test_openai_provider_json_mode():
    """Test OpenAI JSON mode request format."""
    from unittest.mock import MagicMock, Mock

    from docs2synth.agent.providers.openai import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4", api_key="test-key")

    # Mock the client
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = '{"key": "value"}'
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    provider.client = mock_client

    # Test JSON mode
    provider.generate("Generate JSON", response_format="json")

    # Verify response_format was passed
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_anthropic_provider_generate_with_mock():
    """Test Anthropic generate with mocked response."""
    from unittest.mock import MagicMock, Mock

    from docs2synth.agent.providers.anthropic import AnthropicProvider

    # Create mock response
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Claude response"
    mock_response.stop_reason = "end_turn"
    mock_response.usage = Mock()
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 10

    # Create provider and mock client
    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="test-key")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    provider.client = mock_client

    # Test generate
    response = provider.generate("Test prompt", system_prompt="You are helpful")
    assert response.content == "Claude response"
    assert response.usage["total_tokens"] == 25


def test_anthropic_provider_json_mode():
    """Test Anthropic JSON mode adds to system prompt."""
    from unittest.mock import MagicMock, Mock

    from docs2synth.agent.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="test-key")

    # Mock the client
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = '{"result": "success"}'
    mock_response.stop_reason = "end_turn"
    mock_response.usage = Mock()
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 10

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    provider.client = mock_client

    # Test JSON mode
    provider.generate("Generate JSON", response_format="json")

    # Verify system prompt contains JSON instruction
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" in call_kwargs
    system_text = call_kwargs["system"][0]["text"]
    assert "JSON" in system_text


# ============================================================================
# Base provider tests
# ============================================================================


def test_base_provider_is_abstract():
    """Test that BaseLLMProvider is an abstract class."""
    # BaseLLMProvider cannot be instantiated directly
    with pytest.raises(TypeError, match="abstract"):
        BaseLLMProvider(model="test-model")


# ============================================================================
# Factory tests
# ============================================================================


def test_factory_create_provider_with_model():
    """Test factory create_provider with model specified."""
    from docs2synth.agent.providers.openai import OpenAIProvider

    provider = LLMProviderFactory.create_provider(
        "openai", model="gpt-4", api_key="test-key"
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4"


def test_factory_create_provider_without_model():
    """Test factory create_provider without model (uses default)."""
    from docs2synth.agent.providers.openai import OpenAIProvider

    provider = LLMProviderFactory.create_provider("openai", api_key="test-key")
    assert isinstance(provider, OpenAIProvider)
    # Should use default model
    assert provider.model == "gpt-3.5-turbo"


def test_factory_create_provider_case_insensitive():
    """Test that provider names are case insensitive."""
    provider1 = LLMProviderFactory.create_provider("OpenAI", api_key="test-key")
    provider2 = LLMProviderFactory.create_provider("OPENAI", api_key="test-key")
    provider3 = LLMProviderFactory.create_provider("openai", api_key="test-key")

    from docs2synth.agent.providers.openai import OpenAIProvider

    assert isinstance(provider1, OpenAIProvider)
    assert isinstance(provider2, OpenAIProvider)
    assert isinstance(provider3, OpenAIProvider)


def test_factory_invalid_provider_raises_error():
    """Test that invalid provider raises clear error."""
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMProviderFactory.create_provider("invalid_provider_name")


def test_factory_error_message_shows_available_providers():
    """Test that error message lists available providers."""
    try:
        LLMProviderFactory.create_provider("nonexistent")
    except ValueError as e:
        error_msg = str(e)
        assert "Available providers:" in error_msg
        assert "openai" in error_msg.lower()
        assert "anthropic" in error_msg.lower()


def test_factory_create_from_config():
    """Test factory create_from_config method."""
    from docs2synth.agent.providers.openai import OpenAIProvider

    config = {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "test-key",
        "temperature": 0.8,
    }

    provider = LLMProviderFactory.create_from_config(config.copy())
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4"
    assert provider.config.get("temperature") == 0.8


def test_factory_create_from_config_missing_provider():
    """Test that create_from_config requires provider key."""
    config = {
        "model": "gpt-4",
        "api_key": "test-key",
    }

    with pytest.raises(ValueError, match="Configuration must include 'provider'"):
        LLMProviderFactory.create_from_config(config)


def test_factory_creates_all_registered_providers():
    """Test that factory can create all registered providers."""
    from docs2synth.agent.providers import PROVIDER_REGISTRY

    test_configs = {
        "openai": {"api_key": "test-key"},
        "anthropic": {"api_key": "test-key"},
        "gemini": {"api_key": "test-key"},
        "doubao": {"api_key": "test-key"},
    }

    # Test only the providers that don't require special setup
    for provider_name in ["openai", "anthropic", "gemini", "doubao"]:
        if provider_name in PROVIDER_REGISTRY:
            kwargs = test_configs.get(provider_name, {})
            provider = LLMProviderFactory.create_provider(provider_name, **kwargs)
            assert provider is not None
            assert isinstance(provider, BaseLLMProvider)
