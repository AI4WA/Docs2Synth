"""Tests for LLM provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from docs2synth.agent.base import BaseLLMProvider
from docs2synth.agent.factory import LLMProviderFactory


@pytest.fixture
def mock_provider_class():
    """Create a mock provider class."""
    mock_class = MagicMock(spec=BaseLLMProvider)
    mock_instance = MagicMock(spec=BaseLLMProvider)
    mock_class.return_value = mock_instance
    mock_class.__name__ = "MockProvider"
    return mock_class


@patch("docs2synth.agent.factory.PROVIDER_REGISTRY")
def test_create_provider_success(mock_registry, mock_provider_class) -> None:
    """Test successful provider creation."""
    mock_registry.__contains__ = lambda self, key: key == "testprovider"
    mock_registry.__getitem__ = lambda self, key: mock_provider_class

    provider = LLMProviderFactory.create_provider(
        "testprovider", model="test-model", api_key="test-key"
    )

    assert provider is not None
    mock_provider_class.assert_called_once_with(model="test-model", api_key="test-key")


@patch("docs2synth.agent.factory.PROVIDER_REGISTRY")
def test_create_provider_unsupported(mock_registry) -> None:
    """Test creating unsupported provider raises ValueError."""
    mock_registry.__contains__ = lambda self, key: False
    mock_registry.keys.return_value = ["openai", "anthropic", "ollama"]

    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMProviderFactory.create_provider("unsupported_provider")


@patch("docs2synth.agent.factory.PROVIDER_REGISTRY")
def test_create_provider_case_insensitive(mock_registry, mock_provider_class) -> None:
    """Test provider name is case-insensitive."""
    mock_registry.__contains__ = lambda self, key: key == "testprovider"
    mock_registry.__getitem__ = lambda self, key: mock_provider_class

    # Test with uppercase
    provider1 = LLMProviderFactory.create_provider("TESTPROVIDER")
    assert provider1 is not None

    # Test with mixed case
    provider2 = LLMProviderFactory.create_provider("TestProvider")
    assert provider2 is not None


@patch("docs2synth.agent.factory.PROVIDER_REGISTRY")
def test_create_provider_strips_whitespace(mock_registry, mock_provider_class) -> None:
    """Test provider name whitespace is stripped."""
    mock_registry.__contains__ = lambda self, key: key == "testprovider"
    mock_registry.__getitem__ = lambda self, key: mock_provider_class

    provider = LLMProviderFactory.create_provider("  testprovider  ")
    assert provider is not None


@patch("docs2synth.agent.factory.PROVIDER_REGISTRY")
def test_create_provider_without_model(mock_registry, mock_provider_class) -> None:
    """Test creating provider without model uses default."""
    mock_registry.__contains__ = lambda self, key: key == "testprovider"
    mock_registry.__getitem__ = lambda self, key: mock_provider_class

    LLMProviderFactory.create_provider("testprovider", api_key="test-key")

    # Should be called once
    assert mock_provider_class.call_count == 1
    # Model should not be in kwargs when None
    call_kwargs = mock_provider_class.call_args[1]
    assert "model" not in call_kwargs
    assert call_kwargs["api_key"] == "test-key"


@patch("docs2synth.agent.factory.PROVIDER_REGISTRY")
def test_create_provider_with_model(mock_registry, mock_provider_class) -> None:
    """Test creating provider with explicit model."""
    mock_registry.__contains__ = lambda self, key: key == "testprovider"
    mock_registry.__getitem__ = lambda self, key: mock_provider_class

    LLMProviderFactory.create_provider("testprovider", model="custom-model")

    call_kwargs = mock_provider_class.call_args[1]
    assert call_kwargs["model"] == "custom-model"


@patch("docs2synth.agent.factory.PROVIDER_REGISTRY")
def test_create_provider_with_extra_kwargs(mock_registry, mock_provider_class) -> None:
    """Test creating provider with additional kwargs."""
    mock_registry.__contains__ = lambda self, key: key == "testprovider"
    mock_registry.__getitem__ = lambda self, key: mock_provider_class

    LLMProviderFactory.create_provider(
        "testprovider",
        model="model",
        temperature=0.7,
        max_tokens=100,
        custom_param="value",
    )

    call_kwargs = mock_provider_class.call_args[1]
    assert call_kwargs["model"] == "model"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["custom_param"] == "value"


def test_create_from_config_success() -> None:
    """Test creating provider from config dict."""
    with patch(
        "docs2synth.agent.factory.LLMProviderFactory.create_provider"
    ) as mock_create:
        mock_provider = MagicMock()
        mock_create.return_value = mock_provider

        config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
            "temperature": 0.5,
        }

        provider = LLMProviderFactory.create_from_config(config)

        # Should call create_provider with provider extracted and passed separately
        mock_create.assert_called_once_with(
            "openai", model="gpt-4", api_key="test-key", temperature=0.5
        )
        assert provider == mock_provider


def test_create_from_config_missing_provider() -> None:
    """Test creating from config without provider key raises ValueError."""
    config = {"model": "gpt-4", "api_key": "test-key"}

    with pytest.raises(ValueError, match="must include 'provider' key"):
        LLMProviderFactory.create_from_config(config)


def test_create_from_config_provider_extracted() -> None:
    """Test that provider key is removed from config before passing to create_provider."""
    with patch("docs2synth.agent.factory.LLMProviderFactory.create_provider"):
        config = {
            "provider": "anthropic",
            "model": "claude-3",
            "api_key": "key",
        }

        LLMProviderFactory.create_from_config(config)

        # Provider should be popped from config
        assert "provider" not in config
        # But other keys should remain
        assert config["model"] == "claude-3"
        assert config["api_key"] == "key"


def test_create_from_config_empty_dict() -> None:
    """Test creating from empty config dict."""
    with pytest.raises(ValueError, match="must include 'provider' key"):
        LLMProviderFactory.create_from_config({})


def test_create_from_config_only_provider() -> None:
    """Test creating from config with only provider key."""
    with patch(
        "docs2synth.agent.factory.LLMProviderFactory.create_provider"
    ) as mock_create:
        mock_create.return_value = MagicMock()

        config = {"provider": "ollama"}
        LLMProviderFactory.create_from_config(config)

        # Should pass no additional kwargs
        mock_create.assert_called_once_with("ollama")
