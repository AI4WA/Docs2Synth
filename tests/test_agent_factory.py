"""Tests for agent factory module."""

import pytest

from docs2synth.agent.base import BaseLLMProvider
from docs2synth.agent.factory import LLMProviderFactory
from docs2synth.agent.providers import PROVIDER_REGISTRY


class TestLLMProviderFactory:
    """Test suite for LLMProviderFactory."""

    def test_create_provider_openai(self):
        """Test creating OpenAI provider."""
        try:
            provider = LLMProviderFactory.create_provider(
                "openai",
                model="gpt-3.5-turbo",
                api_key="test-key-123",
            )
            assert provider is not None
            assert isinstance(provider, BaseLLMProvider)
            assert provider.model == "gpt-3.5-turbo"
        except ImportError:
            pytest.skip("OpenAI package not installed")

    def test_create_provider_anthropic(self):
        """Test creating Anthropic provider."""
        try:
            provider = LLMProviderFactory.create_provider(
                "anthropic",
                model="claude-3-5-sonnet-20241022",
                api_key="test-key-123",
            )
            assert provider is not None
            assert isinstance(provider, BaseLLMProvider)
            assert provider.model == "claude-3-5-sonnet-20241022"
        except ImportError:
            pytest.skip("Anthropic package not installed")

    def test_create_provider_ollama(self):
        """Test creating Ollama provider."""
        try:
            provider = LLMProviderFactory.create_provider(
                "ollama",
                model="llama2",
                base_url="http://localhost:11434",
            )
            assert provider is not None
            assert isinstance(provider, BaseLLMProvider)
            assert provider.model == "llama2"
        except ImportError:
            pytest.skip("Ollama package not installed")

    def test_create_provider_gemini(self):
        """Test creating Gemini provider."""
        try:
            provider = LLMProviderFactory.create_provider(
                "gemini",
                model="gemini-pro",
                api_key="test-key-123",
            )
            assert provider is not None
            assert isinstance(provider, BaseLLMProvider)
            assert provider.model == "gemini-pro"
        except ImportError:
            pytest.skip("Google Generative AI package not installed")

    def test_create_provider_case_insensitive(self):
        """Test that provider names are case insensitive."""
        try:
            provider1 = LLMProviderFactory.create_provider(
                "openai", model="gpt-3.5-turbo", api_key="test-key"
            )
            provider2 = LLMProviderFactory.create_provider(
                "OpenAI", model="gpt-3.5-turbo", api_key="test-key"
            )
            provider3 = LLMProviderFactory.create_provider(
                "OPENAI", model="gpt-3.5-turbo", api_key="test-key"
            )
            assert provider1.__class__ == provider2.__class__ == provider3.__class__
        except ImportError:
            pytest.skip("OpenAI package not installed")

    def test_create_provider_with_whitespace(self):
        """Test that provider names handle whitespace correctly."""
        try:
            provider = LLMProviderFactory.create_provider(
                "  openai  ", model="gpt-3.5-turbo", api_key="test-key"
            )
            assert provider is not None
            assert isinstance(provider, BaseLLMProvider)
        except ImportError:
            pytest.skip("OpenAI package not installed")

    def test_create_provider_unsupported(self):
        """Test creating provider with unsupported name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LLMProviderFactory.create_provider("unsupported_provider")
        assert "Unsupported provider" in str(exc_info.value)
        assert "unsupported_provider" in str(exc_info.value)
        assert "Available providers" in str(exc_info.value)

    def test_create_provider_without_model(self):
        """Test creating provider without model uses default."""
        try:
            provider = LLMProviderFactory.create_provider(
                "openai", api_key="test-key-123"
            )
            assert provider is not None
            assert provider.model is not None  # Should use provider default
        except ImportError:
            pytest.skip("OpenAI package not installed")

    def test_create_provider_with_kwargs(self):
        """Test creating provider with additional kwargs."""
        try:
            provider = LLMProviderFactory.create_provider(
                "openai",
                model="gpt-3.5-turbo",
                api_key="test-key-123",
                temperature=0.5,
                max_tokens=100,
            )
            assert provider is not None
            assert provider.config.get("temperature") == 0.5
            assert provider.config.get("max_tokens") == 100
        except ImportError:
            pytest.skip("OpenAI package not installed")

    def test_create_from_config_valid(self):
        """Test creating provider from config dictionary."""
        try:
            config = {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "test-key-123",
            }
            provider = LLMProviderFactory.create_from_config(config)
            assert provider is not None
            assert isinstance(provider, BaseLLMProvider)
            assert provider.model == "gpt-3.5-turbo"
        except ImportError:
            pytest.skip("OpenAI package not installed")

    def test_create_from_config_missing_provider(self):
        """Test creating from config without 'provider' key raises ValueError."""
        config = {"model": "gpt-3.5-turbo", "api_key": "test-key-123"}
        with pytest.raises(ValueError) as exc_info:
            LLMProviderFactory.create_from_config(config)
        assert "Configuration must include 'provider' key" in str(exc_info.value)

    def test_create_from_config_pops_provider_key(self):
        """Test that create_from_config pops provider from config dict."""
        try:
            config = {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "test-key-123",
            }
            original_len = len(config)
            provider = LLMProviderFactory.create_from_config(config)
            # Provider should be popped from config
            assert "provider" not in config
            assert len(config) == original_len - 1
            assert provider is not None
        except ImportError:
            pytest.skip("OpenAI package not installed")

    def test_provider_registry_not_empty(self):
        """Test that provider registry is populated."""
        assert len(PROVIDER_REGISTRY) > 0
        assert "openai" in PROVIDER_REGISTRY or "anthropic" in PROVIDER_REGISTRY

    def test_all_registered_providers_are_classes(self):
        """Test that all registered providers are classes."""
        for name, provider_class in PROVIDER_REGISTRY.items():
            assert isinstance(name, str)
            assert isinstance(provider_class, type)
            # Check that it's a subclass of BaseLLMProvider
            assert issubclass(provider_class, BaseLLMProvider)
