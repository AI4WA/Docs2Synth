"""Tests for QA generator base classes."""

from unittest.mock import MagicMock, patch

import pytest

from docs2synth.qa.generators.base import BaseQAGenerator


class ConcreteGenerator(BaseQAGenerator):
    """Concrete implementation for testing."""

    def generate(self, context, target, **kwargs):
        """Implementation for testing."""
        return f"Question about {target}?"


def test_base_generator_with_agent() -> None:
    """Test initializing generator with pre-configured agent."""
    mock_agent = MagicMock()
    mock_agent.provider_type = "test_provider"
    mock_agent.model = "test-model"

    generator = ConcreteGenerator(agent=mock_agent)

    assert generator.agent == mock_agent
    assert "ConcreteGenerator" in repr(generator)
    assert "test_provider" in repr(generator)


@patch("docs2synth.qa.generators.base.AgentWrapper")
def test_base_generator_with_provider(mock_agent_class) -> None:
    """Test initializing generator with provider name."""
    mock_agent = MagicMock()
    mock_agent.provider_type = "openai"
    mock_agent.model = "gpt-4"
    mock_agent_class.return_value = mock_agent

    generator = ConcreteGenerator(provider="openai", model="gpt-4")

    mock_agent_class.assert_called_once_with(provider="openai", model="gpt-4")
    assert generator.agent == mock_agent


def test_base_generator_no_agent_or_provider() -> None:
    """Test that ValueError is raised when neither agent nor provider is specified."""
    with pytest.raises(
        ValueError, match="Either 'agent' or 'provider' must be specified"
    ):
        ConcreteGenerator()


@patch("docs2synth.qa.generators.base.AgentWrapper")
def test_base_generator_with_kwargs(mock_agent_class) -> None:
    """Test initializing generator with additional kwargs."""
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent

    ConcreteGenerator(
        provider="anthropic",
        model="claude-3",
        temperature=0.8,
        api_key="test-key",
    )

    mock_agent_class.assert_called_once_with(
        provider="anthropic",
        model="claude-3",
        temperature=0.8,
        api_key="test-key",
    )


def test_base_generator_repr() -> None:
    """Test string representation of generator."""
    mock_agent = MagicMock()
    mock_agent.provider_type = "gemini"
    mock_agent.model = "gemini-pro"

    generator = ConcreteGenerator(agent=mock_agent)
    repr_str = repr(generator)

    assert "ConcreteGenerator" in repr_str
    assert "gemini" in repr_str
    assert "gemini-pro" in repr_str


def test_concrete_generator_generate() -> None:
    """Test that concrete implementation works."""
    mock_agent = MagicMock()
    generator = ConcreteGenerator(agent=mock_agent)

    question = generator.generate(
        context="Document context",
        target="AI definition",
    )

    assert "Question about AI definition?" == question


def test_base_generator_abstract() -> None:
    """Test that BaseQAGenerator cannot be instantiated without implementing generate."""

    class IncompleteGenerator(BaseQAGenerator):
        """Generator without generate implementation."""

        pass

    mock_agent = MagicMock()

    with pytest.raises(TypeError):
        IncompleteGenerator(agent=mock_agent)


def test_base_generator_generate_not_implemented() -> None:
    """Test that calling generate on base class raises NotImplementedError."""

    # Create a partial implementation that doesn't override generate
    with patch.object(BaseQAGenerator, "__abstractmethods__", set()):
        mock_agent = MagicMock()
        generator = BaseQAGenerator(agent=mock_agent)

        with pytest.raises(
            NotImplementedError, match="Subclasses must implement generate method"
        ):
            generator.generate(
                context="test context",
                target="test target",
            )


def test_generator_with_temperature() -> None:
    """Test generator can be used with custom temperature."""
    mock_agent = MagicMock()
    generator = ConcreteGenerator(agent=mock_agent)

    # The generate method should accept temperature
    question = generator.generate(
        context="context",
        target="target",
        temperature=0.5,
    )

    assert isinstance(question, str)


def test_generator_with_max_tokens() -> None:
    """Test generator can be used with max_tokens."""
    mock_agent = MagicMock()
    generator = ConcreteGenerator(agent=mock_agent)

    # The generate method should accept max_tokens
    question = generator.generate(
        context="context",
        target="target",
        max_tokens=100,
    )

    assert isinstance(question, str)
