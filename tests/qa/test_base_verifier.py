"""Tests for QA verifier base classes."""

from unittest.mock import MagicMock, patch

import pytest

from docs2synth.qa.verifiers.base import BaseQAVerifier


class ConcreteVerifier(BaseQAVerifier):
    """Concrete implementation for testing."""

    def verify(self, question, answer, context=None, image=None, **kwargs):
        """Implementation for testing."""
        return {"response": "Yes", "explanation": "Test verification"}


def test_base_verifier_with_agent() -> None:
    """Test initializing verifier with pre-configured agent."""
    mock_agent = MagicMock()
    mock_agent.provider_type = "test_provider"
    mock_agent.model = "test-model"

    verifier = ConcreteVerifier(agent=mock_agent)

    assert verifier.agent == mock_agent
    assert "ConcreteVerifier" in repr(verifier)
    assert "test_provider" in repr(verifier)


@patch("docs2synth.qa.verifiers.base.AgentWrapper")
def test_base_verifier_with_provider(mock_agent_class) -> None:
    """Test initializing verifier with provider name."""
    mock_agent = MagicMock()
    mock_agent.provider_type = "openai"
    mock_agent.model = "gpt-4"
    mock_agent_class.return_value = mock_agent

    verifier = ConcreteVerifier(provider="openai", model="gpt-4")

    mock_agent_class.assert_called_once_with(provider="openai", model="gpt-4")
    assert verifier.agent == mock_agent


def test_base_verifier_no_agent_or_provider() -> None:
    """Test that ValueError is raised when neither agent nor provider is specified."""
    with pytest.raises(
        ValueError, match="Either 'agent' or 'provider' must be specified"
    ):
        ConcreteVerifier()


@patch("docs2synth.qa.verifiers.base.AgentWrapper")
def test_base_verifier_with_kwargs(mock_agent_class) -> None:
    """Test initializing verifier with additional kwargs."""
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent

    ConcreteVerifier(
        provider="openai",
        model="gpt-4",
        temperature=0.5,
        api_key="test-key",
    )

    mock_agent_class.assert_called_once_with(
        provider="openai",
        model="gpt-4",
        temperature=0.5,
        api_key="test-key",
    )


def test_base_verifier_repr() -> None:
    """Test string representation of verifier."""
    mock_agent = MagicMock()
    mock_agent.provider_type = "anthropic"
    mock_agent.model = "claude-3"

    verifier = ConcreteVerifier(agent=mock_agent)
    repr_str = repr(verifier)

    assert "ConcreteVerifier" in repr_str
    assert "anthropic" in repr_str
    assert "claude-3" in repr_str


def test_concrete_verifier_verify() -> None:
    """Test that concrete implementation works."""
    mock_agent = MagicMock()
    verifier = ConcreteVerifier(agent=mock_agent)

    result = verifier.verify(
        question="What is AI?",
        answer="Artificial Intelligence",
        context="Some context",
    )

    assert result["response"] == "Yes"
    assert "explanation" in result


def test_base_verifier_abstract() -> None:
    """Test that BaseQAVerifier cannot be instantiated without implementing verify."""

    class IncompleteVerifier(BaseQAVerifier):
        """Verifier without verify implementation."""

        pass

    mock_agent = MagicMock()

    with pytest.raises(TypeError):
        IncompleteVerifier(agent=mock_agent)


def test_base_verifier_verify_not_implemented() -> None:
    """Test that calling verify on base class raises NotImplementedError."""

    # Create a partial implementation that doesn't override verify
    with patch.object(BaseQAVerifier, "__abstractmethods__", set()):
        mock_agent = MagicMock()
        verifier = BaseQAVerifier(agent=mock_agent)

        with pytest.raises(
            NotImplementedError, match="Subclasses must implement verify method"
        ):
            verifier.verify(
                question="test",
                answer="test",
            )
