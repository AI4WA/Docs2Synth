"""Advanced tests for agent module including wrapper and QA generator."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from docs2synth.agent import AgentWrapper, LLMResponse, QAGenerator

# ============================================================================
# AgentWrapper tests
# ============================================================================


def test_agent_wrapper_initialization_with_provider():
    """Test AgentWrapper initialization with provider."""
    from unittest.mock import MagicMock

    with patch("docs2synth.agent.wrapper.LLMProviderFactory") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.model = "gpt-3.5-turbo"
        mock_factory.create_provider.return_value = mock_provider

        agent = AgentWrapper(provider="openai", model="gpt-3.5-turbo", api_key="test")

        assert agent.provider_name == "openai"
        assert agent.model == "gpt-3.5-turbo"
        mock_factory.create_provider.assert_called_once()


def test_agent_wrapper_generate():
    """Test AgentWrapper generate method."""
    with patch("docs2synth.agent.wrapper.LLMProviderFactory") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.model = "gpt-4"
        expected_response = LLMResponse(
            content="Test response",
            model="gpt-4",
            usage={"total_tokens": 10},
        )
        mock_provider.generate.return_value = expected_response
        mock_factory.create_provider.return_value = mock_provider

        agent = AgentWrapper(provider="openai", model="gpt-4", api_key="test")
        response = agent.generate("Test prompt", system_prompt="You are helpful")

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        mock_provider.generate.assert_called_once()


def test_agent_wrapper_chat():
    """Test AgentWrapper chat method."""
    with patch("docs2synth.agent.wrapper.LLMProviderFactory") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.model = "gpt-4"
        expected_response = LLMResponse(
            content="Chat response",
            model="gpt-4",
            usage={"total_tokens": 20},
        )
        mock_provider.chat.return_value = expected_response
        mock_factory.create_provider.return_value = mock_provider

        agent = AgentWrapper(provider="openai", model="gpt-4", api_key="test")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        response = agent.chat(messages)

        assert response.content == "Chat response"
        mock_provider.chat.assert_called_once()


def test_agent_wrapper_switch_provider():
    """Test switching providers."""
    with patch("docs2synth.agent.wrapper.LLMProviderFactory") as mock_factory:
        mock_provider1 = MagicMock()
        mock_provider1.model = "gpt-4"
        mock_provider2 = MagicMock()
        mock_provider2.model = "claude-3"

        mock_factory.create_provider.side_effect = [mock_provider1, mock_provider2]

        agent = AgentWrapper(provider="openai", model="gpt-4", api_key="test")
        assert agent.provider_name == "openai"

        agent.switch_provider("anthropic", model="claude-3", api_key="test2")
        assert agent.provider_name == "anthropic"
        assert mock_factory.create_provider.call_count == 2


def test_agent_wrapper_properties():
    """Test AgentWrapper properties."""
    with patch("docs2synth.agent.wrapper.LLMProviderFactory") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.model = "gpt-3.5-turbo"
        mock_factory.create_provider.return_value = mock_provider

        agent = AgentWrapper(provider="openai", model="gpt-3.5-turbo", api_key="test")

        assert agent.model == "gpt-3.5-turbo"
        assert agent.provider_type == "openai"
        assert "AgentWrapper" in repr(agent)
        assert "openai" in repr(agent)


def test_agent_wrapper_from_config():
    """Test AgentWrapper initialization from config file."""
    config_data = """
agent:
  provider: openai
  model: gpt-4
  config:
    temperature: 0.8
    max_tokens: 1000

keys:
  openai_api_key: test-key-from-config
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_data)
        config_path = f.name

    try:
        with patch("docs2synth.agent.wrapper.LLMProviderFactory") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.model = "gpt-4"
            mock_factory.create_provider.return_value = mock_provider

            agent = AgentWrapper(config_path=config_path)

            # Verify provider was created with config values
            assert agent.provider_name == "openai"
            call_kwargs = mock_factory.create_provider.call_args[1]
            assert call_kwargs.get("api_key") == "test-key-from-config"
    finally:
        os.unlink(config_path)


def test_agent_wrapper_config_override():
    """Test that explicit kwargs override config values."""
    config_data = """
agent:
  provider: openai
  model: gpt-3.5-turbo

keys:
  openai_api_key: config-key
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_data)
        config_path = f.name

    try:
        with patch("docs2synth.agent.wrapper.LLMProviderFactory") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.model = "gpt-4"
            mock_factory.create_provider.return_value = mock_provider

            # Override model and api_key
            AgentWrapper(
                config_path=config_path,
                model="gpt-4",
                api_key="explicit-key",
            )

            call_kwargs = mock_factory.create_provider.call_args
            # Model should be overridden
            assert call_kwargs[0][1] == "gpt-4"  # Second positional arg is model
            # API key should be overridden
            assert call_kwargs[1].get("api_key") == "explicit-key"
    finally:
        os.unlink(config_path)


# ============================================================================
# QAGenerator tests
# ============================================================================


def test_qa_generator_initialization_with_agent():
    """Test QAGenerator initialization with existing agent."""
    mock_agent = MagicMock()
    generator = QAGenerator(agent=mock_agent)
    assert generator.agent == mock_agent


def test_qa_generator_initialization_with_provider():
    """Test QAGenerator initialization with provider."""
    with patch("docs2synth.agent.qa.AgentWrapper") as mock_wrapper_class:
        mock_agent = MagicMock()
        mock_wrapper_class.return_value = mock_agent

        generator = QAGenerator(provider="openai", model="gpt-4", api_key="test")

        mock_wrapper_class.assert_called_once_with(
            provider="openai", model="gpt-4", api_key="test"
        )
        assert generator.agent == mock_agent


def test_qa_generator_requires_provider_or_agent():
    """Test that QAGenerator requires either agent or provider."""
    with pytest.raises(ValueError, match="Either 'agent' or 'provider'"):
        QAGenerator()


def test_qa_generator_custom_prompts():
    """Test QAGenerator with custom prompts."""
    mock_agent = MagicMock()
    custom_system = "Custom system prompt"
    custom_template = "Generate QA from: {content}"

    generator = QAGenerator(
        agent=mock_agent,
        system_prompt=custom_system,
        prompt_template=custom_template,
    )

    assert generator.system_prompt == custom_system
    assert generator.prompt_template == custom_template


def test_qa_generator_generate_qa_pair():
    """Test QAGenerator.generate_qa_pair method."""
    mock_agent = MagicMock()
    mock_response = LLMResponse(
        content='{"question": "What is AI?", "answer": "Artificial Intelligence"}',
        model="gpt-4",
        usage={"total_tokens": 50},
    )
    mock_agent.generate.return_value = mock_response

    generator = QAGenerator(agent=mock_agent)
    qa_pair = generator.generate_qa_pair("AI is about making computers smart")

    assert qa_pair["question"] == "What is AI?"
    assert qa_pair["answer"] == "Artificial Intelligence"
    mock_agent.generate.assert_called_once()


def test_qa_generator_handles_json_markdown():
    """Test that QAGenerator handles JSON in markdown blocks."""
    mock_agent = MagicMock()
    mock_response = LLMResponse(
        content='```json\n{"question": "Test Q", "answer": "Test A"}\n```',
        model="gpt-4",
        usage={"total_tokens": 50},
    )
    mock_agent.generate.return_value = mock_response

    generator = QAGenerator(agent=mock_agent)
    qa_pair = generator.generate_qa_pair("Some content")

    assert qa_pair["question"] == "Test Q"
    assert qa_pair["answer"] == "Test A"


def test_qa_generator_invalid_json_raises_error():
    """Test that invalid JSON raises ValueError."""
    mock_agent = MagicMock()
    mock_response = LLMResponse(
        content="This is not JSON",
        model="gpt-4",
        usage={"total_tokens": 10},
    )
    mock_agent.generate.return_value = mock_response

    generator = QAGenerator(agent=mock_agent)

    with pytest.raises(ValueError, match="Invalid JSON response"):
        generator.generate_qa_pair("Some content")


def test_qa_generator_missing_fields_raises_error():
    """Test that missing question/answer fields raises error."""
    mock_agent = MagicMock()
    mock_response = LLMResponse(
        content='{"question": "What is AI?"}',  # Missing answer
        model="gpt-4",
        usage={"total_tokens": 20},
    )
    mock_agent.generate.return_value = mock_response

    generator = QAGenerator(agent=mock_agent)

    with pytest.raises(ValueError, match="must contain 'question' and 'answer'"):
        generator.generate_qa_pair("Some content")


def test_qa_generator_generate_qa_pairs():
    """Test generating multiple QA pairs."""
    mock_agent = MagicMock()

    # Create multiple responses
    responses = [
        LLMResponse(
            content=f'{{"question": "Q{i}", "answer": "A{i}"}}',
            model="gpt-4",
            usage={"total_tokens": 10},
        )
        for i in range(3)
    ]
    mock_agent.generate.side_effect = responses

    generator = QAGenerator(agent=mock_agent)
    contents = ["Content 1", "Content 2", "Content 3"]
    qa_pairs = generator.generate_qa_pairs(contents)

    assert len(qa_pairs) == 3
    assert qa_pairs[0]["question"] == "Q0"
    assert qa_pairs[1]["question"] == "Q1"
    assert qa_pairs[2]["question"] == "Q2"


def test_qa_generator_continues_on_error():
    """Test that generate_qa_pairs continues after errors."""
    mock_agent = MagicMock()

    # Second call raises error
    mock_agent.generate.side_effect = [
        LLMResponse(
            content='{"question": "Q1", "answer": "A1"}',
            model="gpt-4",
            usage={"total_tokens": 10},
        ),
        Exception("API Error"),
        LLMResponse(
            content='{"question": "Q3", "answer": "A3"}',
            model="gpt-4",
            usage={"total_tokens": 10},
        ),
    ]

    generator = QAGenerator(agent=mock_agent)
    contents = ["Content 1", "Content 2", "Content 3"]
    qa_pairs = generator.generate_qa_pairs(contents)

    # Should have 2 successful pairs (1st and 3rd)
    assert len(qa_pairs) == 2
    assert qa_pairs[0]["question"] == "Q1"
    assert qa_pairs[1]["question"] == "Q3"


def test_qa_generator_with_verification():
    """Test QA generation with verification."""
    mock_agent = MagicMock()

    # First call: generate QA pair
    # Second call: meaningfulness check
    # Third call: correctness check
    mock_agent.generate.side_effect = [
        LLMResponse(
            content='{"question": "What is Python?", "answer": "A programming language"}',
            model="gpt-4",
            usage={"total_tokens": 50},
        ),
        LLMResponse(content="yes", model="gpt-4", usage={"total_tokens": 5}),
        LLMResponse(content="yes", model="gpt-4", usage={"total_tokens": 5}),
    ]

    generator = QAGenerator(agent=mock_agent)
    qa_pair = generator.generate_with_verification("Python is a language")

    assert qa_pair is not None
    assert qa_pair["question"] == "What is Python?"
    assert mock_agent.generate.call_count == 3


def test_qa_generator_verification_fails_meaningful():
    """Test verification failure on meaningfulness check."""
    mock_agent = MagicMock()

    # Generate QA pair, then fail meaningfulness check
    mock_agent.generate.side_effect = [
        LLMResponse(
            content='{"question": "Test?", "answer": "Test"}',
            model="gpt-4",
            usage={"total_tokens": 20},
        ),
        LLMResponse(content="no", model="gpt-4", usage={"total_tokens": 5}),
    ]

    generator = QAGenerator(agent=mock_agent)
    qa_pair = generator.generate_with_verification("Some content")

    assert qa_pair is None


def test_qa_generator_verification_fails_correctness():
    """Test verification failure on correctness check."""
    mock_agent = MagicMock()

    # Generate QA pair, pass meaningful check, fail correctness check
    mock_agent.generate.side_effect = [
        LLMResponse(
            content='{"question": "Test?", "answer": "Wrong answer"}',
            model="gpt-4",
            usage={"total_tokens": 20},
        ),
        LLMResponse(content="yes", model="gpt-4", usage={"total_tokens": 5}),
        LLMResponse(content="no", model="gpt-4", usage={"total_tokens": 5}),
    ]

    generator = QAGenerator(agent=mock_agent)
    qa_pair = generator.generate_with_verification("Some content")

    assert qa_pair is None


def test_qa_generator_verification_skips_checks():
    """Test verification with checks disabled."""
    mock_agent = MagicMock()

    # Only one call needed (QA generation)
    mock_agent.generate.return_value = LLMResponse(
        content='{"question": "Q", "answer": "A"}',
        model="gpt-4",
        usage={"total_tokens": 10},
    )

    generator = QAGenerator(agent=mock_agent)
    qa_pair = generator.generate_with_verification(
        "Content",
        meaningful_check=False,
        correctness_check=False,
    )

    assert qa_pair is not None
    assert mock_agent.generate.call_count == 1
