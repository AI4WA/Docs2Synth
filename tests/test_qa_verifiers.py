"""Tests for QA verifier modules."""

from unittest.mock import MagicMock, patch

import pytest

from docs2synth.agent.base import LLMResponse
from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.qa.verifiers.correctness import CorrectnessVerifier


class TestCorrectnessVerifier:
    """Test suite for CorrectnessVerifier."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock(spec=AgentWrapper)
        return agent

    @pytest.fixture
    def verifier(self, mock_agent):
        """Create a CorrectnessVerifier with mock agent."""
        return CorrectnessVerifier(agent=mock_agent)

    def test_initialization_with_agent(self, mock_agent):
        """Test initialization with provided agent."""
        verifier = CorrectnessVerifier(agent=mock_agent)
        assert verifier.agent is mock_agent
        assert verifier.prompt_template == CorrectnessVerifier.DEFAULT_PROMPT_TEMPLATE

    def test_initialization_with_custom_prompt(self, mock_agent):
        """Test initialization with custom prompt template."""
        custom_prompt = "Is '{answer}' correct for '{question}'?"
        verifier = CorrectnessVerifier(agent=mock_agent, prompt_template=custom_prompt)
        assert verifier.prompt_template == custom_prompt

    def test_escape_template_braces_with_json_format(self, verifier):
        """Test escaping braces in JSON format template."""
        template = 'Output: {{"Response": "Yes/No", "Explanation": "xxx"}}.'
        result = verifier._escape_template_braces(template)
        # JSON braces should be escaped
        assert "{{" in result
        assert "}}" in result

    def test_escape_template_braces_with_python_dict_format(self, verifier):
        """Test escaping braces in Python dict format template."""
        template = "Output: {'Response': 'Yes/No', 'Explanation': 'xxx'}."
        result = verifier._escape_template_braces(template)
        # Python dict braces should be escaped
        assert "{{" in result
        assert "}}" in result

    def test_escape_template_braces_preserves_format_placeholders(self, verifier):
        """Test that format placeholders {question} and {answer} are preserved."""
        template = "Is {answer} correct for {question}? Output: {{'Response': 'Yes'}}."
        result = verifier._escape_template_braces(template)
        # Format placeholders should remain as single braces
        assert "{answer}" in result
        assert "{question}" in result

    def test_parse_response_with_valid_json(self, verifier):
        """Test parsing response with valid JSON format."""
        raw_output = '{"Response": "Yes", "Explanation": "The answer is correct."}'
        result = verifier._parse_response(raw_output)
        assert result["Response"] == "Yes"
        assert result["Explanation"] == "The answer is correct."
        assert result["raw_output"] == raw_output

    def test_parse_response_with_python_dict(self, verifier):
        """Test parsing response with Python dict format (single quotes)."""
        raw_output = "{'Response': 'No', 'Explanation': 'The answer is incorrect.'}"
        result = verifier._parse_response(raw_output)
        assert result["Response"] == "No"
        assert result["Explanation"] == "The answer is incorrect."

    def test_parse_response_with_json_in_text(self, verifier):
        """Test parsing response with JSON embedded in text."""
        raw_output = (
            'Here is the result: {"Response": "Yes", "Explanation": "Correct."}'
        )
        result = verifier._parse_response(raw_output)
        assert result["Response"] == "Yes"
        assert result["Explanation"] == "Correct."

    def test_parse_response_fallback_yes(self, verifier):
        """Test fallback parsing when JSON parsing fails - contains 'yes'."""
        raw_output = "Yes, the answer is correct."
        result = verifier._parse_response(raw_output)
        assert result["Response"] == "Yes"
        assert result["raw_output"] == raw_output

    def test_parse_response_fallback_no(self, verifier):
        """Test fallback parsing when JSON parsing fails - contains 'no'."""
        raw_output = "No, the answer is incorrect."
        result = verifier._parse_response(raw_output)
        assert result["Response"] == "No"

    def test_parse_response_fallback_unknown(self, verifier):
        """Test fallback parsing when neither yes nor no is found."""
        raw_output = "Maybe the answer could be correct."
        result = verifier._parse_response(raw_output)
        assert result["Response"] == "Unknown"

    def test_parse_response_with_malformed_json(self, verifier):
        """Test parsing response with malformed JSON."""
        raw_output = '{"Response": "Yes", "Explanation": incomplete'
        result = verifier._parse_response(raw_output)
        # Should fall back to text parsing
        assert "Response" in result
        assert result["raw_output"] == raw_output

    def test_verify_success(self, verifier, mock_agent):
        """Test verify with successful verification."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "Yes", "Explanation": "The answer is correct."}',
            model="test-model",
        )

        result = verifier.verify(
            question="What is 2+2?",
            answer="4",
            temperature=0.7,
        )

        assert result["response"] == "Yes"
        assert result["explanation"] == "The answer is correct."
        mock_agent.generate.assert_called_once()

    def test_verify_with_context(self, verifier, mock_agent):
        """Test verify ignores context as expected for correctness verification."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "Yes", "Explanation": "Correct."}',
            model="test-model",
        )

        result = verifier.verify(
            question="What is the capital?",
            answer="Paris",
            context="This is about France",  # Should be ignored
        )

        assert result["response"] == "Yes"
        # Verify that context is not used in the prompt
        call_args = mock_agent.generate.call_args
        assert "France" not in call_args[1]["prompt"]

    def test_verify_with_image(self, verifier, mock_agent):
        """Test verify with image parameter."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "No", "Explanation": "Incorrect."}',
            model="test-model",
        )

        result = verifier.verify(
            question="What color is this?",
            answer="Red",
            image="test_image.png",
        )

        assert result["response"] == "No"

    def test_verify_response_format_json(self, verifier, mock_agent):
        """Test that verify uses JSON response format."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "Yes", "Explanation": "Correct."}',
            model="test-model",
        )

        verifier.verify(question="Test?", answer="Test")

        call_args = mock_agent.generate.call_args
        assert call_args[1]["response_format"] == "json"

    def test_verify_normalizes_key_names(self, verifier, mock_agent):
        """Test that verify normalizes Response/Explanation to lowercase."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "Yes", "Explanation": "Good."}',
            model="test-model",
        )

        result = verifier.verify(question="Test?", answer="Test")

        # Keys should be normalized to lowercase
        assert "response" in result
        assert "explanation" in result
        assert "Response" not in result
        assert "Explanation" not in result

    def test_verify_with_custom_temperature(self, verifier, mock_agent):
        """Test verify with custom temperature."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "Yes", "Explanation": "Correct."}',
            model="test-model",
        )

        verifier.verify(
            question="Test?",
            answer="Test",
            temperature=0.3,
        )

        call_args = mock_agent.generate.call_args
        assert call_args[1]["temperature"] == 0.3

    def test_verify_with_max_tokens(self, verifier, mock_agent):
        """Test verify with max_tokens parameter."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "Yes", "Explanation": "Correct."}',
            model="test-model",
        )

        verifier.verify(
            question="Test?",
            answer="Test",
            max_tokens=100,
        )

        call_args = mock_agent.generate.call_args
        assert call_args[1]["max_tokens"] == 100

    def test_verify_prompt_formatting(self, verifier, mock_agent):
        """Test that verify correctly formats the prompt."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "Yes", "Explanation": "Correct."}',
            model="test-model",
        )

        question = "What is the capital of France?"
        answer = "Paris"

        verifier.verify(question=question, answer=answer)

        call_args = mock_agent.generate.call_args
        prompt = call_args[1]["prompt"]
        assert question in prompt
        assert answer in prompt

    def test_verify_with_kwargs(self, verifier, mock_agent):
        """Test that verify passes additional kwargs to agent.generate."""
        mock_agent.generate.return_value = LLMResponse(
            content='{"Response": "Yes", "Explanation": "Correct."}',
            model="test-model",
        )

        verifier.verify(
            question="Test?",
            answer="Test",
            custom_param="custom_value",
        )

        call_args = mock_agent.generate.call_args
        assert call_args[1]["custom_param"] == "custom_value"

    def test_initialization_without_agent(self):
        """Test initialization without agent creates one."""
        with patch("docs2synth.qa.verifiers.base.AgentWrapper") as mock_wrapper:
            mock_wrapper.return_value = MagicMock(spec=AgentWrapper)
            verifier = CorrectnessVerifier(provider="openai", model="gpt-3.5-turbo")
            assert verifier.agent is not None

    def test_parse_response_extracts_explanation_with_regex(self, verifier):
        """Test that parse_response can extract explanation using regex."""
        raw_output = 'Response: Yes, Explanation: "This is the explanation"'
        result = verifier._parse_response(raw_output)
        # Should extract explanation
        assert result["Response"] == "Yes"
        # Note: The regex in the code looks for explanation with quotes
        if "Explanation" in result:
            assert "explanation" in result["Explanation"].lower()
