"""Tests for text processing utilities."""

from unittest.mock import patch

from docs2synth.utils.text import (
    CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_WINDOWS,
    DEFAULT_MAX_CONTEXT_LENGTH,
    _estimate_context_window,
    calculate_max_context_length,
    truncate_context,
)


def test_default_constants() -> None:
    """Test default constant values."""
    assert isinstance(DEFAULT_MAX_CONTEXT_LENGTH, int)
    assert DEFAULT_MAX_CONTEXT_LENGTH > 0
    assert isinstance(CHARS_PER_TOKEN, float)
    assert CHARS_PER_TOKEN > 0
    assert isinstance(DEFAULT_CONTEXT_WINDOWS, dict)
    assert "default" in DEFAULT_CONTEXT_WINDOWS


def test_estimate_context_window_with_known_model() -> None:
    """Test estimating context window for known models."""
    # Test OpenAI models
    assert _estimate_context_window(model="gpt-4o") == 128000
    assert _estimate_context_window(model="gpt-4") == 8192
    assert _estimate_context_window(model="gpt-3.5-turbo") == 16385

    # Test Anthropic models
    assert _estimate_context_window(model="claude-3-5-sonnet") == 200000
    assert _estimate_context_window(model="claude-3-opus") == 200000

    # Test Gemini models
    assert _estimate_context_window(model="gemini-1.5-pro") == 2000000
    assert _estimate_context_window(model="gemini-1.5-flash") == 1000000


def test_estimate_context_window_case_insensitive() -> None:
    """Test model matching is case-insensitive."""
    assert _estimate_context_window(model="GPT-4O") == 128000
    assert _estimate_context_window(model="Claude-3-Opus") == 200000


def test_estimate_context_window_by_provider() -> None:
    """Test estimating context window by provider name."""
    assert _estimate_context_window(provider="openai") == 128000
    assert _estimate_context_window(provider="anthropic") == 200000
    assert _estimate_context_window(provider="claude") == 200000
    assert _estimate_context_window(provider="gemini") == 2000000
    assert _estimate_context_window(provider="google") == 2000000
    assert _estimate_context_window(provider="doubao") == 32768
    assert _estimate_context_window(provider="vllm") == 8192
    assert _estimate_context_window(provider="huggingface") == 4096
    assert _estimate_context_window(provider="ollama") == 4096


def test_estimate_context_window_default_fallback() -> None:
    """Test default fallback when no match found."""
    window = _estimate_context_window(provider="unknown", model="unknown-model")
    assert window == DEFAULT_CONTEXT_WINDOWS["default"]
    assert window == 8192


def test_estimate_context_window_no_args() -> None:
    """Test estimation with no arguments returns default."""
    assert _estimate_context_window() == DEFAULT_CONTEXT_WINDOWS["default"]


def test_calculate_max_context_length_default() -> None:
    """Test calculating max context length with defaults."""
    max_chars = calculate_max_context_length()
    assert isinstance(max_chars, int)
    assert max_chars > 0


def test_calculate_max_context_length_with_max_tokens() -> None:
    """Test calculation with custom max_tokens."""
    max_chars = calculate_max_context_length(max_tokens=500, provider="openai")
    assert isinstance(max_chars, int)
    assert max_chars > 0


def test_calculate_max_context_length_with_model() -> None:
    """Test calculation considers model context window."""
    # GPT-4 has smaller context window than GPT-4O
    max_gpt4 = calculate_max_context_length(model="gpt-4")
    max_gpt4o = calculate_max_context_length(model="gpt-4o")
    assert max_gpt4o > max_gpt4


def test_calculate_max_context_length_with_vllm_max_model_len() -> None:
    """Test calculation with vLLM max_model_len override."""
    max_chars = calculate_max_context_length(
        provider="vllm",
        max_model_len=16384,
        max_tokens=500,
    )
    assert isinstance(max_chars, int)
    assert max_chars > 0


def test_calculate_max_context_length_minimum_preserved() -> None:
    """Test that minimum context length is preserved."""
    # Even with very small max_tokens, should get at least some context
    max_chars = calculate_max_context_length(max_tokens=10, model="gpt-4")
    assert max_chars >= 80  # At least 500 tokens * 0.16 chars/token


def test_truncate_context_no_truncation_needed() -> None:
    """Test truncate_context when text is short enough."""
    context = "Short text"
    result, was_truncated = truncate_context(context, max_length=100)

    assert result == context
    assert was_truncated is False


def test_truncate_context_empty_string() -> None:
    """Test truncate_context with empty string."""
    result, was_truncated = truncate_context("")

    assert result == ""
    assert was_truncated is False


def test_truncate_context_with_max_length() -> None:
    """Test truncate_context with explicit max_length."""
    context = "A" * 1000
    result, was_truncated = truncate_context(context, max_length=100)

    assert len(result) == 100
    assert result.endswith("...")
    assert was_truncated is True


def test_truncate_context_with_calculated_length() -> None:
    """Test truncate_context using calculated max_length."""
    context = "A" * 10000
    result, was_truncated = truncate_context(
        context,
        max_tokens=100,
        provider="openai",
        model="gpt-4",
    )

    assert was_truncated is True
    assert result.endswith("...")


def test_truncate_context_warning_enabled() -> None:
    """Test that warning is logged when truncation occurs."""
    context = "A" * 1000

    with patch("docs2synth.utils.text.logger") as mock_logger:
        truncate_context(context, max_length=100, warn=True)
        mock_logger.warning.assert_called_once()


def test_truncate_context_warning_disabled() -> None:
    """Test that warning is not logged when warn=False."""
    context = "A" * 1000

    with patch("docs2synth.utils.text.logger") as mock_logger:
        truncate_context(context, max_length=100, warn=False)
        mock_logger.warning.assert_not_called()


def test_truncate_context_ellipsis_added() -> None:
    """Test that ellipsis is added when truncating."""
    context = "Hello World! " * 100
    result, was_truncated = truncate_context(context, max_length=50)

    assert was_truncated is True
    assert result.endswith("...")
    assert len(result) == 50


def test_truncate_context_preserves_beginning() -> None:
    """Test that beginning of context is preserved."""
    context = "BEGINNING" + ("X" * 1000) + "END"
    result, was_truncated = truncate_context(context, max_length=50)

    assert result.startswith("BEGINNING")
    assert "END" not in result  # End should be cut off


def test_truncate_context_with_vllm() -> None:
    """Test truncate_context with vLLM configuration."""
    context = "A" * 10000
    result, was_truncated = truncate_context(
        context,
        provider="vllm",
        max_model_len=8192,
        max_tokens=500,
    )

    assert isinstance(result, str)
    assert was_truncated is True


def test_truncate_context_exact_boundary() -> None:
    """Test truncate_context when length exactly matches max."""
    context = "A" * 100
    result, was_truncated = truncate_context(context, max_length=100)

    assert result == context
    assert was_truncated is False


def test_truncate_context_just_over_boundary() -> None:
    """Test truncate_context when length is just over max."""
    context = "A" * 101
    result, was_truncated = truncate_context(context, max_length=100)

    assert len(result) == 100
    assert result.endswith("...")
    assert was_truncated is True


def test_calculate_max_context_length_with_all_params() -> None:
    """Test calculation with all parameters specified."""
    max_chars = calculate_max_context_length(
        max_tokens=1000,
        provider="anthropic",
        model="claude-3-opus",
        prompt_template_overhead=500,
    )

    assert isinstance(max_chars, int)
    assert max_chars > 0


def test_estimate_context_window_provider_case_variations() -> None:
    """Test provider matching with various case variations."""
    assert _estimate_context_window(provider="OpenAI") == 128000
    assert _estimate_context_window(provider="ANTHROPIC") == 200000
    assert _estimate_context_window(provider="Gemini") == 2000000


def test_truncate_context_returns_tuple() -> None:
    """Test that truncate_context always returns a tuple."""
    result = truncate_context("test", max_length=100)
    assert isinstance(result, tuple)
    assert len(result) == 2
    text, truncated = result
    assert isinstance(text, str)
    assert isinstance(truncated, bool)
