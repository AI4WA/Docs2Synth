"""Text processing utilities for Docs2Synth.

This module provides utilities for text manipulation including context truncation.
"""

from __future__ import annotations

from typing import Optional

from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)

# Default max context length in characters (approximately 8000 chars ≈ 2000 tokens)
# This is a conservative default that should work with most models
DEFAULT_MAX_CONTEXT_LENGTH = 8000

# Approximate characters per token (varies by language)
# For English: ~4 chars/token, for Chinese: ~1-2 chars/token
# However, actual tokenization can be MUCH denser, especially with:
# - Special characters, punctuation, whitespace
# - Mixed content (English + Chinese + numbers)
# - Image tokens in multimodal models (images can consume 1000-3000+ tokens each)
# Using 0.16 char/token (i.e., ~6 tokens/char) as an extremely conservative estimate
# Based on actual observations: 1930 chars → 11892 tokens ≈ 6.16 tokens/char
# This accounts for both dense text tokenization AND image tokens
CHARS_PER_TOKEN = 0.16  # Means ~6 tokens per character on average (very conservative)

# Default context window sizes for common models (in tokens)
# These are conservative estimates - actual values may vary
DEFAULT_CONTEXT_WINDOWS = {
    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    # Anthropic models
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    # Gemini models
    "gemini-2.5-pro": 2000000,
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-flash": 1000000,
    # Default fallback
    "default": 8192,
}


def _estimate_context_window(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    """Estimate context window size based on provider and model.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet')

    Returns:
        Estimated context window size in tokens
    """
    # Try to match model name first
    if model:
        model_lower = model.lower()
        for model_key, window_size in DEFAULT_CONTEXT_WINDOWS.items():
            if model_key in model_lower:
                return window_size

    # Fallback to provider-based defaults
    if provider:
        provider_lower = provider.lower()
        if "openai" in provider_lower:
            return 128000  # Modern OpenAI models
        elif "anthropic" in provider_lower or "claude" in provider_lower:
            return 200000  # Claude models
        elif "gemini" in provider_lower or "google" in provider_lower:
            return 2000000  # Gemini models
        elif "doubao" in provider_lower:
            return 32768  # Doubao models often have 32k context
        elif "vllm" in provider_lower:
            return 8192  # Default for vLLM, but can be configured
        elif "huggingface" in provider_lower or "ollama" in provider_lower:
            return 4096  # Common for local models

    # Ultimate fallback
    return DEFAULT_CONTEXT_WINDOWS["default"]


def calculate_max_context_length(
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    prompt_template_overhead: int = 300,  # Estimated tokens for prompt template + target
    max_model_len: Optional[
        int
    ] = None,  # For vLLM, can specify max_model_len from config
) -> int:
    """Calculate maximum context length in characters based on model configuration.

    Args:
        max_tokens: Maximum output tokens (from config)
        provider: Provider name (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet')
        prompt_template_overhead: Estimated tokens for prompt template + target (default: 300)
        max_model_len: Maximum model length for vLLM (from config, overrides estimate)

    Returns:
        Maximum context length in characters
    """
    # For vLLM, use max_model_len from config if provided
    if provider and "vllm" in provider.lower() and max_model_len is not None:
        context_window_tokens = max_model_len
        logger.debug(f"Using max_model_len from config: {max_model_len} tokens")
    else:
        # Estimate context window
        context_window_tokens = _estimate_context_window(provider, model)

    # Reserve tokens for:
    # - max_tokens (output)
    # - prompt_template_overhead (template + target text, typically 200-300 tokens)
    # - image_tokens_estimate (for vision models, images can consume significant tokens)
    #   A single high-resolution image can easily consume 1500-3000+ tokens
    #   Based on observations: images are a major contributor to token count
    # - safety margin (for system messages, formatting, etc.)
    image_tokens_estimate = 2000  # Very conservative estimate for image tokens
    safety_margin = 200
    reserved_tokens = (
        (max_tokens or 1000)
        + prompt_template_overhead
        + image_tokens_estimate
        + safety_margin
    )

    # Calculate available tokens for context
    # Ensure at least 500 tokens available (very conservative minimum)
    available_tokens = max(500, context_window_tokens - reserved_tokens)

    # Convert to characters (using extremely conservative 0.16 char/token ≈ 6 tokens/char)
    # Based on actual observations: 1930 chars → 11892 tokens ≈ 6.16 tokens/char
    # This accounts for:
    # - Dense text tokenization (Chinese, special chars, punctuation, whitespace)
    # - Image tokens (which are included in the total prompt token count)
    # - Mixed content tokenization overhead
    max_chars = int(available_tokens * CHARS_PER_TOKEN)

    logger.debug(
        f"Calculated max context length: {max_chars} chars "
        f"(context_window={context_window_tokens} tokens, "
        f"max_tokens={max_tokens}, reserved={reserved_tokens} tokens, "
        f"available={available_tokens} tokens)"
    )

    return max_chars


def truncate_context(
    context: str,
    max_length: Optional[int] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_model_len: Optional[int] = None,  # For vLLM, max_model_len from config
    warn: bool = True,
) -> tuple[str, bool]:
    """Truncate context if it exceeds the maximum length.

    Args:
        context: The context string to potentially truncate
        max_length: Maximum allowed length in characters (optional, calculated if not provided)
        max_tokens: Maximum output tokens (used to calculate max_length if not provided)
        provider: Provider name (used to calculate max_length if not provided)
        model: Model name (used to calculate max_length if not provided)
        warn: Whether to log a warning when truncation occurs (default: True)

    Returns:
        Tuple of (truncated_context, was_truncated)
        - truncated_context: The context, truncated if necessary
        - was_truncated: Boolean indicating if truncation occurred

    Example:
        >>> context = "A very long context..." * 1000
        >>> truncated, was_cut = truncate_context(context, max_tokens=1000, provider="openai")
        >>> if was_cut:
        ...     print("Context was too long and was truncated")
    """
    if not context:
        return context, False

    # Calculate max_length if not provided
    if max_length is None:
        max_length = calculate_max_context_length(
            max_tokens=max_tokens,
            provider=provider,
            model=model,
            max_model_len=max_model_len,
        )

    original_length = len(context)
    if original_length <= max_length:
        return context, False

    # Truncate and add ellipsis to indicate truncation
    truncated = context[: max_length - 3] + "..."
    was_truncated = True

    if warn:
        logger.warning(
            f"Context length ({original_length} chars) exceeds maximum ({max_length} chars). "
            f"Truncated to {len(truncated)} chars. "
            f"Some information may be lost. "
            f"(Based on max_tokens={max_tokens}, provider={provider}, model={model})"
        )

    return truncated, was_truncated
