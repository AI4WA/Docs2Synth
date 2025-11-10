"""Semantic question generation strategy.

Generates natural language questions based on context and target answer.
"""

from __future__ import annotations

from typing import Any, Optional

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.qa.generators.base import BaseQAGenerator
from docs2synth.utils.text import truncate_context


class SemanticQAGenerator(BaseQAGenerator):
    """Generate semantic questions from context and target.

    This strategy generates human-asked short questions where the answer
    matches the target exactly. It uses both document context and visual
    information (if available through the agent).
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Context: {context}\n"
        "Based on the above context and target document image, "
        "generate a human-asked SHORT question (output question only) "
        'of which answer is exactly same as "{target}"'
    )

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize semantic QA generator.

        Args:
            agent: Pre-configured AgentWrapper instance (optional)
            provider: Provider name if creating new agent
            model: Model name if creating new agent
            prompt_template: Custom prompt template (use {context} and {target} placeholders)
            **kwargs: Additional arguments for AgentWrapper initialization
        """
        super().__init__(agent=agent, provider=provider, model=model, **kwargs)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def generate(
        self,
        context: str,
        target: str,
        image: Optional[Any] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a semantic question.

        Args:
            context: Document context (OCR text)
            target: Target answer/object to generate question for
            image: Document image (required for vision models, optional for text-only models)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to agent.generate() (can include 'image' if not passed directly)

        Returns:
            Generated semantic question string

        Note:
            For vision-language models (e.g., Qwen2-VL), image is typically required.
            For text-only models, image is optional and will be ignored.
        """
        # Get image from kwargs if not provided directly
        if image is None:
            image = kwargs.pop("image", None)

        # Truncate context if too long
        # Note: In batch processing, context is already truncated in qa_batch.py
        # This truncation is for direct calls (e.g., from CLI)
        # Get provider and model from agent if available
        provider = None
        model = None
        max_model_len = kwargs.pop(
            "max_model_len", None
        )  # Allow passing max_model_len via kwargs
        if hasattr(self.agent, "provider_name"):
            provider = self.agent.provider_name
        if hasattr(self.agent, "model"):
            model = self.agent.model

        # Always truncate to ensure we don't exceed limits
        # Even if context was truncated before, we need to ensure it's within limits
        # (in case truncation parameters changed or context was truncated with different settings)
        truncated_context, was_truncated = truncate_context(
            context,
            max_tokens=max_tokens,
            provider=provider,
            model=model,
            max_model_len=max_model_len,
        )
        if was_truncated:
            self.logger.warning(
                f"Context truncated in semantic generator (original length: {len(context)} chars, "
                f"truncated to: {len(truncated_context)} chars)"
            )

        # Format prompt
        prompt = self.prompt_template.format(context=truncated_context, target=target)

        # Pass image to agent if provided
        if image is not None:
            kwargs["image"] = image

        # Generate using agent
        response = self.agent.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return response.content.strip()
