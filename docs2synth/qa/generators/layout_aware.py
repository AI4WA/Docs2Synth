"""Layout-aware question generation strategy.

Transforms questions to focus on spatial position/location in the document.
"""

from __future__ import annotations

from typing import Any, Optional

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.qa.generators.base import BaseQAGenerator


class LayoutAwareQAGenerator(BaseQAGenerator):
    """Generate layout-aware questions about spatial position.

    This strategy transforms a given question into a question about finding
    the position/location of the answer in the document image.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Change the question {question} to a very short question "
        "about finding the position of the answer from the input document image. "
        "For example, where is the answer of xx located?"
    )

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize layout-aware QA generator.

        Args:
            agent: Pre-configured AgentWrapper instance (optional)
            provider: Provider name if creating new agent
            model: Model name if creating new agent
            prompt_template: Custom prompt template (use {question} placeholder)
            **kwargs: Additional arguments for AgentWrapper initialization
        """
        super().__init__(agent=agent, provider=provider, model=model, **kwargs)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def generate(
        self,
        context: str = "",
        target: str = "",
        question: Optional[str] = None,
        image: Optional[Any] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a layout-aware question.

        Args:
            context: Document context (not used in this strategy, but kept for interface consistency)
            target: Target answer (not used in this strategy, but kept for interface consistency)
            question: Original question to transform (required for this strategy)
            image: Document image (required for understanding spatial layout and position)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to agent.generate() (can include 'question' or 'image' if not passed directly)

        Returns:
            Generated layout-aware question string

        Raises:
            ValueError: If question is not provided

        Note:
            This strategy transforms a question to ask about spatial position.
            Image is typically required to understand the document layout and answer positions.
            However, for text-only transformation (just rewriting the question format), image can be optional.
        """
        # Get question from kwargs if not provided directly
        if question is None:
            question = kwargs.pop("question", None)

        if question is None:
            raise ValueError(
                "LayoutAwareQAGenerator requires 'question' parameter to transform"
            )

        # Get image from kwargs if not provided directly
        if image is None:
            image = kwargs.pop("image", None)

        # Format prompt
        prompt = self.prompt_template.format(question=question)

        # Pass image to agent if provided (needed for understanding spatial layout)
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
