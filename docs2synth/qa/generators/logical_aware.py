"""Logical-aware question generation strategy.

Transforms questions to focus on document sections/structure.
"""

from __future__ import annotations

from typing import Any, Optional

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.qa.generators.base import BaseQAGenerator


class LogicalAwareQAGenerator(BaseQAGenerator):
    """Generate logical-aware questions about document sections.

    This strategy transforms a given question into a question about finding
    the belonging sections of the answer in the document.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Change the question {question} to a very short question "
        "about finding the belonging sections of the answer from the input document. "
        "For example, which section you could find the information about the xx?"
    )

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize logical-aware QA generator.

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
        """Generate a logical-aware question.

        Args:
            context: Document context (not used in this strategy, but kept for interface consistency)
            target: Target answer (not used in this strategy, but kept for interface consistency)
            question: Original question to transform (required for this strategy)
            image: Document image (optional, can help understand document structure)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to agent.generate() (can include 'question' or 'image' if not passed directly)

        Returns:
            Generated logical-aware question string

        Raises:
            ValueError: If question is not provided

        Note:
            This strategy transforms an existing question. Image is optional:
            - If provided: Can help model understand document structure and sections
            - If not provided: Pure text-based transformation (works without image)
        """
        # Get question from kwargs if not provided directly
        if question is None:
            question = kwargs.pop("question", None)

        if question is None:
            raise ValueError(
                "LogicalAwareQAGenerator requires 'question' parameter to transform"
            )

        # Get image from kwargs if not provided directly
        if image is None:
            image = kwargs.pop("image", None)

        # Format prompt
        prompt = self.prompt_template.format(question=question)

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
