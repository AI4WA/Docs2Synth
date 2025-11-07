"""Meaningful verification strategy.

Verifies whether target information was entered by the form user (not part of form template).
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.qa.verifiers.base import BaseQAVerifier


class MeaningfulVerifier(BaseQAVerifier):
    """Verify meaningfulness of target information.

    This strategy checks whether target information was entered by the form user
    (not part of the form template), based on the document context and image.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Context: {context}\n"
        "Based on the document context and image, determine if the question and answer pair is meaningful. "
        "A meaningful QA pair should:\n"
        "1. Have a clear, answerable question\n"
        "2. Have an answer that can be found or inferred from the document\n"
        "3. Be useful for understanding the document content\n"
        'Output format (JSON with double quotes): {{"Response": "Yes/No", "Explanation": "xxx"}}.'
    )

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize meaningful verifier.

        Args:
            agent: Pre-configured AgentWrapper instance (optional)
            provider: Provider name if creating new agent
            model: Model name if creating new agent
            prompt_template: Custom prompt template (use {context} and {target} placeholders)
            **kwargs: Additional arguments for AgentWrapper initialization
        """
        super().__init__(agent=agent, provider=provider, model=model, **kwargs)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def _parse_response(self, raw_output: str) -> dict[str, Any]:
        """Parse agent response into structured result.

        Args:
            raw_output: Raw response from agent

        Returns:
            Dictionary with parsed response data
        """
        result: dict[str, Any] = {"raw_output": raw_output}
        parsed = None

        if "{" in raw_output and "}" in raw_output:
            # Extract the dict/JSON portion
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            dict_str = raw_output[start:end]

            # First try: Parse as JSON (double quotes)
            try:
                parsed = json.loads(dict_str)
            except json.JSONDecodeError:
                # Second try: Parse as Python dict (single quotes) using ast.literal_eval
                try:
                    parsed = ast.literal_eval(dict_str)
                    if not isinstance(parsed, dict):
                        parsed = None
                except (ValueError, SyntaxError):
                    parsed = None

        if parsed:
            result.update(parsed)
        else:
            # Fallback: try to extract Yes/No from text
            response_lower = raw_output.lower()
            if "yes" in response_lower:
                result["Response"] = "Yes"
            elif "no" in response_lower:
                result["Response"] = "No"
            else:
                result["Response"] = "Unknown"
            # Try to extract explanation if present
            if "explanation" in response_lower or "explanation" in raw_output:
                # Try to find explanation text
                explanation_match = re.search(
                    r"(?:explanation|Explanation)[:\s]+['\"]([^'\"]+)['\"]",
                    raw_output,
                    re.IGNORECASE,
                )
                if explanation_match:
                    result["Explanation"] = explanation_match.group(1)

        return result

    def verify(
        self,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        context: Optional[str] = None,
        image: Optional[Any] = None,
        target: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Verify meaningfulness of question and answer pair.

        Args:
            question: Generated question (required)
            answer: Target answer (required)
            context: Document context (OCR text from document) - optional if image is provided
            image: Document image (required if context is not provided, for vision models)
            target: Deprecated, use 'answer' instead
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to agent.generate()

        Returns:
            Dictionary with verification results:
            - 'response': 'Yes' or 'No'
            - 'explanation': Explanation string (if provided)
            - 'raw_output': Raw model output

        Note:
            Either 'context' or 'image' must be provided. If image is provided,
            context can be empty (vision model can read from image).
        """
        # Use question and answer (target is deprecated)
        if question is None:
            raise ValueError("'question' is required for meaningful verification")
        if answer is None and target is None:
            raise ValueError("Either 'answer' or 'target' must be provided")
        answer_value = answer if answer is not None else target

        # Context is optional if image is provided (vision model can read from image)
        if context is None and image is None:
            raise ValueError(
                "Either 'context' or 'image' must be provided for meaningful verification"
            )

        # Use empty context if not provided but image is available
        context_value = (
            context
            if context is not None
            else "(No text context provided, use image only)"
        )

        # Format prompt
        prompt = self.prompt_template.format(
            question=question, answer=answer_value, context=context_value
        )

        # Pass image to agent if provided
        if image is not None:
            kwargs["image"] = image

        # Generate using agent with JSON response format
        response = self.agent.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
            **kwargs,
        )

        raw_output = response.content.strip()
        result = self._parse_response(raw_output)

        # Normalize key names
        if "Response" in result:
            result["response"] = result.pop("Response")
        if "Explanation" in result:
            result["explanation"] = result.pop("Explanation")

        return result
