"""Correctness verification strategy.

Verifies whether a given answer could be the expected answer to a question.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.qa.verifiers.base import BaseQAVerifier


class CorrectnessVerifier(BaseQAVerifier):
    """Verify correctness of QA pairs.

    This strategy checks whether a target answer could be the expected answer
    to a given question, ignoring context information and domain knowledge.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Ignore the context information and domain knowledge (e.g. ABN or ACN/ARSN). "
        "Just consider whether '{answer}' could be the expected answer to the question '{question}'. "
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
        """Initialize correctness verifier.

        Args:
            agent: Pre-configured AgentWrapper instance (optional)
            provider: Provider name if creating new agent
            model: Model name if creating new agent
            prompt_template: Custom prompt template (use {question} and {answer} placeholders)
            **kwargs: Additional arguments for AgentWrapper initialization
        """
        super().__init__(agent=agent, provider=provider, model=model, **kwargs)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def _escape_template_braces(self, template: str) -> str:
        """Escape braces in template to prevent format string errors.

        Args:
            template: Template string that may contain unescaped braces

        Returns:
            Template with braces properly escaped
        """
        # Pattern 1: Python dict format {'Response': 'Yes/No', 'Explanation': 'xxx'}
        if "'Response'" in template and "'Explanation'" in template:
            pattern = (
                r"(?<!\{)\{'Response':\s*'[^']*',\s*'Explanation':\s*'[^']*'\}(?!\})"
            )

            def escape_braces(match):
                content = match.group(0)
                return content.replace("{", "{{").replace("}", "}}")

            template = re.sub(pattern, escape_braces, template)
        # Pattern 2: JSON format {"Response": "Yes/No", "Explanation": "xxx"}
        if '"Response"' in template and '"Explanation"' in template:
            pattern = (
                r'(?<!\{)\{"Response":\s*"[^"]*",\s*"Explanation":\s*"[^"]*"\}(?!\})'
            )

            def escape_braces(match):
                content = match.group(0)
                return content.replace("{", "{{").replace("}", "}}")

            template = re.sub(pattern, escape_braces, template)
        return template

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
        question: str,
        answer: str,
        context: Optional[str] = None,
        image: Optional[Any] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Verify correctness of a QA pair.

        Args:
            question: Generated question
            answer: Target answer
            context: Document context (ignored for correctness verification)
            image: Document image (optional, typically not needed for correctness)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to agent.generate()

        Returns:
            Dictionary with verification results:
            - 'response': 'Yes' or 'No'
            - 'explanation': Explanation string (if provided)
            - 'raw_output': Raw model output
        """
        # Format prompt with escaped braces
        template = self._escape_template_braces(self.prompt_template)
        try:
            prompt = template.format(question=question, answer=answer)
        except KeyError:
            # If still fails, log the template and re-raise
            self.logger.error(
                f"Failed to format prompt template. Template: {repr(self.prompt_template[:300])}"
            )
            raise

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
