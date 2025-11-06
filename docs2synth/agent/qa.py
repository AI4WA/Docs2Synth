"""QA pair generation using LLM agents."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class QAGenerator:
    """Generate QA pairs from document content using LLM agents."""

    DEFAULT_SYSTEM_PROMPT = """You are an expert at creating high-quality question-answer pairs from document content.
Your task is to generate meaningful, clear, and accurate QA pairs that would be useful for training retrieval systems.

Guidelines:
- Generate questions that are specific and answerable from the given content
- Answers should be concise and directly extracted or derived from the content
- Avoid generic or overly broad questions
- Ensure questions test different aspects (facts, reasoning, concepts)
- Format your response as JSON with "question" and "answer" fields
"""

    DEFAULT_QA_PROMPT_TEMPLATE = """Based on the following document content, generate a high-quality question-answer pair.

Document Content:
{content}

Generate a single QA pair in JSON format:
{{
    "question": "Your question here",
    "answer": "Your answer here"
}}
"""

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize QA generator.

        Args:
            agent: Pre-configured AgentWrapper instance (optional)
            provider: Provider name if creating new agent
            model: Model name if creating new agent
            system_prompt: Custom system prompt for QA generation
            prompt_template: Custom prompt template (use {content} placeholder)
            **kwargs: Additional arguments for AgentWrapper initialization

        Example:
            >>> # Using existing agent
            >>> generator = QAGenerator(agent=my_agent)
            >>>
            >>> # Create new agent
            >>> generator = QAGenerator(provider="openai", model="gpt-4", api_key="sk-...")
        """
        if agent is None and provider is None:
            raise ValueError("Either 'agent' or 'provider' must be specified")

        if agent is None:
            self.agent = AgentWrapper(provider=provider, model=model, **kwargs)
        else:
            self.agent = agent

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.prompt_template = prompt_template or self.DEFAULT_QA_PROMPT_TEMPLATE

    def generate_qa_pair(
        self,
        content: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Generate a single QA pair from content.

        Args:
            content: Document content to generate QA pair from
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for LLM generation

        Returns:
            Dictionary with 'question' and 'answer' keys

        Example:
            >>> generator = QAGenerator(provider="openai", model="gpt-4")
            >>> qa = generator.generate_qa_pair("Python is a programming language...")
            >>> print(f"Q: {qa['question']}")
            >>> print(f"A: {qa['answer']}")
        """
        prompt = self.prompt_template.format(content=content)

        # Automatically use JSON mode for QA generation
        response = self.agent.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",  # Force JSON mode for structured output
            **kwargs,
        )

        # Parse JSON response
        try:
            # Try to extract JSON from response (might have extra text)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            qa_pair = json.loads(content)
            if "question" not in qa_pair or "answer" not in qa_pair:
                raise ValueError("QA pair must contain 'question' and 'answer' fields")

            return {"question": qa_pair["question"], "answer": qa_pair["answer"]}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response.content}")
            raise ValueError(f"Invalid JSON response from LLM: {e}") from e

    def generate_qa_pairs(
        self,
        contents: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """Generate multiple QA pairs from a list of contents.

        Args:
            contents: List of document content strings
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for LLM generation

        Returns:
            List of QA pair dictionaries

        Example:
            >>> contents = ["Content 1...", "Content 2...", "Content 3..."]
            >>> qa_pairs = generator.generate_qa_pairs(contents)
            >>> for qa in qa_pairs:
            ...     print(f"Q: {qa['question']}")
        """
        qa_pairs = []
        for i, content in enumerate(contents):
            logger.info(f"Generating QA pair {i+1}/{len(contents)}")
            try:
                qa_pair = self.generate_qa_pair(
                    content, temperature=temperature, max_tokens=max_tokens, **kwargs
                )
                qa_pairs.append(qa_pair)
            except Exception as e:
                logger.error(f"Failed to generate QA pair for content {i+1}: {e}")
                # Continue with next content instead of failing completely
                continue

        return qa_pairs

    def generate_with_verification(
        self,
        content: str,
        meaningful_check: bool = True,
        correctness_check: bool = True,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Optional[Dict[str, str]]:
        """Generate QA pair with two-step verification.

        Args:
            content: Document content
            meaningful_check: Whether to verify meaningfulness
            correctness_check: Whether to verify correctness
            temperature: Sampling temperature
            **kwargs: Additional arguments for LLM generation

        Returns:
            QA pair dict if passes verification, None otherwise
        """
        # Generate initial QA pair
        qa_pair = self.generate_qa_pair(content, temperature=temperature, **kwargs)

        if meaningful_check:
            if not self._check_meaningful(qa_pair, content):
                logger.warning("QA pair failed meaningfulness check")
                return None

        if correctness_check:
            if not self._check_correctness(qa_pair, content):
                logger.warning("QA pair failed correctness check")
                return None

        return qa_pair

    def _check_meaningful(self, qa_pair: Dict[str, str], content: str) -> bool:
        """Check if QA pair is meaningful."""
        prompt = f"""Is the following QA pair meaningful and useful for a retrieval system?

Question: {qa_pair['question']}
Answer: {qa_pair['answer']}

Respond with only "yes" or "no"."""
        response = self.agent.generate(prompt, temperature=0.1)
        return "yes" in response.content.lower()

    def _check_correctness(self, qa_pair: Dict[str, str], content: str) -> bool:
        """Check if QA pair is correct based on content."""
        prompt = f"""Is the following answer correct based on the given content?

Content: {content}

Question: {qa_pair['question']}
Answer: {qa_pair['answer']}

Respond with only "yes" or "no"."""
        response = self.agent.generate(prompt, temperature=0.1)
        return "yes" in response.content.lower()
