"""Factory for creating QA verifiers based on verifier type."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from docs2synth.agent.wrapper import AgentWrapper
from docs2synth.qa.config import QAVerificationConfig, QAVerifierConfig
from docs2synth.qa.verifiers.base import BaseQAVerifier
from docs2synth.qa.verifiers.correctness import CorrectnessVerifier
from docs2synth.qa.verifiers.meaningful import MeaningfulVerifier
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)

# Verifier registry
VERIFIER_REGISTRY: Dict[str, Type[BaseQAVerifier]] = {
    "correctness": CorrectnessVerifier,
    "meaningful": MeaningfulVerifier,
}


class QAVerifierFactory:
    """Factory for creating QA verifiers based on verifier type."""

    @staticmethod
    def create(
        verifier_type: str,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseQAVerifier:
        """Create a QA verifier instance.

        Args:
            verifier_type: Verifier type (correctness, meaningful)
            agent: Pre-configured AgentWrapper instance (optional)
            provider: Provider name if creating new agent
            model: Model name if creating new agent
            **kwargs: Additional arguments passed to verifier and AgentWrapper

        Returns:
            QA verifier instance

        Raises:
            ValueError: If verifier_type is not registered

        Example:
            >>> # Using existing agent
            >>> agent = AgentWrapper(provider="openai", model="gpt-4")
            >>> verifier = QAVerifierFactory.create(
            ...     verifier_type="correctness",
            ...     agent=agent
            ... )
            >>> result = verifier.verify(question="...", answer="...")
            >>>
            >>> # Or create agent automatically
            >>> verifier = QAVerifierFactory.create(
            ...     verifier_type="meaningful",
            ...     provider="openai",
            ...     model="gpt-4"
            ... )
        """
        verifier_type_lower = verifier_type.lower().strip()

        if verifier_type_lower not in VERIFIER_REGISTRY:
            available = ", ".join(sorted(VERIFIER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown QA verifier type: {verifier_type}. "
                f"Available verifiers: {available}"
            )

        verifier_class = VERIFIER_REGISTRY[verifier_type_lower]
        logger.info(f"Creating {verifier_class.__name__} verifier")

        return verifier_class(
            agent=agent,
            provider=provider,
            model=model,
            **kwargs,
        )

    @staticmethod
    def register(verifier_type: str, verifier_class: Type[BaseQAVerifier]) -> None:
        """Register a new QA verifier type.

        Args:
            verifier_type: Verifier type name
            verifier_class: Verifier class (must inherit from BaseQAVerifier)

        Example:
            >>> class CustomVerifier(BaseQAVerifier):
            ...     def verify(self, ...):
            ...         ...
            >>> QAVerifierFactory.register("custom", CustomVerifier)
        """
        if not issubclass(verifier_class, BaseQAVerifier):
            raise TypeError(
                f"Verifier class must inherit from BaseQAVerifier, "
                f"got {verifier_class}"
            )

        verifier_type_lower = verifier_type.lower().strip()
        VERIFIER_REGISTRY[verifier_type_lower] = verifier_class
        logger.info(f"Registered QA verifier type: {verifier_type_lower}")

    @staticmethod
    def list_verifiers() -> list[str]:
        """List all registered verifier types.

        Returns:
            List of verifier type names
        """
        return sorted(VERIFIER_REGISTRY.keys())

    @staticmethod
    def create_from_config(
        verifier_config: QAVerifierConfig,
        config_path: Optional[str] = None,
    ) -> BaseQAVerifier:
        """Create a QA verifier from configuration.

        Args:
            verifier_config: QAVerifierConfig instance
            config_path: Path to config.yml (optional, for AgentWrapper)

        Returns:
            QA verifier instance

        Example:
            >>> from docs2synth.qa.config import QAVerifierConfig
            >>> config = QAVerifierConfig(
            ...     verifier_type="correctness",
            ...     provider="openai",
            ...     model="gpt-4",
            ...     prompt_template="Custom template..."
            ... )
            >>> verifier = QAVerifierFactory.create_from_config(config)
        """
        # Build kwargs for verifier initialization
        verifier_kwargs: Dict[str, Any] = {}
        if verifier_config.prompt_template:
            verifier_kwargs["prompt_template"] = verifier_config.prompt_template

        # Build kwargs for AgentWrapper
        agent_kwargs: Dict[str, Any] = {}
        if config_path:
            agent_kwargs["config_path"] = config_path
        if verifier_config.temperature is not None:
            agent_kwargs["temperature"] = verifier_config.temperature
        if verifier_config.max_tokens is not None:
            agent_kwargs["max_tokens"] = verifier_config.max_tokens
        agent_kwargs.update(verifier_config.extra_kwargs)

        return QAVerifierFactory.create(
            verifier_type=verifier_config.verifier_type,
            provider=verifier_config.provider,
            model=verifier_config.model,
            **verifier_kwargs,
            **agent_kwargs,
        )

    @staticmethod
    def create_all_from_config(
        verification_config: QAVerificationConfig,
        config_path: Optional[str] = None,
    ) -> List[BaseQAVerifier]:
        """Create all QA verifiers from configuration.

        Args:
            verification_config: QAVerificationConfig instance
            config_path: Path to config.yml (optional, for AgentWrapper)

        Returns:
            List of QA verifier instances

        Example:
            >>> from docs2synth.qa.config import QAVerificationConfig
            >>> verification_config = QAVerificationConfig.from_yaml("config.yml")
            >>> verifiers = QAVerifierFactory.create_all_from_config(verification_config)
            >>> for ver in verifiers:
            ...     print(ver)
        """
        verifiers = []
        for verifier_config in verification_config.verifiers:
            try:
                verifier = QAVerifierFactory.create_from_config(
                    verifier_config, config_path=config_path
                )
                verifiers.append(verifier)
            except Exception as e:
                logger.error(
                    f"Failed to create verifier for type '{verifier_config.verifier_type}': {e}"
                )
                continue
        return verifiers
