"""Configuration loader for QA generation strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from docs2synth.utils.config import Config
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class QAStrategyConfig:
    """Configuration for a single QA generation strategy."""

    def __init__(
        self,
        strategy: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize QA strategy configuration.

        Args:
            strategy: Strategy name (semantic, layout_aware, logical_aware)
            provider: Provider name (openai, anthropic, gemini, etc.)
            model: Model name (optional, uses provider default if not specified)
            prompt_template: Custom prompt template (optional, uses default if not specified)
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional configuration parameters
        """
        self.strategy = strategy
        self.provider = provider
        self.model = model
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "strategy": self.strategy,
        }
        if self.provider:
            result["provider"] = self.provider
        if self.model:
            result["model"] = self.model
        if self.prompt_template:
            result["prompt_template"] = self.prompt_template
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        result.update(self.extra_kwargs)
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> QAStrategyConfig:
        """Create from dictionary."""
        strategy = config_dict.pop("strategy")
        provider = config_dict.pop("provider", None)
        model = config_dict.pop("model", None)
        prompt_template = config_dict.pop("prompt_template", None)
        temperature = config_dict.pop("temperature", None)
        max_tokens = config_dict.pop("max_tokens", None)

        return cls(
            strategy=strategy,
            provider=provider,
            model=model,
            prompt_template=prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
            **config_dict,
        )


class QAGenerationConfig:
    """Configuration for QA generation with multiple strategies."""

    def __init__(self, strategies: List[QAStrategyConfig]):
        """Initialize QA generation configuration.

        Args:
            strategies: List of QA strategy configurations
        """
        self.strategies = strategies

    @classmethod
    def from_config(cls, config: Config) -> Optional[QAGenerationConfig]:
        """Load QA generation configuration from Config object.

        Args:
            config: Config instance

        Returns:
            QAGenerationConfig if qa section exists, None otherwise
        """
        qa_config = config.get("qa")
        if qa_config is None:
            return None

        strategies = []
        if isinstance(qa_config, list):
            # List of strategy configs
            for strategy_dict in qa_config:
                if isinstance(strategy_dict, dict):
                    strategies.append(QAStrategyConfig.from_dict(strategy_dict))
        elif isinstance(qa_config, dict):
            # Single strategy config or dict with strategies key
            if "strategies" in qa_config:
                # Multiple strategies under "strategies" key
                for strategy_dict in qa_config["strategies"]:
                    if isinstance(strategy_dict, dict):
                        strategies.append(QAStrategyConfig.from_dict(strategy_dict))
            else:
                # Single strategy config
                strategies.append(QAStrategyConfig.from_dict(qa_config))

        if not strategies:
            logger.warning("No QA strategies found in config")
            return None

        return cls(strategies=strategies)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Optional[QAGenerationConfig]:
        """Load QA generation configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            QAGenerationConfig if qa section exists, None otherwise
        """
        config = Config.from_yaml(yaml_path)
        return cls.from_config(config)

    def get_strategy_config(self, strategy: str) -> Optional[QAStrategyConfig]:
        """Get configuration for a specific strategy.

        Args:
            strategy: Strategy name

        Returns:
            QAStrategyConfig if found, None otherwise
        """
        for config in self.strategies:
            if config.strategy == strategy:
                return config
        return None

    def list_strategies(self) -> List[str]:
        """List all configured strategy names.

        Returns:
            List of strategy names
        """
        return [s.strategy for s in self.strategies]


class QAVerifierConfig:
    """Configuration for a single QA verifier."""

    def __init__(
        self,
        verifier_type: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize QA verifier configuration.

        Args:
            verifier_type: Verifier type (correctness, meaningful)
            provider: Provider name (openai, anthropic, gemini, etc.)
            model: Model name (optional, uses provider default if not specified)
            prompt_template: Custom prompt template (optional, uses default if not specified)
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional configuration parameters
        """
        self.verifier_type = verifier_type
        self.provider = provider
        self.model = model
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "verifier_type": self.verifier_type,
        }
        if self.provider:
            result["provider"] = self.provider
        if self.model:
            result["model"] = self.model
        if self.prompt_template:
            result["prompt_template"] = self.prompt_template
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        result.update(self.extra_kwargs)
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> QAVerifierConfig:
        """Create from dictionary."""
        verifier_type = config_dict.pop("verifier_type")
        provider = config_dict.pop("provider", None)
        model = config_dict.pop("model", None)
        prompt_template = config_dict.pop("prompt_template", None)
        temperature = config_dict.pop("temperature", None)
        max_tokens = config_dict.pop("max_tokens", None)

        return cls(
            verifier_type=verifier_type,
            provider=provider,
            model=model,
            prompt_template=prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
            **config_dict,
        )


class QAVerificationConfig:
    """Configuration for QA verification with multiple verifiers."""

    def __init__(self, verifiers: List[QAVerifierConfig]):
        """Initialize QA verification configuration.

        Args:
            verifiers: List of QA verifier configurations
        """
        self.verifiers = verifiers

    @classmethod
    def from_config(cls, config: Config) -> Optional[QAVerificationConfig]:
        """Load QA verification configuration from Config object.

        Args:
            config: Config instance

        Returns:
            QAVerificationConfig if verifiers section exists, None otherwise

        Note:
            Verifiers can be at top level (verifiers:) or nested under qa (qa.verifiers:)
        """
        # First check for top-level verifiers
        verifiers_config = config.get("verifiers")

        # If not found, check for nested verifiers under qa
        if verifiers_config is None:
            qa_config = config.get("qa")
            if isinstance(qa_config, dict) and "verifiers" in qa_config:
                verifiers_config = qa_config["verifiers"]
            else:
                return None

        verifiers = []
        if isinstance(verifiers_config, list):
            # List of verifier configs
            for verifier_dict in verifiers_config:
                if isinstance(verifier_dict, dict):
                    verifiers.append(QAVerifierConfig.from_dict(verifier_dict))
        elif isinstance(verifiers_config, dict):
            # Single verifier config
            verifiers.append(QAVerifierConfig.from_dict(verifiers_config))

        if not verifiers:
            logger.warning("No QA verifiers found in config")
            return None

        return cls(verifiers=verifiers)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Optional[QAVerificationConfig]:
        """Load QA verification configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            QAVerificationConfig if qa.verifiers section exists, None otherwise
        """
        config = Config.from_yaml(yaml_path)
        return cls.from_config(config)

    def get_verifier_config(self, verifier_type: str) -> Optional[QAVerifierConfig]:
        """Get configuration for a specific verifier type.

        Args:
            verifier_type: Verifier type name

        Returns:
            QAVerifierConfig if found, None otherwise
        """
        for config in self.verifiers:
            if config.verifier_type == verifier_type:
                return config
        return None

    def list_verifiers(self) -> List[str]:
        """List all configured verifier types.

        Returns:
            List of verifier type names
        """
        return [v.verifier_type for v in self.verifiers]
