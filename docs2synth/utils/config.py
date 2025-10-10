"""Configuration loader for Docs2Synth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .logging import get_logger

logger = get_logger(__name__)


class Config:
    """Configuration manager for Docs2Synth."""

    def __init__(self, config_dict: Dict[str, Any] | None = None):
        """Initialize configuration.

        Args:
            config_dict: Configuration dictionary. If None, uses defaults.
        """
        self._config = config_dict or self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {
                "root_dir": "./data",
                "datasets_dir": "./data/datasets",
                "processed_dir": "./data/processed",
                "qa_pairs_dir": "./data/qa_pairs",
                "models_dir": "./models",
                "logs_dir": "./logs",
            },
        }

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Config:
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance

        Example:
            >>> config = Config.from_yaml("config.yml")
            >>> print(config.get("datasets.output_dir"))
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        logger.info(f"Loading config from {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Merge with defaults
        default_config = cls._get_default_config()
        merged_config = cls._merge_configs(default_config, config_dict or {})

        return cls(merged_config)

    @staticmethod
    def _merge_configs(
        base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Config._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Supports dot notation for nested keys.

        Args:
            key: Configuration key (e.g., "datasets.output_dir")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config.get("datasets.output_dir")
            './data'
            >>> config.get("datasets.cache_downloads")
            True
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.

        Supports dot notation for nested keys.

        Args:
            key: Configuration key (e.g., "datasets.output_dir")
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def save(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving config to {yaml_path}")

        with open(yaml_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config})"


# Global config instance
_global_config: Config | None = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance.

    Args:
        config: Config instance to set as global
    """
    global _global_config
    _global_config = config


def load_config(yaml_path: str | Path) -> Config:
    """Load configuration from YAML and set as global.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Loaded Config instance
    """
    config = Config.from_yaml(yaml_path)
    set_config(config)
    return config
