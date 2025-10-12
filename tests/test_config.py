"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from docs2synth.utils.config import Config, get_config, load_config, set_config


class TestConfig:
    """Tests for Config class."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        config = Config()
        assert config.get("data.root_dir") == "./data"
        assert config.get("data.datasets_dir") == "./data/datasets"
        assert config.get("data.processed_dir") == "./data/processed"

    def test_init_with_custom_dict(self):
        """Test initialization with custom config dict."""
        custom_config = {"data": {"root_dir": "/custom/path"}}
        config = Config(custom_config)
        assert config.get("data.root_dir") == "/custom/path"

    def test_from_yaml_valid_file(self, tmp_path):
        """Test loading config from valid YAML file."""
        config_file = tmp_path / "config.yml"
        config_data = {
            "data": {
                "datasets_dir": "/test/datasets",
                "processed_dir": "/test/processed",
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(config_file)
        assert config.get("data.datasets_dir") == "/test/datasets"
        assert config.get("data.processed_dir") == "/test/processed"
        # Should still have defaults for missing keys
        assert config.get("data.root_dir") == "./data"

    def test_from_yaml_file_not_found(self, tmp_path):
        """Test loading config from non-existent file raises error."""
        config_file = tmp_path / "nonexistent.yml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config.from_yaml(config_file)

    def test_from_yaml_empty_file(self, tmp_path):
        """Test loading config from empty YAML file."""
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")

        config = Config.from_yaml(config_file)
        # Should use defaults
        assert config.get("data.root_dir") == "./data"

    def test_merge_configs_shallow(self):
        """Test merging configs with shallow override."""
        base = {"key1": "value1", "key2": "value2"}
        override = {"key2": "new_value2", "key3": "value3"}

        result = Config._merge_configs(base, override)
        assert result["key1"] == "value1"
        assert result["key2"] == "new_value2"
        assert result["key3"] == "value3"

    def test_merge_configs_nested(self):
        """Test merging configs with nested dicts."""
        base = {"data": {"root_dir": "./data", "cache": True}}
        override = {"data": {"root_dir": "/new/data"}}

        result = Config._merge_configs(base, override)
        assert result["data"]["root_dir"] == "/new/data"
        assert result["data"]["cache"] is True  # Preserved from base

    def test_get_simple_key(self):
        """Test getting config value with simple key."""
        config = Config({"simple_key": "value"})
        assert config.get("simple_key") == "value"

    def test_get_nested_key(self):
        """Test getting config value with nested key."""
        config = Config({"data": {"datasets_dir": "./datasets"}})
        assert config.get("data.datasets_dir") == "./datasets"

    def test_get_deeply_nested_key(self):
        """Test getting config value with deeply nested key."""
        config = Config({"level1": {"level2": {"level3": "value"}}})
        assert config.get("level1.level2.level3") == "value"

    def test_get_nonexistent_key_returns_default(self):
        """Test getting non-existent key returns default value."""
        config = Config()
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "default") == "default"

    def test_get_partial_path_returns_default(self):
        """Test getting partial path that doesn't exist returns default."""
        config = Config({"data": {"root_dir": "./data"}})
        assert config.get("data.nonexistent.nested") is None

    def test_set_simple_key(self):
        """Test setting config value with simple key."""
        config = Config()
        config.set("new_key", "new_value")
        assert config.get("new_key") == "new_value"

    def test_set_nested_key_existing_path(self):
        """Test setting nested key on existing path."""
        config = Config({"data": {"root_dir": "./data"}})
        config.set("data.new_key", "new_value")
        assert config.get("data.new_key") == "new_value"
        assert config.get("data.root_dir") == "./data"  # Existing value preserved

    def test_set_nested_key_new_path(self):
        """Test setting nested key creates new path."""
        config = Config()
        config.set("new.nested.key", "value")
        assert config.get("new.nested.key") == "value"

    def test_set_overwrite_existing_value(self):
        """Test setting existing key overwrites value."""
        config = Config({"key": "old_value"})
        config.set("key", "new_value")
        assert config.get("key") == "new_value"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config_dict = {"data": {"root_dir": "./data"}, "setting": "value"}
        config = Config(config_dict)
        result = config.to_dict()
        assert result == config_dict
        # Should be a copy
        result["new_key"] = "new_value"
        assert "new_key" not in config.to_dict()

    def test_save(self, tmp_path):
        """Test saving config to YAML file."""
        config = Config({"data": {"datasets_dir": "./datasets"}, "key": "value"})
        config_file = tmp_path / "output.yml"

        config.save(config_file)
        assert config_file.exists()

        # Load and verify
        with open(config_file) as f:
            saved_data = yaml.safe_load(f)
        assert saved_data["data"]["datasets_dir"] == "./datasets"
        assert saved_data["key"] == "value"

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test saving config creates parent directories."""
        config = Config({"key": "value"})
        config_file = tmp_path / "nested" / "dir" / "config.yml"

        config.save(config_file)
        assert config_file.exists()
        assert config_file.parent.exists()

    def test_repr(self):
        """Test string representation of Config."""
        config = Config({"key": "value"})
        repr_str = repr(config)
        assert "Config(" in repr_str
        assert "key" in repr_str


class TestGlobalConfig:
    """Tests for global config functions."""

    def test_get_config_default(self):
        """Test getting global config with defaults."""
        # Reset global config
        import docs2synth.utils.config

        docs2synth.utils.config._global_config = None

        with patch("pathlib.Path.exists", return_value=False):
            config = get_config()
            assert isinstance(config, Config)
            assert config.get("data.root_dir") == "./data"

    def test_get_config_from_file(self, tmp_path, monkeypatch):
        """Test getting global config loads from config.yml if exists."""
        import docs2synth.utils.config

        docs2synth.utils.config._global_config = None

        # Create config.yml in current directory
        config_file = tmp_path / "config.yml"
        config_data = {"data": {"datasets_dir": "/custom/datasets"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        config = get_config()
        assert config.get("data.datasets_dir") == "/custom/datasets"

    def test_get_config_caches_instance(self):
        """Test get_config returns same instance on multiple calls."""
        import docs2synth.utils.config

        docs2synth.utils.config._global_config = None

        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_get_config_handles_yaml_load_error(self, tmp_path, monkeypatch):
        """Test get_config handles YAML loading errors."""
        import docs2synth.utils.config

        docs2synth.utils.config._global_config = None

        # Create invalid YAML file
        config_file = tmp_path / "config.yml"
        config_file.write_text("invalid: yaml: content: [")

        monkeypatch.chdir(tmp_path)

        config = get_config()
        # Should fall back to defaults
        assert isinstance(config, Config)
        assert config.get("data.root_dir") == "./data"

    def test_set_config(self):
        """Test setting global config."""
        custom_config = Config({"custom": "value"})
        set_config(custom_config)

        config = get_config()
        assert config is custom_config
        assert config.get("custom") == "value"

    def test_load_config(self, tmp_path):
        """Test load_config loads and sets global config."""
        import docs2synth.utils.config

        docs2synth.utils.config._global_config = None

        config_file = tmp_path / "test_config.yml"
        config_data = {"data": {"datasets_dir": "/loaded/datasets"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loaded_config = load_config(config_file)
        assert loaded_config.get("data.datasets_dir") == "/loaded/datasets"

        # Should be set as global
        global_config = get_config()
        assert global_config is loaded_config
