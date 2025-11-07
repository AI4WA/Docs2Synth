"""Advanced tests for MCP configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch


class TestMCPConfigAdvanced:
    """Advanced configuration tests."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from docs2synth.mcp.config import MCPConfig

        config = MCPConfig()
        data = config.to_dict()

        assert "server" in data
        assert "oauth" in data
        assert "host" in data["server"]
        assert "port" in data["server"]
        assert "discovery_url" in data["oauth"]
        # Should not expose client_secret
        assert "client_secret" not in data["oauth"]

    def test_config_from_file_with_partial_data(self):
        """Test loading config from file with partial data."""
        from docs2synth.mcp.config import MCPConfig

        config_data = """
server:
  port: 9999
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            config_path = f.name

        try:
            config = MCPConfig.from_file(config_path)
            # Should use default host but custom port
            assert config.server.host == "0.0.0.0"
            assert config.server.port == 9999
            # OAuth should use defaults
            assert config.oauth.timeout == 5.0
        finally:
            Path(config_path).unlink()

    def test_config_from_file_empty_file(self):
        """Test loading config from empty file."""
        from docs2synth.mcp.config import MCPConfig

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            config_path = f.name

        try:
            config = MCPConfig.from_file(config_path)
            # Should use all defaults
            assert config.server.port == 8009
            assert config.oauth.timeout == 5.0
        finally:
            Path(config_path).unlink()

    def test_config_from_file_invalid_yaml(self):
        """Test loading config from invalid YAML file."""
        from docs2synth.mcp.config import MCPConfig

        config_data = """
server:
  port: invalid_port
  host: [this is not a string
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            config_path = f.name

        try:
            config = MCPConfig.from_file(config_path)
            # Should fall back to defaults on error
            assert config.server.port == 8009
        finally:
            Path(config_path).unlink()

    def test_config_from_file_nonexistent(self):
        """Test loading config from non-existent file."""
        from docs2synth.mcp.config import MCPConfig

        config = MCPConfig.from_file("/nonexistent/path/config.yaml")
        # Should use defaults
        assert config.server.port == 8009
        assert config.oauth.timeout == 5.0

    def test_config_from_env_all_variables(self):
        """Test loading config from environment with all variables set."""
        from docs2synth.mcp.config import MCPConfig

        env_vars = {
            "MCP_HOST": "127.0.0.1",
            "MCP_PORT": "9876",
            "MCP_BASE_URL": "http://test:9876",
            "MCP_DATA_DIR": "/tmp/test_data",
            "OIDC_DISCOVERY_URL": "http://auth.example.com/.well-known/openid-configuration",
            "OIDC_PUBLIC_BASE_URL": "http://auth.example.com",
            "OIDC_CLIENT_ID": "test_client",
            "OIDC_CLIENT_SECRET": "secret123",
            "OIDC_USE_INTROSPECTION": "false",
            "OIDC_VERIFY_SSL": "true",
            "OIDC_TIMEOUT": "10.5",
        }

        with patch.dict("os.environ", env_vars, clear=False):
            config = MCPConfig.from_env()

            assert config.server.host == "127.0.0.1"
            assert config.server.port == 9876
            assert config.server.base_url == "http://test:9876"
            assert config.server.data_dir == "/tmp/test_data"
            assert config.oauth.client_id == "test_client"
            assert config.oauth.client_secret == "secret123"
            assert config.oauth.use_introspection is False
            assert config.oauth.verify_ssl is True
            assert config.oauth.timeout == 10.5

    def test_config_from_env_boolean_variations(self):
        """Test boolean parsing from environment variables."""
        from docs2synth.mcp.config import MCPConfig

        # Test various truthy values
        for truthy in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
            with patch.dict("os.environ", {"OIDC_VERIFY_SSL": truthy}, clear=False):
                config = MCPConfig.from_env()
                assert config.oauth.verify_ssl is True, f"Failed for value: {truthy}"

        # Test various falsy values
        for falsy in ["false", "False", "FALSE", "0", "no", "No", "NO"]:
            with patch.dict("os.environ", {"OIDC_VERIFY_SSL": falsy}, clear=False):
                config = MCPConfig.from_env()
                assert config.oauth.verify_ssl is False, f"Failed for value: {falsy}"

    def test_config_load_priority(self):
        """Test that env vars override file config."""
        from docs2synth.mcp.config import MCPConfig

        config_data = """
server:
  host: file_host
  port: 1111
oauth:
  client_id: file_client
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            config_path = f.name

        try:
            # Set env vars that should override file
            env_vars = {
                "MCP_PORT": "2222",
                "OIDC_CLIENT_ID": "env_client",
            }

            with patch.dict("os.environ", env_vars, clear=False):
                config = MCPConfig.load(config_path)

                # Host should come from file (no env override)
                assert config.server.host == "file_host"
                # Port should come from env
                assert config.server.port == 2222
                # Client ID should come from env
                assert config.oauth.client_id == "env_client"
        finally:
            Path(config_path).unlink()

    def test_config_load_without_path(self):
        """Test loading config without explicit path."""
        from docs2synth.mcp.config import MCPConfig

        # Should try to load from cwd/config.mcp.yml and fall back to defaults
        config = MCPConfig.load()
        assert config is not None
        assert config.server.port == 8009  # Should use defaults

    def test_oauth_config_ssl_verification_default(self):
        """Test OAuth config SSL verification is False by default."""
        from docs2synth.mcp.config import OAuthConfig

        config = OAuthConfig()
        assert config.verify_ssl is False

    def test_oauth_config_introspection_default(self):
        """Test OAuth config introspection is True by default."""
        from docs2synth.mcp.config import OAuthConfig

        config = OAuthConfig()
        assert config.use_introspection is True

    def test_server_config_data_dir_default(self):
        """Test server config data_dir is None by default."""
        from docs2synth.mcp.config import ServerConfig

        config = ServerConfig()
        assert config.data_dir is None

    def test_config_with_custom_data_dir(self):
        """Test config with custom data directory."""
        from docs2synth.mcp.config import MCPConfig, ServerConfig

        server = ServerConfig(data_dir="/custom/data/path")
        config = MCPConfig(server=server)

        assert config.server.data_dir == "/custom/data/path"
        data = config.to_dict()
        assert data["server"]["data_dir"] == "/custom/data/path"
