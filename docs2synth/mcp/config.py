"""MCP server configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from docs2synth.utils import get_logger

logger = get_logger(__name__)


@dataclass
class OAuthConfig:
    """OAuth/OIDC configuration."""

    discovery_url: str = "http://localhost:8000/o/.well-known/openid-configuration"
    public_base_url: str = "http://localhost:8000/o"
    client_id: str = "qQdiLy6Raw141wfcHQIo6srYuqsl0Y2oJIaqmcwJ"
    client_secret: str = "test"
    use_introspection: bool = True
    verify_ssl: bool = False
    timeout: float = 5.0


@dataclass
class ServerConfig:
    """MCP server configuration."""

    host: str = "0.0.0.0"
    port: int = 8009
    base_url: str = "http://localhost:8009"
    data_dir: str | None = None


@dataclass
class MCPConfig:
    """Main MCP configuration."""

    server: ServerConfig = field(default_factory=ServerConfig)
    oauth: OAuthConfig = field(default_factory=OAuthConfig)

    @classmethod
    def from_file(cls, config_path: str | Path) -> MCPConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config.mcp.yml

        Returns:
            MCPConfig instance
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}

            # Extract server config
            server_data = data.get("server", {})
            server = ServerConfig(
                host=server_data.get("host", "0.0.0.0"),
                port=server_data.get("port", 8009),
                base_url=server_data.get("base_url", "http://localhost:8009"),
                data_dir=server_data.get("data_dir"),
            )

            # Extract OAuth config
            oauth_data = data.get("oauth", {})
            oauth = OAuthConfig(
                discovery_url=oauth_data.get(
                    "discovery_url",
                    "http://localhost:8000/o/.well-known/openid-configuration",
                ),
                public_base_url=oauth_data.get(
                    "public_base_url", "http://localhost:8000/o"
                ),
                client_id=oauth_data.get(
                    "client_id", "qQdiLy6Raw141wfcHQIo6srYuqsl0Y2oJIaqmcwJ"
                ),
                client_secret=oauth_data.get("client_secret", "test"),
                use_introspection=oauth_data.get("use_introspection", True),
                verify_ssl=oauth_data.get("verify_ssl", False),
                timeout=oauth_data.get("timeout", 5.0),
            )

            logger.info(f"Loaded configuration from {config_path}")
            return cls(server=server, oauth=oauth)

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()

    @classmethod
    def from_env(cls) -> MCPConfig:
        """Load configuration from environment variables.

        Environment variables take precedence over config file values.

        Returns:
            MCPConfig instance
        """
        server = ServerConfig(
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8009")),
            base_url=os.getenv("MCP_BASE_URL", "http://localhost:8009"),
            data_dir=os.getenv("MCP_DATA_DIR"),
        )

        oauth = OAuthConfig(
            discovery_url=os.getenv(
                "OIDC_DISCOVERY_URL",
                "http://localhost:8000/o/.well-known/openid-configuration",
            ),
            public_base_url=os.getenv(
                "OIDC_PUBLIC_BASE_URL", "http://localhost:8000/o"
            ),
            client_id=os.getenv(
                "OIDC_CLIENT_ID", "qQdiLy6Raw141wfcHQIo6srYuqsl0Y2oJIaqmcwJ"
            ),
            client_secret=os.getenv("OIDC_CLIENT_SECRET", "test"),
            use_introspection=os.getenv("OIDC_USE_INTROSPECTION", "true").lower()
            in ("true", "1", "yes"),
            verify_ssl=os.getenv("OIDC_VERIFY_SSL", "false").lower()
            in ("true", "1", "yes"),
            timeout=float(os.getenv("OIDC_TIMEOUT", "5.0")),
        )

        return cls(server=server, oauth=oauth)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> MCPConfig:
        """Load configuration with priority: env vars > config file > defaults.

        Args:
            config_path: Optional path to config file. Defaults to ./config.mcp.yml

        Returns:
            MCPConfig instance
        """
        # Start with file config or defaults
        if config_path is None:
            config_path = Path.cwd() / "config.mcp.yml"

        config = cls.from_file(config_path)

        # Override with environment variables
        env_config = cls.from_env()

        # Merge: env vars take precedence
        if os.getenv("MCP_HOST"):
            config.server.host = env_config.server.host
        if os.getenv("MCP_PORT"):
            config.server.port = env_config.server.port
        if os.getenv("MCP_BASE_URL"):
            config.server.base_url = env_config.server.base_url
        if os.getenv("MCP_DATA_DIR"):
            config.server.data_dir = env_config.server.data_dir

        if os.getenv("OIDC_DISCOVERY_URL"):
            config.oauth.discovery_url = env_config.oauth.discovery_url
        if os.getenv("OIDC_PUBLIC_BASE_URL"):
            config.oauth.public_base_url = env_config.oauth.public_base_url
        if os.getenv("OIDC_CLIENT_ID"):
            config.oauth.client_id = env_config.oauth.client_id
        if os.getenv("OIDC_CLIENT_SECRET"):
            config.oauth.client_secret = env_config.oauth.client_secret
        if os.getenv("OIDC_USE_INTROSPECTION"):
            config.oauth.use_introspection = env_config.oauth.use_introspection
        if os.getenv("OIDC_VERIFY_SSL"):
            config.oauth.verify_ssl = env_config.oauth.verify_ssl
        if os.getenv("OIDC_TIMEOUT"):
            config.oauth.timeout = env_config.oauth.timeout

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "base_url": self.server.base_url,
                "data_dir": self.server.data_dir,
            },
            "oauth": {
                "discovery_url": self.oauth.discovery_url,
                "public_base_url": self.oauth.public_base_url,
                "client_id": self.oauth.client_id,
                "use_introspection": self.oauth.use_introspection,
                "verify_ssl": self.oauth.verify_ssl,
                "timeout": self.oauth.timeout,
            },
        }
