"""Authentication provider for Docs2Synth MCP server using external Django token verification."""

from __future__ import annotations

import os
from typing import Optional

import httpx

from docs2synth.utils import get_logger

try:
    from fastmcp.server.auth import AccessToken, TokenVerifier
except ImportError:
    raise ImportError(
        "fastmcp is required for authentication. Install with: pip install 'docs2synth[mcp]'"
    )

logger = get_logger(__name__)


class DjangoTokenVerifier(TokenVerifier):
    """
    Token verifier for Django authentication backend.

    Calls external Django API endpoint to verify tokens.

    Args:
        verify_url: Token verification endpoint URL
        timeout: HTTP request timeout in seconds (default: 5.0)
        verify_ssl: Verify SSL certificates (default: True, override with AUTH_VERIFY_SSL env)
    """

    def __init__(
        self,
        verify_url: str | None = None,
        timeout: float = 5.0,
        verify_ssl: bool = True,
    ):
        """Initialize the Django token verifier."""
        super().__init__()

        self.verify_url = verify_url or os.getenv(
            "AUTH_VERIFY_URL",
            "http://admin.kaiaperth.com/authenticate/api/token/verify/",
        )
        self.timeout = timeout

        # SSL verification control
        ssl_verify_env = os.getenv("AUTH_VERIFY_SSL", "true").lower()
        self.verify_ssl = verify_ssl and (ssl_verify_env in ("true", "1", "yes"))

        if not self.verify_ssl:
            logger.warning(
                "SSL verification DISABLED. Use only for development/testing."
            )

        logger.info(f"Initialized DjangoTokenVerifier: {self.verify_url}")

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        """
        Verify token by calling Django verification endpoint.

        Args:
            token: Token string to verify

        Returns:
            AccessToken if valid, None otherwise
        """
        try:
            async with httpx.AsyncClient(verify=self.verify_ssl) as client:
                response = await client.post(
                    self.verify_url,
                    json={"token": token},
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    logger.info("Token verified successfully")
                    return AccessToken(token=token, client_id="authenticated", scopes=[])

                logger.warning(f"Token verification failed: {response.status_code}")
                return None

        except httpx.TimeoutException:
            logger.error(f"Verification timeout after {self.timeout}s")
            return None
        except httpx.RequestError as e:
            logger.error(f"Verification request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None


def create_auth_verifier() -> DjangoTokenVerifier | None:
    """
    Create auth verifier if authentication is enabled.

    Returns:
        DjangoTokenVerifier instance or None if auth is disabled
    """
    auth_enabled = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    logger.info(f"Authentication enabled: {auth_enabled}")
    if not auth_enabled:
        return None

    return DjangoTokenVerifier()
