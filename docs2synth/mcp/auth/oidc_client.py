"""OIDC Client for validating tokens from external Django OAuth server.

This module acts as an OIDC Resource Server, validating JWT tokens
issued by an external Django OAuth Toolkit server.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient

from docs2synth.utils import get_logger

logger = get_logger(__name__)


class OIDCResourceServer:
    """
    OIDC Resource Server that validates tokens from Django OAuth Toolkit.

    Acts as a resource server in the OAuth 2.0 / OIDC architecture:
    - Discovers endpoints from OIDC provider
    - Validates JWT tokens using JWKS or introspection
    - Protects MCP resources

    Args:
        oidc_discovery_url: OIDC discovery endpoint URL
        client_id: OAuth client ID (optional, for introspection)
        client_secret: OAuth client secret (optional, for introspection)
        use_introspection: Use token introspection instead of JWT validation
        verify_ssl: Enable SSL certificate verification (default: True from env)
        timeout: HTTP request timeout in seconds (default: 5.0 from env)
    """

    def __init__(
        self,
        oidc_discovery_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        use_introspection: bool = False,
        verify_ssl: bool | None = None,
        timeout: float | None = None,
    ):
        """Initialize the OIDC Resource Server."""
        self.oidc_discovery_url = oidc_discovery_url or os.getenv(
            "OIDC_DISCOVERY_URL",
            "http://localhost:8000/o/.well-known/openid-configuration",
        )
        self.client_id = client_id or os.getenv("OIDC_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("OIDC_CLIENT_SECRET")
        self.use_introspection = use_introspection or os.getenv(
            "OIDC_USE_INTROSPECTION", "false"
        ).lower() in ("true", "1", "yes")

        # Discovered endpoints
        self.issuer = None
        self.jwks_uri = None
        self.introspection_endpoint = None
        self.userinfo_endpoint = None

        # JWKS client for JWT validation
        self.jwks_client = None

        # HTTP client settings - config parameter takes precedence over env vars
        if timeout is not None:
            self.timeout = float(timeout)
        else:
            self.timeout = float(os.getenv("OIDC_TIMEOUT", "5.0"))

        if verify_ssl is not None:
            self.verify_ssl = bool(verify_ssl)
        else:
            self.verify_ssl = os.getenv("OIDC_VERIFY_SSL", "true").lower() in (
                "true",
                "1",
                "yes",
            )

        logger.info("Initializing OIDC Resource Server")
        logger.info(f"  Discovery URL: {self.oidc_discovery_url}")
        logger.info(f"  Use Introspection: {self.use_introspection}")
        logger.info(f"  Verify SSL: {self.verify_ssl}")
        logger.info(f"  Timeout: {self.timeout}s")

    def _fix_url(self, url: str, correct_base: str) -> str:
        """Fix URLs that might have incorrect hostnames.

        Django OAuth Toolkit returns URLs with the host from the request,
        which might be 'localhost' when we need 'host.docker.internal' in containers.

        Args:
            url: The URL to fix
            correct_base: The correct base URL (from discovery URL)

        Returns:
            Fixed URL with correct hostname
        """
        from urllib.parse import urlparse

        url_parsed = urlparse(url)
        correct_parsed = urlparse(correct_base)

        # Replace scheme and netloc with correct ones, keep path
        fixed_url = (
            f"{correct_parsed.scheme}://{correct_parsed.netloc}{url_parsed.path}"
        )

        if url != fixed_url:
            logger.info(f"Fixed URL: {url} -> {fixed_url}")

        return fixed_url

    async def discover_endpoints(self) -> bool:
        """
        Discover OIDC endpoints from the provider.

        Returns:
            True if discovery successful, False otherwise
        """
        try:
            async with httpx.AsyncClient(verify=self.verify_ssl) as client:
                response = await client.get(
                    self.oidc_discovery_url,
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    logger.error(f"OIDC discovery failed: HTTP {response.status_code}")
                    return False

                config = response.json()

                self.issuer = config.get("issuer")
                self.jwks_uri = config.get("jwks_uri")
                self.introspection_endpoint = config.get("introspection_endpoint")
                self.userinfo_endpoint = config.get("userinfo_endpoint")

                # Fix URLs to use the same host as the discovery URL
                # Django might return localhost URLs which don't work in containers
                discovery_base = self.oidc_discovery_url.split("/.well-known")[0]
                if self.jwks_uri:
                    self.jwks_uri = self._fix_url(self.jwks_uri, discovery_base)
                if self.introspection_endpoint:
                    self.introspection_endpoint = self._fix_url(
                        self.introspection_endpoint, discovery_base
                    )
                if self.userinfo_endpoint:
                    self.userinfo_endpoint = self._fix_url(
                        self.userinfo_endpoint, discovery_base
                    )

                logger.info("OIDC Discovery successful:")
                logger.info(f"  Issuer: {self.issuer}")
                logger.info(f"  JWKS URI: {self.jwks_uri}")
                logger.info(f"  Introspection: {self.introspection_endpoint}")

                # Fallback: If using introspection but endpoint not discovered, try environment or common paths
                if self.use_introspection and not self.introspection_endpoint:
                    env_introspection = os.getenv("OIDC_INTROSPECTION_ENDPOINT")
                    if env_introspection:
                        self.introspection_endpoint = env_introspection
                        logger.info(
                            f"Using introspection endpoint from env: {self.introspection_endpoint}"
                        )
                    elif discovery_base:
                        # Try standard Django OAuth Toolkit path
                        fallback_introspection = f"{discovery_base}/introspect/"
                        logger.warning(
                            f"Introspection endpoint not in discovery, trying fallback: {fallback_introspection}"
                        )
                        self.introspection_endpoint = fallback_introspection

                # Initialize JWKS client if not using introspection
                if not self.use_introspection and self.jwks_uri:
                    self.jwks_client = PyJWKClient(self.jwks_uri)
                    logger.info("JWKS client initialized")

                return True

        except Exception as e:
            logger.error(f"OIDC discovery error: {e}", exc_info=True)
            return False

    async def verify_token_introspection(self, token: str) -> dict[str, Any] | None:
        """
        Verify token using OAuth 2.0 Token Introspection (RFC 7662).

        Args:
            token: Access token to verify

        Returns:
            Token data if valid, None otherwise
        """
        if not self.introspection_endpoint:
            logger.error("Introspection endpoint not configured")
            return None

        if not self.client_id or not self.client_secret:
            logger.error("Client credentials required for introspection")
            return None

        try:
            async with httpx.AsyncClient(verify=self.verify_ssl) as client:
                response = await client.post(
                    self.introspection_endpoint,
                    data={
                        "token": token,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    logger.warning(f"Introspection failed: HTTP {response.status_code}")
                    return None

                data = response.json()

                if not data.get("active", False):
                    logger.warning("Token is not active")
                    return None

                logger.info(f"Token verified via introspection: {data.get('sub')}")
                return data

        except Exception as e:
            logger.error(f"Token introspection error: {e}", exc_info=True)
            return None

    async def verify_token_jwt(self, token: str) -> dict[str, Any] | None:
        """
        Verify JWT token using JWKS.

        Args:
            token: JWT access token

        Returns:
            Token payload if valid, None otherwise
        """
        if not self.jwks_client:
            logger.error("JWKS client not initialized")
            return None

        try:
            # Try to get signing key from JWT (using kid if present)
            signing_key = None
            try:
                signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            except Exception as e:
                # If kid is missing or not found, try all signing keys
                logger.warning(
                    f"Could not get signing key from JWT header: {e}. Trying all available keys..."
                )
                signing_keys = self.jwks_client.get_signing_keys()
                if not signing_keys:
                    logger.error("No signing keys available in JWKS")
                    return None

                # Try each key until one works
                last_error = None
                for key in signing_keys:
                    try:
                        payload = jwt.decode(
                            token,
                            key.key,
                            algorithms=["RS256", "HS256"],
                            issuer=self.issuer,
                            options={
                                "verify_signature": True,
                                "verify_exp": True,
                                "verify_iss": True,
                            },
                        )
                        logger.info(
                            f"Token verified via JWT (using key: {key.key_id}): {payload.get('sub')}"
                        )
                        return payload
                    except Exception as verify_error:
                        last_error = verify_error
                        continue

                logger.error(f"Failed to verify with any available key: {last_error}")
                return None

            # Decode and verify JWT with the signing key
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "HS256"],
                issuer=self.issuer,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iss": True,
                },
            )

            logger.info(f"Token verified via JWT: {payload.get('sub')}")
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT verification error: {e}", exc_info=True)
            return None

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """
        Verify access token using configured method.

        Args:
            token: Access token to verify

        Returns:
            Token data if valid, None otherwise
        """
        # Ensure endpoints are discovered
        if not self.issuer:
            logger.info("Discovering OIDC endpoints...")
            if not await self.discover_endpoints():
                logger.error("Failed to discover OIDC endpoints")
                return None

        # Use introspection or JWT validation
        if self.use_introspection:
            return await self.verify_token_introspection(token)
        else:
            return await self.verify_token_jwt(token)

    async def get_userinfo(self, token: str) -> dict[str, Any] | None:
        """
        Get user information from the OIDC provider.

        Args:
            token: Access token

        Returns:
            User information if successful, None otherwise
        """
        if not self.userinfo_endpoint:
            logger.error("Userinfo endpoint not configured")
            return None

        try:
            async with httpx.AsyncClient(verify=self.verify_ssl) as client:
                response = await client.get(
                    self.userinfo_endpoint,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    logger.warning(
                        f"Userinfo request failed: HTTP {response.status_code}"
                    )
                    return None

                return response.json()

        except Exception as e:
            logger.error(f"Userinfo request error: {e}", exc_info=True)
            return None


# Global instance
_oidc_resource_server = None


def get_oidc_resource_server() -> OIDCResourceServer:
    """Get or create the global OIDC Resource Server instance."""
    global _oidc_resource_server
    if _oidc_resource_server is None:
        _oidc_resource_server = OIDCResourceServer()
    return _oidc_resource_server
