"""OAuth 2.0 proxy handlers for MCP server.

This module handles OAuth authorization server metadata and request proxying
to the Django OAuth Toolkit backend.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from docs2synth.mcp.config import MCPConfig
from docs2synth.utils import get_logger

logger = get_logger(__name__)


async def handle_oauth_metadata(config: MCPConfig) -> JSONResponse:
    """Return OAuth 2.0 authorization server metadata (RFC 8414)."""
    base_url = config.server.base_url
    return JSONResponse(
        {
            "issuer": f"{base_url}/oauth",
            "authorization_endpoint": f"{base_url}/oauth/authorize/",
            "token_endpoint": f"{base_url}/oauth/token/",
            "registration_endpoint": f"{base_url}/oauth/register",
            "revocation_endpoint": f"{base_url}/oauth/revoke_token/",
            "introspection_endpoint": f"{base_url}/oauth/introspect/",
            "userinfo_endpoint": f"{base_url}/oauth/userinfo/",
            "jwks_uri": f"{base_url}/oauth/.well-known/jwks.json",
            "response_types_supported": [
                "code",
                "token",
                "id_token",
                "code token",
                "code id_token",
                "token id_token",
                "code token id_token",
            ],
            "grant_types_supported": [
                "authorization_code",
                "implicit",
                "client_credentials",
                "refresh_token",
            ],
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
            ],
            "code_challenge_methods_supported": ["S256", "plain"],
            "service_documentation": "https://django-oauth-toolkit.readthedocs.io/",
        }
    )


async def handle_openid_configuration(config: MCPConfig) -> JSONResponse:
    """Return OpenID Connect Discovery metadata."""
    base_url = config.server.base_url
    return JSONResponse(
        {
            "issuer": f"{base_url}/oauth",
            "authorization_endpoint": f"{base_url}/oauth/authorize/",
            "token_endpoint": f"{base_url}/oauth/token/",
            "userinfo_endpoint": f"{base_url}/oauth/userinfo/",
            "jwks_uri": f"{base_url}/oauth/.well-known/jwks.json",
            "registration_endpoint": f"{base_url}/oauth/register",
            "revocation_endpoint": f"{base_url}/oauth/revoke_token/",
            "introspection_endpoint": f"{base_url}/oauth/introspect/",
            "response_types_supported": [
                "code",
                "token",
                "id_token",
                "code token",
                "code id_token",
                "token id_token",
                "code token id_token",
            ],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "scopes_supported": ["openid", "profile", "email", "read", "write"],
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
            ],
            "grant_types_supported": [
                "authorization_code",
                "implicit",
                "client_credentials",
                "refresh_token",
            ],
            "code_challenge_methods_supported": ["S256", "plain"],
        }
    )


async def handle_protected_resource_metadata(
    config: MCPConfig, request: Request
) -> JSONResponse:
    """Return OAuth 2.0 protected resource metadata (RFC 8707)."""
    resource_base = config.server.base_url
    return JSONResponse(
        {
            "resource": resource_base,
            "authorization_servers": [f"{resource_base}/oauth"],
            "bearer_methods_supported": ["header"],
            "resource_signing_alg_values_supported": ["RS256"],
        }
    )


async def handle_client_registration(config: MCPConfig) -> JSONResponse:
    """Handle OAuth 2.0 dynamic client registration (RFC 7591).

    Returns static client credentials for MCP Inspector compatibility.
    """
    return JSONResponse(
        {
            "client_id": config.oauth.client_id,
            "client_secret": config.oauth.client_secret,
            "client_id_issued_at": 1730000000,
            "client_secret_expires_at": 0,
            "client_name": "Docs2Synth MCP",
            "token_endpoint_auth_method": "client_secret_post",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "redirect_uris": [f"{config.server.base_url}/oauth/callback"],
        },
        status_code=201,
    )


async def handle_oauth_callback(
    config: MCPConfig, request: Request
) -> RedirectResponse:
    """Handle OAuth callback from Django and forward to MCP Inspector."""
    params = dict(request.query_params)
    inspector_callback = "http://localhost:6274/oauth/callback/debug"
    redirect_url = f"{inspector_callback}?{urlencode(params)}"

    logger.debug("OAuth callback received, redirecting to Inspector")
    return RedirectResponse(url=redirect_url, status_code=302)


async def proxy_oauth_request(config: MCPConfig, request: Request) -> Response:
    """Proxy OAuth requests to Django OAuth server.

    Intercepts and rewrites redirect_uri to ensure OAuth callback flow works correctly.
    """
    path = request.url.path.replace("/oauth", "/o", 1)
    query_string = request.url.query

    # Intercept authorization requests
    if "/authorize" in path and request.method == "GET":
        params = dict(request.query_params)
        if "redirect_uri" in params:
            params["redirect_uri"] = f"{config.server.base_url}/oauth/callback"
            query_string = urlencode(params)
            logger.debug("Rewrote redirect_uri in authorization request")

    # Intercept token exchange requests
    request_body = None
    if "/token" in path and request.method == "POST":
        request_body = await request.body()
        body_params = parse_qs(request_body.decode("utf-8"))

        if "redirect_uri" in body_params:
            body_params["redirect_uri"] = [f"{config.server.base_url}/oauth/callback"]
            request_body = urlencode({k: v[0] for k, v in body_params.items()}).encode(
                "utf-8"
            )
            logger.debug("Rewrote redirect_uri in token exchange")

    # Forward request to Django
    target_url = f"{config.oauth.public_base_url.rstrip('/o')}{path}"
    if query_string:
        target_url = f"{target_url}?{query_string}"

    try:
        async with httpx.AsyncClient(verify=config.oauth.verify_ssl) as client:
            headers = dict(request.headers)
            headers.pop("host", None)

            if request.method == "GET":
                response = await client.get(target_url, headers=headers)
            elif request.method == "POST":
                if request_body is not None:
                    headers.pop("content-length", None)
                else:
                    request_body = await request.body()

                response = await client.post(
                    target_url, headers=headers, content=request_body
                )
            else:
                return Response(
                    f"Method {request.method} not supported", status_code=405
                )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

    except Exception as e:
        logger.error(f"OAuth proxy error: {e}")
        return Response(f"Proxy error: {str(e)}", status_code=502)
