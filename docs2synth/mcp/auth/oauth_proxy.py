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


def _rewrite_set_cookie_headers(
    headers: dict[str, str], use_https: bool
) -> dict[str, str]:
    """Rewrite Set-Cookie attributes to bind cookies to localhost:8009.

    - Remove Domain so cookies are host-only (localhost)
    - Drop Secure if not https
    - Ensure SameSite=Lax for login flow
    """
    new_headers = dict(headers)
    set_cookie = new_headers.get("set-cookie") or new_headers.get("Set-Cookie")
    if not set_cookie:
        return new_headers

    def transform(cookie: str) -> str:
        parts = [p.strip() for p in cookie.split(";")]
        out = []
        for p in parts:
            if p.lower().startswith("domain="):
                # strip Domain to make it host-only (localhost)
                continue
            if not use_https and p.lower() == "secure":
                # remove Secure on http
                continue
            out.append(p)
        # ensure SameSite=Lax
        if not any(p.lower().startswith("samesite=") for p in out):
            out.append("SameSite=Lax")
        return "; ".join(out)

    # Multiple Set-Cookie headers may be concatenated by httpx with commas.
    cookies = [c for c in set_cookie.split(",") if c.strip()]
    rewritten = ", ".join(transform(c) for c in cookies)
    new_headers["set-cookie"] = rewritten
    return new_headers


def _rewrite_location(location: str, config: MCPConfig) -> str:
    """Rewrite absolute redirects from IdP to local proxy paths.

    Handles mapping of /o/* -> /oauth/*, Django accounts and login routes
    back to their local equivalents to avoid cross-host redirects.
    """
    idp_base = config.oauth.public_base_url.rstrip("/o")

    # Map OAuth endpoints
    if location.startswith(idp_base + "/o/"):
        location = location.replace(idp_base + "/o/", "/oauth/", 1)
    elif location.startswith("/o/"):
        location = location.replace("/o/", "/oauth/", 1)

    # Map Django Accounts (general first)
    if location.startswith(idp_base + "/accounts/"):
        location = location.replace(idp_base + "/accounts/", "/accounts/", 1)
    elif location.startswith("/accounts/"):
        # already relative; keep as-is
        pass

    # Normalize login routes specifically to /login
    if location.startswith(idp_base + "/accounts/login"):
        location = location.replace(idp_base + "/accounts/login", "/login", 1)
    elif location.startswith("/accounts/login"):
        location = location.replace("/accounts/login", "/login", 1)

    if location.startswith(idp_base + "/login"):
        location = location.replace(idp_base + "/login", "/login", 1)
    elif location.startswith("/login"):
        # already relative; keep as-is
        pass

    return location


def _strip_and_rewrite_cookies(
    upstream_headers: httpx.Headers, use_https: bool
) -> list[str]:
    """Extract Set-Cookie values from upstream and rewrite attributes for localhost.

    - Remove Domain
    - Drop Secure when not https
    - Ensure SameSite=Lax
    """
    try:
        cookie_values = upstream_headers.get_list("set-cookie")
    except Exception:
        cookie_values = []

    rewritten_cookies: list[str] = []
    for cookie in cookie_values:
        parts = [p.strip() for p in cookie.split(";")]
        out: list[str] = []
        for p in parts:
            if p.lower().startswith("domain="):
                continue
            if (not use_https) and p.lower() == "secure":
                continue
            out.append(p)
        if not any(p.lower().startswith("samesite=") for p in out):
            out.append("SameSite=Lax")
        rewritten_cookies.append("; ".join(out))
    return rewritten_cookies


async def _forward_and_build_response(
    config: MCPConfig,
    request: Request,
    target_url: str,
    content_override: bytes | None = None,
) -> Response:
    """Forward the incoming request to target_url and build a proxied response.

    Handles header sanitation, optional content override (for POST bodies),
    redirect Location rewriting and Set-Cookie normalization.
    """
    async with httpx.AsyncClient(verify=config.oauth.verify_ssl) as client:
        headers = dict(request.headers)
        headers.pop("host", None)

        if request.method == "GET":
            response = await client.get(target_url, headers=headers)
        elif request.method == "POST":
            body = content_override
            if body is None:
                body = await request.body()
            # content-length may be wrong after re-encoding; drop it so httpx sets correctly
            headers.pop("content-length", None)
            response = await client.post(target_url, headers=headers, content=body)
        else:
            return Response(f"Method {request.method} not supported", status_code=405)

    # Rewrite redirect Location headers to stay on this host
    resp_headers = dict(response.headers)
    location = resp_headers.get("location") or resp_headers.get("Location")
    if location:
        resp_headers["location"] = _rewrite_location(location, config)

    # Remove any upstream Set-Cookie headers; we'll append rewritten ones
    resp_headers.pop("set-cookie", None)
    resp_headers.pop("Set-Cookie", None)

    proxied = Response(
        content=response.content,
        status_code=response.status_code,
        headers=resp_headers,
    )

    for cookie in _strip_and_rewrite_cookies(
        response.headers, use_https=config.server.base_url.startswith("https://")
    ):
        proxied.headers.append("set-cookie", cookie)

    return proxied


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
            "scopes_supported": ["openid", "profile", "email", "read", "write"],
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
        # Ensure scope is present - Django OAuth Toolkit requires it
        if "scope" not in params or not params.get("scope"):
            params["scope"] = "read"  # Default scope for MCP access
            logger.debug("Added default scope to authorization request")
        query_string = urlencode(params)
        logger.debug("Rewrote redirect_uri in authorization request")

    # Intercept token exchange requests
    request_body: bytes | None = None
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
        return await _forward_and_build_response(
            config, request, target_url, content_override=request_body
        )
    except Exception as e:
        logger.error(f"OAuth proxy error: {e}")
        return Response(f"Proxy error: {str(e)}", status_code=502)


async def proxy_accounts_request(config: MCPConfig, request: Request) -> Response:
    """Proxy /accounts/* requests to the Django OAuth server.

    This allows accessing the IdP's login and related Django views via the MCP host,
    e.g. GET /accounts/login/?next=... on port 8009.
    """
    # Build target URL on the IdP, preserving path and query string
    path = request.url.path  # e.g., /accounts/login/
    if path.startswith("/accounts/login"):
        path = "/login" + path[len("/accounts/login") :]
    query_string = request.url.query

    target_base = config.oauth.public_base_url.rstrip("/o")
    target_url = f"{target_base}{path}"
    if query_string:
        target_url = f"{target_url}?{query_string}"

    try:
        return await _forward_and_build_response(config, request, target_url)
    except Exception as e:
        logger.error(f"Accounts proxy error: {e}")
        return Response(f"Proxy error: {str(e)}", status_code=502)


async def proxy_login_request(config: MCPConfig, request: Request) -> Response:
    """Proxy /login and /login/ requests to the IdP.

    Renders the Django login form via the proxy, avoiding redirect loops.
    """
    path = request.url.path  # /login or /login/
    query_string = request.url.query

    target_base = config.oauth.public_base_url.rstrip("/o")
    target_url = f"{target_base}{path}"
    if query_string:
        target_url = f"{target_url}?{query_string}"

    try:
        return await _forward_and_build_response(config, request, target_url)
    except Exception as e:
        logger.error(f"Login proxy error: {e}")
        return Response(f"Proxy error: {str(e)}", status_code=502)


async def proxy_static_request(config: MCPConfig, request: Request) -> Response:
    """Proxy /static/* assets from the IdP so CSS/JS load under 8009."""
    path = request.url.path  # /static/...
    query_string = request.url.query

    target_base = config.oauth.public_base_url.rstrip("/o")
    target_url = f"{target_base}{path}"
    if query_string:
        target_url = f"{target_url}?{query_string}"

    try:
        async with httpx.AsyncClient(verify=config.oauth.verify_ssl) as client:
            headers = dict(request.headers)
            headers.pop("host", None)
            response = await client.get(target_url, headers=headers)

            resp_headers = dict(response.headers)
            # Static responses may set long cache headers; pass-through fine
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=resp_headers,
            )
    except Exception as e:
        logger.error(f"Static proxy error: {e}")
        return Response(f"Proxy error: {str(e)}", status_code=502)
