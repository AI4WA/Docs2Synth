"""OAuth 2.0 proxy handlers for MCP server.

This module handles OAuth authorization server metadata and request proxying
to the Django OAuth Toolkit backend.
"""

from __future__ import annotations

import base64
import html
import json
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from docs2synth.mcp.config import MCPConfig
from docs2synth.utils import get_logger

logger = get_logger(__name__)


def _validate_redirect_uri(redirect_uri: str) -> bool:
    """Validate redirect_uri to prevent open redirect attacks.

    Allows:
    - Custom protocols (e.g., claude://, mcp://)
    - HTTP/HTTPS URLs (with warnings for HTTP in production)

    Blocks:
    - javascript:, data:, and other dangerous protocols
    - Invalid URLs
    """
    if not redirect_uri:
        return False

    try:
        parsed = urlparse(redirect_uri)
    except Exception:
        return False

    # Allow custom protocols (common for desktop apps)
    if parsed.scheme and len(parsed.scheme) > 0:
        # Allow common safe protocols
        if parsed.scheme.lower() in ["claude", "mcp", "http", "https"]:
            # For HTTP, warn but allow (useful for development)
            if parsed.scheme.lower() == "http":
                logger.warning(
                    f"HTTP redirect_uri used (consider HTTPS): {redirect_uri}"
                )
            return True

        # Block dangerous protocols
        dangerous_protocols = ["javascript", "data", "vbscript", "file", "about"]
        if parsed.scheme.lower() in dangerous_protocols:
            logger.warning(f"Blocked dangerous redirect_uri protocol: {parsed.scheme}")
            return False

    return False


def _rewrite_set_cookie_headers(
    headers: dict[str, str], use_https: bool
) -> dict[str, str]:
    """Rewrite Set-Cookie attributes for proper cookie handling.

    - Remove Domain so cookies are host-only
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
                # strip Domain to make it host-only
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
    """Extract Set-Cookie values from upstream and rewrite attributes.

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

    Returns static client credentials for MCP client compatibility.
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


async def handle_oauth_callback(config: MCPConfig, request: Request) -> Response:
    """Handle OAuth callback from Django.

    Extracts original redirect_uri from state and redirects back to the
    client (Claude) with the authorization code.
    """
    params = dict(request.query_params)
    code = params.get("code")
    state = params.get("state")
    error = params.get("error")
    error_description = params.get("error_description")

    # Extract original redirect_uri from state if present
    # Format: original_state|redirect_uri_base64
    original_redirect_uri = None
    clean_state = state
    if state and "|" in state:
        try:
            parts = state.rsplit("|", 1)
            if len(parts) == 2:
                clean_state = parts[0]
                encoded_redirect = parts[1]
                # Add padding if needed
                padding = 4 - len(encoded_redirect) % 4
                if padding != 4:
                    encoded_redirect += "=" * padding
                original_redirect_uri = base64.urlsafe_b64decode(
                    encoded_redirect.encode("utf-8")
                ).decode("utf-8")
                logger.debug(
                    f"Extracted original redirect_uri from state: {original_redirect_uri}"
                )
        except Exception as e:
            logger.warning(f"Failed to decode redirect_uri from state: {e}")

    # If we have original redirect_uri, redirect back to client with code
    if original_redirect_uri:
        # Validate redirect_uri again (safety check)
        if not _validate_redirect_uri(original_redirect_uri):
            logger.error(
                f"Invalid redirect_uri in callback state: {original_redirect_uri}"
            )
            return JSONResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Invalid redirect_uri in state parameter",
                },
                status_code=400,
            )

        if error:
            # Redirect error back to client
            redirect_params = {"error": error}
            if error_description:
                redirect_params["error_description"] = error_description
            if clean_state:
                redirect_params["state"] = clean_state
            redirect_url = f"{original_redirect_uri}?{urlencode(redirect_params)}"
            logger.debug(f"Redirecting error back to client: {redirect_url}")
            return RedirectResponse(url=redirect_url, status_code=302)
        else:
            # Redirect success with code back to client
            redirect_params = {"code": code}
            if clean_state:
                redirect_params["state"] = clean_state
            redirect_url = f"{original_redirect_uri}?{urlencode(redirect_params)}"
            logger.debug(
                f"Redirecting authorization code back to client: {original_redirect_uri}"
            )
            return RedirectResponse(url=redirect_url, status_code=302)

    # Real client: return HTML page that can be used by clients
    # This allows clients to extract the code via postMessage or direct access
    # Use JSON encoding to safely escape values for JavaScript
    if error:
        # Safely escape values for JavaScript
        error_js = json.dumps(error)
        error_desc_js = json.dumps(error_description or "")
        state_js = json.dumps(state or "")

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OAuth Authorization Error</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 500px;
            width: 100%;
            text-align: center;
        }}
        .icon {{
            width: 80px;
            height: 80px;
            margin: 0 auto 20px;
            background: #fee;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
        }}
        h1 {{
            color: #d32f2f;
            margin-bottom: 16px;
            font-size: 24px;
        }}
        p {{
            color: #666;
            margin-bottom: 12px;
            line-height: 1.6;
        }}
        .error-box {{
            background: #fee;
            border-left: 4px solid #d32f2f;
            padding: 12px;
            margin: 20px 0;
            text-align: left;
            border-radius: 4px;
        }}
        .error-box strong {{
            color: #d32f2f;
            display: block;
            margin-bottom: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">⚠️</div>
        <h1>Authorization Error</h1>
        <div class="error-box">
            <strong>Error Code:</strong>
            <span>{html.escape(error)}</span>
        </div>
        <div class="error-box">
            <strong>Description:</strong>
            <span>{html.escape(error_description or 'No description provided')}</span>
        </div>
        <p style="margin-top: 24px; font-size: 14px; color: #999;">
            This window can be closed safely.
        </p>
    </div>
    <script>
        // Send error to parent window if in iframe
        if (window.parent !== window) {{
            window.parent.postMessage({{
                type: 'oauth_error',
                error: {error_js},
                error_description: {error_desc_js},
                state: {state_js}
            }}, '*');
        }}
        // Try to close the window if it was opened by JavaScript
        try {{
            if (window.opener) {{
                setTimeout(() => window.close(), 2000);
            }}
        }} catch (e) {{
            // Ignore errors when trying to close
        }}
    </script>
</body>
</html>
"""
    else:
        # Safely escape values for JavaScript
        code_js = json.dumps(code or "")
        state_js = json.dumps(state or "")
        code_display = html.escape(code or "None")
        callback_url = f"{config.server.base_url}/oauth/callback"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Authorization Successful</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 500px;
            width: 100%;
            text-align: center;
            animation: slideIn 0.3s ease-out;
        }}
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        .icon {{
            width: 80px;
            height: 80px;
            margin: 0 auto 20px;
            background: #e8f5e9;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            animation: checkmark 0.6s ease-in-out;
        }}
        @keyframes checkmark {{
            0% {{
                transform: scale(0);
            }}
            50% {{
                transform: scale(1.1);
            }}
            100% {{
                transform: scale(1);
            }}
        }}
        h1 {{
            color: #2e7d32;
            margin-bottom: 16px;
            font-size: 24px;
            font-weight: 600;
        }}
        .message {{
            color: #666;
            margin-bottom: 24px;
            line-height: 1.6;
        }}
        .code-box {{
            background: #f5f5f5;
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 16px;
            margin: 20px 0;
            word-break: break-all;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            font-size: 14px;
            color: #333;
        }}
        .code-label {{
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .spinner {{
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .spinner.show {{
            display: block;
        }}
        .status {{
            font-size: 14px;
            color: #999;
            margin-top: 16px;
        }}
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 12px;
            margin: 20px 0;
            text-align: left;
            border-radius: 4px;
            font-size: 12px;
            color: #555;
        }}
        .info-box code {{
            background: #fff;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">✓</div>
        <h1>Authorization Successful</h1>
        <p class="message">
            Your authorization code has been received.<br>
            This window will close automatically...
        </p>
        <div class="code-box">
            <div class="code-label">Authorization Code</div>
            <div>{code_display}</div>
        </div>
        <div class="info-box" id="infoBox" style="display: none;">
            <strong>📋 Next Steps:</strong><br>
            The authorization code is in the URL. Your client should:<br>
            1. Extract code from URL<br>
            2. POST to <code>/oauth/token</code> with:<br>
            &nbsp;&nbsp;• <code>grant_type=authorization_code</code><br>
            &nbsp;&nbsp;• <code>code={code_display}</code><br>
            &nbsp;&nbsp;• <code>redirect_uri={html.escape(callback_url)}</code><br>
            &nbsp;&nbsp;• <code>client_id</code> and <code>client_secret</code>
        </div>
        <div class="spinner" id="spinner"></div>
        <div class="status" id="status">Processing...</div>
    </div>
    <script>
        (function() {{
            // Extract code and state from URL (OAuth standard - clients should read from URL)
            const urlParams = new URLSearchParams(window.location.search);
            const codeFromUrl = urlParams.get('code') || {code_js};
            const stateFromUrl = urlParams.get('state') || {state_js};

            const code = codeFromUrl;
            const state = stateFromUrl;
            const spinner = document.getElementById('spinner');
            const status = document.getElementById('status');

            // Important: Code is available in URL for OAuth clients to extract
            // Standard OAuth flow: Client reads code from URL → POST to /oauth/token
            console.log('OAuth callback - Code available in URL:', code ? 'Yes' : 'No');
            console.log('Code:', code);
            console.log('State:', state);

            // Try multiple methods to communicate with parent/opener
            let handled = false;

            function markHandled(message) {{
                if (handled) return;
                handled = true;
                status.textContent = message;
                spinner.classList.add('show');
            }}

            // Method 1: postMessage to parent (for iframe)
            if (window.parent !== window) {{
                try {{
                    window.parent.postMessage({{
                        type: 'oauth_callback',
                        code: code,
                        state: state,
                        source: 'mcp_oauth_callback'
                    }}, '*');
                    markHandled('Code sent to parent window. Closing...');
                    setTimeout(() => {{
                        try {{
                            window.close();
                        }} catch (e) {{
                            // Window might not be closable
                        }}
                    }}, 1500);
                }} catch (e) {{
                    console.error('postMessage to parent failed:', e);
                }}
            }}

            // Method 2: Communicate with opener (for popup windows)
            if (window.opener && !window.opener.closed) {{
                try {{
                    window.opener.postMessage({{
                        type: 'oauth_callback',
                        code: code,
                        state: state,
                        source: 'mcp_oauth_callback'
                    }}, '*');
                    if (!handled) {{
                        markHandled('Code sent to opener. Closing...');
                        setTimeout(() => {{
                            try {{
                                window.close();
                            }} catch (e) {{
                                // Window might not be closable
                            }}
                        }}, 1500);
                    }}
                }} catch (e) {{
                    console.error('postMessage to opener failed:', e);
                }}
            }}

            // Method 3: Dispatch custom event (for advanced clients)
            try {{
                window.dispatchEvent(new CustomEvent('oauth_callback', {{
                    detail: {{ code: code, state: state }}
                }}));
            }} catch (e) {{
                console.error('Custom event dispatch failed:', e);
            }}

            // Method 4: Store in localStorage as fallback (if same origin)
            try {{
                if (window.localStorage) {{
                    localStorage.setItem('oauth_code', code);
                    localStorage.setItem('oauth_state', state || '');
                    localStorage.setItem('oauth_timestamp', Date.now().toString());
                    localStorage.setItem('oauth_callback_url', window.location.href);
                }}
            }} catch (e) {{
                // Cross-origin or storage disabled
            }}

            // Method 5: Try to close after delay if opened by script
            setTimeout(() => {{
                if (!handled) {{
                    // For MCP clients: They should read code from URL and exchange for token
                    status.innerHTML = '✅ Code is in URL. Client should exchange for token at <code>/oauth/token</code>';
                    spinner.classList.remove('show');
                    // Show info box
                    const infoBox = document.getElementById('infoBox');
                    if (infoBox) {{
                        infoBox.style.display = 'block';
                    }}
                }} else {{
                    try {{
                        window.close();
                    }} catch (e) {{
                        // Ignore - window might not be closable
                        status.textContent = 'Window will remain open. You can close it manually.';
                    }}
                }}
            }}, 2000);
        }})();
    </script>
</body>
</html>
"""

    logger.debug(
        f"OAuth callback received: code={'present' if code else 'missing'}, state={state}, error={error}"
    )
    return Response(content=html_content, media_type="text/html")


async def proxy_oauth_request(config: MCPConfig, request: Request) -> Response:
    """Proxy OAuth requests to Django OAuth server.

    Intercepts and rewrites redirect_uri to ensure OAuth callback flow works correctly.
    """
    path = request.url.path.replace("/oauth", "/o", 1)
    query_string = request.url.query

    # Intercept authorization requests
    if "/authorize" in path and request.method == "GET":
        params = dict(request.query_params)
        original_redirect_uri = params.get("redirect_uri")

        # Store original redirect_uri in state for later retrieval
        # Validate redirect_uri first to prevent open redirect attacks
        if original_redirect_uri and not _validate_redirect_uri(original_redirect_uri):
            logger.warning(f"Invalid redirect_uri rejected: {original_redirect_uri}")
            return JSONResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Invalid redirect_uri parameter",
                },
                status_code=400,
            )

        # If no state provided, create one that includes the original redirect_uri
        if "state" in params and original_redirect_uri:
            # Encode original redirect_uri in state (base64-like encoding)
            # Format: original_state|redirect_uri_base64
            state_value = params["state"]
            encoded_redirect = (
                base64.urlsafe_b64encode(original_redirect_uri.encode("utf-8"))
                .decode("utf-8")
                .rstrip("=")
            )
            params["state"] = f"{state_value}|{encoded_redirect}"

        # Rewrite redirect_uri to MCP server callback (to capture the code)
        # But we'll forward it back to original_redirect_uri in callback handler
        if "redirect_uri" in params:
            params["redirect_uri"] = f"{config.server.base_url}/oauth/callback"
        # Ensure scope is present - Django OAuth Toolkit requires it
        if "scope" not in params or not params.get("scope"):
            params["scope"] = "read"  # Default scope for MCP access
            logger.debug("Added default scope to authorization request")
        query_string = urlencode(params)
        logger.debug(
            f"Rewrote redirect_uri to MCP callback (original was: {original_redirect_uri})"
        )

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
