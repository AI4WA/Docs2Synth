"""MCP server implementation with StreamableHTTP transport and OAuth 2.0 authentication.

This module provides a Model Context Protocol server for Docs2Synth with:
- StreamableHTTP transport (MCP protocol 2025-03-26)
- OAuth 2.0/OIDC authentication via Django OAuth Toolkit
- Document search and retrieval tools
"""

from __future__ import annotations

from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from docs2synth import __version__
from docs2synth.mcp.auth import oauth_proxy
from docs2synth.mcp.auth.oidc_client import OIDCResourceServer
from docs2synth.mcp.config import MCPConfig
from docs2synth.utils import get_logger

from .prompts import register_prompts
from .resources import register_resources
from .tools import register_tools

logger = get_logger(__name__)

SERVER_NAME = "Docs2Synth MCP"
SERVER_INSTRUCTIONS = (
    "This MCP server provides search and document retrieval capabilities for ChatGPT's "
    "chat and deep research features. Use the search tool to find relevant documents "
    "based on keywords, then use the fetch tool to retrieve complete document content "
    "with citations. The server also exposes Docs2Synth utilities for dataset discovery "
    "and configuration management."
)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate OAuth tokens for protected endpoints."""

    def __init__(
        self, app, oidc_resource_server: OIDCResourceServer, protected_paths: list[str]
    ):
        super().__init__(app)
        self.oidc_resource_server = oidc_resource_server
        self.protected_paths = protected_paths

    async def dispatch(self, request: Request, call_next):
        """Validate token for protected endpoints."""
        if not any(request.url.path.startswith(p) for p in self.protected_paths):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {
                    "error": "unauthorized",
                    "error_description": "Missing or invalid Authorization header",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="Docs2Synth MCP"'},
            )

        token = auth_header[7:]
        token_data = await self.oidc_resource_server.verify_token(token)

        if not token_data:
            return JSONResponse(
                {
                    "error": "unauthorized",
                    "error_description": "Invalid or expired token",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="Docs2Synth MCP"'},
            )

        request.state.token_data = token_data
        request.state.user_id = token_data.get("sub") or token_data.get("username")
        logger.debug(f"Authenticated request from user: {request.state.user_id}")

        return await call_next(request)


def create_mcp_server() -> Server:
    """Create and configure the MCP server instance."""
    server = Server(
        name=SERVER_NAME, version=__version__, instructions=SERVER_INSTRUCTIONS
    )

    register_tools(server)
    register_resources(server)
    register_prompts(server)
    return server


def create_asgi_app(
    mcp_server: Server,
    oidc_resource_server: OIDCResourceServer,
    http_transport: StreamableHTTPServerTransport,
    config: MCPConfig,
) -> Starlette:
    """Create the ASGI application with MCP endpoints protected by OAuth."""

    async def handle_mcp_endpoint(request: Request) -> Response:
        """Handle MCP connections via StreamableHTTP transport."""
        await http_transport.handle_request(
            request.scope, request.receive, request._send
        )
        return Response()

    async def handle_health(request: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse(
            {"status": "healthy", "service": SERVER_NAME, "version": __version__}
        )

    async def handle_metadata(request: Request) -> JSONResponse:
        """Return MCP server metadata for discovery."""
        return JSONResponse(
            {
                "name": SERVER_NAME,
                "version": __version__,
                "instructions": SERVER_INSTRUCTIONS,
                "capabilities": {
                    "tools": True,
                    "resources": True,
                    "resourceTemplates": True,
                    "prompts": True,
                },
                "authentication": {
                    "required": True,
                    "type": "oauth2",
                    "discovery": {
                        "oauth": "/.well-known/oauth-authorization-server",
                        "oidc": "/.well-known/openid-configuration",
                        "resource": "/.well-known/oauth-protected-resource",
                    },
                },
                "endpoints": {"mcp": "/mcp", "health": "/health"},
                "transport": {
                    "type": "streamable-http",
                    "description": "StreamableHTTP transport (MCP protocol 2025-03-26)",
                },
            }
        )

    # OAuth proxy endpoint wrappers - properly await coroutines
    async def handle_oauth_metadata(request: Request) -> JSONResponse:
        return await oauth_proxy.handle_oauth_metadata(config)

    async def handle_protected_resource(request: Request) -> JSONResponse:
        return await oauth_proxy.handle_protected_resource_metadata(config, request)

    async def handle_openid_config(request: Request) -> JSONResponse:
        return await oauth_proxy.handle_openid_configuration(config)

    async def handle_client_reg(request: Request) -> JSONResponse:
        return await oauth_proxy.handle_client_registration(config)

    async def handle_callback(request: Request) -> Response:
        return await oauth_proxy.handle_oauth_callback(config, request)

    async def handle_oauth_proxy(request: Request) -> Response:
        return await oauth_proxy.proxy_oauth_request(config, request)

    async def handle_accounts_proxy(request: Request) -> Response:
        return await oauth_proxy.proxy_accounts_request(config, request)

    async def handle_login_proxy(request: Request) -> Response:
        return await oauth_proxy.proxy_login_request(config, request)

    async def handle_static_proxy(request: Request) -> Response:
        return await oauth_proxy.proxy_static_request(config, request)

    routes = [
        Route("/health", endpoint=handle_health, methods=["GET"]),
        Route("/", endpoint=handle_metadata, methods=["GET", "HEAD"]),
        Route("/metadata", endpoint=handle_metadata, methods=["GET"]),
        Route(
            "/.well-known/oauth-authorization-server",
            endpoint=handle_oauth_metadata,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/.well-known/oauth-protected-resource",
            endpoint=handle_protected_resource,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/.well-known/openid-configuration",
            endpoint=handle_openid_config,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/oauth/register", endpoint=handle_client_reg, methods=["POST", "OPTIONS"]
        ),
        Route("/oauth/callback", endpoint=handle_callback, methods=["GET", "OPTIONS"]),
        Route(
            "/oauth/{path:path}",
            endpoint=handle_oauth_proxy,
            methods=["GET", "POST", "OPTIONS"],
        ),
        Route(
            "/accounts/{path:path}",
            endpoint=handle_accounts_proxy,
            methods=["GET", "POST", "OPTIONS"],
        ),
        Route(
            "/login", endpoint=handle_login_proxy, methods=["GET", "POST", "OPTIONS"]
        ),
        Route(
            "/login/", endpoint=handle_login_proxy, methods=["GET", "POST", "OPTIONS"]
        ),
        Route("/static/{path:path}", endpoint=handle_static_proxy, methods=["GET"]),
        Route("/mcp", endpoint=handle_mcp_endpoint, methods=["GET", "POST", "DELETE"]),
    ]

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        ),
        Middleware(
            JWTAuthMiddleware,
            oidc_resource_server=oidc_resource_server,
            protected_paths=["/mcp"],
        ),
    ]

    app = Starlette(routes=routes, middleware=middleware)
    logger.info("ASGI application created with OAuth authentication")
    return app


def build_server(
    config: MCPConfig | None = None,
) -> tuple[Starlette, StreamableHTTPServerTransport, Server]:
    """Build and configure the complete MCP server with OAuth integration.

    Args:
        config: MCP configuration. If None, loads from config.mcp.yml and environment variables.

    Returns:
        Tuple of (Starlette ASGI app, StreamableHTTP transport, MCP Server)
    """
    if config is None:
        config = MCPConfig.load()

    logger.info(f"Building MCP server at {config.server.base_url}")
    logger.info("Using StreamableHTTP transport (MCP protocol 2025-03-26)")
    logger.info(
        f"Token validation: {'Introspection' if config.oauth.use_introspection else 'JWT'}"
    )

    oidc_resource_server = OIDCResourceServer(
        oidc_discovery_url=config.oauth.discovery_url,
        client_id=config.oauth.client_id,
        client_secret=config.oauth.client_secret,
        use_introspection=config.oauth.use_introspection,
        verify_ssl=config.oauth.verify_ssl,
        timeout=config.oauth.timeout,
    )

    mcp_server = create_mcp_server()
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=False,
        event_store=None,
    )

    app = create_asgi_app(mcp_server, oidc_resource_server, http_transport, config)
    return app, http_transport, mcp_server
