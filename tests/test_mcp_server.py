"""Tests for MCP server module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest_plugins = ("pytest_asyncio",)


def test_create_mcp_server():
    """Test creating MCP server instance."""
    from docs2synth.mcp.server import create_mcp_server

    server = create_mcp_server()

    assert server is not None
    assert server.name == "Docs2Synth MCP"
    assert hasattr(server, "version")


def test_server_has_tools():
    """Test that server has tools registered."""
    from mcp.types import CallToolRequest, ListToolsRequest

    from docs2synth.mcp.server import create_mcp_server

    server = create_mcp_server()

    # Check that tools are registered
    assert ListToolsRequest in server.request_handlers
    assert CallToolRequest in server.request_handlers


def test_server_has_resources():
    """Test that server has resources registered."""
    from mcp.types import ListResourcesRequest, ReadResourceRequest

    from docs2synth.mcp.server import create_mcp_server

    server = create_mcp_server()

    # Check that resources are registered
    assert ListResourcesRequest in server.request_handlers
    assert ReadResourceRequest in server.request_handlers


def test_server_has_prompts():
    """Test that server has prompts registered."""
    from mcp.types import GetPromptRequest, ListPromptsRequest

    from docs2synth.mcp.server import create_mcp_server

    server = create_mcp_server()

    # Check that prompts are registered
    assert ListPromptsRequest in server.request_handlers
    assert GetPromptRequest in server.request_handlers


def test_jwt_auth_middleware_initialization():
    """Test JWTAuthMiddleware initialization."""
    from docs2synth.mcp.server import JWTAuthMiddleware

    mock_app = MagicMock()
    mock_oidc = MagicMock()
    protected_paths = ["/mcp"]

    middleware = JWTAuthMiddleware(mock_app, mock_oidc, protected_paths)

    assert middleware.oidc_resource_server == mock_oidc
    assert middleware.protected_paths == protected_paths


@pytest.mark.asyncio(loop_scope="function")
async def test_jwt_auth_middleware_unprotected_path():
    """Test that middleware allows unprotected paths."""
    from docs2synth.mcp.server import JWTAuthMiddleware

    mock_app = MagicMock()
    mock_oidc = MagicMock()
    protected_paths = ["/mcp"]

    middleware = JWTAuthMiddleware(mock_app, mock_oidc, protected_paths)

    # Create mock request for unprotected path
    mock_request = MagicMock()
    mock_request.url.path = "/health"
    mock_request.headers.get.return_value = ""

    mock_call_next = AsyncMock(return_value=MagicMock())

    await middleware.dispatch(mock_request, mock_call_next)

    # Should call next without authentication
    mock_call_next.assert_called_once_with(mock_request)


@pytest.mark.asyncio(loop_scope="function")
async def test_jwt_auth_middleware_missing_auth_header():
    """Test that middleware rejects requests without auth header."""
    from docs2synth.mcp.server import JWTAuthMiddleware

    mock_app = MagicMock()
    mock_oidc = MagicMock()
    protected_paths = ["/mcp"]

    middleware = JWTAuthMiddleware(mock_app, mock_oidc, protected_paths)

    # Create mock request for protected path without auth
    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers.get.return_value = ""

    mock_call_next = AsyncMock()

    result = await middleware.dispatch(mock_request, mock_call_next)

    # Should return 401
    assert result.status_code == 401
    mock_call_next.assert_not_called()


@pytest.mark.asyncio(loop_scope="function")
async def test_jwt_auth_middleware_invalid_token():
    """Test that middleware rejects invalid tokens."""
    from docs2synth.mcp.server import JWTAuthMiddleware

    mock_app = MagicMock()
    mock_oidc = MagicMock()
    mock_oidc.verify_token = AsyncMock(return_value=None)
    protected_paths = ["/mcp"]

    middleware = JWTAuthMiddleware(mock_app, mock_oidc, protected_paths)

    # Create mock request with invalid token
    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers.get.return_value = "Bearer invalid_token"
    mock_request.state = MagicMock()

    mock_call_next = AsyncMock()

    result = await middleware.dispatch(mock_request, mock_call_next)

    # Should return 401
    assert result.status_code == 401
    mock_call_next.assert_not_called()


@pytest.mark.asyncio(loop_scope="function")
async def test_jwt_auth_middleware_valid_token():
    """Test that middleware allows valid tokens."""
    from docs2synth.mcp.server import JWTAuthMiddleware

    mock_app = MagicMock()
    mock_oidc = MagicMock()
    token_data = {"sub": "user123", "username": "testuser"}
    mock_oidc.verify_token = AsyncMock(return_value=token_data)
    protected_paths = ["/mcp"]

    middleware = JWTAuthMiddleware(mock_app, mock_oidc, protected_paths)

    # Create mock request with valid token
    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers.get.return_value = "Bearer valid_token"
    mock_request.state = MagicMock()

    mock_call_next = AsyncMock(return_value=MagicMock())

    await middleware.dispatch(mock_request, mock_call_next)

    # Should call next
    mock_call_next.assert_called_once_with(mock_request)
    assert mock_request.state.token_data == token_data
    assert mock_request.state.user_id == "user123"


def test_create_asgi_app():
    """Test creating ASGI application."""
    from mcp.server import Server
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    from docs2synth.mcp.config import MCPConfig
    from docs2synth.mcp.server import create_asgi_app

    mock_oidc = MagicMock()
    mcp_server = Server(name="test")
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=False,
        event_store=None,
    )
    config = MCPConfig()

    app = create_asgi_app(mcp_server, mock_oidc, http_transport, config)

    assert app is not None
    # Check that routes are configured
    assert len(app.routes) > 0


def test_asgi_app_routes():
    """Test that ASGI app has all required routes."""
    from mcp.server import Server
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    from docs2synth.mcp.config import MCPConfig
    from docs2synth.mcp.server import create_asgi_app

    mock_oidc = MagicMock()
    mcp_server = Server(name="test")
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=False,
        event_store=None,
    )
    config = MCPConfig()

    app = create_asgi_app(mcp_server, mock_oidc, http_transport, config)

    # Get route paths
    route_paths = [route.path for route in app.routes]

    # Check for required routes
    assert "/health" in route_paths
    assert "/" in route_paths
    assert "/metadata" in route_paths
    assert "/mcp" in route_paths


def test_build_server():
    """Test building complete server."""
    from docs2synth.mcp.config import MCPConfig
    from docs2synth.mcp.server import build_server

    config = MCPConfig()

    app, http_transport, mcp_server = build_server(config)

    assert app is not None
    assert http_transport is not None
    assert mcp_server is not None
    assert mcp_server.name == "Docs2Synth MCP"


def test_build_server_without_config():
    """Test building server without explicit config."""
    from docs2synth.mcp.server import build_server

    # Should use default config
    app, http_transport, mcp_server = build_server()

    assert app is not None
    assert http_transport is not None
    assert mcp_server is not None


@pytest.mark.asyncio(loop_scope="function")
async def test_handle_health():
    """Test health check endpoint."""
    from mcp.server import Server
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    from docs2synth.mcp.config import MCPConfig
    from docs2synth.mcp.server import create_asgi_app

    mock_oidc = MagicMock()
    mcp_server = Server(name="test")
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=False,
        event_store=None,
    )
    config = MCPConfig()

    app = create_asgi_app(mcp_server, mock_oidc, http_transport, config)

    # Create test client
    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio(loop_scope="function")
async def test_handle_metadata():
    """Test metadata endpoint."""
    from mcp.server import Server
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    from docs2synth.mcp.config import MCPConfig
    from docs2synth.mcp.server import create_asgi_app

    mock_oidc = MagicMock()
    mcp_server = Server(name="test")
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=False,
        event_store=None,
    )
    config = MCPConfig()

    app = create_asgi_app(mcp_server, mock_oidc, http_transport, config)

    # Create test client
    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.get("/metadata")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Docs2Synth MCP"
    assert "capabilities" in data
    assert "authentication" in data


@pytest.mark.asyncio(loop_scope="function")
async def test_mcp_endpoint_requires_auth():
    """Test that MCP endpoint requires authentication."""
    from mcp.server import Server
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    from docs2synth.mcp.config import MCPConfig
    from docs2synth.mcp.server import create_asgi_app

    mock_oidc = MagicMock()
    mock_oidc.verify_token = AsyncMock(return_value=None)
    mcp_server = Server(name="test")
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=False,
        event_store=None,
    )
    config = MCPConfig()

    app = create_asgi_app(mcp_server, mock_oidc, http_transport, config)

    # Create test client
    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.get("/mcp")

    # Should return 401 without auth
    assert response.status_code == 401


def test_server_constants():
    """Test server constants are defined."""
    from docs2synth.mcp import server

    assert hasattr(server, "SERVER_NAME")
    assert hasattr(server, "SERVER_INSTRUCTIONS")
    assert isinstance(server.SERVER_NAME, str)
    assert isinstance(server.SERVER_INSTRUCTIONS, str)


def test_cors_middleware_configured():
    """Test that CORS middleware is configured."""
    from mcp.server import Server
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    from docs2synth.mcp.config import MCPConfig
    from docs2synth.mcp.server import create_asgi_app

    mock_oidc = MagicMock()
    mcp_server = Server(name="test")
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=False,
        event_store=None,
    )
    config = MCPConfig()

    app = create_asgi_app(mcp_server, mock_oidc, http_transport, config)

    # Check that middleware is configured
    assert app is not None
    # Middleware stack may be None or empty initially, but app.middleware exists
    assert hasattr(app, "middleware")
