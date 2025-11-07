"""Tests for MCP prompts module."""

from __future__ import annotations

import pytest

pytest_plugins = ("pytest_asyncio",)


def test_register_prompts():
    """Test registering prompts with server."""
    from mcp.server import Server
    from mcp.types import GetPromptRequest, ListPromptsRequest

    from docs2synth.mcp.prompts import register_prompts

    server = Server(name="test")
    register_prompts(server)

    # Check that handlers are registered
    assert ListPromptsRequest in server.request_handlers
    assert GetPromptRequest in server.request_handlers


def test_prompt_specs_structure():
    """Test that all prompt specs have required fields."""
    from docs2synth.mcp.prompts import PROMPT_SPECS

    assert len(PROMPT_SPECS) >= 2

    for spec in PROMPT_SPECS:
        assert "name" in spec
        assert "description" in spec
        assert isinstance(spec["name"], str)
        assert isinstance(spec["description"], str)


def test_prompt_handlers_structure():
    """Test that all prompt handlers are properly registered."""
    from docs2synth.mcp.prompts import PROMPT_HANDLERS, PROMPT_SPECS

    # All specs should have handlers
    for spec in PROMPT_SPECS:
        assert spec["name"] in PROMPT_HANDLERS
        handler = PROMPT_HANDLERS[spec["name"]]
        assert callable(handler)


@pytest.mark.asyncio(loop_scope="function")
async def test_hello_prompt_module():
    """Test hello prompt module directly."""
    from docs2synth.mcp.prompts import hello

    assert hasattr(hello, "PROMPT_SPEC")
    assert hasattr(hello, "get_hello_prompt")

    spec = hello.PROMPT_SPEC
    assert spec["name"] == "hello"
    assert "description" in spec

    # Test the handler
    result = await hello.get_hello_prompt()
    assert result is not None
    assert hasattr(result, "messages")


@pytest.mark.asyncio(loop_scope="function")
async def test_help_prompt_module():
    """Test help prompt module directly."""
    from docs2synth.mcp.prompts import help as help_prompt

    assert hasattr(help_prompt, "PROMPT_SPEC")
    assert hasattr(help_prompt, "get_help_prompt")

    spec = help_prompt.PROMPT_SPEC
    assert spec["name"] == "help"
    assert "description" in spec

    # Test the handler
    result = await help_prompt.get_help_prompt()
    assert result is not None
    assert hasattr(result, "messages")


def test_prompt_modules_list():
    """Test that PROMPT_MODULES is properly defined."""
    from docs2synth.mcp.prompts import PROMPT_MODULES

    assert isinstance(PROMPT_MODULES, list)
    assert len(PROMPT_MODULES) >= 2

    # All modules should have PROMPT_SPEC
    for module in PROMPT_MODULES:
        assert hasattr(module, "PROMPT_SPEC")


def test_hello_prompt_spec():
    """Test hello prompt specification."""
    from docs2synth.mcp.prompts.hello import PROMPT_SPEC

    assert PROMPT_SPEC["name"] == "hello"
    assert isinstance(PROMPT_SPEC["description"], str)
    assert len(PROMPT_SPEC["description"]) > 0


def test_help_prompt_spec():
    """Test help prompt specification."""
    from docs2synth.mcp.prompts.help import PROMPT_SPEC

    assert PROMPT_SPEC["name"] == "help"
    assert isinstance(PROMPT_SPEC["description"], str)
    assert len(PROMPT_SPEC["description"]) > 0


@pytest.mark.asyncio(loop_scope="function")
async def test_hello_prompt_returns_valid_result():
    """Test that hello prompt returns valid GetPromptResult."""
    from mcp.types import PromptMessage

    from docs2synth.mcp.prompts.hello import get_hello_prompt

    result = await get_hello_prompt()

    assert result is not None
    assert hasattr(result, "messages")
    assert len(result.messages) > 0
    assert isinstance(result.messages[0], PromptMessage)


@pytest.mark.asyncio(loop_scope="function")
async def test_help_prompt_returns_valid_result():
    """Test that help prompt returns valid GetPromptResult."""
    from mcp.types import PromptMessage

    from docs2synth.mcp.prompts.help import get_help_prompt

    result = await get_help_prompt()

    assert result is not None
    assert hasattr(result, "messages")
    assert len(result.messages) > 0
    assert isinstance(result.messages[0], PromptMessage)


def test_all_prompts_have_handlers():
    """Test that all registered prompts have corresponding handlers."""
    from docs2synth.mcp.prompts import PROMPT_HANDLERS, PROMPT_SPECS

    prompt_names = {spec["name"] for spec in PROMPT_SPECS}
    handler_names = set(PROMPT_HANDLERS.keys())

    assert prompt_names == handler_names


def test_prompt_handler_naming_convention():
    """Test that prompt handlers follow naming convention."""
    from docs2synth.mcp.prompts import PROMPT_MODULES

    for module in PROMPT_MODULES:
        prompt_name = module.PROMPT_SPEC["name"]
        expected_handler = f"get_{prompt_name}_prompt"
        assert hasattr(module, expected_handler)
