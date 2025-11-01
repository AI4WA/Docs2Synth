"""Hello test prompt."""

from __future__ import annotations

from mcp.types import GetPromptResult, PromptMessage, TextContent


async def get_hello_prompt() -> GetPromptResult:
    """Return a simple hello prompt for testing."""
    return GetPromptResult(
        description="Simple hello prompt for testing",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="Hello! This is a simple test prompt from Docs2Synth MCP server.",
                ),
            )
        ],
    )


PROMPT_SPEC = {
    "name": "hello",
    "description": "Simple hello prompt for testing.",
}
