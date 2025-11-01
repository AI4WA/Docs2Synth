"""MCP prompts registration and integration."""

from __future__ import annotations

from mcp.server import Server
from mcp.types import GetPromptResult, Prompt

from . import hello
from . import help as help_prompt

# Collect all prompt modules
PROMPT_MODULES = [
    hello,
    help_prompt,
]

# Collect prompt specifications and handlers
PROMPT_SPECS = [module.PROMPT_SPEC for module in PROMPT_MODULES]
PROMPT_HANDLERS = {
    module.PROMPT_SPEC["name"]: getattr(
        module, f"get_{module.PROMPT_SPEC['name']}_prompt"
    )
    for module in PROMPT_MODULES
}


def register_prompts(server: Server) -> None:
    """Register all MCP prompts with the server."""

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        """List all available prompts."""
        return [
            Prompt(name=spec["name"], description=spec["description"])
            for spec in PROMPT_SPECS
        ]

    @server.get_prompt()
    async def get_prompt(
        name: str, arguments: dict[str, str] | None = None
    ) -> GetPromptResult:
        """Handle prompt retrieval by routing to the appropriate handler."""
        handler = PROMPT_HANDLERS.get(name)
        if not handler:
            raise ValueError(f"Unknown prompt: {name}")
        return await handler()
