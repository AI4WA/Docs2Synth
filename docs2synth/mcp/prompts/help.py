"""Help information prompt."""

from __future__ import annotations

from mcp.types import GetPromptResult, PromptMessage, TextContent


async def get_help_prompt() -> GetPromptResult:
    """Return help information about the MCP server."""
    return GetPromptResult(
        description="Help information about the Docs2Synth MCP server",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        "# Docs2Synth MCP Help\n\n"
                        "This server provides:\n"
                        "- **Tools**: list_datasets, dataset_info, active_config, search, fetch\n"
                        "- **Resources**: info, status\n"
                        "- **Resource Templates**: document/{doc_id}, dataset/{dataset_name}\n"
                        "- **Prompts**: hello, help\n\n"
                        "Use the tools to:\n"
                        "- **list_datasets**: Get available dataset names\n"
                        "- **dataset_info**: Get download info for a specific dataset\n"
                        "- **active_config**: View current configuration\n"
                        "- **search**: Search through processed documents by query\n"
                        "- **fetch**: Retrieve full document content by ID\n\n"
                        "## Authentication\n"
                        "This server uses OpenID Connect (OIDC) for authentication. Clients must:\n"
                        "1. Discover endpoints via /.well-known/openid-configuration\n"
                        "2. Authorize via /authorize endpoint\n"
                        "3. Exchange authorization code for access token via /token\n"
                        "4. Use Bearer token in Authorization header for MCP requests\n"
                    ),
                ),
            )
        ],
    )


PROMPT_SPEC = {
    "name": "help",
    "description": "Help prompt with basic server information.",
}
