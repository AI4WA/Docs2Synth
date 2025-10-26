# MCP Server Integration

Expose Docs2Synth functionality to AI agents (Claude Desktop, ChatGPT, Cursor) via Model Context Protocol.

## Installation & Running

```bash
# Install
pip install -e ".[mcp]"

# Run SSE transport (for ChatGPT)
docs2synth-mcp sse --host 0.0.0.0 --port 8009

# Or STDIO transport (for CLI tools)
docs2synth-mcp stdio
```

## Docker Deployment

```bash
docker-compose -f docker-compose.mcp.yml up
```

Server available at: `http://localhost:8009/sse`

## Authentication

Enable in `docker-compose.mcp.yml`:

```yaml
environment:
  - AUTH_ENABLED=true
  - AUTH_VERIFY_URL=http://admin.kaiaperth.com/authenticate/api/token/verify/
  - AUTH_VERIFY_SSL=false  # true in production
```

**Usage:** Include `Authorization: YOUR_TOKEN` in all requests.


## Client Configuration


**Claude Desktop / Cursor:** Add to configuration, local:
```json
{
  "mcpServers": {
    "docs2synth": {
      "command": "/path/to/venv/bin/docs2synth-mcp",
      "args": ["stdio"]
    }
  }
}
```

If it is the remote server, you can setup it up with the instruction from the corresponding documentation.

