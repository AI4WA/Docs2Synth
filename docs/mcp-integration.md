# MCP Server Integration

Expose Docs2Synth functionality to AI agents (Claude Desktop, ChatGPT, Cursor) via Model Context Protocol.

## Remote Mode (SSE)

Docs2Synth MCP server runs in remote mode using SSE (Server-Sent Events) transport.

### Installation

```bash
pip install -e ".[mcp]"
```

### Run Server

```bash
# Start MCP server
docs2synth-mcp sse --host 0.0.0.0 --port 8009

# Custom port
docs2synth-mcp sse --host 0.0.0.0 --port 8080
```

Server available at: `http://localhost:8009/sse`

---

## Docker Deployment

```bash
# Start with docker-compose
docker-compose -f docker-compose.mcp.yml up

# Background mode
docker-compose -f docker-compose.mcp.yml up -d
```

Server available at: `http://localhost:8009/sse`

---

## Authentication

Enable authentication in `docker-compose.mcp.yml`:

```yaml
environment:
  - AUTH_ENABLED=true
  - AUTH_VERIFY_URL=http://admin.kaiaperth.com/authenticate/api/token/verify/
  - AUTH_VERIFY_SSL=false  # Set true in production
```

**Usage:** Include `Authorization: Bearer YOUR_TOKEN` in request headers.

---

## Client Configuration

### Claude Desktop

Add to Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "docs2synth": {
      "url": "http://localhost:8009/sse"
    }
  }
}
```

With authentication:
```json
{
  "mcpServers": {
    "docs2synth": {
      "url": "http://localhost:8009/sse",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

### Cursor

Add to Cursor settings (`.cursor/mcp_config.json`):

```json
{
  "mcpServers": {
    "docs2synth": {
      "url": "http://localhost:8009/sse"
    }
  }
}
```

### ChatGPT / Custom Clients

Connect to: `http://your-server:8009/sse`

Include authentication header if enabled:
```
Authorization: Bearer YOUR_TOKEN
```

---

## Available Tools

The MCP server exposes these tools:

- **preprocess**: Process documents with OCR
- **qa_generation**: Generate QA pairs
- **verify_qa**: Verify QA quality
- **retriever_train**: Train retriever models
- **rag_query**: Query RAG system

See [CLI Reference](cli-reference.md) for tool parameters.

---

## Health Check

```bash
# Check server status
curl http://localhost:8009/health

# Expected response
{"status": "ok"}
```

---

## Configuration

Create `config.mcp.yml` from `config.mcp.example.yml`:

```yaml
mcp:
  host: 0.0.0.0
  port: 8009
  auth_enabled: false

# API keys
agent:
  keys:
    openai_api_key: "sk-..."
    anthropic_api_key: "sk-ant-..."
```

---

## Troubleshooting

### Port already in use

```bash
# Check what's using port 8009
lsof -i :8009

# Kill process
kill -9 <PID>

# Or use different port
docs2synth-mcp sse --port 8010
```

### Connection refused

```bash
# Check server is running
curl http://localhost:8009/sse

# Check firewall
sudo ufw allow 8009
```

### Authentication fails

Verify token with:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8009/health
```

---

## References

- [Model Context Protocol Docs](https://modelcontextprotocol.io/)
- [Claude Desktop Configuration](https://claude.ai/docs)
- [CLI Reference](cli-reference.md)
