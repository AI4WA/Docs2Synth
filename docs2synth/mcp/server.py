"""MCP server definition for Docs2Synth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from docs2synth import __version__
from docs2synth.utils import get_config, get_logger

SERVER_NAME = "Docs2Synth MCP"
SERVER_INSTRUCTIONS = (
    "This MCP server provides search and document retrieval capabilities for ChatGPT's "
    "chat and deep research features. Use the search tool to find relevant documents "
    "based on keywords, then use the fetch tool to retrieve complete document content "
    "with citations. The server also exposes Docs2Synth utilities for dataset discovery "
    "and configuration management."
)

logger = get_logger(__name__)


class DocumentIndex:
    """Simple document index for searching processed documents."""

    def __init__(self, data_dir: str | Path = None):
        """Initialize the document index.

        Args:
            data_dir: Path to the data directory containing processed documents
        """
        if data_dir is None:
            config = get_config()
            data_dir = config.get("data.processed_dir", "./data/processed")

        self.data_dir = Path(data_dir)
        self.documents = {}
        self._load_documents()

    def _load_documents(self):
        """Load all processed documents from the data directory."""
        processed_dir = self.data_dir / "images"
        if not processed_dir.exists():
            logger.warning(f"Processed documents directory not found: {processed_dir}")
            return

        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract text content from objects
                text_content = []
                if "objects" in data:
                    for obj in data["objects"].values():
                        if "text" in obj and obj["text"].strip():
                            text_content.append(obj["text"].strip())

                # Create document entry
                doc_id = json_file.stem
                self.documents[doc_id] = {
                    "id": doc_id,
                    "title": f"Document {doc_id}",
                    "text": " ".join(text_content),
                    "url": f"file://{json_file.absolute()}",
                    "metadata": {
                        "source_file": str(json_file),
                        "object_count": len(data.get("objects", {})),
                        "processor": (
                            "PaddleOCR" if "_easyocr" not in doc_id else "EasyOCR"
                        ),
                    },
                }

            except Exception as e:
                logger.warning(f"Failed to load document {json_file}: {e}")

        logger.info(f"Loaded {len(self.documents)} documents from {processed_dir}")

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for documents containing the query text.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching documents
        """
        if not query.strip():
            return []

        query_lower = query.lower()
        results = []

        for doc in self.documents.values():
            # Simple text matching
            if (
                query_lower in doc["text"].lower()
                or query_lower in doc["title"].lower()
            ):
                results.append(doc)

        # Sort by relevance (simple: number of query matches)
        def relevance_score(doc):
            text_lower = doc["text"].lower()
            title_lower = doc["title"].lower()
            return text_lower.count(query_lower) + title_lower.count(query_lower)

        results.sort(key=relevance_score, reverse=True)
        return results[:limit]

    def fetch(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a specific document by ID.

        Args:
            doc_id: Document ID to fetch

        Returns:
            Document data or None if not found
        """
        return self.documents.get(doc_id)


# Global document index instance
_document_index = None


def get_document_index() -> DocumentIndex:
    """Get or create the global document index."""
    global _document_index
    if _document_index is None:
        _document_index = DocumentIndex()
    return _document_index


def build_server() -> FastMCP:  # noqa: C901
    """Create and configure the FastMCP server instance."""
    server = FastMCP(
        name=SERVER_NAME,
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
        include_fastmcp_meta=False,
    )

    @server.tool(description="List dataset names registered for Docs2Synth downloads.")
    def list_datasets() -> list[str]:
        from docs2synth.datasets.downloader import DATASETS

        names = sorted(DATASETS.keys())
        logger.info("MCP tool list_datasets invoked (%d datasets)", len(names))
        return names

    @server.tool(
        description="Describe the download URL and suggested command for a dataset.",
    )
    def dataset_info(name: str) -> dict[str, Any]:
        from docs2synth.datasets.downloader import DATASETS

        datasets = {key.lower(): key for key in DATASETS.keys()}
        key = datasets.get(name.lower())
        if key is None:
            raise ValueError(
                f"Unknown dataset '{name}'. Use list_datasets to see available values."
            )

        download_url = DATASETS[key]
        logger.info("MCP tool dataset_info invoked for dataset '%s'", key)
        return {
            "name": key,
            "download_url": download_url,
            "cli_example": f"docs2synth datasets download {key}",
        }

    @server.tool(
        description="Return Docs2Synth default configuration values and active overrides."
    )
    def active_config() -> dict[str, Any]:
        # Config returns nested dict that is JSON serialisable
        logger.info("MCP tool active_config invoked")
        config = get_config()
        return config.to_dict()

    @server.tool(
        description="Search through processed documents for relevant content based on a query string."
    )
    def search(query: str) -> dict[str, Any]:
        """Search for documents containing the query text.

        Args:
            query: Search query string

        Returns:
            Dictionary with 'results' key containing array of matching documents
        """
        logger.info(f"MCP tool search invoked with query: '{query}'")

        try:
            doc_index = get_document_index()
            results = doc_index.search(query, limit=10)

            # Format results for MCP response
            formatted_results = []
            for doc in results:
                formatted_results.append(
                    {"id": doc["id"], "title": doc["title"], "url": doc["url"]}
                )

            return {"results": formatted_results}

        except Exception as e:
            logger.error(f"Error in search tool: {e}")
            return {"results": []}

    @server.tool(
        description="Fetch the full content of a specific document by its unique identifier."
    )
    def fetch(doc_id: str) -> dict[str, Any]:
        """Fetch a specific document by ID.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            Document data with full content
        """
        logger.info(f"MCP tool fetch invoked for document ID: '{doc_id}'")

        try:
            doc_index = get_document_index()
            doc = doc_index.fetch(doc_id)

            if doc is None:
                logger.warning(f"Document not found: {doc_id}")
                return {
                    "id": doc_id,
                    "title": "Document not found",
                    "text": "",
                    "url": "",
                    "metadata": {"error": "Document not found"},
                }

            return doc

        except Exception as e:
            logger.error(f"Error in fetch tool: {e}")
            return {
                "id": doc_id,
                "title": "Error fetching document",
                "text": "",
                "url": "",
                "metadata": {"error": str(e)},
            }

    # Add simple mock resources using decorator approach
    @server.resource(
        "resource://docs2synth/info",
        description="Simple info resource for testing.",
        mime_type="text/plain",
    )
    def info_resource() -> str:
        logger.info("MCP resource info requested")
        return "This is a simple info resource for testing Docs2Synth MCP server."

    @server.resource(
        "resource://docs2synth/status",
        description="Server status resource for testing.",
        mime_type="application/json",
    )
    def status_resource() -> str:
        logger.info("MCP resource status requested")
        return '{"status": "running", "version": "0.1.0", "server": "Docs2Synth MCP"}'

    # Add simple mock prompts
    @server.prompt(
        "hello",
        description="Simple hello prompt for testing.",
    )
    def hello_prompt() -> str:
        logger.info("MCP prompt hello invoked")
        return "Hello! This is a simple test prompt from Docs2Synth MCP server."

    @server.prompt(
        "help",
        description="Help prompt with basic server information.",
    )
    def help_prompt() -> str:
        logger.info("MCP prompt help invoked")
        return """# Docs2Synth MCP Help

This server provides:
- **Tools**: list_datasets, dataset_info, active_config, search, fetch
- **Resources**: info, status
- **Prompts**: hello, help

Use the tools to:
- **list_datasets**: Get available dataset names
- **dataset_info**: Get download info for a specific dataset
- **active_config**: View current configuration
- **search**: Search through processed documents by query
- **fetch**: Retrieve full document content by ID

The search and fetch tools enable deep research capabilities by allowing you to search through processed document content and retrieve full documents for analysis.

## Transport Options
- **SSE**: Use `docs2synth-mcp sse` for Server-Sent Events transport (recommended for ChatGPT)
- **STDIO**: Use `docs2synth-mcp stdio` for stdio transport

For ChatGPT integration, use the SSE transport and configure the connector with your server URL + `/sse`."""

    return server
