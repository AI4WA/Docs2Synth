from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import List

from mcp.server import Server
from mcp.types import Resource, ResourceTemplate

from docs2synth import __version__

from ..common.document_index import get_document_index


def register_resources(server: Server) -> None:
    @server.list_resources()
    async def list_resources() -> List[Resource]:
        return [
            Resource(
                uri="resource://docs2synth/info",
                name="Server Information",
                description="Simple info resource for testing.",
                mimeType="text/plain",
            ),
            Resource(
                uri="resource://docs2synth/status",
                name="Server Status",
                description="Server status resource for testing.",
                mimeType="application/json",
            ),
        ]

    @server.list_resource_templates()
    async def list_resource_templates() -> List[ResourceTemplate]:
        return [
            ResourceTemplate(
                uriTemplate="resource://docs2synth/document/{doc_id}",
                name="Document by ID",
                description=(
                    "Retrieve a specific document by its unique identifier. Returns full document content including text, metadata, and source information."
                ),
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="resource://docs2synth/dataset/{dataset_name}",
                name="Dataset Information",
                description=(
                    "Get detailed information about a specific dataset, including download URL and CLI usage examples."
                ),
                mimeType="application/json",
            ),
        ]

    def _read_info() -> str:
        return "This is a simple info resource for testing Docs2Synth MCP server."

    def _read_status() -> str:
        return json.dumps(
            {
                "status": "running",
                "version": __version__,
                "server": "Docs2Synth MCP",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _read_document_by_id(doc_id: str) -> str:
        doc_index = get_document_index()
        doc = doc_index.fetch(doc_id)
        if doc is None:
            return json.dumps(
                {
                    "error": "Document not found",
                    "error_code": "DOCUMENT_NOT_FOUND",
                    "document_id": doc_id,
                    "message": f"No document found with ID '{doc_id}'. Use the search tool to find available documents.",
                },
                indent=2,
            )
        return json.dumps(doc, indent=2)

    def _read_dataset_by_name(dataset_name: str) -> str:
        from docs2synth.datasets.downloader import DATASETS

        datasets = {key.lower(): key for key in DATASETS.keys()}
        key = datasets.get(dataset_name.lower())
        if key is None:
            available_datasets = sorted(DATASETS.keys())
            return json.dumps(
                {
                    "error": "Dataset not found",
                    "error_code": "DATASET_NOT_FOUND",
                    "dataset_name": dataset_name,
                    "message": f"Unknown dataset '{dataset_name}'. Use list_datasets tool to see available datasets.",
                    "available_datasets": available_datasets,
                },
                indent=2,
            )
        return json.dumps(
            {
                "name": key,
                "download_url": DATASETS[key],
                "cli_example": f"docs2synth datasets download {key}",
            },
            indent=2,
        )

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        uri_str = str(uri)
        if uri_str == "resource://docs2synth/info":
            return _read_info()
        if uri_str == "resource://docs2synth/status":
            return _read_status()

        doc_match = re.match(r"^resource://docs2synth/document/(.+)$", uri_str)
        if doc_match:
            return _read_document_by_id(doc_match.group(1))

        dataset_match = re.match(r"^resource://docs2synth/dataset/(.+)$", uri_str)
        if dataset_match:
            return _read_dataset_by_name(dataset_match.group(1))

        return json.dumps(
            {
                "error": "Unknown resource",
                "error_code": "UNKNOWN_RESOURCE",
                "uri": uri_str,
                "message": (
                    "Resource URI '"
                    + uri_str
                    + "' is not recognized. Use list_resources or list_resource_templates to see available resources."
                ),
            },
            indent=2,
        )
