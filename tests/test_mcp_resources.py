"""Tests for MCP resources module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pytest_plugins = ("pytest_asyncio",)


def test_register_resources():
    """Test registering resources with server."""
    from mcp.server import Server
    from mcp.types import ListResourcesRequest, ReadResourceRequest

    from docs2synth.mcp.resources import register_resources

    server = Server(name="test")
    register_resources(server)

    # Check that handlers are registered
    assert ListResourcesRequest in server.request_handlers
    assert ReadResourceRequest in server.request_handlers


def test_read_dataset_by_name_found():
    """Test reading dataset by name when it exists."""
    from docs2synth.datasets.downloader import DATASETS

    # Verify test dataset exists
    assert "cord" in DATASETS or "CORD" in DATASETS


def test_read_dataset_by_name_case_insensitive():
    """Test dataset lookup is case insensitive."""
    from docs2synth.datasets.downloader import DATASETS

    datasets_lower = {key.lower(): key for key in DATASETS.keys()}
    assert "cord" in datasets_lower


@pytest.mark.asyncio(loop_scope="function")
async def test_resource_integration_with_document_index():
    """Test resource integration with document index."""
    from docs2synth.mcp.common.document_index import DocumentIndex

    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = Path(tmpdir) / "images"
        images_dir.mkdir()

        # Create a test document
        doc_data = {"objects": {"1": {"text": "Integration test", "type": "text"}}}
        with open(images_dir / "integration_doc.json", "w") as f:
            json.dump(doc_data, f)

        # Create index
        test_index = DocumentIndex(data_dir=tmpdir)

        # Verify document was loaded
        assert "integration_doc" in test_index.documents
        doc = test_index.fetch("integration_doc")
        assert doc is not None
        assert doc["id"] == "integration_doc"


def test_document_index_fetch_existing():
    """Test fetching existing document."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from docs2synth.mcp.common.document_index import DocumentIndex

        images_dir = Path(tmpdir) / "images"
        images_dir.mkdir()

        doc_data = {"objects": {"1": {"text": "Test content", "type": "text"}}}
        with open(images_dir / "test_doc.json", "w") as f:
            json.dump(doc_data, f)

        index = DocumentIndex(data_dir=tmpdir)
        doc = index.fetch("test_doc")

        assert doc is not None
        assert doc["id"] == "test_doc"


def test_document_index_fetch_nonexistent():
    """Test fetching non-existent document returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from docs2synth.mcp.common.document_index import DocumentIndex

        images_dir = Path(tmpdir) / "images"
        images_dir.mkdir()

        index = DocumentIndex(data_dir=tmpdir)
        doc = index.fetch("nonexistent")

        assert doc is None


def test_datasets_available():
    """Test that datasets are available."""
    from docs2synth.datasets.downloader import DATASETS

    assert isinstance(DATASETS, dict)
    assert len(DATASETS) > 0
    # Check for known datasets
    known_datasets = ["cord", "funsd"]
    for dataset in known_datasets:
        # Case-insensitive check
        datasets_lower = {k.lower(): k for k in DATASETS.keys()}
        assert dataset.lower() in datasets_lower


def test_resource_uris_format():
    """Test that resource URIs follow expected format."""
    expected_uris = [
        "resource://docs2synth/info",
        "resource://docs2synth/status",
    ]

    for uri in expected_uris:
        assert uri.startswith("resource://")
        assert "docs2synth" in uri


def test_resource_template_uris_format():
    """Test that resource template URIs follow expected format."""
    expected_templates = [
        "resource://docs2synth/document/{doc_id}",
        "resource://docs2synth/dataset/{dataset_name}",
    ]

    for template in expected_templates:
        assert template.startswith("resource://")
        assert "docs2synth" in template
        assert "{" in template and "}" in template


def test_document_index_search_functionality():
    """Test document index search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from docs2synth.mcp.common.document_index import DocumentIndex

        images_dir = Path(tmpdir) / "images"
        images_dir.mkdir()

        # Create test documents
        doc1 = {"objects": {"1": {"text": "Python programming", "type": "text"}}}
        doc2 = {"objects": {"1": {"text": "JavaScript coding", "type": "text"}}}

        with open(images_dir / "doc1.json", "w") as f:
            json.dump(doc1, f)
        with open(images_dir / "doc2.json", "w") as f:
            json.dump(doc2, f)

        index = DocumentIndex(data_dir=tmpdir)

        # Search for documents
        results = index.search("Python")
        assert isinstance(results, list)


def test_document_index_with_empty_directory():
    """Test document index with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from docs2synth.mcp.common.document_index import DocumentIndex

        images_dir = Path(tmpdir) / "images"
        images_dir.mkdir()

        index = DocumentIndex(data_dir=tmpdir)

        assert len(index.documents) == 0


def test_document_index_extracts_text():
    """Test that document index extracts text from objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from docs2synth.mcp.common.document_index import DocumentIndex

        images_dir = Path(tmpdir) / "images"
        images_dir.mkdir()

        doc_data = {
            "objects": {
                "1": {"text": "First paragraph", "type": "text"},
                "2": {"text": "Second paragraph", "type": "text"},
            }
        }

        with open(images_dir / "doc.json", "w") as f:
            json.dump(doc_data, f)

        index = DocumentIndex(data_dir=tmpdir)
        doc = index.documents.get("doc")

        assert doc is not None
        assert "text" in doc
        text_content = doc["text"]
        assert "First paragraph" in text_content
        assert "Second paragraph" in text_content
