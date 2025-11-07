"""Tests for MCP tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio(loop_scope="function")
async def test_list_datasets_tool():
    """Test list_datasets tool returns dataset names."""
    from docs2synth.mcp.tools.list_datasets import handle_list_datasets

    result = await handle_list_datasets({})

    assert len(result) == 1
    assert result[0].type == "text"
    data = json.loads(result[0].text)
    assert "datasets" in data
    assert isinstance(data["datasets"], list)
    # Should have at least the known datasets
    assert len(data["datasets"]) > 0


@pytest.mark.asyncio(loop_scope="function")
async def test_dataset_info_tool_valid_dataset():
    """Test dataset_info tool with valid dataset."""
    from docs2synth.mcp.tools.dataset_info import handle_dataset_info

    result = await handle_dataset_info({"name": "cord"})

    assert len(result) == 1
    assert result[0].type == "text"
    data = json.loads(result[0].text)
    assert data["name"] == "cord"
    assert "download_url" in data
    assert "cli_example" in data
    assert "docs2synth datasets download cord" in data["cli_example"]


@pytest.mark.asyncio(loop_scope="function")
async def test_dataset_info_tool_case_insensitive():
    """Test dataset_info tool is case insensitive."""
    from docs2synth.mcp.tools.dataset_info import handle_dataset_info

    result = await handle_dataset_info({"name": "CORD"})

    assert len(result) == 1
    data = json.loads(result[0].text)
    assert data["name"] == "cord"


@pytest.mark.asyncio(loop_scope="function")
async def test_dataset_info_tool_invalid_dataset():
    """Test dataset_info tool with invalid dataset."""
    from docs2synth.mcp.tools.dataset_info import handle_dataset_info

    result = await handle_dataset_info({"name": "nonexistent_dataset"})

    assert len(result) == 1
    assert result[0].type == "text"
    data = json.loads(result[0].text)
    assert "error" in data
    assert "nonexistent_dataset" in data["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_dataset_info_tool_no_name():
    """Test dataset_info tool without name parameter."""
    from docs2synth.mcp.tools.dataset_info import handle_dataset_info

    result = await handle_dataset_info({})

    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "error" in data


@pytest.mark.asyncio(loop_scope="function")
async def test_search_tool_with_query():
    """Test search tool with a query string."""
    from docs2synth.mcp.tools.search import handle_search

    mock_doc_index = MagicMock()
    mock_doc_index.search.return_value = [
        {"id": "doc1", "title": "Test Document", "url": "http://example.com/doc1"},
        {"id": "doc2", "title": "Another Doc", "url": "http://example.com/doc2"},
    ]

    with patch(
        "docs2synth.mcp.tools.search.get_document_index", return_value=mock_doc_index
    ):
        result = await handle_search({"query": "test query"})

    assert len(result) == 1
    assert result[0].type == "text"
    data = json.loads(result[0].text)
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == "doc1"
    mock_doc_index.search.assert_called_once_with("test query", limit=10)


@pytest.mark.asyncio(loop_scope="function")
async def test_search_tool_no_query():
    """Test search tool without query parameter."""
    from docs2synth.mcp.tools.search import handle_search

    mock_doc_index = MagicMock()
    mock_doc_index.search.return_value = []

    with patch(
        "docs2synth.mcp.tools.search.get_document_index", return_value=mock_doc_index
    ):
        result = await handle_search({})

    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "results" in data
    assert data["results"] == []
    mock_doc_index.search.assert_called_once_with("", limit=10)


@pytest.mark.asyncio(loop_scope="function")
async def test_active_config_tool():
    """Test active_config tool returns current configuration."""
    from docs2synth.mcp.tools.active_config import handle_active_config

    mock_config = MagicMock()
    mock_config.to_dict.return_value = {
        "server": {"host": "0.0.0.0", "port": 8009},
        "oauth": {"client_id": "test"},
    }

    with patch(
        "docs2synth.mcp.tools.active_config.get_config", return_value=mock_config
    ):
        result = await handle_active_config({})

    assert len(result) == 1
    assert result[0].type == "text"
    data = json.loads(result[0].text)
    assert "server" in data
    assert "oauth" in data


@pytest.mark.asyncio(loop_scope="function")
async def test_fetch_tool_valid_document():
    """Test fetch tool with valid document ID."""
    from docs2synth.mcp.tools.fetch import handle_fetch

    mock_doc_index = MagicMock()
    mock_doc_index.fetch.return_value = {
        "id": "doc1",
        "title": "Test Document",
        "url": "http://example.com/doc1",
        "text": "This is test content",
    }

    with patch(
        "docs2synth.mcp.tools.fetch.get_document_index", return_value=mock_doc_index
    ):
        result = await handle_fetch({"doc_id": "doc1"})

    assert len(result) == 1
    assert result[0].type == "text"
    data = json.loads(result[0].text)
    assert data["id"] == "doc1"
    assert data["title"] == "Test Document"


@pytest.mark.asyncio(loop_scope="function")
async def test_fetch_tool_invalid_document():
    """Test fetch tool with invalid document ID."""
    from docs2synth.mcp.tools.fetch import handle_fetch

    mock_doc_index = MagicMock()
    mock_doc_index.fetch.return_value = None

    with patch(
        "docs2synth.mcp.tools.fetch.get_document_index", return_value=mock_doc_index
    ):
        result = await handle_fetch({"doc_id": "nonexistent"})

    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "error" in data


def test_tool_specs_structure():
    """Test that all tool specs have required fields."""
    from docs2synth.mcp.tools import (
        active_config,
        dataset_info,
        fetch,
        list_datasets,
        search,
    )

    tools = [list_datasets, dataset_info, search, active_config, fetch]

    for tool in tools:
        assert hasattr(tool, "TOOL_SPEC")
        spec = tool.TOOL_SPEC
        assert "name" in spec
        assert "description" in spec
        assert "inputSchema" in spec
        assert isinstance(spec["name"], str)
        assert isinstance(spec["description"], str)
        assert isinstance(spec["inputSchema"], dict)
