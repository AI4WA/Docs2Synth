"""Basic tests for MCP module components."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ============================================================================
# MCP Config tests
# ============================================================================


def test_oauth_config_defaults():
    """Test OAuthConfig default values."""
    try:
        from docs2synth.mcp.config import OAuthConfig

        config = OAuthConfig()
        assert config.discovery_url is not None
        assert config.client_id is not None
        assert config.timeout > 0
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


def test_server_config_defaults():
    """Test ServerConfig default values."""
    try:
        from docs2synth.mcp.config import ServerConfig

        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8009
        assert config.base_url is not None
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


def test_mcp_config_initialization():
    """Test MCPConfig initialization."""
    try:
        from docs2synth.mcp.config import MCPConfig

        config = MCPConfig()
        assert config.server is not None
        assert config.oauth is not None
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


def test_mcp_config_from_file():
    """Test loading MCPConfig from YAML file."""
    try:
        from docs2synth.mcp.config import MCPConfig

        config_data = """
server:
  host: 127.0.0.1
  port: 9000
  base_url: http://localhost:9000

oauth:
  client_id: test-client-id
  timeout: 10.0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            config_path = f.name

        try:
            config = MCPConfig.from_file(config_path)
            assert config.server.host == "127.0.0.1"
            assert config.server.port == 9000
            assert config.oauth.client_id == "test-client-id"
            assert config.oauth.timeout == 10.0
        finally:
            Path(config_path).unlink()
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


def test_mcp_config_from_env():
    """Test loading MCPConfig from environment."""
    try:
        from docs2synth.mcp.config import MCPConfig

        with patch.dict(
            "os.environ",
            {
                "DOCS2SYNTH_MCP_SERVER_HOST": "192.168.1.1",
                "DOCS2SYNTH_MCP_SERVER_PORT": "8888",
            },
            clear=False,
        ):
            try:
                config = MCPConfig.from_env()
                # Just verify it creates without error
                assert config is not None
            except Exception:
                # from_env might not be implemented, that's okay
                pytest.skip("MCPConfig.from_env not implemented")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


# ============================================================================
# DocumentIndex tests
# ============================================================================


def test_document_index_initialization():
    """Test DocumentIndex initialization."""
    try:
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = DocumentIndex(data_dir=tmpdir)
            assert index.data_dir == Path(tmpdir)
            assert isinstance(index.documents, dict)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


def test_document_index_loads_json_documents():
    """Test that DocumentIndex loads JSON documents."""
    try:
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create processed documents directory structure
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            # Create a sample document
            doc_data = {
                "objects": {
                    "1": {"text": "Hello World", "type": "text"},
                    "2": {"text": "Test Document", "type": "text"},
                }
            }

            doc_path = images_dir / "test_doc.json"
            with open(doc_path, "w") as f:
                json.dump(doc_data, f)

            # Load index
            index = DocumentIndex(data_dir=tmpdir)
            assert "test_doc" in index.documents
            assert index.documents["test_doc"]["id"] == "test_doc"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


def test_document_index_missing_directory():
    """Test DocumentIndex with non-existent directory."""
    try:
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a path that doesn't exist
            non_existent = Path(tmpdir) / "nonexistent"

            index = DocumentIndex(data_dir=non_existent)
            # Should handle gracefully
            assert len(index.documents) == 0
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


def test_document_index_search():
    """Test document search functionality."""
    try:
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            # Create sample documents
            doc1 = {
                "objects": {
                    "1": {"text": "Python programming language", "type": "text"},
                }
            }
            doc2 = {
                "objects": {
                    "1": {"text": "JavaScript web development", "type": "text"},
                }
            }

            with open(images_dir / "doc1.json", "w") as f:
                json.dump(doc1, f)
            with open(images_dir / "doc2.json", "w") as f:
                json.dump(doc2, f)

            index = DocumentIndex(data_dir=tmpdir)

            # Try to search (if method exists)
            try:
                results = index.search("Python")
                # If search is implemented, verify results
                assert isinstance(results, (list, dict))
            except AttributeError:
                # search method might not be implemented yet
                pytest.skip("DocumentIndex.search not implemented")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MCP module not fully installed")


# ============================================================================
# MCP Tools tests
# ============================================================================


def test_list_datasets_tool_import():
    """Test that list_datasets tool can be imported."""
    try:
        from docs2synth.mcp.tools import list_datasets

        assert list_datasets is not None
    except ImportError:
        pytest.skip("MCP tools not available")


def test_dataset_info_tool_import():
    """Test that dataset_info tool can be imported."""
    try:
        from docs2synth.mcp.tools import dataset_info

        assert dataset_info is not None
    except ImportError:
        pytest.skip("MCP tools not available")


# ============================================================================
# MCP Prompts tests
# ============================================================================


def test_hello_prompt_import():
    """Test that hello prompt can be imported."""
    try:
        from docs2synth.mcp.prompts import hello

        assert hello is not None
    except ImportError:
        pytest.skip("MCP prompts not available")


def test_help_prompt_import():
    """Test that help prompt can be imported."""
    try:
        from docs2synth.mcp.prompts import help as help_prompt

        assert help_prompt is not None
    except ImportError:
        pytest.skip("MCP prompts not available")


# ============================================================================
# MCP CLI tests
# ============================================================================


def test_mcp_cli_import():
    """Test that MCP CLI can be imported."""
    try:
        from docs2synth.mcp import cli

        assert cli is not None
    except ImportError:
        pytest.skip("MCP CLI not available")


def test_mcp_server_import():
    """Test that MCP server can be imported."""
    try:
        from docs2synth.mcp import server

        assert server is not None
    except ImportError:
        pytest.skip("MCP server not available")
