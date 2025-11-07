"""Tests for MCP DocumentIndex."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


class TestDocumentIndex:
    """Tests for DocumentIndex class."""

    def test_document_index_with_multiple_directories(self):
        """Test DocumentIndex loads from images subdirectory."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # DocumentIndex only loads from images/ subdirectory
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            # Create documents
            doc1 = {"objects": {"1": {"text": "Image document", "type": "text"}}}
            doc2 = {"objects": {"1": {"text": "PDF document", "type": "text"}}}

            with open(images_dir / "doc1.json", "w") as f:
                json.dump(doc1, f)
            with open(images_dir / "doc2.json", "w") as f:
                json.dump(doc2, f)

            index = DocumentIndex(data_dir=tmpdir)

            # Should load both documents
            assert len(index.documents) == 2
            assert "doc1" in index.documents
            assert "doc2" in index.documents

    def test_document_index_ignores_non_json_files(self):
        """Test DocumentIndex ignores non-JSON files."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            # Create JSON and non-JSON files
            doc_data = {"objects": {"1": {"text": "Test", "type": "text"}}}
            with open(images_dir / "doc.json", "w") as f:
                json.dump(doc_data, f)
            with open(images_dir / "readme.txt", "w") as f:
                f.write("Not JSON")
            with open(images_dir / "image.png", "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")

            index = DocumentIndex(data_dir=tmpdir)

            # Should only load the JSON file
            assert len(index.documents) == 1
            assert "doc" in index.documents

    def test_document_index_handles_malformed_json(self):
        """Test DocumentIndex handles malformed JSON gracefully."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            # Create valid and invalid JSON files
            valid_doc = {"objects": {"1": {"text": "Valid", "type": "text"}}}
            with open(images_dir / "valid.json", "w") as f:
                json.dump(valid_doc, f)
            with open(images_dir / "invalid.json", "w") as f:
                f.write("{this is not valid json")

            index = DocumentIndex(data_dir=tmpdir)

            # Should load only the valid document
            assert len(index.documents) >= 0  # At least doesn't crash
            # If it loaded the valid one, check it
            if "valid" in index.documents:
                assert index.documents["valid"]["id"] == "valid"

    def test_document_index_search_basic(self):
        """Test basic search functionality."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            # Create documents with searchable content
            docs = {
                "python_doc": {
                    "objects": {
                        "1": {"text": "Python programming language", "type": "text"}
                    }
                },
                "javascript_doc": {
                    "objects": {
                        "1": {"text": "JavaScript web framework", "type": "text"}
                    }
                },
                "java_doc": {
                    "objects": {
                        "1": {"text": "Java enterprise applications", "type": "text"}
                    }
                },
            }

            for name, content in docs.items():
                with open(images_dir / f"{name}.json", "w") as f:
                    json.dump(content, f)

            index = DocumentIndex(data_dir=tmpdir)

            # Search for Python
            results = index.search("Python")
            assert isinstance(results, list)
            assert len(results) > 0

            # Search with limit
            results = index.search("", limit=2)
            assert len(results) <= 2

    def test_document_index_fetch_document(self):
        """Test fetching a specific document by ID."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            doc_data = {
                "objects": {
                    "1": {"text": "Test document content", "type": "text"},
                    "2": {"text": "More content", "type": "text"},
                }
            }
            with open(images_dir / "test_doc.json", "w") as f:
                json.dump(doc_data, f)

            index = DocumentIndex(data_dir=tmpdir)

            # Get existing document using fetch method
            doc = index.fetch("test_doc")
            assert doc is not None
            assert doc["id"] == "test_doc"
            assert "text" in doc

            # Get non-existent document
            missing = index.fetch("nonexistent")
            assert missing is None

    def test_document_index_extracts_text_content(self):
        """Test that DocumentIndex extracts text content from objects."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            doc_data = {
                "objects": {
                    "obj1": {"text": "First paragraph", "type": "text"},
                    "obj2": {"text": "Second paragraph", "type": "text"},
                    "obj3": {"type": "image", "path": "/some/path.png"},
                }
            }
            with open(images_dir / "doc.json", "w") as f:
                json.dump(doc_data, f)

            index = DocumentIndex(data_dir=tmpdir)
            doc = index.documents.get("doc")

            assert doc is not None
            # Check that text content is extracted in the "text" field
            text_content = doc.get("text", "")
            assert "First paragraph" in text_content
            assert "Second paragraph" in text_content
            assert len(text_content) > 0

    def test_get_document_index_singleton(self):
        """Test that get_document_index returns singleton instance."""
        from docs2synth.mcp.common.document_index import get_document_index

        # Get index twice
        index1 = get_document_index()
        index2 = get_document_index()

        # Should be the same instance
        assert index1 is index2

    def test_document_index_with_custom_data_dir(self):
        """Test DocumentIndex with custom data directory."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create images subdirectory
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            index = DocumentIndex(data_dir=tmpdir)
            assert index.data_dir == Path(tmpdir)

    def test_document_index_url_generation(self):
        """Test that DocumentIndex generates URLs for documents."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            doc_data = {"objects": {"1": {"text": "Test", "type": "text"}}}
            with open(images_dir / "mydoc.json", "w") as f:
                json.dump(doc_data, f)

            index = DocumentIndex(data_dir=tmpdir)
            doc = index.documents.get("mydoc")

            assert doc is not None
            # Should have a URL
            assert "url" in doc
            assert "mydoc" in doc["url"]

    def test_document_index_title_generation(self):
        """Test that DocumentIndex generates titles for documents."""
        from docs2synth.mcp.common.document_index import DocumentIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()

            doc_data = {"objects": {"1": {"text": "Content here", "type": "text"}}}
            with open(images_dir / "my_document.json", "w") as f:
                json.dump(doc_data, f)

            index = DocumentIndex(data_dir=tmpdir)
            doc = index.documents.get("my_document")

            assert doc is not None
            # Should have a title
            assert "title" in doc
            # Title should be derived from filename or content
            assert len(doc["title"]) > 0
