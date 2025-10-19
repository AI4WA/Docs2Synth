"""Tests for PDFPlumber processor."""

from __future__ import annotations

import pytest

from docs2synth.preprocess.pdfplumber_proc import (
    _PDFPLUMBER_AVAILABLE,
    PDFPlumberProcessor,
)
from docs2synth.preprocess.schema import LabelType

pytestmark = pytest.mark.skipif(
    not _PDFPLUMBER_AVAILABLE, reason="pdfplumber not installed"
)


def test_pdfplumber_processor_runs_minimal(monkeypatch, tmp_path):
    """Test that PDFPlumberProcessor can process a mocked PDF."""

    # Mock pdfplumber to avoid needing a real PDF
    class DummyPage:
        def __init__(self):
            self.width = 612.0
            self.height = 792.0

        def extract_words(self, **kwargs):
            return [
                {
                    "text": "Hello",
                    "x0": 100.0,
                    "top": 50.0,
                    "x1": 150.0,
                    "bottom": 70.0,
                },
                {
                    "text": "World",
                    "x0": 160.0,
                    "top": 50.0,
                    "x1": 210.0,
                    "bottom": 70.0,
                },
            ]

        def extract_tables(self):
            return []

    class DummyPDF:
        def __init__(self, path):
            self.path = path
            self.pages = [DummyPage()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def dummy_open(path):
        return DummyPDF(path)

    monkeypatch.setattr(
        "docs2synth.preprocess.pdfplumber_proc.pdfplumber.open", dummy_open
    )

    # Create a dummy PDF file
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")  # Minimal PDF header

    proc = PDFPlumberProcessor()
    result = proc.process(str(pdf_path))

    # Verify basic structure
    assert result.process_metadata.processor_name == "pdfplumber"
    assert result.document_metadata.source == str(pdf_path)
    assert result.document_metadata.page_count == 1
    assert result.document_metadata.mime_type == "application/pdf"

    # Verify extracted text
    assert len(result.objects) == 2
    assert result.objects[0].text == "Hello"
    assert result.objects[1].text == "World"
    assert result.objects[0].label == LabelType.TEXT
    assert result.context == "Hello World"


def test_pdfplumber_processor_handles_empty_pdf(monkeypatch, tmp_path):
    """Test that PDFPlumberProcessor handles PDFs with no text."""

    class DummyPage:
        def __init__(self):
            self.width = 612.0
            self.height = 792.0

        def extract_words(self, **kwargs):
            return []

        def extract_tables(self):
            return []

    class DummyPDF:
        def __init__(self, path):
            self.pages = [DummyPage()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def dummy_open(path):
        return DummyPDF(path)

    monkeypatch.setattr(
        "docs2synth.preprocess.pdfplumber_proc.pdfplumber.open", dummy_open
    )

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    proc = PDFPlumberProcessor()
    result = proc.process(str(pdf_path))

    assert len(result.objects) == 0
    assert result.context == ""


def test_pdfplumber_processor_extracts_tables(monkeypatch, tmp_path):
    """Test that PDFPlumberProcessor can extract tables when enabled."""

    class DummyPage:
        def __init__(self):
            self.width = 612.0
            self.height = 792.0

        def extract_words(self, **kwargs):
            return []

        def extract_tables(self):
            return [
                [
                    ["Header1", "Header2"],
                    ["Value1", "Value2"],
                ]
            ]

    class DummyPDF:
        def __init__(self, path):
            self.pages = [DummyPage()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def dummy_open(path):
        return DummyPDF(path)

    monkeypatch.setattr(
        "docs2synth.preprocess.pdfplumber_proc.pdfplumber.open", dummy_open
    )

    pdf_path = tmp_path / "table.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    proc = PDFPlumberProcessor(extract_tables=True)
    result = proc.process(str(pdf_path))

    assert len(result.objects) == 1
    assert "Header1" in result.objects[0].text
    assert "Value1" in result.objects[0].text
    assert result.objects[0].label == LabelType.OTHER


def test_pdfplumber_processor_skips_non_pdf_files(tmp_path):
    """Test that PDFPlumberProcessor skips non-PDF files (images)."""
    # Create a PNG file
    png_path = tmp_path / "image.png"
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    png_path.write_bytes(png_data)

    proc = PDFPlumberProcessor(skip_non_pdf=True)
    result = proc.process(str(png_path))

    # Should return empty result
    assert len(result.objects) == 0
    assert result.context == ""
    assert result.document_metadata.page_count == 0


def test_pdfplumber_processor_raises_on_non_pdf_if_configured(tmp_path):
    """Test that PDFPlumberProcessor raises error for non-PDF when skip_non_pdf=False."""
    # Create a PNG file
    png_path = tmp_path / "image.png"
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    png_path.write_bytes(png_data)

    proc = PDFPlumberProcessor(skip_non_pdf=False)

    with pytest.raises(ValueError, match="File is not a PDF"):
        proc.process(str(png_path))


def test_pdfplumber_processor_multipage(monkeypatch, tmp_path):
    """Test that PDFPlumberProcessor handles multi-page PDFs."""

    class DummyPage:
        def __init__(self, page_num):
            self.width = 612.0
            self.height = 792.0
            self.page_num = page_num

        def extract_words(self, **kwargs):
            return [
                {
                    "text": f"Page{self.page_num}",
                    "x0": 100.0,
                    "top": 50.0,
                    "x1": 150.0,
                    "bottom": 70.0,
                }
            ]

        def extract_tables(self):
            return []

    class DummyPDF:
        def __init__(self, path):
            self.pages = [DummyPage(0), DummyPage(1), DummyPage(2)]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def dummy_open(path):
        return DummyPDF(path)

    monkeypatch.setattr(
        "docs2synth.preprocess.pdfplumber_proc.pdfplumber.open", dummy_open
    )

    pdf_path = tmp_path / "multipage.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    proc = PDFPlumberProcessor()
    result = proc.process(str(pdf_path))

    assert result.document_metadata.page_count == 3
    assert len(result.objects) == 3
    assert result.objects[0].page == 0
    assert result.objects[1].page == 1
    assert result.objects[2].page == 2
    assert result.context == "Page0 Page1 Page2"
