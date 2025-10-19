"""Tests for preprocessing runner and orchestration."""

from __future__ import annotations

from pathlib import Path

from docs2synth.preprocess.runner import run_preprocess


def test_run_preprocess_includes_processor_in_filename(monkeypatch, tmp_path):
    """Test that output filenames include the processor name."""

    # Mock PaddleOCR processor
    class DummyProcessor:
        def process(self, path, **kwargs):
            from docs2synth.preprocess.schema import (
                DocumentMetadata,
                DocumentProcessResult,
                ProcessMetadata,
            )

            return DocumentProcessResult(
                objects={},
                object_list=[],
                bbox_list=[],
                context="test",
                reading_order_ids=[],
                process_metadata=ProcessMetadata(
                    processor_name="paddleocr",
                    timestamp="2024-01-01T00:00:00Z",
                    latency=0.0,
                ),
                document_metadata=DocumentMetadata(
                    source=path,
                    filename="test.png",
                    page_count=1,
                    size_bytes=100,
                    mime_type="image/png",
                    language="en",
                    width=100,
                    height=100,
                ),
            )

    def mock_get_processor(name):
        return DummyProcessor()

    monkeypatch.setattr(
        "docs2synth.preprocess.runner._get_processor", mock_get_processor
    )

    # Create a test file
    input_file = tmp_path / "document.png"
    input_file.write_bytes(b"fake image data")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Process with paddleocr
    num_success, num_failed, outputs = run_preprocess(
        input_file, processor="paddleocr", output_dir=output_dir
    )

    assert num_success == 1
    assert num_failed == 0
    assert len(outputs) == 1

    # Check filename includes processor name
    output_path = outputs[0]
    assert output_path.name == "document_paddleocr.json"
    assert output_path.exists()


def test_run_preprocess_different_processors_different_files(monkeypatch, tmp_path):
    """Test that different processors produce different output files."""

    class DummyProcessor:
        def __init__(self, name):
            self.name = name

        def process(self, path, **kwargs):
            from docs2synth.preprocess.schema import (
                DocumentMetadata,
                DocumentProcessResult,
                ProcessMetadata,
            )

            return DocumentProcessResult(
                objects={},
                object_list=[],
                bbox_list=[],
                context=f"processed by {self.name}",
                reading_order_ids=[],
                process_metadata=ProcessMetadata(
                    processor_name=self.name,
                    timestamp="2024-01-01T00:00:00Z",
                    latency=0.0,
                ),
                document_metadata=DocumentMetadata(
                    source=path,
                    filename="test.pdf",
                    page_count=1,
                    size_bytes=100,
                    mime_type="application/pdf",
                    language=None,
                    width=None,
                    height=None,
                ),
            )

    def mock_get_processor(name):
        return DummyProcessor(name)

    monkeypatch.setattr(
        "docs2synth.preprocess.runner._get_processor", mock_get_processor
    )

    # Create a test file
    input_file = tmp_path / "document.pdf"
    input_file.write_bytes(b"%PDF-1.4\n")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Process with pdfplumber
    num_success1, _, outputs1 = run_preprocess(
        input_file, processor="pdfplumber", output_dir=output_dir
    )

    # Process with paddleocr
    num_success2, _, outputs2 = run_preprocess(
        input_file, processor="paddleocr", output_dir=output_dir
    )

    # Process with easyocr
    num_success3, _, outputs3 = run_preprocess(
        input_file, processor="easyocr", output_dir=output_dir
    )

    assert num_success1 == 1
    assert num_success2 == 1
    assert num_success3 == 1

    # Check all three output files exist with different names
    assert outputs1[0].name == "document_pdfplumber.json"
    assert outputs2[0].name == "document_paddleocr.json"
    assert outputs3[0].name == "document_easyocr.json"

    assert outputs1[0].exists()
    assert outputs2[0].exists()
    assert outputs3[0].exists()

    # Verify content is different
    import json

    with open(outputs1[0]) as f:
        data1 = json.load(f)
    with open(outputs2[0]) as f:
        data2 = json.load(f)
    with open(outputs3[0]) as f:
        data3 = json.load(f)

    assert data1["context"] == "processed by pdfplumber"
    assert data2["context"] == "processed by paddleocr"
    assert data3["context"] == "processed by easyocr"


def test_run_preprocess_batch_includes_processor_in_all_filenames(
    monkeypatch, tmp_path
):
    """Test that batch processing includes processor name in all output filenames."""

    class DummyProcessor:
        def process(self, path, **kwargs):
            from docs2synth.preprocess.schema import (
                DocumentMetadata,
                DocumentProcessResult,
                ProcessMetadata,
            )

            return DocumentProcessResult(
                objects={},
                object_list=[],
                bbox_list=[],
                context="test",
                reading_order_ids=[],
                process_metadata=ProcessMetadata(
                    processor_name="easyocr",
                    timestamp="2024-01-01T00:00:00Z",
                    latency=0.0,
                ),
                document_metadata=DocumentMetadata(
                    source=path,
                    filename=Path(path).name,
                    page_count=1,
                    size_bytes=100,
                    mime_type="image/png",
                    language="en",
                    width=100,
                    height=100,
                ),
            )

    def mock_get_processor(name):
        return DummyProcessor()

    monkeypatch.setattr(
        "docs2synth.preprocess.runner._get_processor", mock_get_processor
    )

    # Create multiple test files
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    (input_dir / "doc1.png").write_bytes(b"data1")
    (input_dir / "doc2.png").write_bytes(b"data2")
    (input_dir / "doc3.png").write_bytes(b"data3")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Process directory
    num_success, num_failed, outputs = run_preprocess(
        input_dir, processor="easyocr", output_dir=output_dir
    )

    assert num_success == 3
    assert num_failed == 0
    assert len(outputs) == 3

    # Check all filenames include processor name
    output_names = sorted([p.name for p in outputs])
    assert output_names == [
        "doc1_easyocr.json",
        "doc2_easyocr.json",
        "doc3_easyocr.json",
    ]

    # Verify all files exist
    for output_path in outputs:
        assert output_path.exists()
