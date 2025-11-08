"""Tests for preprocessing runner and orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_get_processor_paddleocr(monkeypatch):
    """Test _get_processor returns PaddleOCRProcessor."""
    from docs2synth.preprocess.runner import _get_processor

    # Mock PaddleOCRProcessor where it's actually imported
    class MockPaddleOCR:
        pass

    monkeypatch.setattr(
        "docs2synth.preprocess.paddleocr.PaddleOCRProcessor", MockPaddleOCR
    )

    processor = _get_processor("paddleocr")
    assert isinstance(processor, MockPaddleOCR)


def test_get_processor_pdfplumber(monkeypatch):
    """Test _get_processor returns PDFPlumberProcessor."""
    from docs2synth.preprocess.runner import _get_processor

    # Mock PDFPlumberProcessor where it's actually imported
    class MockPDFPlumber:
        pass

    monkeypatch.setattr(
        "docs2synth.preprocess.pdfplumber_proc.PDFPlumberProcessor", MockPDFPlumber
    )

    processor = _get_processor("pdfplumber")
    assert isinstance(processor, MockPDFPlumber)


def test_get_processor_easyocr(monkeypatch):
    """Test _get_processor returns EasyOCRProcessor."""
    from docs2synth.preprocess.runner import _get_processor

    # Mock EasyOCRProcessor where it's actually imported
    class MockEasyOCR:
        pass

    monkeypatch.setattr(
        "docs2synth.preprocess.easyocr_proc.EasyOCRProcessor", MockEasyOCR
    )

    processor = _get_processor("easyocr")
    assert isinstance(processor, MockEasyOCR)


def test_get_processor_case_insensitive(monkeypatch):
    """Test _get_processor is case insensitive."""
    from docs2synth.preprocess.runner import _get_processor

    class MockProcessor:
        pass

    monkeypatch.setattr(
        "docs2synth.preprocess.paddleocr.PaddleOCRProcessor", MockProcessor
    )

    # Should work with various cases
    assert isinstance(_get_processor("PADDLEOCR"), MockProcessor)
    assert isinstance(_get_processor("PaddleOCR"), MockProcessor)
    assert isinstance(_get_processor("paddleocr"), MockProcessor)


def test_get_processor_unsupported():
    """Test _get_processor raises ValueError for unsupported processor."""
    from docs2synth.preprocess.runner import _get_processor

    with pytest.raises(ValueError) as exc_info:
        _get_processor("unsupported_processor")
    assert "Unsupported processor" in str(exc_info.value)


def test_determine_output_dir_with_explicit_path(tmp_path):
    """Test _determine_output_dir with explicit output_dir."""
    from docs2synth.preprocess.runner import _determine_output_dir
    from docs2synth.utils.config import Config

    input_path = tmp_path / "input.png"
    input_path.touch()
    output_dir = tmp_path / "explicit_output"

    config = Config()
    result = _determine_output_dir(input_path, output_dir, config)

    assert result == output_dir.resolve()
    assert result.exists()


def test_determine_output_dir_from_config_preprocess(tmp_path):
    """Test _determine_output_dir uses preprocess.output_dir from config."""
    from docs2synth.preprocess.runner import _determine_output_dir
    from docs2synth.utils.config import Config

    input_path = tmp_path / "input.png"
    input_path.touch()
    config_output = tmp_path / "config_output"

    config = Config({"preprocess": {"output_dir": str(config_output)}})
    result = _determine_output_dir(input_path, None, config)

    assert result == config_output.resolve()
    assert result.exists()


def test_determine_output_dir_from_config_data_processed(tmp_path):
    """Test _determine_output_dir falls back to data.processed_dir."""
    from docs2synth.preprocess.runner import _determine_output_dir
    from docs2synth.utils.config import Config

    input_path = tmp_path / "input.png"
    input_path.touch()
    processed_dir = tmp_path / "processed"

    config = Config({"data": {"processed_dir": str(processed_dir)}})
    result = _determine_output_dir(input_path, None, config)

    assert result == processed_dir.resolve()
    assert result.exists()


def test_determine_output_dir_for_directory_input(tmp_path):
    """Test _determine_output_dir creates subfolder for directory input."""
    from docs2synth.preprocess.runner import _determine_output_dir
    from docs2synth.utils.config import Config

    input_dir = tmp_path / "my_dataset"
    input_dir.mkdir()
    output_base = tmp_path / "output"

    config = Config()
    result = _determine_output_dir(input_dir, output_base, config)

    # Should create output/my_dataset/
    assert result == (output_base).resolve()
    assert result.exists()


def test_get_file_list_single_file(tmp_path):
    """Test _get_file_list with single file."""
    from docs2synth.preprocess.runner import _get_file_list

    test_file = tmp_path / "test.png"
    test_file.touch()

    files = _get_file_list(test_file)
    assert files == [test_file]


def test_get_file_list_directory(tmp_path):
    """Test _get_file_list with directory."""
    from docs2synth.preprocess.runner import _get_file_list

    # Create test files
    (tmp_path / "file1.png").touch()
    (tmp_path / "file2.jpg").touch()
    (tmp_path / "file3.pdf").touch()

    files = _get_file_list(tmp_path)
    assert len(files) == 3
    # Files should be sorted
    assert files[0].name == "file1.png"
    assert files[1].name == "file2.jpg"
    assert files[2].name == "file3.pdf"


def test_run_preprocess_nonexistent_path():
    """Test run_preprocess raises FileNotFoundError for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        run_preprocess("/nonexistent/path/to/file.png")


def test_run_preprocess_with_custom_config(monkeypatch, tmp_path):
    """Test run_preprocess with custom config."""

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

    input_file = tmp_path / "test.png"
    input_file.write_bytes(b"fake data")

    custom_output = tmp_path / "custom_output"
    from docs2synth.utils.config import Config

    config = Config({"preprocess": {"output_dir": str(custom_output)}})

    num_success, num_failed, outputs = run_preprocess(
        input_file, processor="paddleocr", config=config
    )

    assert num_success == 1
    assert num_failed == 0
    # Output should be in custom directory
    assert outputs[0].parent == custom_output.resolve()
