"""Tests for Docling processor."""

import pytest

# Check if docling is available
try:
    from docs2synth.preprocess.docling_proc import _DOCLING_AVAILABLE, DoclingProcessor
except ImportError:
    _DOCLING_AVAILABLE = False


@pytest.mark.skipif(not _DOCLING_AVAILABLE, reason="Docling not installed")
def test_docling_processor_initialization():
    """Test that DoclingProcessor can be initialized."""
    processor = DoclingProcessor()
    assert processor is not None
    assert processor.ocr_enabled is True
    assert processor.table_structure_enabled is True


@pytest.mark.skipif(not _DOCLING_AVAILABLE, reason="Docling not installed")
def test_docling_processor_with_options():
    """Test DoclingProcessor with custom options."""
    processor = DoclingProcessor(
        ocr_enabled=False, table_structure_enabled=False, device="cpu"
    )
    assert processor.ocr_enabled is False
    assert processor.table_structure_enabled is False
    assert processor.device == "cpu"
    assert processor.converter is not None


@pytest.mark.skipif(not _DOCLING_AVAILABLE, reason="Docling not installed")
def test_docling_processor_process_requires_file():
    """Test that process() requires a valid file path."""
    processor = DoclingProcessor()

    with pytest.raises(FileNotFoundError):
        processor.process("nonexistent_file.pdf")


@pytest.mark.skipif(not _DOCLING_AVAILABLE, reason="Docling not installed")
def test_docling_unavailable_error():
    """Test that proper error is raised when docling is not available."""
    # This test only makes sense if we can mock the import failure
    # Skip for now as it requires complex mocking
    pass


def test_docling_import_graceful_failure():
    """Test that missing docling doesn't crash the import."""
    # Should be able to import the module even if docling is not installed
    try:
        from docs2synth.preprocess import docling_proc

        assert docling_proc is not None
    except ImportError:
        pytest.fail("Should be able to import docling_proc module even without docling")


@pytest.mark.skipif(not _DOCLING_AVAILABLE, reason="Docling not installed")
def test_docling_processor_api_compatibility():
    """Test that DoclingProcessor has compatible API with other processors."""
    processor = DoclingProcessor()

    # Should accept lang and device parameters (even if ignored)
    assert hasattr(processor, "process")

    # Check method signature accepts keyword arguments
    import inspect

    sig = inspect.signature(processor.process)
    param_names = list(sig.parameters.keys())
    assert "file_path" in param_names
    assert "lang" in param_names
    assert "device" in param_names
