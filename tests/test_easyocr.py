"""Tests for EasyOCR processor."""

from __future__ import annotations

import pytest

from docs2synth.preprocess.easyocr_proc import _EASYOCR_AVAILABLE, EasyOCRProcessor
from docs2synth.preprocess.schema import LabelType

pytestmark = pytest.mark.skipif(not _EASYOCR_AVAILABLE, reason="easyocr not installed")


def test_easyocr_processor_runs_minimal(monkeypatch, tmp_path):
    """Test that EasyOCRProcessor can process a mocked image."""

    # Mock EasyOCR Reader
    class DummyReader:
        def __init__(self, *args, **kwargs):
            pass

        def readtext(self, image_path, **kwargs):
            # Return mock results: List[Tuple[bbox_points, text, confidence]]
            return [
                (
                    [[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]],
                    "Hello",
                    0.95,
                ),
                (
                    [[110.0, 20.0], [200.0, 20.0], [200.0, 50.0], [110.0, 50.0]],
                    "World",
                    0.92,
                ),
            ]

    def dummy_reader_constructor(*args, **kwargs):
        return DummyReader(*args, **kwargs)

    monkeypatch.setattr(
        "docs2synth.preprocess.easyocr_proc.easyocr.Reader", dummy_reader_constructor
    )

    # Create a dummy image file
    img_path = tmp_path / "test.png"
    # Create a minimal valid PNG (1x1 pixel, black)
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path.write_bytes(png_data)

    proc = EasyOCRProcessor(lang_list=["en"], gpu=False, verbose=False)
    result = proc.process(str(img_path))

    # Verify basic structure
    assert result.process_metadata.processor_name == "easyocr"
    assert result.document_metadata.source == str(img_path)
    assert result.document_metadata.language == "en"

    # Verify extracted text
    assert len(result.objects) == 2
    assert result.objects[0].text == "Hello"
    assert result.objects[1].text == "World"
    assert result.objects[0].label == LabelType.TEXT
    assert result.objects[0].score == 0.95
    assert result.objects[1].score == 0.92

    # Verify bounding boxes (converted from quadrilateral to axis-aligned)
    assert result.objects[0].bbox == (10.0, 20.0, 100.0, 50.0)
    assert result.objects[1].bbox == (110.0, 20.0, 200.0, 50.0)

    assert result.context == "Hello World"


def test_easyocr_processor_handles_empty_result(monkeypatch, tmp_path):
    """Test that EasyOCRProcessor handles images with no text."""

    class DummyReader:
        def __init__(self, *args, **kwargs):
            pass

        def readtext(self, image_path, **kwargs):
            return []

    def dummy_reader_constructor(*args, **kwargs):
        return DummyReader(*args, **kwargs)

    monkeypatch.setattr(
        "docs2synth.preprocess.easyocr_proc.easyocr.Reader", dummy_reader_constructor
    )

    img_path = tmp_path / "empty.png"
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path.write_bytes(png_data)

    proc = EasyOCRProcessor(lang_list=["en"], gpu=False)
    result = proc.process(str(img_path))

    assert len(result.objects) == 0
    assert result.context == ""


def test_easyocr_processor_caches_reader(monkeypatch, tmp_path):
    """Test that EasyOCR Reader instances are cached."""

    init_count = {"count": 0}

    class DummyReader:
        def __init__(self, *args, **kwargs):
            init_count["count"] += 1

        def readtext(self, image_path, **kwargs):
            return []

    def dummy_reader_constructor(*args, **kwargs):
        return DummyReader(*args, **kwargs)

    monkeypatch.setattr(
        "docs2synth.preprocess.easyocr_proc.easyocr.Reader", dummy_reader_constructor
    )

    img_path = tmp_path / "test.png"
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path.write_bytes(png_data)

    proc = EasyOCRProcessor(lang_list=["en"], gpu=False)

    # Process same image twice
    proc.process(str(img_path))
    proc.process(str(img_path))

    # Reader should only be initialized once
    assert init_count["count"] == 1


def test_easyocr_processor_multilang(monkeypatch, tmp_path):
    """Test that EasyOCRProcessor supports multiple languages."""

    class DummyReader:
        def __init__(self, lang_list, *args, **kwargs):
            self.lang_list = lang_list

        def readtext(self, image_path, **kwargs):
            return [
                (
                    [[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]],
                    "Bonjour",
                    0.90,
                ),
            ]

    def dummy_reader_constructor(*args, **kwargs):
        return DummyReader(*args, **kwargs)

    monkeypatch.setattr(
        "docs2synth.preprocess.easyocr_proc.easyocr.Reader", dummy_reader_constructor
    )

    img_path = tmp_path / "french.png"
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path.write_bytes(png_data)

    proc = EasyOCRProcessor(lang_list=["en", "fr"], gpu=False)
    result = proc.process(str(img_path))

    assert len(result.objects) == 1
    assert result.objects[0].text == "Bonjour"
    # Language should be set to first in list
    assert result.document_metadata.language == "en"


def test_easyocr_processor_skips_empty_text(monkeypatch, tmp_path):
    """Test that EasyOCRProcessor skips empty or whitespace-only text."""

    class DummyReader:
        def __init__(self, *args, **kwargs):
            pass

        def readtext(self, image_path, **kwargs):
            return [
                (
                    [[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]],
                    "",
                    0.95,
                ),
                (
                    [[110.0, 20.0], [200.0, 20.0], [200.0, 50.0], [110.0, 50.0]],
                    "   ",
                    0.92,
                ),
                (
                    [[210.0, 20.0], [300.0, 20.0], [300.0, 50.0], [210.0, 50.0]],
                    "Valid",
                    0.93,
                ),
            ]

    def dummy_reader_constructor(*args, **kwargs):
        return DummyReader(*args, **kwargs)

    monkeypatch.setattr(
        "docs2synth.preprocess.easyocr_proc.easyocr.Reader", dummy_reader_constructor
    )

    img_path = tmp_path / "test.png"
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path.write_bytes(png_data)

    proc = EasyOCRProcessor(lang_list=["en"], gpu=False)
    result = proc.process(str(img_path))

    # Should only have 1 valid object
    assert len(result.objects) == 1
    assert result.objects[0].text == "Valid"
    assert result.context == "Valid"
