import pytest
from pathlib import Path
from typing import Any, Dict

from docs2synth.preprocess.schema import LabelType
from docs2synth.preprocess.docling_processor import _DOCLING_AVAILABLE, DoclingProcessor


def test_docling_device_and_lang_overrides(monkeypatch: Any, tmp_path: Path):
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        pytest.skip("Pillow not available to generate test image")

    img_path = tmp_path / "docling_test.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(img_path)

    class DummyDoclingConverter:
        def __init__(self, *args, **kwargs):
            pass

        def convert(self, path: str) -> Any:
            dummy_elem = type(
                "Elem",
                (),
                {
                    "text": "test lang override",
                    "confidence": 0.9,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "type": "text",
                },
            )()
            dummy_page = type("Page", (), {"elements": [dummy_elem]})()
            dummy_doc = type("Document", (), {"pages": [dummy_page]})()
            return type("ConvertResult", (), {"document": dummy_doc})()

    calls = {"gpu_checked": 0}

    def fake_gpu_available(self) -> bool:
        calls["gpu_checked"] += 1
        return True

    monkeypatch.setattr(
        "docs2synth.preprocess.docling_processor.DocumentConverter",
        DummyDoclingConverter,
        raising=True,
    )
    monkeypatch.setattr(DoclingProcessor, "_gpu_available", fake_gpu_available)

    proc = DoclingProcessor(lang="en")
    result = proc.process(str(img_path), lang="fr", device="gpu")
    assert calls["gpu_checked"] >= 1
    assert result.document_metadata.language == "fr"
    assert len(result.objects) == 1

    proc2 = DoclingProcessor(lang="en", device="cpu")
    result2 = proc2.process(str(img_path))
    assert result2.process_metadata.processor_name == "docling"
    assert len(result2.objects) == 1


def test_docling_caches_instance(monkeypatch: Any, tmp_path: Path):
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        pytest.skip("Pillow not available to generate test image")

    img_path = tmp_path / "docling_cache.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(img_path)

    created = {"count": 0}

    class DummyDoclingConverter:
        def __init__(self, *args, **kwargs):
            created["count"] += 1

        def convert(self, path: str) -> Any:
            dummy_elem = type(
                "Elem",
                (),
                {
                    "text": "cache test",
                    "confidence": 0.9,
                    "bbox": (0.0, 0.0, 10.0, 10.0),
                    "type": "text",
                },
            )()
            dummy_page = type("Page", (), {"elements": [dummy_elem]})()
            dummy_doc = type("Document", (), {"pages": [dummy_page]})()
            return type("ConvertResult", (), {"document": dummy_doc})()

    def mock_resolve_device(self, override: Any) -> str:
        return "cpu"

    monkeypatch.setattr(
        "docs2synth.preprocess.docling_processor.DocumentConverter",
        DummyDoclingConverter,
        raising=True,
    )
    monkeypatch.setattr(DoclingProcessor, "_resolve_device", mock_resolve_device)

    proc = DoclingProcessor(lang="en", ocr_engine="tesseract")
    proc.process(str(img_path), lang=None, device="cpu")
    proc.process(str(img_path), lang=None, device="cpu")
    assert created["count"] == 1


pytestmark = pytest.mark.skipif(
    not _DOCLING_AVAILABLE, reason="Docling not installed - skip integration tests"
)


def test_docling_processor_runs_minimal(monkeypatch: Any, tmp_path: Path):
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        pytest.skip("Pillow not available to generate test image")

    img_path = tmp_path / "docling_blank.png"
    Image.new("RGB", (100, 40), (255, 255, 255)).save(img_path)

    class DummyDoclingConverter:
        def convert(self, path: str) -> Any:
            dummy_elem = type(
                "Elem",
                (),
                {
                    "text": "minimal test",
                    "confidence": 0.95,
                    "bbox": (10.0, 10.0, 50.0, 30.0),
                    "type": "text",
                },
            )()
            dummy_page = type("Page", (), {"elements": [dummy_elem]})()
            dummy_doc = type("Document", (), {"pages": [dummy_page]})()
            return type("ConvertResult", (), {"document": dummy_doc})()

    def dummy_init_docling(self, *args, **kwargs) -> DummyDoclingConverter:
        return DummyDoclingConverter()

    monkeypatch.setattr(
        "docs2synth.preprocess.docling_processor.DocumentConverter",
        DummyDoclingConverter,
        raising=True,
    )
    monkeypatch.setattr(DoclingProcessor, "_init_docling", dummy_init_docling)

    proc = DoclingProcessor(use_layout_analysis=True)
    res = proc.process(str(img_path))

    assert res.process_metadata.processor_name == "docling"
    assert res.document_metadata.source == str(img_path)
    assert isinstance(res.bbox_list, list)
    assert isinstance(res.objects, dict)
    assert len(res.objects) == 1
    assert res.document_metadata.page_count == 1


def test_docling_processor_handles_empty_result(monkeypatch: Any, tmp_path: Path):
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        pytest.skip("Pillow not available to generate test image")

    img_path = tmp_path / "docling_empty.png"
    Image.new("RGB", (50, 20), (255, 255, 255)).save(img_path)

    class DummyDoclingConverter:
        def convert(self, path: str) -> Any:
            dummy_page = type("Page", (), {"elements": []})()
            dummy_doc = type("Document", (), {"pages": [dummy_page]})()
            return type("ConvertResult", (), {"document": dummy_doc})()

    def dummy_init_docling(self, *args, **kwargs) -> DummyDoclingConverter:
        return DummyDoclingConverter()

    monkeypatch.setattr(
        "docs2synth.preprocess.docling_processor.DocumentConverter",
        DummyDoclingConverter,
        raising=True,
    )
    monkeypatch.setattr(DoclingProcessor, "_init_docling", dummy_init_docling)

    proc = DoclingProcessor()
    res = proc.process(str(img_path))

    assert res.objects == {}
    assert res.bbox_list == []
    assert res.context == ""
    assert res.document_metadata.page_count == 1


def test_docling_processor_parses_layout_elements(monkeypatch: Any, tmp_path: Path):
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        pytest.skip("Pillow not available to generate test image")

    img_path = tmp_path / "docling_layout.png"
    Image.new("RGB", (200, 100), (255, 255, 255)).save(img_path)

    class DummyDoclingConverter:
        def convert(self, path: str) -> Any:
            heading_elem = type(
                "HeadingElem",
                (),
                {
                    "text": "Layout Test",
                    "confidence": 0.98,
                    "bbox": (10.0, 10.0, 100.0, 25.0),
                    "type": "heading",
                },
            )()
            table_elem = type(
                "TableElem",
                (),
                {
                    "text": "Name | Age",
                    "confidence": 0.95,
                    "bbox": (10.0, 30.0, 150.0, 60.0),
                    "type": "table",
                },
            )()
            picture_elem = type(
                "PictureElem",
                (),
                {
                    "text": "",
                    "confidence": None,
                    "bbox": (10.0, 70.0, 180.0, 85.0),
                    "type": "picture",
                },
            )()
            dummy_page = type(
                "Page", (), {"elements": [heading_elem, table_elem, picture_elem]}
            )()
            dummy_doc = type("Document", (), {"pages": [dummy_page]})()
            return type("ConvertResult", (), {"document": dummy_doc})()

    def dummy_init_docling(self, *args, **kwargs) -> DummyDoclingConverter:
        return DummyDoclingConverter()

    monkeypatch.setattr(
        "docs2synth.preprocess.docling_processor.DocumentConverter",
        DummyDoclingConverter,
        raising=True,
    )
    monkeypatch.setattr(DoclingProcessor, "_init_docling", dummy_init_docling)

    proc = DoclingProcessor(use_layout_analysis=True)
    res = proc.process(str(img_path))

    assert len(res.objects) == 3
    obj_list = list(res.objects.values())
    assert obj_list[0].label == LabelType.TEXT
    assert obj_list[1].label == LabelType.TEXT
    assert obj_list[2].label == LabelType.OTHER
    assert "Layout Test" in obj_list[0].text
    assert "Name | Age" in obj_list[1].text


def test_docling_processor_supports_pdf_multi_page(monkeypatch: Any, tmp_path: Path):
    pdf_path = tmp_path / "docling_multi.pdf"
    with open(pdf_path, "wb") as f:
        f.write(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \n0000000173 00000 n \n0000000279 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n395\n%%EOF"
        )

    class DummyDoclingConverter:
        def convert(self, path: str) -> Any:
            page1_elem1 = type(
                "Elem1",
                (),
                {
                    "text": "Page 1",
                    "confidence": 0.9,
                    "bbox": (10.0, 10.0, 80.0, 25.0),
                    "type": "text",
                },
            )()
            page1 = type("Page1", (), {"elements": [page1_elem1]})()

            page2_elem1 = type(
                "Elem3",
                (),
                {
                    "text": "Page 2",
                    "confidence": 0.93,
                    "bbox": (10.0, 10.0, 90.0, 40.0),
                    "type": "text",
                },
            )()
            page2 = type("Page2", (), {"elements": [page2_elem1]})()

            dummy_doc = type("Document", (), {"pages": [page1, page2]})()
            return type("ConvertResult", (), {"document": dummy_doc})()

    def dummy_init_docling(self, *args, **kwargs) -> DummyDoclingConverter:
        return DummyDoclingConverter()

    monkeypatch.setattr(
        "docs2synth.preprocess.docling_processor.DocumentConverter",
        DummyDoclingConverter,
        raising=True,
    )
    monkeypatch.setattr(DoclingProcessor, "_init_docling", dummy_init_docling)

    proc = DoclingProcessor(lang="en")
    res = proc.process(str(pdf_path))

    assert res.document_metadata.page_count == 2
    assert len(res.objects) == 2
    page1_objs = [obj for obj in res.objects.values() if obj.page == 0]
    assert len(page1_objs) == 1
    page2_objs = [obj for obj in res.objects.values() if obj.page == 1]
    assert len(page2_objs) == 1
