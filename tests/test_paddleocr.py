import pytest

from docs2synth.preprocess.paddleocr import _PADDLE_AVAILABLE, PaddleOCRProcessor


def test_paddleocr_device_and_lang_overrides(monkeypatch, tmp_path):
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("Pillow not available to generate a test image")

    img_path = tmp_path / "text.png"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img_path)

    class DummyOCR:
        def __init__(self, *args, **kwargs):
            pass

        def ocr(self, path):
            return [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]

    # Provide a dummy 'paddle' module surface so set_device works
    class DummyPaddle:
        def set_device(self, *_args, **_kwargs):
            return None

    # Track calls to _gpu_available to ensure override path is used
    calls = {"gpu_checked": 0}

    def fake_gpu_available(self):
        calls["gpu_checked"] += 1
        return True

    # Monkeypatch environment so _init_ocr can run its normal logic
    monkeypatch.setattr(
        "docs2synth.preprocess.paddleocr.paddle", DummyPaddle(), raising=False
    )
    monkeypatch.setattr(
        "docs2synth.preprocess.paddleocr.PaddleOCR", DummyOCR, raising=True
    )
    monkeypatch.setattr(PaddleOCRProcessor, "_gpu_available", fake_gpu_available)

    # device override forces GPU path when available
    proc = PaddleOCRProcessor(lang="en")
    _ = proc.process(str(img_path), lang="fr", device="gpu")
    assert calls["gpu_checked"] >= 1

    # instance-level device override respected (should not raise)
    proc2 = PaddleOCRProcessor(lang="en", device="cpu")
    _ = proc2.process(str(img_path))


def test_paddleocr_caches_ocr_instance(monkeypatch, tmp_path):
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("Pillow not available to generate a test image")

    img_path = tmp_path / "text.png"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img_path)

    class DummyOCR:
        def __init__(self, *args, **kwargs):
            pass

        def ocr(self, path):
            return [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]

    created = {"count": 0}

    def counting_constructor(*args, **kwargs):
        created["count"] += 1
        return DummyOCR(*args, **kwargs)

    # Ensure stable device resolution so the cache key stays the same
    monkeypatch.setattr(
        PaddleOCRProcessor, "_resolve_device", lambda self, override: "cpu"
    )
    monkeypatch.setattr(
        "docs2synth.preprocess.paddleocr.PaddleOCR", counting_constructor, raising=True
    )

    proc = PaddleOCRProcessor(lang="en")
    _ = proc.process(str(img_path), lang=None, device="cpu")
    _ = proc.process(str(img_path), lang=None, device="cpu")
    assert created["count"] == 1, "OCR instance should be cached per (lang, device)"


pytestmark = pytest.mark.skipif(not _PADDLE_AVAILABLE, reason="paddleocr not installed")


def test_paddleocr_processor_runs_minimal(tmp_path):
    # Create a tiny blank image using PIL if available; otherwise skip
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("Pillow not available to generate a test image")

    img_path = tmp_path / "blank.png"
    Image.new("RGB", (100, 40), color=(255, 255, 255)).save(img_path)

    proc = PaddleOCRProcessor(lang="en", det=True, rec=True, show_log=False)
    res = proc.process(str(img_path))

    # Schema basics
    assert res.process_metadata.processor_name == "paddleocr"
    assert res.document_metadata.source == str(img_path)
    assert isinstance(res.bbox_list, list)
    assert isinstance(res.objects, dict)


def test_paddleocr_processor_handles_empty_result(monkeypatch, tmp_path):
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("Pillow not available to generate a test image")

    img_path = tmp_path / "blank.png"
    Image.new("RGB", (50, 20), color=(255, 255, 255)).save(img_path)

    # Mock PaddleOCR and ocr() to return empty list
    class DummyOCR:
        def ocr(self, path):
            return []

    def dummy_init(self, *args, **kwargs):
        return DummyOCR()

    monkeypatch.setattr("docs2synth.preprocess.paddleocr.PaddleOCR", DummyOCR)
    proc = PaddleOCRProcessor()
    # Also monkeypatch the internal initializer to bypass paddle import logic
    monkeypatch.setattr(PaddleOCRProcessor, "_init_ocr", dummy_init)

    res = proc.process(str(img_path))
    assert res.objects == {}
    assert res.bbox_list == []
    assert res.context == ""


def test_paddleocr_processor_parses_dict_format(monkeypatch, tmp_path):
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("Pillow not available to generate a test image")

    img_path = tmp_path / "text.png"
    Image.new("RGB", (100, 40), color=(255, 255, 255)).save(img_path)

    class DummyOCR:
        def ocr(self, path):
            return [
                {
                    "rec_texts": ["Hello", "World"],
                    "rec_scores": [0.9, 0.8],
                    "rec_polys": [
                        [[1, 1], [20, 1], [20, 10], [1, 10]],
                        [[30, 1], [60, 1], [60, 10], [30, 10]],
                    ],
                }
            ]

    def dummy_init(self, *args, **kwargs):
        return DummyOCR()

    monkeypatch.setattr("docs2synth.preprocess.paddleocr.PaddleOCR", DummyOCR)
    proc = PaddleOCRProcessor()
    monkeypatch.setattr(PaddleOCRProcessor, "_init_ocr", dummy_init)

    res = proc.process(str(img_path))
    assert len(res.objects) == 2
    assert res.document_metadata.filename == img_path.name
    assert "Hello" in res.context and "World" in res.context


def test_paddleocr_processor_parses_list_format(monkeypatch, tmp_path):
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("Pillow not available to generate a test image")

    img_path = tmp_path / "text.png"
    Image.new("RGB", (100, 40), color=(255, 255, 255)).save(img_path)

    class DummyOCR:
        def ocr(self, path):
            # Old PaddleOCR 2.x style: list of detections per page
            return [
                [
                    (
                        [[1, 1], [20, 1], [20, 10], [1, 10]],
                        ("Foo", 0.95),
                    ),
                    (
                        [[30, 1], [60, 1], [60, 10], [30, 10]],
                        ("Bar", 0.85),
                    ),
                ]
            ]

    def dummy_init(self, *args, **kwargs):
        return DummyOCR()

    monkeypatch.setattr("docs2synth.preprocess.paddleocr.PaddleOCR", DummyOCR)
    proc = PaddleOCRProcessor()
    monkeypatch.setattr(PaddleOCRProcessor, "_init_ocr", dummy_init)

    res = proc.process(str(img_path))
    assert len(res.objects) == 2
    assert "Foo" in res.context and "Bar" in res.context
