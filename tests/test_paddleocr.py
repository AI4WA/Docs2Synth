import pytest

from docs2synth.preprocess.paddleocr import _PADDLE_AVAILABLE, PaddleOCRProcessor

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
