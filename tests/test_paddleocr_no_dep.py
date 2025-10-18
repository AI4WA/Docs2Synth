import pytest

import docs2synth.preprocess.paddleocr as paddleocr_mod


def test_paddleocr_raises_when_dependency_missing(monkeypatch):
    # Simulate dependency missing flag
    monkeypatch.setattr(paddleocr_mod, "_PADDLE_AVAILABLE", False, raising=True)

    proc = paddleocr_mod.PaddleOCRProcessor()
    with pytest.raises(RuntimeError):
        proc.process("/a/nonexistent.png")
