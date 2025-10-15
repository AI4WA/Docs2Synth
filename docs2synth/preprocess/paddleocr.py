"""PaddleOCR-based processor producing DocumentProcessResult outputs.

This module defines a thin wrapper over PaddleOCR that reads a document image
from a file path and converts the OCR detections into the schema defined in
`docs2synth.preprocess.schema`.
"""

from __future__ import annotations

import mimetypes
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from docs2synth.utils.logging import get_logger

from .schema import (
    DocumentMetadata,
    DocumentObject,
    DocumentProcessResult,
    LabelType,
    ProcessMetadata,
)

logger = get_logger(__name__)

try:
    from paddleocr import PaddleOCR  # type: ignore

    _PADDLE_AVAILABLE = True
except Exception:  # pragma: no cover - runtime availability check
    PaddleOCR = None  # type: ignore
    _PADDLE_AVAILABLE = False
    logger.error(
        "PaddleOCR is not installed. Please install paddleocr to use PaddleOCRProcessor."
    )


@dataclass
class PaddleOCRProcessor:
    """OCR processor using PaddleOCR.

    Parameters
    ----------
    lang : str
        Language code passed to PaddleOCR (e.g., "en").
    use_angle_cls : bool
        Whether to enable angle classification in PaddleOCR.
    det : bool
        Enable text detection.
    rec : bool
        Enable text recognition.
    show_log : bool
        Control PaddleOCR internal logging.
    """

    lang: str = "en"
    use_angle_cls: bool = True
    det: bool = True
    rec: bool = True
    show_log: bool = False
    # Optional device preference: "cpu" or "gpu"/"cuda". If None, auto-detect GPU.
    device: Optional[str] = None
    _ocr_cache: Dict[str, "PaddleOCR"] = field(
        default_factory=dict, init=False, repr=False
    )

    def _gpu_available(self) -> bool:
        try:
            import paddle  # type: ignore

            if (
                getattr(paddle, "is_compiled_with_cuda", None)
                and paddle.is_compiled_with_cuda()
            ):
                try:
                    from paddle.device import cuda as paddle_cuda  # type: ignore

                    return paddle_cuda.device_count() > 0
                except Exception:
                    return True
        except Exception:
            return False
        return False

    def _resolve_device(self, device_override: Optional[str]) -> str:
        if device_override is not None:
            dev = device_override.lower()
            if dev in ("gpu", "cuda"):
                return "gpu" if self._gpu_available() else "cpu"
            return "cpu"
        if self.device is not None:
            logger.info(f"Device override: {self.device}")
            dev = self.device.lower()
            if dev in ("gpu", "cuda"):
                return "gpu" if self._gpu_available() else "cpu"
            return "cpu"
        return "gpu" if self._gpu_available() else "cpu"

    def _init_ocr(
        self, lang_override: Optional[str] = None, device_override: Optional[str] = None
    ) -> PaddleOCR:
        if not _PADDLE_AVAILABLE:
            raise RuntimeError(
                "paddleocr is not installed. Please install paddleocr to use PaddleOCRProcessor."
            )
        # Determine device string ("gpu" or "cpu")
        logger.debug(f"GPU available: {self._gpu_available()}")
        device_str = self._resolve_device(device_override)

        # Cache PaddleOCR instances per (language, device) to avoid re-initialization cost
        lang_key = lang_override or self.lang
        cache_key = f"lang={lang_key}|device={device_str}"
        cached = self._ocr_cache.get(cache_key)
        if cached is not None:
            return cached
        # Attempt to set Paddle device prior to constructing PaddleOCR
        try:
            import paddle  # type: ignore

            paddle.set_device(device_str)
        except Exception:
            pass
        # Note: show_log is not supported in PaddleOCR 3.x
        ocr = PaddleOCR(
            lang=lang_key,
            use_angle_cls=self.use_angle_cls,
        )
        self._ocr_cache[cache_key] = ocr
        return ocr

    def process(  # noqa: C901
        self,
        image_path: str,
        *,
        lang: Optional[str] = None,
        device: Optional[str] = None,
    ) -> DocumentProcessResult:
        """Run OCR on a file path and return schema-compliant results.

        Notes
        -----
        - Bounding boxes from PaddleOCR are quadrilaterals; here we convert to
          axis-aligned `(x1, y1, x2, y2)` by taking min/max on the 4 points.
        - All recognized segments are labeled as `LabelType.TEXT`.

        Parameters
        ----------
        image_path : str
            Path to the image to OCR.
        lang : Optional[str]
            Optional language override for this call (defaults to the instance's
            `lang` value, which itself defaults to "en").
        """
        ocr = self._init_ocr(lang_override=lang, device_override=device)
        start = time.time()

        # Call PaddleOCR
        result = ocr.ocr(image_path)
        end = time.time()

        # Debug logging
        logger.debug(f"PaddleOCR raw result type: {type(result)}")
        logger.debug(f"PaddleOCR raw result: {result}")

        objects: Dict[int, DocumentObject] = {}
        bbox_list: List[Tuple[float, float, float, float]] = []
        context_parts: List[str] = []
        reading_order_ids: List[int] = []

        current_id = 0

        # Handle None or empty results
        if result is None or not result:
            logger.warning(f"PaddleOCR returned empty result for {image_path}")
        else:
            # PaddleOCR 3.x returns a list of dictionaries (one per page)
            # Each dict has keys: 'rec_texts', 'rec_scores', 'rec_polys', etc.
            for page_idx, page_result in enumerate(result):
                if page_result is None:
                    logger.warning(f"Page {page_idx} is None")
                    continue

                # Check if it's the new dict format (PaddleOCR 3.x)
                if isinstance(page_result, dict):
                    rec_texts = page_result.get("rec_texts", [])
                    rec_scores = page_result.get("rec_scores", [])
                    rec_polys = page_result.get("rec_polys", [])

                    logger.info(
                        f"Processing page {page_idx} with {len(rec_texts)} detections (dict format)"
                    )

                    # Iterate through all detections
                    for idx in range(len(rec_texts)):
                        try:
                            text = rec_texts[idx]
                            score = (
                                float(rec_scores[idx])
                                if idx < len(rec_scores)
                                else None
                            )
                            points = rec_polys[idx] if idx < len(rec_polys) else None

                            if points is None or text is None:
                                continue

                            # Convert quadrilateral to axis-aligned bounding box
                            xs = [float(p[0]) for p in points]
                            ys = [float(p[1]) for p in points]
                            bbox = (min(xs), min(ys), max(xs), max(ys))

                            obj = DocumentObject(
                                object_id=current_id,
                                text=str(text) if text else "",
                                bbox=bbox,
                                label=LabelType.TEXT,
                                page=page_idx,
                                score=score,
                            )

                            objects[current_id] = obj
                            bbox_list.append(bbox)
                            context_parts.append(obj.text)
                            reading_order_ids.append(current_id)
                            current_id += 1

                        except Exception as e:
                            logger.error(
                                f"Error processing detection {idx} on page {page_idx}: {e}"
                            )
                            continue

                # Fall back to old list format (PaddleOCR 2.x)
                elif isinstance(page_result, list):
                    logger.info(
                        f"Processing page {page_idx} with {len(page_result)} detections (list format)"
                    )

                    for det_idx, det in enumerate(page_result):
                        if det is None:
                            continue

                        try:
                            # det format: [bbox_points, (text, confidence)]
                            points = det[0]
                            text_tuple = det[1]
                            text = text_tuple[0]
                            score = (
                                float(text_tuple[1])
                                if text_tuple[1] is not None
                                else None
                            )

                            # Convert quadrilateral to axis-aligned bounding box
                            xs = [float(p[0]) for p in points]
                            ys = [float(p[1]) for p in points]
                            bbox = (min(xs), min(ys), max(xs), max(ys))

                            obj = DocumentObject(
                                object_id=current_id,
                                text=str(text) if text is not None else "",
                                bbox=bbox,
                                label=LabelType.TEXT,
                                page=page_idx,
                                score=score,
                            )

                            objects[current_id] = obj
                            bbox_list.append(bbox)
                            context_parts.append(obj.text)
                            reading_order_ids.append(current_id)
                            current_id += 1

                        except Exception as e:
                            logger.error(
                                f"Error processing detection {det_idx} on page {page_idx}: {e}"
                            )
                            continue
                else:
                    logger.warning(
                        f"Unknown result format for page {page_idx}: {type(page_result)}"
                    )

        logger.info(f"Extracted {current_id} text objects from {image_path}")

        process_metadata = ProcessMetadata(
            processor_name="paddleocr",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
            latency=(end - start) * 1000.0,
        )

        # Populate document metadata from the input image
        detected_lang: Optional[str] = (
            (lang or self.lang) if (lang or self.lang) else None
        )
        filename: Optional[str] = None
        size_bytes: Optional[int] = None
        mime_type: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None

        try:
            filename = os.path.basename(image_path)
        except Exception:  # pragma: no cover - defensive
            filename = None

        try:
            size_bytes = os.path.getsize(image_path)
        except Exception:
            size_bytes = None

        try:
            guessed_type, _ = mimetypes.guess_type(image_path)
            mime_type = guessed_type
        except Exception:
            mime_type = None

        # Try to get image dimensions via Pillow, fall back silently if unavailable
        try:
            from PIL import Image  # type: ignore

            with Image.open(image_path) as img:
                width, height = img.size  # type: ignore[assignment]
        except Exception:
            width = None
            height = None

        document_metadata = DocumentMetadata(
            source=image_path,
            filename=filename,
            page_count=1,
            size_bytes=size_bytes,
            mime_type=mime_type,
            language=detected_lang,
            width=width,
            height=height,
        )

        return DocumentProcessResult(
            objects=objects,
            object_list=[],
            bbox_list=bbox_list,
            context=" ".join(context_parts).strip() + (" " if context_parts else ""),
            reading_order_ids=reading_order_ids,
            process_metadata=process_metadata,
            document_metadata=document_metadata,
        )
