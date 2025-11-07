"""PaddleOCR-based processor producing DocumentProcessResult outputs.

This module defines a thin wrapper over PaddleOCR that reads a document image
from a file path and converts the OCR detections into the schema defined in
`docs2synth.preprocess.schema`.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
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


logging.getLogger("ppocr").setLevel(logging.INFO)

try:
    from paddleocr import PaddleOCR  # type: ignore

    _PADDLE_AVAILABLE = True
except Exception:  # pragma: no cover - runtime availability check
    PaddleOCR = None  # type: ignore
    _PADDLE_AVAILABLE = False
    logger.error(
        "PaddleOCR is not installed. Please install paddleocr to use PaddleOCRProcessor."
    )

try:
    import fitz  # PyMuPDF  # type: ignore

    _PYMUPDF_AVAILABLE = True
except Exception:  # pragma: no cover - runtime availability check
    fitz = None  # type: ignore
    _PYMUPDF_AVAILABLE = False
    logger.debug("PyMuPDF is not installed. PDF support will be unavailable.")


@dataclass
class PaddleOCRProcessor:
    """OCR processor using PaddleOCR.

    Supports both image files and PDF files. PDFs are automatically converted to images
    before OCR processing.

    Parameters
    ----------
    lang : str
        Language code passed to PaddleOCR (e.g., "en").
    use_textline_orientation : bool
        Whether to enable textline orientation detection in PaddleOCR.
    det : bool
        Enable text detection.
    rec : bool
        Enable text recognition.
    show_log : bool
        Control PaddleOCR internal logging.
    pdf_dpi : int
        DPI (resolution) for PDF to image conversion (default: 300).
        Higher values produce better quality but slower processing.
    save_pdf_images : bool
        If True, save PDF page images to a structured directory for later use
        (e.g., QA generation). Images are saved as {pdf_dir}/{pdf_stem}/1.png, 2.png, ...
        If False, use temporary files that are cleaned up after processing (default: True).
    """

    lang: str = "en"
    use_textline_orientation: bool = True
    det: bool = True
    rec: bool = True
    show_log: bool = False
    pdf_dpi: int = 300
    save_pdf_images: bool = (
        True  # Save PDF page images for later use (e.g., QA generation)
    )
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

    def _is_pdf_file(self, file_path: str) -> bool:
        """Check if the file is a PDF based on extension and/or magic bytes.

        Parameters
        ----------
        file_path : str
            Path to the file to check.

        Returns
        -------
        bool
            True if file appears to be a PDF, False otherwise.
        """
        # Check file extension
        if not file_path.lower().endswith(".pdf"):
            return False

        # Check PDF magic bytes (%PDF)
        try:
            with open(file_path, "rb") as f:
                header = f.read(5)
                return header.startswith(b"%PDF-")
        except Exception as e:
            logger.warning(f"Could not read file header for {file_path}: {e}")
            return False

    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to image files.

        If `save_pdf_images` is True, images are saved to a structured directory.
        Otherwise, temporary files are created.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        List[str]
            List of image file paths (one per page).

        Raises
        ------
        RuntimeError
            If PyMuPDF is not available or PDF conversion fails.
        """
        if not _PYMUPDF_AVAILABLE:
            raise RuntimeError(
                "PyMuPDF is not installed. Please install pymupdf to enable PDF support."
            )

        try:
            doc = fitz.open(pdf_path)  # type: ignore
            image_paths: List[str] = []

            # Determine where to save images
            if self.save_pdf_images:
                from docs2synth.utils.pdf_images import get_pdf_images_dir

                images_dir = get_pdf_images_dir(pdf_path)
                images_dir.mkdir(parents=True, exist_ok=True)
                save_dir = str(images_dir)
                logger.info(f"Saving PDF page images to {images_dir}")
            else:
                save_dir = tempfile.mkdtemp(prefix="paddleocr_pdf_")

            try:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Convert page to image with specified DPI
                    mat = fitz.Matrix(self.pdf_dpi / 72, self.pdf_dpi / 72)  # type: ignore
                    pix = page.get_pixmap(matrix=mat)
                    # Save as PNG file (numbered from 1)
                    image_filename = f"{page_num + 1}.png"
                    image_path = os.path.join(save_dir, image_filename)
                    pix.save(image_path)
                    image_paths.append(image_path)
                    logger.debug(f"Converted PDF page {page_num + 1} to {image_path}")

                logger.info(f"Converted {len(image_paths)} PDF pages to images")
                return image_paths
            finally:
                doc.close()
        except Exception as e:
            logger.error(f"Failed to convert PDF {pdf_path} to images: {e}")
            raise RuntimeError(f"PDF to image conversion failed: {e}") from e

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
            use_textline_orientation=self.use_textline_orientation,
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

        Supports both image files (PNG, JPG, etc.) and PDF files. PDFs are
        automatically converted to images before OCR processing.

        Notes
        -----
        - Bounding boxes from PaddleOCR are quadrilaterals; here we convert to
          axis-aligned `(x1, y1, x2, y2)` by taking min/max on the 4 points.
        - All recognized segments are labeled as `LabelType.TEXT`.
        - For PDF files, each page is processed separately and page numbers are
          preserved in the results.

        Parameters
        ----------
        image_path : str
            Path to the image or PDF file to OCR.
        lang : Optional[str]
            Optional language override for this call (defaults to the instance's
            `lang` value, which itself defaults to "en").
        device : Optional[str]
            Optional device override ("cpu" or "gpu"/"cuda").
        """
        ocr = self._init_ocr(lang_override=lang, device_override=device)
        start = time.time()

        # Check if input is a PDF and convert if necessary
        # Note: If PDFs were pre-converted (e.g., in batch processing), use those images
        is_pdf = self._is_pdf_file(image_path)
        temp_image_paths: List[str] = []
        input_path = image_path
        # Track if this is a single page image from a PDF (for page numbering)
        is_pdf_page_image = False
        pdf_source_path = image_path

        if is_pdf:
            # Check if PDF images already exist (from pre-conversion)
            from docs2synth.utils.pdf_images import get_pdf_images

            existing_images = get_pdf_images(image_path)
            if existing_images:
                logger.info(
                    f"Using pre-converted images for PDF {image_path} ({len(existing_images)} pages)"
                )
                temp_image_paths = [str(img) for img in existing_images]
                input_path = temp_image_paths[0]
            else:
                # Convert PDF to images on-the-fly
                logger.info(f"Detected PDF file: {image_path}, converting to images...")
                temp_image_paths = self._pdf_to_images(image_path)
                if not temp_image_paths:
                    raise RuntimeError(f"Failed to convert PDF {image_path} to images")
                input_path = temp_image_paths[0]
                logger.info(f"Converted PDF to {len(temp_image_paths)} image(s)")
        else:
            # Check if this is a PDF page image (e.g., document/1.png)
            # If so, find all pages from the same PDF
            from docs2synth.utils.pdf_images import get_pdf_images

            img_path = Path(image_path)
            # Check if this is a PDF page image (e.g., document/1.png)
            # Look for PDF file with matching stem in parent's parent directory
            parent_dir = img_path.parent
            if parent_dir.is_dir():
                grandparent = parent_dir.parent
                pdf_candidate = grandparent / f"{parent_dir.name}.pdf"
                if pdf_candidate.exists():
                    pdf_images = get_pdf_images(pdf_candidate)
                    if pdf_images and img_path in pdf_images:
                        # This is a PDF page image
                        is_pdf_page_image = True
                        pdf_source_path = str(pdf_candidate)
                        temp_image_paths = [str(p) for p in pdf_images]
                        # Find the page index
                        page_num = pdf_images.index(img_path)
                        logger.debug(
                            f"Detected PDF page image: {img_path.name} (page {page_num + 1} of {len(pdf_images)})"
                        )

        # Call PaddleOCR (suppress deprecation warning about ocr() method)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Please use `predict` instead")
            # Process each page separately and combine results
            if (is_pdf or is_pdf_page_image) and len(temp_image_paths) > 1:
                # Process all pages one by one
                result = []
                logger.info(f"Processing {len(temp_image_paths)} PDF pages...")
                for page_idx, page_image_path in enumerate(temp_image_paths):
                    logger.info(
                        f"Processing PDF page {page_idx + 1}/{len(temp_image_paths)}"
                    )
                    page_result = ocr.ocr(page_image_path)
                    # ocr() returns a list, so we need to extract the first element
                    if page_result and len(page_result) > 0:
                        result.append(page_result[0])
                    else:
                        result.append(None)
            else:
                # Single image or single-page PDF
                result = ocr.ocr(input_path)
                # Ensure result is a list for consistent processing
                if not isinstance(result, list):
                    result = [result]

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
        # For PDFs, use the first converted image
        dimension_source = temp_image_paths[0] if temp_image_paths else image_path
        try:
            from PIL import Image  # type: ignore

            with Image.open(dimension_source) as img:
                width, height = img.size  # type: ignore[assignment]
        except Exception:
            width = None
            height = None

        # Determine page count and source path
        page_count = 1
        source_path = image_path
        if is_pdf or is_pdf_page_image:
            page_count = len(temp_image_paths) if temp_image_paths else 1
            if is_pdf_page_image:
                source_path = pdf_source_path  # Use original PDF path
        elif result and isinstance(result, list):
            # PaddleOCR 3.x returns list of pages
            page_count = len(result)

        document_metadata = DocumentMetadata(
            source=source_path,
            filename=(
                os.path.basename(source_path) if source_path != image_path else filename
            ),
            page_count=page_count,
            size_bytes=size_bytes,
            mime_type=mime_type
            or ("application/pdf" if (is_pdf or is_pdf_page_image) else None),
            language=detected_lang,
            width=width,
            height=height,
        )

        result_obj = DocumentProcessResult(
            objects=objects,
            object_list=[],
            bbox_list=bbox_list,
            context=" ".join(context_parts),
            reading_order_ids=reading_order_ids,
            process_metadata=process_metadata,
            document_metadata=document_metadata,
        )

        # Clean up temporary image files if PDF was converted and not saving images
        # Do this after all processing is complete
        if temp_image_paths and not self.save_pdf_images:
            try:
                temp_dir = os.path.dirname(temp_image_paths[0])
                for temp_path in temp_image_paths:
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass
                try:
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")

        return result_obj
