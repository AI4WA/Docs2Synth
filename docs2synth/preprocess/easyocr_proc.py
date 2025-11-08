"""EasyOCR-based processor producing DocumentProcessResult outputs.

This module defines a wrapper over EasyOCR that reads document images or scanned PDFs
from a file path and converts the OCR detections into the schema defined in
`docs2synth.preprocess.schema`.

EasyOCR supports 80+ languages with pre-trained models and GPU acceleration.
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
    import easyocr  # type: ignore

    _EASYOCR_AVAILABLE = True
except Exception:  # pragma: no cover - runtime availability check
    easyocr = None  # type: ignore
    _EASYOCR_AVAILABLE = False
    logger.error(
        "EasyOCR is not installed. Please install easyocr to use EasyOCRProcessor."
    )


@dataclass
class EasyOCRProcessor:
    """OCR processor using EasyOCR.

    EasyOCR is a PyTorch-based OCR library supporting 80+ languages with
    pre-trained models. It provides both text detection and recognition.

    Parameters
    ----------
    lang_list : List[str]
        List of language codes to use (e.g., ["en"], ["en", "fr"]).
        See EasyOCR documentation for supported language codes.
    gpu : bool
        Whether to use GPU acceleration (default: True if available).
    model_storage_directory : Optional[str]
        Directory to store downloaded models (default: ~/.EasyOCR/).
    download_enabled : bool
        Whether to download models if not present (default: True).
    detector : bool
        Enable text detection (default: True).
    recognizer : bool
        Enable text recognition (default: True).
    verbose : bool
        Enable verbose logging from EasyOCR (default: False).
    """

    lang_list: List[str] = field(default_factory=lambda: ["en"])
    gpu: bool = True
    model_storage_directory: Optional[str] = None
    download_enabled: bool = True
    detector: bool = True
    recognizer: bool = True
    verbose: bool = False
    _reader_cache: Dict[str, "easyocr.Reader"] = field(
        default_factory=dict, init=False, repr=False
    )

    def _get_reader(self) -> "easyocr.Reader":
        """Get or create an EasyOCR Reader instance.

        Readers are cached by language list to avoid re-initialization.
        """
        if not _EASYOCR_AVAILABLE:
            raise RuntimeError(
                "easyocr is not installed. Please install easyocr to use EasyOCRProcessor."
            )

        # Create cache key from sorted language list
        cache_key = "|".join(sorted(self.lang_list))

        if cache_key in self._reader_cache:
            logger.debug(f"Using cached EasyOCR reader for languages: {self.lang_list}")
            return self._reader_cache[cache_key]

        logger.info(f"Initializing EasyOCR reader for languages: {self.lang_list}")

        # Initialize EasyOCR Reader
        reader = easyocr.Reader(
            lang_list=self.lang_list,
            gpu=self.gpu,
            model_storage_directory=self.model_storage_directory,
            download_enabled=self.download_enabled,
            detector=self.detector,
            recognizer=self.recognizer,
            verbose=self.verbose,
        )

        self._reader_cache[cache_key] = reader
        return reader

    def process(
        self,
        image_path: str,
        *,
        lang: Optional[str] = None,
        device: Optional[str] = None,
        paragraph: bool = False,
        min_size: int = 10,
        contrast_ths: float = 0.1,
        adjust_contrast: float = 0.5,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
    ) -> DocumentProcessResult:
        """Run OCR on an image file and return schema-compliant results.

        Parameters
        ----------
        image_path : str
            Path to the image file to process.
        lang : Optional[str]
            Language override (ignored - EasyOCR language is set at initialization).
            This parameter exists for API compatibility with other processors.
        device : Optional[str]
            Device override (ignored - EasyOCR device is set at initialization).
            This parameter exists for API compatibility with other processors.
        paragraph : bool
            Whether to combine results into paragraphs (default: False).
        min_size : int
            Minimum size for text detection (default: 10).
        contrast_ths : float
            Text vs background contrast threshold (default: 0.1).
        adjust_contrast : float
            Contrast adjustment factor (default: 0.5).
        text_threshold : float
            Text confidence threshold (default: 0.7).
        low_text : float
            Low text threshold for detection (default: 0.4).
        link_threshold : float
            Link threshold for combining text boxes (default: 0.4).
        canvas_size : int
            Maximum image size for processing (default: 2560).
        mag_ratio : float
            Image magnification ratio (default: 1.0).

        Returns
        -------
        DocumentProcessResult
            Structured document processing result.

        Notes
        -----
        - Bounding boxes are in (x1, y1, x2, y2) format.
        - All recognized text segments are labeled as LabelType.TEXT.
        - Confidence scores are provided by EasyOCR.
        - The lang and device parameters are ignored; these must be set when
          constructing the EasyOCRProcessor instance.
        """
        reader = self._get_reader()
        start = time.time()

        # Run EasyOCR
        try:
            result = reader.readtext(
                image_path,
                paragraph=paragraph,
                min_size=min_size,
                contrast_ths=contrast_ths,
                adjust_contrast=adjust_contrast,
                text_threshold=text_threshold,
                low_text=low_text,
                link_threshold=link_threshold,
                canvas_size=canvas_size,
                mag_ratio=mag_ratio,
            )
        except Exception as e:
            logger.error(f"EasyOCR failed on {image_path}: {e}")
            raise

        end = time.time()

        # Process results
        objects: Dict[int, DocumentObject] = {}
        bbox_list: List[Tuple[float, float, float, float]] = []
        context_parts: List[str] = []
        reading_order_ids: List[int] = []

        current_id = 0

        # EasyOCR returns: List[Tuple[bbox_points, text, confidence]]
        # bbox_points is a list of 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        for detection in result:
            if len(detection) < 3:
                continue

            bbox_points, text, confidence = detection

            if not text or not text.strip():
                continue

            # Convert points to polygon format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            polygon = [(float(point[0]), float(point[1])) for point in bbox_points]

            # Also compute axis-aligned bounding box for backward compatibility
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            obj = DocumentObject(
                object_id=current_id,
                text=str(text),
                bbox=bbox,
                polygon=polygon,
                label=LabelType.TEXT,
                page=0,  # EasyOCR processes single images, no page concept
                score=float(confidence) if confidence is not None else None,
            )

            objects[current_id] = obj
            bbox_list.append(bbox)
            context_parts.append(obj.text)
            reading_order_ids.append(current_id)
            current_id += 1

        logger.info(f"EasyOCR extracted {current_id} text objects from {image_path}")

        # Create process metadata
        process_metadata = ProcessMetadata(
            processor_name="easyocr",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
            latency=(end - start) * 1000.0,
        )

        # Create document metadata
        filename: Optional[str] = None
        size_bytes: Optional[int] = None
        mime_type: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None

        try:
            filename = os.path.basename(image_path)
        except Exception:
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

        # Try to get image dimensions
        try:
            from PIL import Image  # type: ignore

            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            width = None
            height = None

        # Language detection from lang_list
        detected_lang = self.lang_list[0] if self.lang_list else None

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
            context=" ".join(context_parts),
            reading_order_ids=reading_order_ids,
            process_metadata=process_metadata,
            document_metadata=document_metadata,
        )
