"""PDFPlumber-based processor for parsed PDFs.

This module defines a processor that uses pdfplumber to extract text, bounding boxes,
and layout information from parsed (text-based) PDFs. It converts the extracted data
into the schema defined in `docs2synth.preprocess.schema`.

Best for: PDFs with embedded text (not scanned images requiring OCR).
"""

from __future__ import annotations

import mimetypes
import os
import time
from dataclasses import dataclass
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
    import pdfplumber  # type: ignore

    _PDFPLUMBER_AVAILABLE = True
except Exception:  # pragma: no cover - runtime availability check
    pdfplumber = None  # type: ignore
    _PDFPLUMBER_AVAILABLE = False
    logger.error(
        "pdfplumber is not installed. Please install pdfplumber to use PDFPlumberProcessor."
    )


@dataclass
class PDFPlumberProcessor:
    """PDF text extraction processor using pdfplumber.

    This processor extracts text from parsed PDFs (those with embedded text layers).
    For scanned PDFs without text, consider using OCR-based processors like
    PaddleOCR or EasyOCR instead.

    Parameters
    ----------
    extract_tables : bool
        Whether to extract tables as separate objects (default: False).
    x_tolerance : int
        Horizontal tolerance for grouping characters into words (default: 3).
    y_tolerance : int
        Vertical tolerance for grouping characters into lines (default: 3).
    skip_non_pdf : bool
        If True, skip non-PDF files (images, etc.) and return empty result.
        If False, raise an error for non-PDF files (default: True).
    min_text_threshold : int
        Minimum number of characters required to consider PDF as text-based.
        If fewer characters are extracted, a warning is logged (default: 10).
    """

    extract_tables: bool = False
    x_tolerance: int = 3
    y_tolerance: int = 3
    skip_non_pdf: bool = True
    min_text_threshold: int = 10

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

    def _has_extractable_text(self, pdf_obj) -> bool:
        """Check if PDF has extractable text (not a scanned image).

        Parameters
        ----------
        pdf_obj : pdfplumber.PDF
            Opened PDF object.

        Returns
        -------
        bool
            True if PDF contains extractable text, False otherwise.
        """
        try:
            # Check first few pages for any text
            pages_to_check = min(3, len(pdf_obj.pages))
            total_chars = 0

            for i in range(pages_to_check):
                text = pdf_obj.pages[i].extract_text()
                if text:
                    total_chars += len(text.strip())

            return total_chars >= self.min_text_threshold
        except Exception as e:
            logger.warning(f"Error checking for extractable text: {e}")
            return False

    def process(  # noqa: C901
        self,
        pdf_path: str,
    ) -> DocumentProcessResult:
        """Extract text and layout information from a PDF file.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to process.

        Returns
        -------
        DocumentProcessResult
            Structured document processing result with text objects and metadata.

        Notes
        -----
        - Each word or text chunk is represented as a DocumentObject with bounding box.
        - All text objects are labeled as LabelType.TEXT.
        - If extract_tables is True, table cells are extracted as separate objects.
        """
        if not _PDFPLUMBER_AVAILABLE:
            raise RuntimeError(
                "pdfplumber is not installed. Please install pdfplumber to use PDFPlumberProcessor."
            )

        # Validate that the file is a PDF
        if not self._is_pdf_file(pdf_path):
            if self.skip_non_pdf:
                logger.warning(
                    f"Skipping non-PDF file: {pdf_path}. "
                    "PDFPlumber only processes PDF files with embedded text. "
                    "For images or scanned documents, use EasyOCR or PaddleOCR instead."
                )
                return self._create_empty_result(pdf_path)
            else:
                raise ValueError(
                    f"File is not a PDF: {pdf_path}. "
                    "PDFPlumber requires PDF files. Set skip_non_pdf=True to skip such files."
                )

        start = time.time()

        objects: Dict[int, DocumentObject] = {}
        bbox_list: List[Tuple[float, float, float, float]] = []
        context_parts: List[str] = []
        reading_order_ids: List[int] = []

        current_id = 0
        page_count = 0
        total_width = 0
        total_height = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Check if PDF has extractable text
                if not self._has_extractable_text(pdf):
                    logger.warning(
                        f"PDF appears to be scanned or has minimal text: {pdf_path}. "
                        f"Extracted fewer than {self.min_text_threshold} characters. "
                        "Consider using EasyOCR or PaddleOCR for scanned documents."
                    )
                page_count = len(pdf.pages)

                for page_idx, page in enumerate(pdf.pages):
                    # Track page dimensions
                    page_width = page.width
                    page_height = page.height
                    total_width = max(total_width, page_width)
                    total_height = max(total_height, page_height)

                    # Extract words with bounding boxes
                    words = page.extract_words(
                        x_tolerance=self.x_tolerance,
                        y_tolerance=self.y_tolerance,
                        keep_blank_chars=False,
                    )

                    logger.info(
                        f"Page {page_idx}: extracted {len(words)} words "
                        f"(dimensions: {page_width}x{page_height})"
                    )

                    for word in words:
                        text = word.get("text", "")
                        if not text or not text.strip():
                            continue

                        # pdfplumber uses (x0, top, x1, bottom) coordinates
                        x0 = float(word.get("x0", 0))
                        top = float(word.get("top", 0))
                        x1 = float(word.get("x1", 0))
                        bottom = float(word.get("bottom", 0))

                        # Convert to (x1, y1, x2, y2) format
                        bbox = (x0, top, x1, bottom)

                        obj = DocumentObject(
                            object_id=current_id,
                            text=text,
                            bbox=bbox,
                            label=LabelType.TEXT,
                            page=page_idx,
                            score=None,  # pdfplumber doesn't provide confidence scores
                        )

                        objects[current_id] = obj
                        bbox_list.append(bbox)
                        context_parts.append(text)
                        reading_order_ids.append(current_id)
                        current_id += 1

                    # Extract tables if requested
                    if self.extract_tables:
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables):
                            if not table:
                                continue

                            # Flatten table into text
                            table_text = "\n".join(
                                [
                                    " | ".join(
                                        [str(cell) if cell else "" for cell in row]
                                    )
                                    for row in table
                                ]
                            )

                            if not table_text.strip():
                                continue

                            # Try to get table bounding box (not always available)
                            # Use page dimensions as fallback
                            bbox = (0.0, 0.0, page_width, page_height)

                            obj = DocumentObject(
                                object_id=current_id,
                                text=table_text,
                                bbox=bbox,
                                label=LabelType.OTHER,  # Tables marked as OTHER
                                page=page_idx,
                                score=None,
                            )

                            objects[current_id] = obj
                            bbox_list.append(bbox)
                            context_parts.append(table_text)
                            reading_order_ids.append(current_id)
                            current_id += 1

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise

        end = time.time()

        logger.info(f"Extracted {current_id} text objects from {pdf_path}")

        # Create process metadata
        process_metadata = ProcessMetadata(
            processor_name="pdfplumber",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
            latency=(end - start) * 1000.0,
        )

        # Create document metadata
        filename: Optional[str] = None
        size_bytes: Optional[int] = None
        mime_type: Optional[str] = None

        try:
            filename = os.path.basename(pdf_path)
        except Exception:
            filename = None

        try:
            size_bytes = os.path.getsize(pdf_path)
        except Exception:
            size_bytes = None

        try:
            guessed_type, _ = mimetypes.guess_type(pdf_path)
            mime_type = guessed_type or "application/pdf"
        except Exception:
            mime_type = "application/pdf"

        document_metadata = DocumentMetadata(
            source=pdf_path,
            filename=filename,
            page_count=page_count,
            size_bytes=size_bytes,
            mime_type=mime_type,
            language=None,  # pdfplumber doesn't detect language
            width=int(total_width) if total_width > 0 else None,
            height=int(total_height) if total_height > 0 else None,
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

    def _create_empty_result(self, file_path: str) -> DocumentProcessResult:
        """Create an empty DocumentProcessResult for skipped files.

        Parameters
        ----------
        file_path : str
            Path to the skipped file.

        Returns
        -------
        DocumentProcessResult
            Empty result with basic metadata.
        """
        filename: Optional[str] = None
        size_bytes: Optional[int] = None
        mime_type: Optional[str] = None

        try:
            filename = os.path.basename(file_path)
        except Exception:
            filename = None

        try:
            size_bytes = os.path.getsize(file_path)
        except Exception:
            size_bytes = None

        try:
            guessed_type, _ = mimetypes.guess_type(file_path)
            mime_type = guessed_type
        except Exception:
            mime_type = None

        process_metadata = ProcessMetadata(
            processor_name="pdfplumber",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            latency=0.0,
        )

        document_metadata = DocumentMetadata(
            source=file_path,
            filename=filename,
            page_count=0,
            size_bytes=size_bytes,
            mime_type=mime_type,
            language=None,
            width=None,
            height=None,
        )

        return DocumentProcessResult(
            objects={},
            object_list=[],
            bbox_list=[],
            context="",
            reading_order_ids=[],
            process_metadata=process_metadata,
            document_metadata=document_metadata,
        )
