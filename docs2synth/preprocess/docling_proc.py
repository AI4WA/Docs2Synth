"""Docling-based processor for document understanding.

This module defines a processor that uses Docling (IBM's document understanding library)
to extract text, layout, and structure information from documents. It converts the extracted
data into the schema defined in `docs2synth.preprocess.schema`.

Docling supports: PDF, DOCX, PPTX, images, HTML, and more.
"""

from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass
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

# Try to import docling
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    _DOCLING_AVAILABLE = True
except ImportError:
    DocumentConverter = None
    PdfFormatOption = None
    InputFormat = None
    PdfPipelineOptions = None
    _DOCLING_AVAILABLE = False
    logger.warning("Docling is not installed. Please install with: pip install docling")


@dataclass
class DoclingProcessor:
    """Document processor using Docling.

    Docling is IBM's advanced document understanding library that supports
    multiple formats and provides rich document structure extraction.

    Parameters
    ----------
    ocr_enabled : bool
        Enable OCR for scanned documents (default: True).
    table_structure_enabled : bool
        Enable table structure recognition (default: True).
    output_format : str
        Output format: 'markdown', 'json', or 'text' (default: 'json').
    device : Optional[str]
        Device for processing: 'cpu', 'cuda', or None for auto-detect.
    do_cell_matching : bool
        Enable cell matching for tables (default: True).
    """

    ocr_enabled: bool = True
    table_structure_enabled: bool = True
    output_format: str = "json"
    device: Optional[str] = None
    do_cell_matching: bool = True

    def __post_init__(self):
        """Initialize the Docling converter."""
        if not _DOCLING_AVAILABLE:
            raise RuntimeError(
                "Docling is not installed. Please install with: pip install docling"
            )

        # Configure pipeline options for PDF processing
        pipeline_options = PdfPipelineOptions(
            do_ocr=self.ocr_enabled,
            do_table_structure=self.table_structure_enabled,
        )

        # Create format options
        pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)

        # Initialize converter with format options
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.HTML,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: pdf_format_option,
            },
        )

        logger.info("DoclingProcessor initialized")
        logger.info(f"  OCR enabled: {self.ocr_enabled}")
        logger.info(f"  Table structure: {self.table_structure_enabled}")

    def _process_text_item(
        self,
        text_item,
        current_id: int,
        page_heights: Dict[int, float],
    ) -> Optional[DocumentObject]:
        """Process a single text item from Docling result.

        Parameters
        ----------
        text_item
            Text item from Docling document.
        current_id : int
            Current object ID to assign.
        page_heights : Dict[int, float]
            Dictionary mapping page numbers to page heights.

        Returns
        -------
        Optional[DocumentObject]
            Created DocumentObject or None if processing failed.
        """
        try:
            # Get text content
            text = text_item.text if hasattr(text_item, "text") else str(text_item)

            if not text or not text.strip():
                return None

            # Get bounding box and page from provenance
            bbox = None
            polygon = None
            page_idx = 0

            if hasattr(text_item, "prov") and text_item.prov:
                # Use the first provenance item
                prov = text_item.prov[0]
                page_idx = prov.page_no - 1  # Convert to 0-indexed

                if hasattr(prov, "bbox"):
                    # Docling bbox format: BoundingBox with l, t, r, b
                    # Note: Docling uses bottom-left origin, we need to convert to top-left
                    bb = prov.bbox
                    page_no = prov.page_no

                    # Convert from bottom-left to top-left origin
                    if page_no in page_heights:
                        page_height = page_heights[page_no]
                        x0 = float(bb.l)
                        y0_bottom = float(bb.b)  # bottom in bottom-left coords
                        x1 = float(bb.r)
                        y1_bottom = float(bb.t)  # top in bottom-left coords

                        # Convert to top-left origin: y_top = page_height - y_bottom
                        y0_top = page_height - y1_bottom  # top edge in top-left coords
                        y1_top = (
                            page_height - y0_bottom
                        )  # bottom edge in top-left coords

                        bbox = (x0, y0_top, x1, y1_top)
                    else:
                        # Fallback: use raw coordinates
                        bbox = (float(bb.l), float(bb.t), float(bb.r), float(bb.b))

            # If no bbox, use default placeholder
            if bbox is None:
                bbox = (0.0, 0.0, 0.0, 0.0)

            # Determine label type based on item type
            label = LabelType.TEXT
            if hasattr(text_item, "label"):
                label_str = str(text_item.label).lower()
                if "paragraph" in label_str:
                    label = LabelType.PARAGRAPH
                elif "sentence" in label_str:
                    label = LabelType.SENTENCE
                elif "title" in label_str or "heading" in label_str:
                    label = LabelType.TEXT  # Keep as TEXT for now

            # Create DocumentObject
            obj = DocumentObject(
                object_id=current_id,
                text=str(text).strip(),
                bbox=bbox,
                polygon=polygon,
                label=label,
                page=page_idx,
                score=None,  # Docling doesn't provide confidence scores
            )

            return obj

        except Exception as e:
            logger.error(f"Error processing text item: {e}")
            logger.debug(f"  Item: {text_item}")
            return None

    def _process_single_image(
        self, image_path: Path, page_idx: int
    ) -> List[DocumentObject]:
        """Process a single image and return objects.

        Parameters
        ----------
        image_path : Path
            Path to the image file.
        page_idx : int
            Page index (0-based) for this image.

        Returns
        -------
        List[DocumentObject]
            List of DocumentObjects from this page.
        """
        result = self.converter.convert(str(image_path))
        doc = result.document

        # Get page height for coordinate conversion
        # When processing a single image, docling treats it as page 1
        page_height = None
        for page_no, page_item in doc.pages.items():
            if hasattr(page_item, "size") and hasattr(page_item.size, "height"):
                page_height = float(page_item.size.height)
                break

        # Create page_heights dict for coordinate conversion
        # Docling uses page 1 for single images
        page_heights = {1: page_height} if page_height is not None else {}

        # Process all text items from this page
        page_objects: List[DocumentObject] = []
        temp_id = 0  # Temporary ID, will be reassigned later
        for text_item in doc.texts:
            obj = self._process_text_item(text_item, temp_id, page_heights)
            if obj is not None:
                # Update page index to match our document structure
                obj.page = page_idx
                page_objects.append(obj)
            temp_id += 1

        return page_objects

    def process(
        self,
        file_path: str,
        *,
        lang: Optional[str] = None,
        device: Optional[str] = None,
    ) -> DocumentProcessResult:
        """Process a document using Docling.

        For PDF files, if pre-converted images exist (e.g., from preprocessing
        with image resizing), those images will be used instead of the PDF to
        ensure bbox coordinates match the actual images used for annotation.

        Parameters
        ----------
        file_path : str
            Path to the document file (PDF, image, DOCX, etc.).
        lang : Optional[str]
            Language hint (ignored - for API compatibility).
        device : Optional[str]
            Device override (ignored - for API compatibility).

        Returns
        -------
        DocumentProcessResult
            Structured document processing result.
        """
        if not _DOCLING_AVAILABLE:
            raise RuntimeError("Docling is not installed")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if this is a PDF and if pre-converted images exist
        is_pdf = file_path.suffix.lower() == ".pdf"
        pdf_images: Optional[List[Path]] = None

        if is_pdf:
            from docs2synth.utils.pdf_images import get_pdf_images

            pdf_images = get_pdf_images(file_path)
            if pdf_images:
                logger.info(
                    f"Using pre-converted images for PDF {file_path} "
                    f"({len(pdf_images)} pages) to ensure bbox coordinates match"
                )

        start = time.time()

        # Process using images if available, otherwise use original file
        if is_pdf and pdf_images:
            # Process each page image separately and combine results
            all_objects: Dict[int, DocumentObject] = {}
            bbox_list: List[Tuple[float, float, float, float]] = []
            context_parts: List[str] = []
            reading_order_ids: List[int] = []

            current_id = 0
            for page_idx, image_path in enumerate(pdf_images):
                logger.debug(
                    f"Processing PDF page {page_idx + 1}/{len(pdf_images)}: {image_path.name}"
                )
                page_objects = self._process_single_image(image_path, page_idx)

                # Add objects from this page
                for obj in page_objects:
                    # Update object ID to be unique across all pages
                    obj.object_id = current_id
                    all_objects[current_id] = obj
                    bbox_list.append(obj.bbox)
                    context_parts.append(obj.text)
                    reading_order_ids.append(current_id)
                    current_id += 1

            end = time.time()
            logger.info(
                f"Docling processing completed in {end - start:.2f}s "
                f"(processed {len(pdf_images)} page images)"
            )
            logger.info(f"Extracted {current_id} text objects from {file_path.name}")

            # Create metadata
            process_metadata = ProcessMetadata(
                processor_name="docling",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
                latency=(end - start) * 1000.0,
            )

            # Document metadata (use original PDF file info)
            mime_type = mimetypes.guess_type(str(file_path))[0]
            size_bytes = file_path.stat().st_size if file_path.exists() else None

            document_metadata = DocumentMetadata(
                filename=file_path.name,
                size_bytes=size_bytes,
                mime_type=mime_type,
            )

            # Full context text
            full_context = "\n".join(context_parts)

            # Create object_list from objects dict in reading order
            object_list = [
                all_objects[obj_id]
                for obj_id in reading_order_ids
                if obj_id in all_objects
            ]

            return DocumentProcessResult(
                objects=all_objects,
                object_list=object_list,
                bbox_list=bbox_list,
                context=full_context,
                reading_order_ids=reading_order_ids,
                process_metadata=process_metadata,
                document_metadata=document_metadata,
            )
        else:
            # Original behavior: process file directly (PDF, image, DOCX, etc.)
            logger.info(f"Processing document with Docling: {file_path}")
            result = self.converter.convert(str(file_path))

            end = time.time()
            logger.info(f"Docling processing completed in {end - start:.2f}s")

            # Extract objects from Docling result
            objects: Dict[int, DocumentObject] = {}
            bbox_list: List[Tuple[float, float, float, float]] = []
            context_parts: List[str] = []
            reading_order_ids: List[int] = []

            current_id = 0

            # Process document structure
            doc = result.document

            # Get page heights for coordinate conversion (Docling uses bottom-left origin)
            page_heights = {}
            for page_no, page_item in doc.pages.items():
                if hasattr(page_item, "size") and hasattr(page_item.size, "height"):
                    page_heights[page_no] = float(page_item.size.height)

            # Iterate through all text items in the document
            for text_item in doc.texts:
                obj = self._process_text_item(text_item, current_id, page_heights)
                if obj is not None:
                    objects[current_id] = obj
                    bbox_list.append(obj.bbox)
                    context_parts.append(obj.text)
                    reading_order_ids.append(current_id)
                    current_id += 1

            logger.info(f"Extracted {current_id} text objects from {file_path.name}")

            # Create metadata
            process_metadata = ProcessMetadata(
                processor_name="docling",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
                latency=(end - start) * 1000.0,
            )

            # Document metadata
            mime_type = mimetypes.guess_type(str(file_path))[0]
            size_bytes = file_path.stat().st_size if file_path.exists() else None

            document_metadata = DocumentMetadata(
                filename=file_path.name,
                size_bytes=size_bytes,
                mime_type=mime_type,
            )

            # Full context text
            full_context = "\n".join(context_parts)

            # Create object_list from objects dict in reading order
            object_list = [
                objects[obj_id] for obj_id in reading_order_ids if obj_id in objects
            ]

            return DocumentProcessResult(
                objects=objects,
                object_list=object_list,
                bbox_list=bbox_list,
                context=full_context,
                reading_order_ids=reading_order_ids,
                process_metadata=process_metadata,
                document_metadata=document_metadata,
            )


# For backward compatibility
def create_docling_processor(**kwargs) -> DoclingProcessor:
    """Factory function to create a DoclingProcessor instance.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to DoclingProcessor constructor.

    Returns
    -------
    DoclingProcessor
        Configured processor instance.
    """
    return DoclingProcessor(**kwargs)
