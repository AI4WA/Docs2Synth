from __future__ import annotations

import logging
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


logging.getLogger("docling").setLevel(logging.INFO)
logging.getLogger("docling.document_converter").setLevel(logging.WARN)

try:
    from docling.document_converter import DocumentConverter

    _DOCLING_AVAILABLE = True
    logger.info("Docling imported successfully")
except ImportError as e:
    DocumentConverter = None  # type: ignore
    _DOCLING_AVAILABLE = False
    logger.error(
        f"Failed to import Docling: {str(e)}. Please install it with: pip install docling",
        exc_info=True,
    )


@dataclass
class DoclingProcessor:
    lang: str = "en"
    use_layout_analysis: bool = True
    ocr_engine: str = "tesseract"
    show_log: bool = False
    device: Optional[str] = None
    _docling_cache: Dict[str, "DocumentConverter"] = field(
        default_factory=dict, init=False, repr=False
    )
    _max_cache_size: int = field(default=10, init=False, repr=False)

    # PaddleOCR逻辑
    def _gpu_available(self) -> bool:
        try:
            from docling.utils.gpu import is_gpu_available

            return is_gpu_available()
        except ImportError:
            try:
                import torch  # type: ignore

                return torch.cuda.is_available()
            except ImportError:
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

    def _init_docling(
        self, lang_override: Optional[str] = None, device_override: Optional[str] = None
    ) -> "DocumentConverter":
        if not _DOCLING_AVAILABLE:
            raise RuntimeError("Docling is not installed: pip install docling")

        device_str = self._resolve_device(device_override)
        lang_key = (
            lang_override.strip()
            if (lang_override and isinstance(lang_override, str))
            else self.lang
        )
        cache_key = f"lang={lang_key}|device={device_str}|ocr={self.ocr_engine}"

        if cache_key in self._docling_cache:
            logger.debug(f"Using cached Docling instance: {cache_key}")
            return self._docling_cache[cache_key]

        if len(self._docling_cache) >= self._max_cache_size:
            oldest_key = next(iter(self._docling_cache.keys()))
            del self._docling_cache[oldest_key]
            logger.debug(f"Cache full, removing oldest instance: {oldest_key}")

        # Initialize new instance
        logger.debug(f"Initializing new Docling instance: {cache_key}")
        try:
            converter_config = {
                "language": lang_key,
                "ocr": {
                    "engine": self.ocr_engine,
                    "enabled": self.ocr_engine != "none",
                },
                "layout_analysis": {
                    "enabled": self.use_layout_analysis,
                    "model": "heron",
                },
                "rendering": {"enabled": False},
                "device": device_str,
            }

            docling_instance = DocumentConverter(**converter_config)
            self._docling_cache[cache_key] = docling_instance
            return docling_instance

        except Exception as e:
            logger.error(f"Failed to initialize Docling: {str(e)}", exc_info=True)
            raise

    def _map_label(self, elem) -> LabelType:
        """将 Docling 元素类型映射为 schema 支持的 LabelType"""
        if hasattr(elem, "type"):
            elem_type = str(elem.type).lower()
            if any(k in elem_type for k in ["picture", "formula"]):
                return LabelType.OTHER
            return LabelType.TEXT

        elem_class = (
            elem.__class__.__name__.lower() if hasattr(elem, "__class__") else ""
        )
        if any(k in elem_class for k in ["picture", "formula"]):
            return LabelType.OTHER
        return LabelType.TEXT

    def process(
        self,
        doc_path: str,
        *,
        lang: Optional[str] = None,
        device: Optional[str] = None,
    ) -> DocumentProcessResult:
        """处理文档并返回 schema 兼容结果"""
        if not isinstance(doc_path, str) or not doc_path.strip():
            raise ValueError(f"Invalid document path: {doc_path}")
        doc_path = doc_path.strip()
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document not found: {doc_path}")

        docling_converter = self._init_docling(
            lang_override=lang, device_override=device
        )
        start_time = time.time()

        try:
            convert_result = docling_converter.convert(doc_path)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to access document: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Docling processing failed: {str(e)}", exc_info=True)
            raise
        end_time = time.time()

        try:
            document = convert_result.document
            if hasattr(document, "pages") and document.pages:
                pages = document.pages
            else:
                pages = [
                    type(
                        "Page",
                        (),
                        {
                            "elements": document.elements
                            if hasattr(document, "elements")
                            else []
                        },
                    )()
                ]
            logger.debug(f"Number of pages processed: {len(pages)}")
        except AttributeError as e:
            logger.warning(f"Failed to parse result: {str(e)}", exc_info=True)
            pages = []

        objects: Dict[int, DocumentObject] = {}
        bbox_list: List[Tuple[float, float, float, float]] = []
        context_parts: List[str] = []
        reading_order_ids: List[int] = []
        current_id = 0

        if not pages:
            logger.warning(f"No valid pages: {doc_path}")
        else:
            for page_idx, page in enumerate(pages):
                elements = (
                    page.elements
                    if (hasattr(page, "elements") and page.elements)
                    else []
                )
                if not elements:
                    logger.warning(f"No elements on page {page_idx}")
                    continue

                logger.info(f"Processing page {page_idx}: {len(elements)} elements")
                for elem_idx, elem in enumerate(elements):
                    try:
                        text = (
                            elem.text.strip()
                            if (hasattr(elem, "text") and elem.text)
                            else ""
                        )
                        is_picture = self._map_label(elem) == LabelType.OTHER
                        if not text and not is_picture:
                            continue
                        score = None
                        if hasattr(elem, "confidence") and elem.confidence is not None:
                            try:
                                score = float(elem.confidence)
                            except (ValueError, TypeError):
                                score = None

                        bbox = (0.0, 0.0, 0.0, 0.0)
                        if hasattr(elem, "bbox") and elem.bbox:
                            try:
                                bbox_tuple = tuple(elem.bbox)
                                if len(bbox_tuple) == 4:
                                    bbox = tuple(map(float, bbox_tuple))
                                    bbox = (
                                        max(0.0, bbox[0]),
                                        max(0.0, bbox[1]),
                                        max(bbox[0] + 1e-6, bbox[2]),
                                        max(bbox[1] + 1e-6, bbox[3]),
                                    )
                            except (ValueError, TypeError, IndexError):
                                bbox = (0.0, 0.0, 0.0, 0.0)

                        obj = DocumentObject(
                            object_id=current_id,
                            text=text,
                            bbox=bbox,
                            label=self._map_label(elem),
                            page=page_idx,
                            score=score,
                        )

                        objects[current_id] = obj
                        bbox_list.append(bbox)
                        if text:
                            context_parts.append(text)
                        reading_order_ids.append(current_id)
                        current_id += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to process element (page {page_idx}, element {elem_idx}): {str(e)}",
                            exc_info=True,
                        )
                        continue

        logger.info(f"Extracted {current_id} elements total: {doc_path}")

        process_metadata = ProcessMetadata(
            processor_name="docling",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            latency=(end_time - start_time) * 1000.0,
        )

        detected_lang = lang or self.lang
        filename = os.path.basename(doc_path) if os.path.exists(doc_path) else None
        size_bytes = os.path.getsize(doc_path) if os.path.exists(doc_path) else None
        mime_type = (
            mimetypes.guess_type(doc_path)[0] if os.path.exists(doc_path) else None
        )
        width, height = None, None
        page_count = len(pages) if pages else 1

        if mime_type and mime_type.startswith("image/"):
            try:
                from PIL import Image  # type: ignore

                with Image.open(doc_path) as img:
                    width, height = img.size
            except Exception:
                width, height = None, None

        document_metadata = DocumentMetadata(
            source=doc_path,
            filename=filename,
            page_count=page_count,
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
