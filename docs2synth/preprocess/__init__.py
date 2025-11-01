"""Preprocessing modules for document analysis."""

from .paddleocr import PaddleOCRProcessor
from .docling_processor import DoclingProcessor
from .schema import DocumentProcessResult, DocumentObject, DocumentMetadata, ProcessMetadata, LabelType

__all__ = [
    "PaddleOCRProcessor",
    "DoclingProcessor",
    "DocumentProcessResult",
    "DocumentObject",
    "DocumentMetadata",
    "ProcessMetadata",
    "LabelType",
]