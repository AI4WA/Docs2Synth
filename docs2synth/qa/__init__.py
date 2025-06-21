"""Question‐Answer pair generation utilities.

This subpackage will provide functions to automatically generate QA pairs from documents
using language models or heuristic approaches.
"""

from .dataloaders import load_documents
from .generator import generate_qa_pairs

__all__: list[str] = [
    "load_documents",
    "generate_qa_pairs",
] 