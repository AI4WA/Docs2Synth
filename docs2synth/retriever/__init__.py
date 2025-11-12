"""Information retriever training and evaluation utilities.

This subpackage contains dataset builders, model wrappers, and training loops
for dense/embedding-based retrieval systems.
"""

from docs2synth.retriever.dataset import (
    LayoutLMQADataset,
    create_dataloader_from_verified_qa,
    is_qa_verified,
    load_verified_qa_pairs,
)
from docs2synth.retriever.metrics import calculate_anls
from docs2synth.retriever.model import LayoutLMForDocumentQA, create_model_for_qa
from docs2synth.retriever.training import (
    evaluate,
    evaluate_layout,
    pretrain_layout,
    train,
    train_layout,
    train_layout_coarse_grained,
    train_layout_gemini,
)

__all__: list[str] = [
    "LayoutLMForDocumentQA",
    "LayoutLMQADataset",
    "calculate_anls",
    "create_dataloader_from_verified_qa",
    "create_model_for_qa",
    "evaluate",
    "evaluate_layout",
    "is_qa_verified",
    "load_verified_qa_pairs",
    "pretrain_layout",
    "train",
    "train_layout",
    "train_layout_coarse_grained",
    "train_layout_gemini",
]
