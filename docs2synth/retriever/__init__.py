"""Information retriever training and evaluation utilities.

This subpackage contains dataset loaders, model wrappers, and training loops
for dense/embedding-based retrieval systems.

Workflow:
    1. Preprocess: Load QA pairs and create DataLoader pickle
       >>> from docs2synth.retriever.dataset import load_verified_qa_pairs
       >>> qa_pairs = load_verified_qa_pairs(data_dir, processor_name="docling")

       Or use CLI:
       $ docs2synth retriever preprocess --json-dir <dir> --output <file.pkl>

    2. Train: Load preprocessed pickle and train model
       $ docs2synth retriever train --data-path <file.pkl>
"""

from docs2synth.retriever.dataset import is_qa_verified, load_verified_qa_pairs
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
    # Dataset utilities
    "is_qa_verified",
    "load_verified_qa_pairs",
    # Model
    "LayoutLMForDocumentQA",
    "create_model_for_qa",
    # Metrics
    "calculate_anls",
    # Training functions
    "train",
    "train_layout",
    "train_layout_gemini",
    "train_layout_coarse_grained",
    "pretrain_layout",
    # Evaluation functions
    "evaluate",
    "evaluate_layout",
]
