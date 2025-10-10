"""Information retriever training and evaluation utilities.

This subpackage will contain dataset builders, model wrappers, and training loops
for dense/embedding-based retrieval systems.
"""

from .dataloaders import load_qa_pairs
from .evaluation import evaluate_retriever
from .inference import retrieve
from .models import Retriever
from .train import train_retriever

__all__: list[str] = [
    "load_qa_pairs",
    "Retriever",
    "train_retriever",
    "retrieve",
    "evaluate_retriever",
]
