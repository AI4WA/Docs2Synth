"""Our retriever model wrapper for RAG pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoProcessor

from docs2synth.rag.embeddings import EmbeddingModel
from docs2synth.retriever.model import LayoutLMForDocumentQA, create_model_for_qa
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class OurRetrieverEmbeddingModel(EmbeddingModel):
    """Embedding model that uses a trained LayoutLM retriever for encoding."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        """Initialize our retriever embedding model.

        Args:
            model_path: Path to trained retriever model checkpoint (.pth file)
            device: Device to run model on (default: auto-detect)
            normalize: Whether to normalize embeddings (default: True)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Retriever model not found: {model_path}")

        # Auto-detect device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        logger.info(f"Loading our retriever model from {model_path}")
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

        # Load processor for tokenization
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )

        self.normalize = normalize
        self._dimension: int | None = None

    def _load_model(self) -> LayoutLMForDocumentQA:
        """Load trained retriever model from checkpoint."""
        # Create base model
        model = create_model_for_qa()

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        return model

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text using the LayoutLMv3 backbone.

        Note: This uses only the text encoder part, without image input.
        For full LayoutLMv3 capabilities, you would need image + bbox.
        """
        # LayoutLMv3 tokenizer requires pre-tokenized input (List[str]) and bounding boxes
        # Split text into words (simple whitespace tokenization)
        words = text.split()
        if not words:
            # Handle empty text
            words = [""]

        # Create dummy bounding boxes for each word (all zeros)
        # LayoutLMv3 expects bboxes in format [x0, y0, x1, y1] for each word
        word_boxes = [[0, 0, 0, 0] for _ in words]

        # Tokenize with bounding boxes
        # LayoutLMv3 tokenizer automatically handles pre-tokenized input when boxes are provided
        encoding = self.processor.tokenizer(
            words,
            boxes=word_boxes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Create dummy bbox (all zeros) - LayoutLMv3 requires this
        seq_len = input_ids.shape[1]
        bbox = torch.zeros((1, seq_len, 4), dtype=torch.long, device=self.device)

        # Create dummy pixel_values (black image) - LayoutLMv3 requires this
        pixel_values = torch.zeros(
            (1, 3, 224, 224), dtype=torch.float, device=self.device
        )

        # Create token_type_ids
        token_type_ids = torch.zeros_like(input_ids)

        # Extract embeddings from LayoutLMv3 backbone
        with torch.no_grad():
            outputs = self.model.layoutlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                bbox=bbox,
                pixel_values=pixel_values,
                return_dict=True,
            )

            # Use CLS token embedding as document representation
            # [batch_size, hidden_size]
            embedding = (
                outputs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
            )

        if self.normalize:
            # L2 normalize
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = embedding / (norm + 1e-8)

        return embedding[0]  # Return first (and only) item

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of texts."""
        if not texts:
            if self._dimension is None:
                # Probe dimension
                test_emb = self._encode_text("test")
                self._dimension = test_emb.shape[0]
            return np.empty((0, self._dimension), dtype="float32")

        embeddings = []
        for text in texts:
            emb = self._encode_text(text)
            embeddings.append(emb)

        result = np.array(embeddings, dtype="float32")
        return result

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query and return a 1D vector."""
        return self._encode_text(text)

    @property
    def dimension(self) -> int:
        """Return the dimensionality of embeddings produced by the model."""
        if self._dimension is None:
            test = self.embed_texts(["dimension-probe"])
            self._dimension = (
                int(test.shape[1]) if test.size else 768
            )  # Default LayoutLMv3 size
        return self._dimension

    def __repr__(self) -> str:
        return f"OurRetrieverEmbeddingModel(model_path={self.model_path}, device={self.device})"
