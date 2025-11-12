"""Custom LayoutLM model for document QA.

This module implements a span-based QA model based on LayoutLMv3 that performs:
- Span-based QA - predicting exact start/end positions of the answer
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import LayoutLMv3Model

# Default model dimensions
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_NUM_OBJECTS = 50
DEFAULT_DROPOUT_RATE = 0.1


class LayoutLMForDocumentQA(nn.Module):
    """LayoutLM model for span-based document QA.

    This model combines:
    - LayoutLMv3 backbone for document understanding
    - QA heads for predicting answer spans (start/end positions)
    """

    def __init__(
        self,
        layoutlm_model: LayoutLMv3Model,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_objects: int = DEFAULT_NUM_OBJECTS,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
    ):
        """Initialize the model.

        Args:
            layoutlm_model: Pre-trained LayoutLMv3 model
            hidden_size: Hidden dimension size (default: 768)
            num_objects: Maximum number of objects per document (kept for compatibility)
            dropout_rate: Dropout rate for regularization (default: 0.1, unused currently)
        """
        super().__init__()
        self.layoutlm = layoutlm_model
        self.hidden_size = hidden_size
        self.num_objects = num_objects  # Keep for compatibility

        # QA heads for span prediction
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start and end logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        bbox: torch.Tensor,
        encoded_token_objt_ids: torch.Tensor = None,
        bert_cls: torch.Tensor = None,
        visual_feat: torch.Tensor = None,
        norm_bbox: torch.Tensor = None,
        object_mask: torch.Tensor = None,
        positional_encoding: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            pixel_values: Image pixels [batch_size, 3, H, W]
            bbox: Token bounding boxes [batch_size, seq_len, 4]
            encoded_token_objt_ids: Token to object mapping (kept for compatibility, unused)
            bert_cls: BERT CLS embeddings (kept for compatibility, unused)
            visual_feat: Visual features (kept for compatibility, unused)
            norm_bbox: Normalized object bboxes (kept for compatibility, unused)
            object_mask: Object attention mask (kept for compatibility, unused)
            positional_encoding: Positional encodings (kept for compatibility, unused)

        Returns:
            Dictionary with:
                - start_logits: Answer start logits [batch_size, seq_len]
                - end_logits: Answer end logits [batch_size, seq_len]
        """
        # Run LayoutLMv3 backbone
        outputs = self.layoutlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            return_dict=True,
        )

        # Get sequence output [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state

        # Span-based QA task
        # Use sequence output from LayoutLMv3
        qa_logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, 2]

        start_logits = qa_logits[:, :, 0]  # [batch_size, seq_len]
        end_logits = qa_logits[:, :, 1]  # [batch_size, seq_len]

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }


def create_model_for_qa(
    base_model_name: str = "microsoft/layoutlmv3-base",
    num_objects: int = DEFAULT_NUM_OBJECTS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> LayoutLMForDocumentQA:
    """Create a LayoutLM model for document QA.

    Args:
        base_model_name: HuggingFace model name or path
        num_objects: Maximum number of objects per document (default: 50)
        dropout_rate: Dropout rate for regularization (default: 0.1)

    Returns:
        Initialized LayoutLMForDocumentQA model
    """
    # Load base LayoutLMv3 model
    layoutlm_base = LayoutLMv3Model.from_pretrained(base_model_name)

    # Get hidden size from config
    hidden_size = layoutlm_base.config.hidden_size

    # Create wrapped model
    model = LayoutLMForDocumentQA(
        layoutlm_model=layoutlm_base,
        hidden_size=hidden_size,
        num_objects=num_objects,
        dropout_rate=dropout_rate,
    )

    return model
