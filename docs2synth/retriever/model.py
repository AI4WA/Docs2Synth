"""Custom LayoutLM model for document QA with entity retrieval.

This module implements a multi-task model based on LayoutLMv3 that performs:
1. Entity retrieval - identifying which object/region contains the answer
2. Span-based QA - predicting exact start/end positions of the answer
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import LayoutLMv3Model


class LayoutLMForDocumentQA(nn.Module):
    """Multi-task LayoutLM model for document QA.

    This model combines:
    - LayoutLMv3 backbone for document understanding
    - Entity retrieval head for identifying answer-containing objects
    - QA heads for predicting answer spans

    Stage 1 (MVP): Basic architecture with simplified entity retrieval
    Stage 2: Will add full visual features and object-level reasoning
    """

    def __init__(
        self,
        layoutlm_model: LayoutLMv3Model,
        hidden_size: int = 768,
        num_objects: int = 50,
    ):
        """Initialize the model.

        Args:
            layoutlm_model: Pre-trained LayoutLMv3 model
            hidden_size: Hidden dimension size (default: 768)
            num_objects: Maximum number of objects per document (default: 50)
        """
        super().__init__()
        self.layoutlm = layoutlm_model
        self.hidden_size = hidden_size
        self.num_objects = num_objects

        # Entity retrieval head
        # Maps from combined features to entity scores
        self.entity_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

        # QA heads for span prediction
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start and end logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        bbox: torch.Tensor,
        encoded_token_objt_ids: torch.Tensor,
        bert_cls: torch.Tensor,
        visual_feat: torch.Tensor,
        norm_bbox: torch.Tensor,
        object_mask: torch.Tensor,
        positional_encoding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            pixel_values: Image pixels [batch_size, 3, H, W]
            bbox: Token bounding boxes [batch_size, seq_len, 4]
            encoded_token_objt_ids: Token to object mapping [batch_size, seq_len]
            bert_cls: BERT CLS embeddings [batch_size, hidden_size]
            visual_feat: Visual features [batch_size, num_objects, hidden_size]
            norm_bbox: Normalized object bboxes [batch_size, num_objects, 4]
            object_mask: Object attention mask [batch_size, num_objects]
            positional_encoding: Positional encodings [batch_size, num_objects, 128]

        Returns:
            Dictionary with:
                - entity_logits: Entity scores [batch_size, num_objects, 1]
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

        # Entity retrieval task
        # Stage 1: Simple approach - combine visual features with bert_cls

        # Expand bert_cls to match num_objects dimension
        bert_cls_expanded = bert_cls.unsqueeze(1).expand(-1, self.num_objects, -1)

        # Concatenate visual features with bert_cls for each object
        entity_features = torch.cat([visual_feat, bert_cls_expanded], dim=-1)

        # Predict entity scores
        entity_logits = self.entity_head(
            entity_features
        )  # [batch_size, num_objects, 1]

        # Span-based QA task
        # Use sequence output from LayoutLMv3
        qa_logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, 2]

        start_logits = qa_logits[:, :, 0]  # [batch_size, seq_len]
        end_logits = qa_logits[:, :, 1]  # [batch_size, seq_len]

        return {
            "entity_logits": entity_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }


def create_model_for_qa(
    base_model_name: str = "microsoft/layoutlmv3-base",
    num_objects: int = 50,
) -> LayoutLMForDocumentQA:
    """Create a LayoutLM model for document QA.

    Args:
        base_model_name: HuggingFace model name or path
        num_objects: Maximum number of objects per document

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
    )

    return model
