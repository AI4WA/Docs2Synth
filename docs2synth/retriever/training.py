"""Training functions for retriever models.

This module provides training loops for LayoutLM-based retriever models,
supporting various training configurations including layout pretraining,
entity retrieval, and span-based QA tasks.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from docs2synth.retriever.metrics import calculate_anls
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)

# Global device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global processor (initialized lazily)
_processor: Optional[AutoProcessor] = None


def get_processor() -> AutoProcessor:
    """Get or create the LayoutLMv3 processor.

    Returns:
        AutoProcessor instance for LayoutLMv3
    """
    global _processor
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )
    return _processor


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    lr: float,
    loss_function: Optional[torch.nn.Module] = None,
) -> Tuple[float, List[str], List[str], List[int], List[int]]:
    """Train model with entity retrieval and span-based QA tasks.

    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        lr: Learning rate for optimizer
        loss_function: Loss function to use (defaults to CrossEntropyLoss)

    Returns:
        Tuple containing:
            - average_anls: Average ANLS score over all examples
            - predict_text_list: List of predicted text strings
            - target_text_list: List of ground truth text strings
            - predict_entity_list: List of predicted entity IDs
            - target_id_list: List of target entity IDs
    """
    if loss_function is None:
        loss_function = CrossEntropyLoss()

    model.train()

    predict_text_list: List[str] = []
    target_text_list: List[str] = []
    anls_scores: List[float] = []
    predict_entity_list: List[int] = []
    target_id_list: List[int] = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    processor = get_processor()

    for _, data in tqdm(enumerate(train_dataloader, 0), desc="Training"):
        # Convert tensors to the correct types
        input_ids = data["input_ids"].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data["attention_mask"].to(device, dtype=torch.float)
        pixel_values = data["pixel_values"].to(device, dtype=torch.float)
        bbox = data["bbox"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        encoded_token_objt_ids = data["token_objt_ids"].to(
            device, dtype=torch.long
        )  # For token aggregate into entities

        visual_feat = data["visual_feat"].to(device, dtype=torch.float)
        bert_cls = data["bert_cls"].to(device, dtype=torch.float)
        positional_encoding = data["positional_encoding"].to(device, dtype=torch.float)
        norm_bbox = data["norm_bbox"].to(device, dtype=torch.float)
        object_mask = data["object_mask"].to(device, dtype=torch.float)

        # Entity Retrieving Target
        entity_targets = data["target"].to(device, dtype=torch.float)

        # Convert start and end positions to torch.long
        start_positions = data["start_id"].to(device, dtype=torch.long)
        end_positions = data["end_id"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs_dict = model(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            bbox,
            encoded_token_objt_ids,
            bert_cls,
            visual_feat,
            norm_bbox,
            object_mask,
            positional_encoding,
        )

        # Entity Retrieving Task
        entity_logits = outputs_dict["entity_logits"]
        entity_logits = entity_logits.squeeze(2)
        entity_loss = loss_function(entity_logits, entity_targets)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list.extend(list(big_idx.cpu().numpy()))

        _, target_idx = torch.max(entity_targets.data, dim=1)
        target_id_list.extend(list(target_idx.cpu().numpy()))

        # Span-based QA Predicted Logits
        start_logits = outputs_dict["start_logits"]
        end_logits = outputs_dict["end_logits"]
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Compute loss with CrossEntropyLoss
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        token_loss = (start_loss + end_loss) / 2

        total_loss = token_loss + entity_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        pred_start_positions = torch.argmax(start_logits, dim=1)
        pred_end_positions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):  # Loop through each example in the batch
            pred_tokens = input_ids[i][
                pred_start_positions[i] : pred_end_positions[i] + 1
            ]
            gt_tokens = input_ids[i][start_positions[i] : end_positions[i] + 1]

            # Decode tokens to text using processor
            pred_text = processor.tokenizer.decode(
                pred_tokens, skip_special_tokens=True
            )
            gt_text = processor.tokenizer.decode(gt_tokens, skip_special_tokens=True)

            # Calculate ANLS
            anls_score = calculate_anls(pred_text, gt_text)
            anls_scores.append(anls_score)

            predict_text_list.append(pred_text)
            target_text_list.append(gt_text)

    # Calculate the average ANLS over all examples
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0

    return (
        average_anls,
        predict_text_list,
        target_text_list,
        predict_entity_list,
        target_id_list,
    )


def train_layout(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    lr: float,
    loss_function: Optional[torch.nn.Module] = None,
) -> Tuple[float, List[str], List[str], List[int], List[int]]:
    """Train model with grid representations for layout pretraining.

    Updated training function with grid representations for layout pretraining.

    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        lr: Learning rate for optimizer
        loss_function: Loss function to use (defaults to CrossEntropyLoss)

    Returns:
        Tuple containing:
            - average_anls: Average ANLS score over all examples
            - predict_text_list: List of predicted text strings
            - target_text_list: List of ground truth text strings
            - predict_entity_list: List of predicted entity IDs
            - target_id_list: List of target entity IDs
    """
    if loss_function is None:
        loss_function = CrossEntropyLoss()

    model.train()

    predict_text_list: List[str] = []
    target_text_list: List[str] = []
    anls_scores: List[float] = []
    predict_entity_list: List[int] = []
    target_id_list: List[int] = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    processor = get_processor()

    for _, data in tqdm(enumerate(train_dataloader, 0), desc="Training (Layout)"):
        # Convert tensors to the correct types
        input_ids = data["input_ids"].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data["attention_mask"].to(device, dtype=torch.float)
        pixel_values = data["pixel_values"].to(device, dtype=torch.float)
        bbox = data["bbox"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        encoded_token_objt_ids = data["token_objt_ids"].to(
            device, dtype=torch.long
        )  # For token aggregate into entities

        visual_feat = data["visual_feat"].to(device, dtype=torch.float)
        bert_cls = data["bert_cls"].to(device, dtype=torch.float)
        positional_encoding = data["positional_encoding"].to(device, dtype=torch.float)
        norm_bbox = data["norm_bbox"].to(device, dtype=torch.float)
        object_mask = data["object_mask"].to(device, dtype=torch.float)

        # Entity Retrieving Target
        entity_targets = data["target"].to(device, dtype=torch.float)
        grid_emb = data["grid_emb"].to(device, dtype=torch.float)

        # Convert start and end positions to torch.long
        start_positions = data["start_id"].to(device, dtype=torch.long)
        end_positions = data["end_id"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs_dict = model(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            bbox,
            encoded_token_objt_ids,
            bert_cls,
            visual_feat,
            norm_bbox,
            object_mask,
            positional_encoding,
            grid_emb,
        )

        # Entity Retrieving Task
        entity_logits = outputs_dict["entity_logits"]
        entity_logits = entity_logits.squeeze(2)
        entity_loss = loss_function(entity_logits, entity_targets)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list.extend(list(big_idx.cpu().numpy()))

        _, target_idx = torch.max(entity_targets.data, dim=1)
        target_id_list.extend(list(target_idx.cpu().numpy()))

        # Span-based QA Predicted Logits
        start_logits = outputs_dict["start_logits"]
        end_logits = outputs_dict["end_logits"]
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Compute loss with CrossEntropyLoss
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        token_loss = (start_loss + end_loss) / 2

        total_loss = token_loss + entity_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        pred_start_positions = torch.argmax(start_logits, dim=1)
        pred_end_positions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):  # Loop through each example in the batch
            pred_tokens = input_ids[i][
                pred_start_positions[i] : pred_end_positions[i] + 1
            ]
            gt_tokens = input_ids[i][start_positions[i] : end_positions[i] + 1]

            # Decode tokens to text using processor
            pred_text = processor.tokenizer.decode(
                pred_tokens, skip_special_tokens=True
            )
            gt_text = processor.tokenizer.decode(gt_tokens, skip_special_tokens=True)

            # Calculate ANLS
            anls_score = calculate_anls(pred_text, gt_text)
            anls_scores.append(anls_score)

            predict_text_list.append(pred_text)
            target_text_list.append(gt_text)

    # Calculate the average ANLS over all examples
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0

    return (
        average_anls,
        predict_text_list,
        target_text_list,
        predict_entity_list,
        target_id_list,
    )


def train_layout_gemini(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    lr: float,
    loss_function: Optional[torch.nn.Module] = None,
) -> Tuple[float, List[str], List[str], List[int], List[int]]:
    """Train model with grid representations (Gemini variant).

    Similar to train_layout but with Gemini-specific configuration.

    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        lr: Learning rate for optimizer
        loss_function: Loss function to use (defaults to CrossEntropyLoss)

    Returns:
        Tuple containing:
            - average_anls: Average ANLS score over all examples
            - predict_text_list: List of predicted text strings
            - target_text_list: List of ground truth text strings
            - predict_entity_list: List of predicted entity IDs
            - target_id_list: List of target entity IDs
    """
    if loss_function is None:
        loss_function = CrossEntropyLoss()

    model.train()

    predict_text_list: List[str] = []
    target_text_list: List[str] = []
    anls_scores: List[float] = []
    predict_entity_list: List[int] = []
    target_id_list: List[int] = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    processor = get_processor()

    for _, data in tqdm(
        enumerate(train_dataloader, 0), desc="Training (Layout Gemini)"
    ):
        # Convert tensors to the correct types
        input_ids = data["input_ids"].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data["attention_mask"].to(device, dtype=torch.float)
        pixel_values = data["pixel_values"].to(device, dtype=torch.float)
        bbox = data["bbox"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        encoded_token_objt_ids = data["token_objt_ids"].to(
            device, dtype=torch.long
        )  # For token aggregate into entities

        visual_feat = data["visual_feat"].to(device, dtype=torch.float)
        bert_cls = data["bert_cls"].to(device, dtype=torch.float)
        positional_encoding = data["positional_encoding"].to(device, dtype=torch.float)
        norm_bbox = data["norm_bbox"].to(device, dtype=torch.float)
        object_mask = data["object_mask"].to(device, dtype=torch.float)

        # Entity Retrieving Target
        entity_targets = data["target"].to(device, dtype=torch.float)
        grid_emb = data["grid_emb"].to(device, dtype=torch.float)

        # Convert start and end positions to torch.long
        start_positions = data["start_id"].to(device, dtype=torch.long)
        end_positions = data["end_id"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs_dict = model(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            bbox,
            encoded_token_objt_ids,
            bert_cls,
            visual_feat,
            norm_bbox,
            object_mask,
            positional_encoding,
            grid_emb,
        )

        # Entity Retrieving Task
        entity_logits = outputs_dict["entity_logits"]
        entity_logits = entity_logits.squeeze(2)
        entity_loss = loss_function(entity_logits, entity_targets)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list.extend(list(big_idx.cpu().numpy()))

        _, target_idx = torch.max(entity_targets.data, dim=1)
        target_id_list.extend(list(target_idx.cpu().numpy()))

        # Span-based QA Predicted Logits
        start_logits = outputs_dict["start_logits"]
        end_logits = outputs_dict["end_logits"]
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Compute loss with CrossEntropyLoss
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        token_loss = (start_loss + end_loss) / 2

        total_loss = token_loss + entity_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        pred_start_positions = torch.argmax(start_logits, dim=1)
        pred_end_positions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):  # Loop through each example in the batch
            pred_tokens = input_ids[i][
                pred_start_positions[i] : pred_end_positions[i] + 1
            ]
            gt_tokens = input_ids[i][start_positions[i] : end_positions[i] + 1]

            # Decode tokens to text using processor
            pred_text = processor.tokenizer.decode(
                pred_tokens, skip_special_tokens=True
            )
            gt_text = processor.tokenizer.decode(gt_tokens, skip_special_tokens=True)

            # Calculate ANLS
            anls_score = calculate_anls(pred_text, gt_text)
            anls_scores.append(anls_score)

            predict_text_list.append(pred_text)
            target_text_list.append(gt_text)

    # Calculate the average ANLS over all examples
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0

    return (
        average_anls,
        predict_text_list,
        target_text_list,
        predict_entity_list,
        target_id_list,
    )


def train_layout_coarse_grained(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    lr: float,
    loss_function: Optional[torch.nn.Module] = None,
) -> Tuple[float, List[str], List[str], List[int], List[int]]:
    """Train model with coarse-grained entity loss only.

    This variant only uses entity loss, skipping token-level QA loss.

    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        lr: Learning rate for optimizer
        loss_function: Loss function to use (defaults to CrossEntropyLoss)

    Returns:
        Tuple containing:
            - average_anls: Average ANLS score over all examples
            - predict_text_list: List of predicted text strings
            - target_text_list: List of ground truth text strings
            - predict_entity_list: List of predicted entity IDs
            - target_id_list: List of target entity IDs
    """
    if loss_function is None:
        loss_function = CrossEntropyLoss()

    model.train()

    predict_text_list: List[str] = []
    target_text_list: List[str] = []
    anls_scores: List[float] = []
    predict_entity_list: List[int] = []
    target_id_list: List[int] = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    processor = get_processor()

    for _, data in tqdm(
        enumerate(train_dataloader, 0), desc="Training (Coarse-grained)"
    ):
        # Convert tensors to the correct types
        input_ids = data["input_ids"].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data["attention_mask"].to(device, dtype=torch.float)
        pixel_values = data["pixel_values"].to(device, dtype=torch.float)
        bbox = data["bbox"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        encoded_token_objt_ids = data["token_objt_ids"].to(
            device, dtype=torch.long
        )  # For token aggregate into entities

        visual_feat = data["visual_feat"].to(device, dtype=torch.float)
        bert_cls = data["bert_cls"].to(device, dtype=torch.float)
        positional_encoding = data["positional_encoding"].to(device, dtype=torch.float)
        norm_bbox = data["norm_bbox"].to(device, dtype=torch.float)
        object_mask = data["object_mask"].to(device, dtype=torch.float)

        # Entity Retrieving Target
        entity_targets = data["target"].to(device, dtype=torch.float)
        grid_emb = data["grid_emb"].to(device, dtype=torch.float)

        # Convert start and end positions to torch.long
        start_positions = data["start_id"].to(device, dtype=torch.long)
        end_positions = data["end_id"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs_dict = model(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            bbox,
            encoded_token_objt_ids,
            bert_cls,
            visual_feat,
            norm_bbox,
            object_mask,
            positional_encoding,
            grid_emb,
        )

        # Entity Retrieving Task
        entity_logits = outputs_dict["entity_logits"]
        entity_logits = entity_logits.squeeze(2)
        entity_loss = loss_function(entity_logits, entity_targets)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list.extend(list(big_idx.cpu().numpy()))

        _, target_idx = torch.max(entity_targets.data, dim=1)
        target_id_list.extend(list(target_idx.cpu().numpy()))

        # Span-based QA Predicted Logits (for evaluation only, not used in loss)
        start_logits = outputs_dict["start_logits"]
        end_logits = outputs_dict["end_logits"]
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Only use entity loss (coarse-grained training)
        total_loss = entity_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        pred_start_positions = torch.argmax(start_logits, dim=1)
        pred_end_positions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):  # Loop through each example in the batch
            pred_tokens = input_ids[i][
                pred_start_positions[i] : pred_end_positions[i] + 1
            ]
            gt_tokens = input_ids[i][start_positions[i] : end_positions[i] + 1]

            # Decode tokens to text using processor
            pred_text = processor.tokenizer.decode(
                pred_tokens, skip_special_tokens=True
            )
            gt_text = processor.tokenizer.decode(gt_tokens, skip_special_tokens=True)

            # Calculate ANLS
            anls_score = calculate_anls(pred_text, gt_text)
            anls_scores.append(anls_score)

            predict_text_list.append(pred_text)
            target_text_list.append(gt_text)

    # Calculate the average ANLS over all examples
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0

    return (
        average_anls,
        predict_text_list,
        target_text_list,
        predict_entity_list,
        target_id_list,
    )


def pretrain_layout(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    lr: float,
    loss_function: Optional[torch.nn.Module] = None,
) -> Tuple[List[int], List[int], float]:
    """Pretrain model with grid-based layout representation.

    This function performs layout pretraining using grid embeddings and grid IDs.

    Args:
        model: PyTorch model to pretrain
        train_dataloader: DataLoader for training data
        lr: Learning rate for optimizer
        loss_function: Loss function to use (defaults to CrossEntropyLoss)

    Returns:
        Tuple containing:
            - predict_entity_list: List of predicted entity IDs
            - target_id_list: List of target entity IDs
            - total_loss: Total loss accumulated over the epoch
    """
    if loss_function is None:
        loss_function = CrossEntropyLoss()

    model.train()

    predict_entity_list: List[int] = []
    target_id_list: List[int] = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    total_loss = 0.0

    for _, data in tqdm(enumerate(train_dataloader, 0), desc="Pretraining (Layout)"):
        # Convert tensors to the correct types
        input_ids = data["input_ids"].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data["attention_mask"].to(device, dtype=torch.float)
        pixel_values = data["pixel_values"].to(device, dtype=torch.float)
        bbox = data["bbox"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        encoded_token_objt_ids = data["token_objt_ids"].to(
            device, dtype=torch.long
        )  # For token aggregate into entities

        visual_feat = data["visual_feat"].to(device, dtype=torch.float)
        bert_cls = data["bert_cls"].to(device, dtype=torch.float)
        positional_encoding = data["positional_encoding"].to(device, dtype=torch.float)
        norm_bbox = data["norm_bbox"].to(device, dtype=torch.float)
        object_mask = data["object_mask"].to(device, dtype=torch.float)

        grid_emb = data["grid_emb"].to(device, dtype=torch.float)
        grid_idx = data["grid_ids"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model (pretrain stage)
        grid_logits = model(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            bbox,
            encoded_token_objt_ids,
            bert_cls,
            visual_feat,
            norm_bbox,
            object_mask,
            positional_encoding,
            grid_emb,
            train_stage="pretrain",
        )

        # Entity Retrieving Task
        entity_logits = grid_logits.squeeze(2)
        entity_loss = loss_function(entity_logits, grid_idx)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list.extend(list(big_idx.cpu().numpy()))

        _, target_idx = torch.max(grid_idx.data, dim=1)
        target_id_list.extend(list(target_idx.cpu().numpy()))

        total_loss += entity_loss.item()  # Accumulate the loss

        # Backpropagation
        entity_loss.backward()
        optimizer.step()

    return predict_entity_list, target_id_list, total_loss

