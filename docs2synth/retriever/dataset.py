"""Dataset loading utilities for retriever training.

This module provides utilities for loading training data from processed JSON files,
filtering QA pairs based on verification results, and creating DataLoaders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docs2synth.preprocess.schema import DocumentProcessResult, QAPair
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


def is_qa_verified(qa_pair: QAPair) -> bool:
    """Check if a QA pair has all verifiers responding 'Yes'.

    Args:
        qa_pair: QA pair to check

    Returns:
        True if all verifiers responded 'Yes', False otherwise
    """
    if not qa_pair.verification:
        return False

    # Check all verifiers
    for verifier_name, verifier_result in qa_pair.verification.items():
        if not isinstance(verifier_result, dict):
            continue

        # Get response (case-insensitive)
        response = verifier_result.get("Response") or verifier_result.get("response")
        if not response:
            # If no Response field, check if it's a simple dict with yes/no
            response_lower = str(verifier_result).lower()
            if "yes" in response_lower and "no" not in response_lower:
                continue
            elif "no" in response_lower:
                return False
            else:
                # Unknown format, skip this verifier or be conservative
                continue

        # Check if response is "Yes" (case-insensitive)
        if str(response).lower() != "yes":
            return False

    return True


def _find_json_files(
    data_dir: Path, processor_name: Optional[str] = None
) -> List[Path]:
    """Find JSON files in data directory, optionally filtered by processor name."""
    if processor_name:
        pattern = f"*_{processor_name.lower()}.json"
        json_files = list(data_dir.glob(pattern))
        logger.info(
            f"Found {len(json_files)} JSON files matching processor '{processor_name}' in {data_dir}"
        )
    else:
        json_files = list(data_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {data_dir}")
    return json_files


def _is_qa_pair_verified(qa_pair: Any, require_all_verifiers: bool) -> bool:
    """Check if QA pair meets verification requirements."""
    if require_all_verifiers:
        return is_qa_verified(qa_pair)
    # At least one verifier must respond 'Yes'
    for verifier_name, verifier_result in qa_pair.verification.items():
        if not isinstance(verifier_result, dict):
            continue
        response = verifier_result.get("Response") or verifier_result.get("response")
        if response and str(response).lower() == "yes":
            return True
    return False


def _create_qa_pair_dict(
    json_file: Path,
    obj_id: str,
    qa_idx: int,
    qa_pair: Any,
    obj: Any,
    result: Any,
) -> Dict[str, Any]:
    """Create a dictionary representation of a verified QA pair."""
    return {
        "json_file": str(json_file),
        "object_id": obj_id,
        "qa_idx": qa_idx,
        "question": qa_pair.question,
        "answer": qa_pair.answer or obj.text,
        "object_text": obj.text,
        "bbox": obj.bbox,
        "polygon": obj.polygon,
        "page": obj.page,
        "context": result.context,
        "strategy": qa_pair.strategy,
        "verification": qa_pair.verification,
        "extra": qa_pair.extra,
    }


def _process_json_file(
    json_file: Path,
    require_all_verifiers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Process a single JSON file and extract verified QA pairs.

    Returns:
        Tuple of (verified_qa_pairs, statistics_dict)
    """
    verified_qa_pairs: List[Dict[str, Any]] = []
    stats = {
        "qa_pairs": 0,
        "verified": 0,
        "objects_with_qa": 0,
        "objects_without_qa": 0,
        "qa_without_verification": 0,
    }

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = DocumentProcessResult.from_dict(data)

        # Process each object
        for obj_id, obj in result.objects.items():
            if not obj.qa:
                stats["objects_without_qa"] += 1
                continue

            stats["objects_with_qa"] += 1
            for qa_idx, qa_pair in enumerate(obj.qa):
                stats["qa_pairs"] += 1

                # Check verification status
                if not qa_pair.verification:
                    stats["qa_without_verification"] += 1
                    continue

                # Check if QA pair meets verification requirements
                if _is_qa_pair_verified(qa_pair, require_all_verifiers):
                    stats["verified"] += 1
                    verified_qa_pairs.append(
                        _create_qa_pair_dict(
                            json_file, obj_id, qa_idx, qa_pair, obj, result
                        )
                    )

    except Exception as e:
        logger.error(f"Failed to process {json_file}: {e}")

    return verified_qa_pairs, stats


def load_verified_qa_pairs(
    data_dir: Path,
    processor_name: Optional[str] = None,
    require_all_verifiers: bool = True,
) -> List[Dict[str, Any]]:
    """Load verified QA pairs from processed JSON files.

    Args:
        data_dir: Directory containing processed JSON files
        processor_name: Name of processor to filter by (e.g., "paddleocr", "docling").
                       If provided, only loads JSON files ending with _{processor_name}.json.
                       If None, loads all JSON files.
        require_all_verifiers: If True, all verifiers must respond 'Yes' (default: True).
                              If False, at least one verifier must respond 'Yes'.

    Returns:
        List of dictionaries containing QA pair data and associated object/document info
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Find JSON files, optionally filtered by processor name
    json_files = _find_json_files(data_dir, processor_name)

    if not json_files:
        if processor_name:
            logger.warning(
                f"No JSON files found matching processor '{processor_name}' in {data_dir}. "
                f"Expected pattern: *_{processor_name.lower()}.json"
            )
        else:
            logger.warning(f"No JSON files found in {data_dir}")
        return []

    verified_qa_pairs: List[Dict[str, Any]] = []
    total_stats = {
        "qa_pairs": 0,
        "verified": 0,
        "objects_with_qa": 0,
        "objects_without_qa": 0,
        "qa_without_verification": 0,
    }

    for json_file in json_files:
        file_pairs, file_stats = _process_json_file(json_file, require_all_verifiers)
        verified_qa_pairs.extend(file_pairs)
        for key in total_stats:
            total_stats[key] += file_stats[key]

    # Log detailed statistics
    logger.info(
        f"Data loading statistics: "
        f"{total_stats['objects_with_qa']} objects with QA, "
        f"{total_stats['objects_without_qa']} objects without QA, "
        f"{total_stats['qa_pairs']} total QA pairs, "
        f"{total_stats['qa_without_verification']} QA pairs without verification, "
        f"{total_stats['verified']} verified QA pairs"
    )

    if total_stats["qa_pairs"] > 0:
        logger.info(
            f"Loaded {total_stats['verified']} verified QA pairs from {total_stats['qa_pairs']} total QA pairs "
            f"({total_stats['verified']/total_stats['qa_pairs']*100:.1f}% verified)"
        )
    else:
        logger.warning(
            f"No QA pairs found in any JSON files in {data_dir}. "
            "Please ensure JSON files contain objects with QA pairs."
        )

    return verified_qa_pairs


class LayoutLMQADataset:
    """Dataset class for LayoutLM-based QA retriever training.

    This dataset preprocesses verified QA pairs into the tensor format
    required by LayoutLM models for document understanding tasks.

    IMPORTANT: This is a placeholder implementation. For actual training, you need to:
    1. Extend this class with your preprocessing logic in __getitem__
    2. Or use a pre-processed DataLoader pickle file with `--data-path /path/to/dataloader.pkl`

    The training functions expect data batches with these fields:
    - input_ids, attention_mask, token_type_ids, pixel_values, bbox
    - token_objt_ids, visual_feat, bert_cls, positional_encoding
    - norm_bbox, object_mask, target, start_id, end_id
    - grid_emb (for layout training modes)
    """

    def __init__(
        self,
        qa_pairs: List[Dict[str, Any]],
        processor: Optional[Any] = None,
        max_length: int = 512,
        image_dir: Optional[Path] = None,
    ):
        """Initialize the dataset.

        Args:
            qa_pairs: List of verified QA pair dictionaries
            processor: LayoutLM processor (e.g., AutoProcessor)
            max_length: Maximum sequence length for tokenization
            image_dir: Directory containing document images
        """
        self.qa_pairs = qa_pairs
        self.processor = processor
        self.max_length = max_length
        self.image_dir = Path(image_dir) if image_dir else None

        logger.warning(
            "\n" + "=" * 70 + "\n"
            "⚠️  LayoutLMQADataset preprocessing is NOT implemented!\n"
            "=" * 70 + "\n"
            "This dataset returns raw QA pairs, not preprocessed tensors.\n"
            "The training will fail when it tries to create batches.\n\n"
            "To fix this, you have two options:\n"
            "1. Implement preprocessing in LayoutLMQADataset.__getitem__()\n"
            "   - Convert QA pairs to model input tensors\n"
            "   - See training.py for required tensor fields\n\n"
            "2. Use a pre-processed DataLoader pickle:\n"
            "   - Preprocess your data offline\n"
            "   - Save DataLoader: pickle.dump(dataloader, open('train.pkl', 'wb'))\n"
            "   - Use: docs2synth retriever train --data-path /path/to/train.pkl\n"
            "=" * 70 + "\n"
        )

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single preprocessed example.

        Returns:
            Dictionary containing preprocessed tensors for model input

        Raises:
            NotImplementedError: This method needs to be implemented
        """
        raise NotImplementedError(
            "LayoutLMQADataset preprocessing is not implemented!\n"
            "You need to implement this method to convert QA pairs to model tensors.\n"
            "Or use a pre-processed DataLoader pickle file instead:\n"
            "  docs2synth retriever train --data-path /path/to/preprocessed_dataloader.pkl"
        )


def create_dataloader_from_verified_qa(
    verified_qa_pairs: List[Dict[str, Any]],
    batch_size: int = 8,
    shuffle: bool = True,
    processor: Optional[Any] = None,
    max_length: int = 512,
    image_dir: Optional[Path] = None,
    dataset_class: Optional[type] = None,
    **dataloader_kwargs: Any,
) -> Any:
    """Create a PyTorch DataLoader from verified QA pairs.

    Args:
        verified_qa_pairs: List of verified QA pair dictionaries
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        processor: LayoutLM processor (e.g., AutoProcessor from transformers)
        max_length: Maximum sequence length for tokenization
        image_dir: Directory containing document images
        dataset_class: Custom Dataset class to use (defaults to LayoutLMQADataset)
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        PyTorch DataLoader instance

    Example:
        >>> from transformers import AutoProcessor
        >>> from docs2synth.retriever.dataset import load_verified_qa_pairs, create_dataloader_from_verified_qa
        >>>
        >>> # Load data
        >>> qa_pairs = load_verified_qa_pairs(Path("./data/processed"))
        >>>
        >>> # Create processor
        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>>
        >>> # Create dataloader
        >>> dataloader = create_dataloader_from_verified_qa(
        ...     qa_pairs,
        ...     batch_size=16,
        ...     processor=processor,
        ...     image_dir=Path("./data/images")
        ... )
    """
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError(
            "PyTorch is required for creating DataLoaders. Install with: pip install torch"
        )

    # Use custom dataset class if provided, otherwise use default
    if dataset_class is None:
        dataset_class = LayoutLMQADataset

    dataset = dataset_class(
        qa_pairs=verified_qa_pairs,
        processor=processor,
        max_length=max_length,
        image_dir=image_dir,
    )

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs
    )
