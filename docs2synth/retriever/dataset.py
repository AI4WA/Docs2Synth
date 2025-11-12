"""Dataset loading utilities for retriever training.

This module provides utilities for loading training data from processed JSON files,
filtering QA pairs based on verification results, and creating DataLoaders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    if processor_name:
        # Filter by processor: files ending with _{processor_name}.json
        pattern = f"*_{processor_name.lower()}.json"
        json_files = list(data_dir.glob(pattern))
        logger.info(
            f"Found {len(json_files)} JSON files matching processor '{processor_name}' in {data_dir}"
        )
    else:
        # Load all JSON files
        json_files = list(data_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {data_dir}")

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
    total_qa_pairs = 0
    total_verified = 0
    total_objects_with_qa = 0
    total_objects_without_qa = 0
    total_qa_without_verification = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            result = DocumentProcessResult.from_dict(data)

            # Process each object
            for obj_id, obj in result.objects.items():
                if not obj.qa:
                    total_objects_without_qa += 1
                    continue

                total_objects_with_qa += 1
                for qa_idx, qa_pair in enumerate(obj.qa):
                    total_qa_pairs += 1

                    # Check verification status
                    if not qa_pair.verification:
                        total_qa_without_verification += 1
                        continue

                    # Check if QA pair meets verification requirements
                    is_verified = False
                    if require_all_verifiers:
                        # All verifiers must respond 'Yes'
                        is_verified = is_qa_verified(qa_pair)
                    else:
                        # At least one verifier must respond 'Yes'
                        for verifier_name, verifier_result in qa_pair.verification.items():
                            if not isinstance(verifier_result, dict):
                                continue
                            response = verifier_result.get("Response") or verifier_result.get(
                                "response"
                            )
                            if response and str(response).lower() == "yes":
                                is_verified = True
                                break  # Found at least one 'Yes'

                    # Add verified QA pair
                    if is_verified:
                        total_verified += 1
                        verified_qa_pairs.append(
                            {
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
                        )

        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
            continue

    # Log detailed statistics
    logger.info(
        f"Data loading statistics: "
        f"{total_objects_with_qa} objects with QA, "
        f"{total_objects_without_qa} objects without QA, "
        f"{total_qa_pairs} total QA pairs, "
        f"{total_qa_without_verification} QA pairs without verification, "
        f"{total_verified} verified QA pairs"
    )

    if total_qa_pairs > 0:
        logger.info(
            f"Loaded {total_verified} verified QA pairs from {total_qa_pairs} total QA pairs "
            f"({total_verified/total_qa_pairs*100:.1f}% verified)"
        )
    else:
        logger.warning(
            f"No QA pairs found in any JSON files in {data_dir}. "
            "Please ensure JSON files contain objects with QA pairs."
        )

    return verified_qa_pairs


def create_dataloader_from_verified_qa(
    verified_qa_pairs: List[Dict[str, Any]],
    batch_size: int = 8,
    shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> Any:
    """Create a PyTorch DataLoader from verified QA pairs.

    Note: This is a placeholder function. Users need to implement their own
    Dataset class that converts the QA pair dictionaries into the required
    tensor format for their model.

    Args:
        verified_qa_pairs: List of verified QA pair dictionaries
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        PyTorch DataLoader instance

    Example:
        >>> from docs2synth.retriever.dataset import load_verified_qa_pairs, create_dataloader_from_verified_qa
        >>> qa_pairs = load_verified_qa_pairs(Path("./data/processed"))
        >>> dataloader = create_dataloader_from_verified_qa(qa_pairs, batch_size=16)
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        raise ImportError(
            "PyTorch is required for creating DataLoaders. Install with: pip install torch"
        )

    # Create a simple dataset wrapper
    class VerifiedQADataset(Dataset):
        def __init__(self, qa_pairs: List[Dict[str, Any]]):
            self.qa_pairs = qa_pairs

        def __len__(self) -> int:
            return len(self.qa_pairs)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.qa_pairs[idx]

    dataset = VerifiedQADataset(verified_qa_pairs)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs
    )

