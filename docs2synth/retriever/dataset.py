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


def _extract_verifier_response(verifier_result: Any) -> Optional[str]:
    """Extract response from a verifier result dictionary.

    Args:
        verifier_result: Verifier result (dict or other)

    Returns:
        Response string ('yes', 'no', etc.) in lowercase, or None if not found
    """
    if not isinstance(verifier_result, dict):
        return None

    # Try both capitalized and lowercase keys
    response = verifier_result.get("Response") or verifier_result.get("response")
    if response:
        return str(response).lower()

    # Fallback: check if dict contains yes/no in string representation
    response_str = str(verifier_result).lower()
    if "yes" in response_str and "no" not in response_str:
        return "yes"
    elif "no" in response_str:
        return "no"

    return None


def is_qa_verified(qa_pair: QAPair, require_all: bool = True) -> bool:
    """Check if a QA pair meets verification requirements.

    Args:
        qa_pair: QA pair to check
        require_all: If True, all verifiers must respond 'Yes'.
                    If False, at least one verifier must respond 'Yes'.

    Returns:
        True if verification requirements are met, False otherwise
    """
    if not qa_pair.verification:
        return False

    yes_count = 0
    total_verifiers = 0

    for verifier_name, verifier_result in qa_pair.verification.items():
        total_verifiers += 1
        response = _extract_verifier_response(verifier_result)

        if response == "yes":
            yes_count += 1
        elif response == "no" and require_all:
            # Early exit if requiring all and found a "no"
            return False

    if require_all:
        # All verifiers must say yes
        return yes_count == total_verifiers
    else:
        # At least one verifier must say yes
        return yes_count > 0


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
                if is_qa_verified(qa_pair, require_all=require_all_verifiers):
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


# Note: LayoutLMQADataset and create_dataloader_from_verified_qa have been removed.
# Use PreprocessedQADataset from preprocess.py and create preprocessed pickle files:
#   docs2synth retriever preprocess --json-dir <dir> --image-dir <dir> --output <file.pkl>
