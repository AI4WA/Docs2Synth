"""Metrics for retriever evaluation.

This module provides evaluation metrics for document retrieval tasks,
including ANLS (Average Normalized Levenshtein Similarity) for QA evaluation.
"""

from __future__ import annotations


def calculate_anls(pred_text: str, gt_text: str) -> float:
    """Calculate Average Normalized Levenshtein Similarity (ANLS) score.

    ANLS is a metric commonly used in document understanding tasks that measures
    the similarity between predicted and ground truth text using normalized
    Levenshtein distance.

    Args:
        pred_text: Predicted text string
        gt_text: Ground truth text string

    Returns:
        ANLS score between 0.0 and 1.0, where 1.0 indicates perfect match

    Example:
        >>> calculate_anls("hello world", "hello world")
        1.0
        >>> calculate_anls("hello", "world")
        0.0
    """
    if not gt_text:
        return 1.0 if not pred_text else 0.0

    if not pred_text:
        return 0.0

    # Normalize texts (lowercase, strip whitespace)
    pred_normalized = pred_text.lower().strip()
    gt_normalized = gt_text.lower().strip()

    if pred_normalized == gt_normalized:
        return 1.0

    # Calculate Levenshtein distance
    distance = _levenshtein_distance(pred_normalized, gt_normalized)

    # Normalize by maximum length
    max_len = max(len(pred_normalized), len(gt_normalized))
    if max_len == 0:
        return 1.0

    normalized_distance = distance / max_len
    anls_score = 1.0 - normalized_distance

    # ANLS is typically thresholded at 0.5
    # If normalized distance > 0.5, score is 0
    return max(0.0, anls_score) if normalized_distance <= 0.5 else 0.0


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Levenshtein distance (minimum number of single-character edits)
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
