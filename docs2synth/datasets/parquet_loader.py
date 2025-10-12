"""Load and extract images from parquet format datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def extract_images_from_parquet(
    parquet_path: str | Path,
    output_dir: str | Path,
    image_column: str = "image",
    ground_truth_column: str | None = "ground_truth",
    image_format: str = "png",
) -> list[dict[str, Any]]:
    """Extract images from a parquet file and save them to disk.

    Args:
        parquet_path: Path to the parquet file
        output_dir: Directory to save extracted images
        image_column: Name of the column containing image data (default: "image")
        ground_truth_column: Name of the column containing ground truth data (default: "ground_truth")
        image_format: Format to save images as (default: "png")

    Returns:
        List of dictionaries containing image paths and metadata

    Example:
        >>> extract_images_from_parquet(
        ...     "data/train.parquet",
        ...     "data/images",
        ...     image_column="image",
        ...     ground_truth_column="ground_truth"
        ... )
        [{'image_path': 'data/images/image_0001.png', 'ground_truth': {...}}, ...]
    """
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if image_column not in df.columns:
        raise ValueError(
            f"Column '{image_column}' not found in parquet file. "
            f"Available columns: {df.columns.tolist()}"
        )

    results = []
    total = len(df)

    for idx, row in df.iterrows():
        image_data = row[image_column]

        # Handle different image data formats
        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
            original_path = image_data.get("path", f"image_{idx:04d}")
        elif isinstance(image_data, bytes):
            image_bytes = image_data
            original_path = f"image_{idx:04d}"
        else:
            logger.warning(f"Skipping row {idx}: unexpected image data format")
            continue

        # Create output filename
        if isinstance(original_path, str) and "." in original_path:
            # Use original filename with its extension if available
            filename = Path(original_path).name
        else:
            filename = f"image_{idx:04d}.{image_format}"

        output_path = output_dir / filename

        # Save image bytes to file
        output_path.write_bytes(image_bytes)

        # Collect metadata
        result = {"image_path": str(output_path), "index": idx}

        if ground_truth_column and ground_truth_column in df.columns:
            gt_data = row[ground_truth_column]
            # Parse if string, otherwise keep as is
            if isinstance(gt_data, str):
                try:
                    gt_data = json.loads(gt_data)
                except json.JSONDecodeError:
                    pass
            result["ground_truth"] = gt_data

        results.append(result)

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{total} images")

    logger.info(f"Extracted {len(results)} images to {output_dir}")
    return results


def load_parquet_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    splits: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load a parquet dataset and extract all images.

    Args:
        dataset_dir: Directory containing parquet files (e.g., data/datasets/cord/cord/data)
        output_dir: Directory to save extracted images. If None, saves to dataset_dir/images
        splits: List of splits to process (e.g., ["train", "validation"]). If None, processes all.

    Returns:
        Dictionary mapping split names to lists of image metadata

    Example:
        >>> load_parquet_dataset("data/datasets/cord/cord/data")
        {'train': [...], 'validation': [...]}
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    if output_dir is None:
        output_dir = dataset_dir.parent / "images"

    output_dir = Path(output_dir)

    # Find all parquet files
    parquet_files = list(dataset_dir.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {dataset_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files in {dataset_dir}")

    # Group by split (train, validation, test, etc.)
    splits_data = {}

    for parquet_file in parquet_files:
        # Extract split name from filename (e.g., "train-00000-of-00004.parquet" -> "train")
        filename = parquet_file.stem
        split_name = filename.split("-")[0]

        # Filter by requested splits
        if splits is not None and split_name not in splits:
            logger.debug(f"Skipping {split_name} (not in requested splits)")
            continue

        # Create split-specific output directory
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {split_name} split: {parquet_file.name}")

        # Extract images from this parquet file
        results = extract_images_from_parquet(parquet_file, split_output_dir)

        # Accumulate results by split
        if split_name not in splits_data:
            splits_data[split_name] = []
        splits_data[split_name].extend(results)

    # Save metadata for each split
    for split_name, data in splits_data.items():
        metadata_path = output_dir / f"{split_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {split_name} metadata to {metadata_path}")

    logger.info(f"Dataset loading complete. Total splits: {len(splits_data)}")
    for split_name, data in splits_data.items():
        logger.info(f"  {split_name}: {len(data)} images")

    return splits_data
