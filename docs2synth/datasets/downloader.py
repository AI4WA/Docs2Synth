"""Simple dataset downloader."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from ..utils import get_config, get_logger

logger = get_logger(__name__)

# Dataset registry: name -> download URL
DATASETS = {
    "vrd-iu2024-tracka": "https://ai4wa.s3.us-east-1.amazonaws.com/Docs2Synth/datasets/vrd-iu2024-tracka.zip",
    "vrd-iu2024-trackb": "https://ai4wa.s3.us-east-1.amazonaws.com/Docs2Synth/datasets/vrd-iu2024-trackb.zip",
    "funsd": "https://ai4wa.s3.us-east-1.amazonaws.com/Docs2Synth/datasets/funsd.zip",
    "cord": "https://ai4wa.s3.us-east-1.amazonaws.com/Docs2Synth/datasets/cord.zip"
}


def download_dataset(name: str, output_dir: str | Path | None = None) -> Path:
    """Download a dataset by name to a specific directory.

    Args:
        name: Dataset name (e.g., "vrd-iu2024-tracka", "vrd-iu2024-trackb")
        output_dir: Directory to save the dataset. If None, uses config setting.

    Returns:
        Path to the downloaded dataset folder

    Example:
        >>> download_dataset("vrd-iu2024-tracka")  # Uses config
        Path('./data/datasets/vrd-iu2024-tracka')

        >>> download_dataset("vrd-iu2024-tracka", "./my_data")  # Custom dir
        Path('./my_data/vrd-iu2024-tracka')
    """
    if name not in DATASETS:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {list(DATASETS.keys())}"
        )

    # Use config if output_dir not specified
    if output_dir is None:
        config = get_config()
        output_dir = config.get("data.datasets_dir", "./data/datasets")
        logger.info(f"Using datasets directory from config: {output_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = DATASETS[name]
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Download file
    filename = url.split("/")[-1]
    download_path = dataset_dir / filename

    logger.info(f"Downloading dataset '{name}' from {url}")
    logger.debug(f"Saving to: {download_path}")
    urlretrieve(url, download_path)
    logger.info(f"Download complete: {download_path}")

    # Extract if zip
    if filename.endswith(".zip"):
        logger.info(f"Extracting {filename}...")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        logger.info("Extraction complete")

    logger.info(f"Dataset '{name}' ready at: {dataset_dir}")
    return dataset_dir
