"""Tests for datasets module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_datasets_import():
    """Test that datasets module can be imported."""
    from docs2synth.datasets import download_dataset

    assert download_dataset is not None


def test_datasets_registry():
    """Test dataset registry contains expected datasets."""
    from docs2synth.datasets.downloader import DATASETS

    assert isinstance(DATASETS, dict)
    assert "vrd-iu2024-tracka" in DATASETS
    assert "vrd-iu2024-trackb" in DATASETS
    assert len(DATASETS) >= 2


def test_download_dataset_invalid_name():
    """Test that invalid dataset name raises ValueError."""
    from docs2synth.datasets import download_dataset

    with pytest.raises(ValueError, match="Dataset .* not found"):
        download_dataset("invalid_dataset_name")


def test_download_dataset_with_custom_dir():
    """Test download_dataset with custom output directory."""
    from docs2synth.datasets import download_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock urlretrieve to avoid actual download
        with patch("docs2synth.datasets.downloader.urlretrieve") as mock_retrieve:
            # Create a fake zip file
            test_zip = Path(tmpdir) / "vrd-iu2024-tracka" / "test.zip"
            test_zip.parent.mkdir(parents=True, exist_ok=True)
            test_zip.write_bytes(b"fake zip content")

            # Mock zipfile extraction
            with patch("zipfile.ZipFile") as mock_zipfile:
                mock_zip = MagicMock()
                mock_zipfile.return_value.__enter__.return_value = mock_zip

                result = download_dataset("vrd-iu2024-tracka", tmpdir)

                # Verify download was called
                assert mock_retrieve.called
                # Verify result is a Path
                assert isinstance(result, Path)
                # Verify path contains dataset name
                assert "vrd-iu2024-tracka" in str(result)


def test_download_dataset_uses_config():
    """Test that download_dataset uses config when output_dir is None."""
    from docs2synth.datasets import download_dataset
    from docs2synth.utils import Config, set_config

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set custom config
        config = Config()
        config.set("data.datasets_dir", tmpdir)
        set_config(config)

        # Mock urlretrieve and zipfile
        with patch("docs2synth.datasets.downloader.urlretrieve"):
            with patch("zipfile.ZipFile"):
                result = download_dataset("vrd-iu2024-tracka")

                # Verify result path uses config directory
                assert tmpdir in str(result)


def test_download_dataset_creates_directories():
    """Test that download_dataset creates necessary directories."""
    from docs2synth.datasets import download_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "nested" / "path"

        with patch("docs2synth.datasets.downloader.urlretrieve"):
            with patch("zipfile.ZipFile"):
                result = download_dataset("vrd-iu2024-tracka", str(output_dir))

                # Verify directories were created
                assert output_dir.exists()
                assert result.parent.exists()


def test_logger_usage():
    """Test that downloader uses logger instead of print."""
    from docs2synth.datasets import download_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("docs2synth.datasets.downloader.urlretrieve"):
            with patch("zipfile.ZipFile"):
                with patch("docs2synth.datasets.downloader.logger") as mock_logger:
                    download_dataset("vrd-iu2024-tracka", tmpdir)

                    # Verify logger was called
                    assert mock_logger.info.called
                    # Verify multiple log messages
                    assert mock_logger.info.call_count >= 3
