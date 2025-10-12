"""Tests for datasets module."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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
                    download_dataset("vrd-iu2024-tracka", tmpdir, extract_images=False)

                    # Verify logger was called
                    assert mock_logger.info.called
                    # Verify multiple log messages
                    assert mock_logger.info.call_count >= 3


# Tests for parquet_loader module


def test_extract_images_from_parquet_with_dict_format():
    """Test extracting images from parquet with dict format (bytes + path)."""
    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        # Create test data with image dict format
        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame(
            {
                "image": [
                    {"bytes": test_image_bytes, "path": "test_image.png"},
                    {"bytes": test_image_bytes, "path": "another_image.jpg"},
                ],
                "ground_truth": ['{"key": "value1"}', '{"key": "value2"}'],
            }
        )
        test_data.to_parquet(parquet_path)

        # Extract images
        results = extract_images_from_parquet(parquet_path, output_dir)

        # Verify results
        assert len(results) == 2
        assert all("image_path" in r for r in results)
        assert all("ground_truth" in r for r in results)
        assert all("index" in r for r in results)

        # Verify files were created
        assert (output_dir / "test_image.png").exists()
        assert (output_dir / "another_image.jpg").exists()

        # Verify content
        assert (output_dir / "test_image.png").read_bytes() == test_image_bytes

        # Verify ground truth is parsed
        assert results[0]["ground_truth"] == {"key": "value1"}


def test_extract_images_from_parquet_with_bytes_format():
    """Test extracting images from parquet with raw bytes format."""
    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        # Create test data with raw bytes format
        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame(
            {
                "image": [test_image_bytes, test_image_bytes],
                "ground_truth": [{"key": "value1"}, {"key": "value2"}],
            }
        )
        test_data.to_parquet(parquet_path)

        # Extract images
        results = extract_images_from_parquet(parquet_path, output_dir)

        # Verify results
        assert len(results) == 2

        # Verify files were created with default naming
        assert (output_dir / "image_0000.png").exists()
        assert (output_dir / "image_0001.png").exists()


def test_extract_images_from_parquet_missing_column():
    """Test that missing image column raises ValueError."""
    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file without image column
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        test_data = pd.DataFrame({"other_column": ["data1", "data2"]})
        test_data.to_parquet(parquet_path)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Column 'image' not found"):
            extract_images_from_parquet(parquet_path, output_dir)


def test_extract_images_from_parquet_custom_columns():
    """Test extracting images with custom column names."""
    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file with custom columns
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame(
            {
                "custom_image": [test_image_bytes],
                "custom_gt": [{"label": "test"}],
            }
        )
        test_data.to_parquet(parquet_path)

        # Extract with custom column names
        results = extract_images_from_parquet(
            parquet_path,
            output_dir,
            image_column="custom_image",
            ground_truth_column="custom_gt",
        )

        # Verify results
        assert len(results) == 1
        assert results[0]["ground_truth"] == {"label": "test"}


def test_extract_images_from_parquet_no_ground_truth():
    """Test extracting images without ground truth column."""
    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file without ground truth
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame({"image": [test_image_bytes]})
        test_data.to_parquet(parquet_path)

        # Extract without ground truth
        results = extract_images_from_parquet(
            parquet_path, output_dir, ground_truth_column=None
        )

        # Verify results
        assert len(results) == 1
        assert "ground_truth" not in results[0]
        assert "image_path" in results[0]


def test_extract_images_from_parquet_custom_format():
    """Test extracting images with custom image format."""
    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        test_image_bytes = b"fake jpg data"
        test_data = pd.DataFrame({"image": [test_image_bytes]})
        test_data.to_parquet(parquet_path)

        # Extract as jpg
        results = extract_images_from_parquet(
            parquet_path, output_dir, image_format="jpg"
        )

        # Verify jpg extension
        assert results[0]["image_path"].endswith(".jpg")
        assert (output_dir / "image_0000.jpg").exists()


def test_load_parquet_dataset_single_split():
    """Test loading a parquet dataset with a single split."""
    from docs2synth.datasets.parquet_loader import load_parquet_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet files
        dataset_dir = Path(tmpdir) / "data"
        dataset_dir.mkdir()

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame(
            {
                "image": [{"bytes": test_image_bytes, "path": "img1.png"}],
                "ground_truth": ['{"key": "value"}'],
            }
        )

        # Create train split
        train_file = dataset_dir / "train-00000-of-00001.parquet"
        test_data.to_parquet(train_file)

        # Load dataset
        result = load_parquet_dataset(dataset_dir)

        # Verify result
        assert "train" in result
        assert len(result["train"]) == 1
        assert (Path(tmpdir) / "images" / "train" / "img1.png").exists()

        # Verify metadata file was created
        metadata_file = Path(tmpdir) / "images" / "train_metadata.json"
        assert metadata_file.exists()

        # Verify metadata content
        with open(metadata_file) as f:
            metadata = json.load(f)
        assert len(metadata) == 1
        assert metadata[0]["ground_truth"] == {"key": "value"}


def test_load_parquet_dataset_multiple_splits():
    """Test loading a parquet dataset with multiple splits."""
    from docs2synth.datasets.parquet_loader import load_parquet_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet files
        dataset_dir = Path(tmpdir) / "data"
        dataset_dir.mkdir()

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame(
            {
                "image": [test_image_bytes, test_image_bytes],
                "ground_truth": [{"key": "value1"}, {"key": "value2"}],
            }
        )

        # Create multiple splits
        (dataset_dir / "train-00000-of-00001.parquet").write_text("")
        test_data.to_parquet(dataset_dir / "train-00000-of-00001.parquet")

        (dataset_dir / "validation-00000-of-00001.parquet").write_text("")
        test_data.to_parquet(dataset_dir / "validation-00000-of-00001.parquet")

        (dataset_dir / "test-00000-of-00001.parquet").write_text("")
        test_data.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

        # Load dataset
        result = load_parquet_dataset(dataset_dir)

        # Verify all splits were loaded
        assert "train" in result
        assert "validation" in result
        assert "test" in result
        assert len(result["train"]) == 2
        assert len(result["validation"]) == 2
        assert len(result["test"]) == 2


def test_load_parquet_dataset_filter_splits():
    """Test loading only specific splits from a parquet dataset."""
    from docs2synth.datasets.parquet_loader import load_parquet_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet files
        dataset_dir = Path(tmpdir) / "data"
        dataset_dir.mkdir()

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame({"image": [test_image_bytes]})

        # Create multiple splits
        test_data.to_parquet(dataset_dir / "train-00000-of-00001.parquet")
        test_data.to_parquet(dataset_dir / "validation-00000-of-00001.parquet")
        test_data.to_parquet(dataset_dir / "test-00000-of-00001.parquet")

        # Load only train and validation
        result = load_parquet_dataset(dataset_dir, splits=["train", "validation"])

        # Verify only requested splits were loaded
        assert "train" in result
        assert "validation" in result
        assert "test" not in result


def test_load_parquet_dataset_custom_output_dir():
    """Test loading parquet dataset with custom output directory."""
    from docs2synth.datasets.parquet_loader import load_parquet_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet files
        dataset_dir = Path(tmpdir) / "data"
        dataset_dir.mkdir()

        custom_output = Path(tmpdir) / "custom_images"

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame({"image": [test_image_bytes]})
        test_data.to_parquet(dataset_dir / "train-00000-of-00001.parquet")

        # Load with custom output directory
        load_parquet_dataset(dataset_dir, output_dir=custom_output)

        # Verify images were saved to custom directory
        assert (custom_output / "train").exists()
        assert len(list((custom_output / "train").glob("*.png"))) == 1


def test_load_parquet_dataset_no_parquet_files():
    """Test that loading from directory without parquet files raises error."""
    from docs2synth.datasets.parquet_loader import load_parquet_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty directory
        dataset_dir = Path(tmpdir) / "data"
        dataset_dir.mkdir()

        # Should raise ValueError
        with pytest.raises(ValueError, match="No parquet files found"):
            load_parquet_dataset(dataset_dir)


def test_load_parquet_dataset_nonexistent_dir():
    """Test that loading from nonexistent directory raises error."""
    from docs2synth.datasets.parquet_loader import load_parquet_dataset

    with pytest.raises(ValueError, match="Dataset directory not found"):
        load_parquet_dataset("/nonexistent/path")


def test_download_dataset_with_extract_images_true():
    """Test download_dataset with extract_images=True."""
    from docs2synth.datasets import download_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "vrd-iu2024-tracka"
        dataset_dir.mkdir(parents=True)

        # Create a fake parquet file structure
        data_dir = dataset_dir / "data"
        data_dir.mkdir()

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame({"image": [test_image_bytes]})
        test_data.to_parquet(data_dir / "train-00000-of-00001.parquet")

        # Create a fake zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data/train-00000-of-00001.parquet", "fake content")

        def fake_urlretrieve(url, path):
            # Copy our fake zip to the download location
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(zip_path.read_bytes())

        with patch(
            "docs2synth.datasets.downloader.urlretrieve", side_effect=fake_urlretrieve
        ):
            # Mock the parquet extraction to avoid actual file operations
            with patch(
                "docs2synth.datasets.parquet_loader.load_parquet_dataset"
            ) as mock_load:
                mock_load.return_value = {"train": [{"image_path": "test.png"}]}

                download_dataset("vrd-iu2024-tracka", tmpdir, extract_images=True)

                # Verify parquet loader was called
                assert mock_load.called


def test_download_dataset_with_extract_images_false():
    """Test download_dataset with extract_images=False."""
    from docs2synth.datasets import download_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("docs2synth.datasets.downloader.urlretrieve"):
            with patch("zipfile.ZipFile"):
                with patch(
                    "docs2synth.datasets.parquet_loader.load_parquet_dataset"
                ) as mock_load:
                    download_dataset("vrd-iu2024-tracka", tmpdir, extract_images=False)

                    # Verify parquet loader was NOT called
                    assert not mock_load.called


def test_download_dataset_no_parquet_files():
    """Test download_dataset when no parquet files exist."""
    from docs2synth.datasets import download_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake zip without parquet files
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data/readme.txt", "no parquet here")

        def fake_urlretrieve(url, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(zip_path.read_bytes())

        with patch(
            "docs2synth.datasets.downloader.urlretrieve", side_effect=fake_urlretrieve
        ):
            # Should not raise error, just log a debug message
            result = download_dataset("vrd-iu2024-tracka", tmpdir, extract_images=True)
            assert result.exists()


def test_download_dataset_parquet_extraction_failure():
    """Test download_dataset handles parquet extraction failure gracefully."""
    from docs2synth.datasets import download_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "vrd-iu2024-tracka"
        dataset_dir.mkdir(parents=True)

        # Create a fake parquet file structure
        data_dir = dataset_dir / "data"
        data_dir.mkdir()

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame({"image": [test_image_bytes]})
        test_data.to_parquet(data_dir / "train-00000-of-00001.parquet")

        # Create a fake zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data/train-00000-of-00001.parquet", "fake content")

        def fake_urlretrieve(url, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(zip_path.read_bytes())

        with patch(
            "docs2synth.datasets.downloader.urlretrieve", side_effect=fake_urlretrieve
        ):
            # Mock the parquet extraction to raise an exception
            with patch(
                "docs2synth.datasets.parquet_loader.load_parquet_dataset"
            ) as mock_load:
                mock_load.side_effect = Exception("Parquet extraction failed")

                # Should not raise error, just log warning
                result = download_dataset(
                    "vrd-iu2024-tracka", tmpdir, extract_images=True
                )
                assert result.exists()


def test_extract_images_from_parquet_with_invalid_image_format():
    """Test extracting images with unexpected image data format (mock test)."""
    from unittest.mock import patch

    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file with valid data
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        test_image_bytes = b"fake image data"
        test_data = pd.DataFrame(
            {
                "image": [test_image_bytes],
                "ground_truth": [{"key": "value1"}],
            }
        )
        test_data.to_parquet(parquet_path)

        # Mock the dataframe iteration to return an invalid format
        original_read = pd.read_parquet

        def mock_read_parquet(path):
            df = original_read(path)
            # Create a mock dataframe that will have invalid image data
            mock_df = pd.DataFrame(
                {
                    "image": [12345, test_image_bytes],  # First row invalid
                    "ground_truth": [{"key": "value1"}, {"key": "value2"}],
                }
            )
            # Copy attributes from real df
            mock_df.columns = df.columns
            return mock_df

        with patch("pandas.read_parquet", side_effect=mock_read_parquet):
            with patch("docs2synth.datasets.parquet_loader.logger") as mock_logger:
                # This should log a warning for the invalid row
                extract_images_from_parquet(parquet_path, output_dir)

                # Check that warning was logged for invalid format
                warning_calls = [call for call in mock_logger.warning.call_args_list]
                assert any(
                    "unexpected image data format" in str(call).lower()
                    for call in warning_calls
                )


def test_extract_images_from_parquet_progress_logging():
    """Test that progress is logged every 100 images."""
    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file with many rows
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        test_image_bytes = b"fake image data"
        # Create 101 rows to trigger progress logging
        test_data = pd.DataFrame(
            {
                "image": [test_image_bytes] * 101,
            }
        )
        test_data.to_parquet(parquet_path)

        # Extract images
        with patch("docs2synth.datasets.parquet_loader.logger") as mock_logger:
            results = extract_images_from_parquet(parquet_path, output_dir)

            # Verify progress was logged (should log at index 100)
            assert len(results) == 101
            # Check that progress logging occurred
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            progress_logged = any("Processed 100/" in str(call) for call in info_calls)
            assert progress_logged


def test_extract_images_from_parquet_with_json_decode_error():
    """Test extracting images with ground truth that fails JSON parsing."""
    from docs2synth.datasets.parquet_loader import extract_images_from_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test parquet file
        parquet_path = Path(tmpdir) / "test.parquet"
        output_dir = Path(tmpdir) / "images"

        test_image_bytes = b"fake image data"
        # Create test data with invalid JSON string that can't be parsed
        test_data = pd.DataFrame(
            {
                "image": [test_image_bytes],
                "ground_truth": ["not valid json {"],
            }
        )
        test_data.to_parquet(parquet_path)

        # Extract images - should keep string as-is when JSON parsing fails
        results = extract_images_from_parquet(parquet_path, output_dir)

        assert len(results) == 1
        # Ground truth should remain as string since JSON parsing failed
        assert results[0]["ground_truth"] == "not valid json {"
