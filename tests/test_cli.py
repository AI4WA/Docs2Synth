"""Tests for CLI commands."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from docs2synth.cli import cli, main


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = MagicMock()
    config.get.return_value = "./data/datasets"
    return config


class TestCLIMain:
    """Tests for main CLI command."""

    def test_cli_version_option(self, runner):
        """Test --version flag displays version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_help(self, runner):
        """Test --help flag displays help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Docs2Synth" in result.output
        assert "datasets" in result.output

    def test_cli_with_config_file(self, runner, tmp_path):
        """Test CLI with config file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("data:\n  datasets_dir: ./test_data\n")

        with patch("docs2synth.cli.load_config") as mock_load:
            mock_load.return_value = MagicMock()
            result = runner.invoke(
                cli, ["--config", str(config_file), "datasets", "list"]
            )
            assert result.exit_code == 0
            mock_load.assert_called_once()

    def test_cli_verbose_flag(self, runner):
        """Test -v flag increases verbosity."""
        with patch("docs2synth.cli.setup_cli_logging") as mock_setup:
            result = runner.invoke(cli, ["-v", "datasets", "list"])
            assert result.exit_code == 0
            mock_setup.assert_called_once()

    def test_cli_multiple_verbose_flags(self, runner):
        """Test -vv flag increases verbosity further."""
        with patch("docs2synth.cli.setup_cli_logging") as mock_setup:
            result = runner.invoke(cli, ["-vv", "datasets", "list"])
            assert result.exit_code == 0
            mock_setup.assert_called_once()


class TestDatasetsCommand:
    """Tests for datasets command."""

    def test_datasets_list(self, runner):
        """Test listing available datasets."""
        result = runner.invoke(cli, ["datasets", "list"])
        assert result.exit_code == 0
        assert "Available datasets:" in result.output
        # Should list some dataset names
        assert "cord" in result.output or "funsd" in result.output

    def test_datasets_download_missing_name(self, runner):
        """Test download without dataset name fails."""
        result = runner.invoke(cli, ["datasets", "download"])
        assert result.exit_code == 1
        assert "Error: NAME required for download" in result.output

    def test_datasets_download_single_dataset(self, runner, tmp_path):
        """Test downloading a single dataset."""
        with patch("docs2synth.datasets.downloader.download_dataset") as mock_download:
            mock_download.return_value = tmp_path / "cord"
            result = runner.invoke(
                cli, ["datasets", "download", "cord", "--output-dir", str(tmp_path)]
            )
            assert result.exit_code == 0
            assert "Downloading cord" in result.output
            assert "Dataset saved to" in result.output
            mock_download.assert_called_once_with("cord", str(tmp_path))

    def test_datasets_download_single_no_output_dir(self, runner, tmp_path):
        """Test downloading dataset without specifying output dir."""
        with patch("docs2synth.datasets.downloader.download_dataset") as mock_download:
            mock_download.return_value = tmp_path / "cord"
            result = runner.invoke(cli, ["datasets", "download", "cord"])
            assert result.exit_code == 0
            assert "Downloading cord..." in result.output
            mock_download.assert_called_once_with("cord", None)

    def test_datasets_download_all(self, runner, tmp_path):
        """Test downloading all datasets."""
        with patch("docs2synth.datasets.downloader.download_dataset") as mock_download:
            with patch(
                "docs2synth.datasets.downloader.DATASETS",
                {"cord": "url1", "funsd": "url2"},
            ):
                mock_download.return_value = tmp_path / "dataset"
                result = runner.invoke(cli, ["datasets", "download", "all"])
                assert result.exit_code == 0
                assert "Downloading all datasets" in result.output
                assert "All datasets downloaded!" in result.output
                # Should be called for each dataset
                assert mock_download.call_count == 2

    def test_datasets_download_all_with_output_dir(self, runner, tmp_path):
        """Test downloading all datasets with output dir."""
        with patch("docs2synth.datasets.downloader.download_dataset") as mock_download:
            with patch("docs2synth.datasets.downloader.DATASETS", {"cord": "url1"}):
                mock_download.return_value = tmp_path / "dataset"
                result = runner.invoke(
                    cli, ["datasets", "download", "all", "--output-dir", str(tmp_path)]
                )
                assert result.exit_code == 0
                assert f"Downloading all datasets to {tmp_path}" in result.output

    def test_datasets_download_invalid_name(self, runner):
        """Test downloading invalid dataset name."""
        with patch("docs2synth.datasets.downloader.download_dataset") as mock_download:
            mock_download.side_effect = ValueError("Unknown dataset: invalid")
            result = runner.invoke(cli, ["datasets", "download", "invalid"])
            assert result.exit_code == 1
            assert "Error: Unknown dataset: invalid" in result.output

    def test_datasets_download_general_exception(self, runner):
        """Test handling general exceptions during download."""
        with patch("docs2synth.datasets.downloader.download_dataset") as mock_download:
            mock_download.side_effect = RuntimeError("Network error")
            result = runner.invoke(cli, ["datasets", "download", "cord"])
            assert result.exit_code == 1
            assert "Error: Network error" in result.output


class TestMainFunction:
    """Tests for main entry point function."""

    def test_main_with_no_args(self):
        """Test main function with no arguments."""
        with patch("docs2synth.cli.cli") as mock_cli:
            with patch.object(sys, "argv", ["docs2synth"]):
                main()
                mock_cli.assert_called_once()

    def test_main_with_args(self):
        """Test main function with custom arguments."""
        with patch("docs2synth.cli.cli") as mock_cli:
            main(["datasets", "list"])
            mock_cli.assert_called_once_with(args=["datasets", "list"])
