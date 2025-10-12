"""Tests for logging utilities."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from docs2synth.utils.logging import (
    DEFAULT_FORMAT,
    DETAILED_FORMAT,
    SIMPLE_FORMAT,
    LoggerContext,
    ProgressLogger,
    configure_third_party_loggers,
    get_logger,
    log_function_call,
    setup_cli_logging,
    setup_logging,
    setup_logging_from_config,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_defaults(self):
        """Test setup_logging with default parameters."""
        logger = logging.getLogger("docs2synth.test")
        setup_logging()

        assert logger.level <= logging.INFO
        # Should have at least one handler
        root = logging.getLogger("docs2synth")
        assert len(root.handlers) > 0

    def test_setup_logging_debug_level(self):
        """Test setup_logging with DEBUG level."""
        setup_logging(level=logging.DEBUG)
        logger = logging.getLogger("docs2synth")
        assert logger.level == logging.DEBUG

    def test_setup_logging_string_level(self):
        """Test setup_logging with string level."""
        setup_logging(level="WARNING")
        logger = logging.getLogger("docs2synth")
        assert logger.level == logging.WARNING

    def test_setup_logging_with_log_file(self, tmp_path):
        """Test setup_logging writes to file."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)

        logger = get_logger("test_module")
        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_setup_logging_creates_log_dir(self, tmp_path):
        """Test setup_logging creates parent directories."""
        log_file = tmp_path / "nested" / "dir" / "test.log"
        setup_logging(log_file=log_file)

        logger = get_logger("test_module")
        logger.info("Test message")

        assert log_file.exists()

    def test_setup_logging_console_false(self):
        """Test setup_logging without console output."""
        setup_logging(console=False)
        root = logging.getLogger("docs2synth")

        # Should have no StreamHandler
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
        ]
        # May have file handlers but not stdout stream handlers
        assert all(h.stream != sys.stdout for h in stream_handlers)

    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format."""
        custom_format = "%(name)s - %(message)s"
        setup_logging(format_string=custom_format)

        root = logging.getLogger("docs2synth")
        handler = root.handlers[0]
        assert handler.formatter._fmt == custom_format


class TestSetupLoggingFromConfig:
    """Tests for setup_logging_from_config function."""

    def test_setup_from_config_defaults(self):
        """Test setup_logging_from_config with default config."""
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            "logging.level": "INFO",
            "logging.format": "default",
            "logging.console.level": "INFO",
            "logging.file.level": "DEBUG",
            "logging.console.enabled": True,
            "logging.file.enabled": False,
            "logging.third_party.level": "WARNING",
            "logging.third_party.loggers": ["urllib3"],
        }.get(key, default)

        setup_logging_from_config(mock_config)

        root = logging.getLogger("docs2synth")
        assert len(root.handlers) > 0

    def test_setup_from_config_with_file_handler(self, tmp_path):
        """Test setup_logging_from_config enables file handler."""
        log_file = tmp_path / "app.log"
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            "logging.level": "DEBUG",
            "logging.format": "simple",
            "logging.console.level": "INFO",
            "logging.file.level": "DEBUG",
            "logging.console.enabled": True,
            "logging.file.enabled": True,
            "logging.file.path": str(log_file),
            "logging.file.max_bytes": 1024,
            "logging.file.backup_count": 3,
            "logging.file.use_timestamp": False,
            "logging.third_party.level": "WARNING",
            "logging.third_party.loggers": [],
        }.get(key, default)

        setup_logging_from_config(mock_config)

        logger = get_logger("test")
        logger.debug("Test message")

        assert log_file.exists()

    def test_setup_from_config_with_timestamp(self, tmp_path):
        """Test setup_logging_from_config with timestamp in filename."""
        log_file = tmp_path / "app.log"
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            "logging.level": "DEBUG",
            "logging.format": "default",
            "logging.console.level": "INFO",
            "logging.file.level": "DEBUG",
            "logging.console.enabled": False,
            "logging.file.enabled": True,
            "logging.file.path": str(log_file),
            "logging.file.max_bytes": 1024,
            "logging.file.backup_count": 3,
            "logging.file.use_timestamp": True,
            "logging.third_party.level": "WARNING",
            "logging.third_party.loggers": [],
        }.get(key, default)

        setup_logging_from_config(mock_config)

        # Should create file with timestamp
        log_files = list(tmp_path.glob("app_*.log"))
        assert len(log_files) > 0

    def test_setup_from_config_no_config(self):
        """Test setup_logging_from_config without config loads default."""
        with patch("docs2synth.utils.config.get_config") as mock_get_config:
            mock_config = MagicMock()

            # Set up proper return values for all config.get() calls
            def get_side_effect(key, default=None):
                config_values = {
                    "logging.level": "INFO",
                    "logging.format": "default",
                    "logging.console.level": "INFO",
                    "logging.file.level": "DEBUG",
                    "logging.console.enabled": True,
                    "logging.file.enabled": False,
                    "logging.third_party.level": "WARNING",
                    "logging.third_party.loggers": [],
                }
                return config_values.get(key, default)

            mock_config.get.side_effect = get_side_effect
            mock_get_config.return_value = mock_config

            setup_logging_from_config(None)
            mock_get_config.assert_called_once()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_creates_logger(self):
        """Test get_logger creates logger with correct name."""
        logger = get_logger("my_module")
        assert logger.name == "docs2synth.my_module"

    def test_get_logger_returns_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("my_module")
        logger2 = get_logger("my_module")
        assert logger1 is logger2


class TestLoggerContext:
    """Tests for LoggerContext context manager."""

    def test_logger_context_changes_level(self):
        """Test LoggerContext temporarily changes log level."""
        logger = get_logger("test_context")
        logger.setLevel(logging.INFO)

        with LoggerContext(logger, logging.DEBUG):
            assert logger.level == logging.DEBUG

        assert logger.level == logging.INFO

    def test_logger_context_returns_logger(self):
        """Test LoggerContext returns logger on enter."""
        logger = get_logger("test_context")

        with LoggerContext(logger, logging.DEBUG) as ctx_logger:
            assert ctx_logger is logger


class TestConfigureThirdPartyLoggers:
    """Tests for configure_third_party_loggers function."""

    def test_configure_third_party_loggers_default(self):
        """Test configure_third_party_loggers sets WARNING level."""
        configure_third_party_loggers()

        # Check some common loggers
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("requests").level == logging.WARNING

    def test_configure_third_party_loggers_custom_level(self):
        """Test configure_third_party_loggers with custom level."""
        configure_third_party_loggers(logging.ERROR)

        assert logging.getLogger("urllib3").level == logging.ERROR


class TestLogFunctionCall:
    """Tests for log_function_call decorator."""

    def test_log_function_call_logs_execution(self):
        """Test decorator logs function calls."""
        logger = get_logger("test_decorator")
        logger.setLevel(logging.DEBUG)

        @log_function_call(logger)
        def sample_function(x, y):
            return x + y

        with patch.object(logger, "log") as mock_log:
            result = sample_function(1, 2)
            assert result == 3
            assert mock_log.call_count >= 2  # Called and returned

    def test_log_function_call_with_kwargs(self):
        """Test decorator logs function calls with kwargs."""
        logger = get_logger("test_decorator")
        logger.setLevel(logging.DEBUG)

        @log_function_call(logger)
        def sample_function(x, y=10):
            return x + y

        with patch.object(logger, "log") as mock_log:
            result = sample_function(5, y=15)
            assert result == 20
            assert mock_log.call_count >= 2

    def test_log_function_call_logs_exceptions(self):
        """Test decorator logs exceptions."""
        logger = get_logger("test_decorator")

        @log_function_call(logger)
        def failing_function():
            raise ValueError("Test error")

        with patch.object(logger, "exception") as mock_exception:
            with pytest.raises(ValueError):
                failing_function()
            mock_exception.assert_called_once()


class TestProgressLogger:
    """Tests for ProgressLogger class."""

    def test_progress_logger_initialization(self):
        """Test ProgressLogger initialization."""
        progress = ProgressLogger("Test operation", total=100)
        assert progress.name == "Test operation"
        assert progress.total == 100
        assert progress.current == 0

    def test_progress_logger_update_with_value(self):
        """Test ProgressLogger update with specific value."""
        progress = ProgressLogger("Test", total=100, log_interval=10)
        progress.update(50)
        assert progress.current == 50

    def test_progress_logger_update_increment(self):
        """Test ProgressLogger update increments by 1."""
        progress = ProgressLogger("Test", total=100)
        progress.update()
        assert progress.current == 1
        progress.update()
        assert progress.current == 2

    def test_progress_logger_logs_at_intervals(self):
        """Test ProgressLogger logs at specified intervals."""
        logger = get_logger("test_progress")
        progress = ProgressLogger("Test", total=100, logger=logger, log_interval=20)

        with patch.object(logger, "info") as mock_info:
            progress.update(15)  # 15% - should not log
            assert mock_info.call_count == 0

            progress.update(25)  # 25% - should log
            assert mock_info.call_count == 1

            progress.update(30)  # 30% - should not log (within interval)
            assert mock_info.call_count == 1

            progress.update(50)  # 50% - should log
            assert mock_info.call_count == 2

    def test_progress_logger_complete(self):
        """Test ProgressLogger complete method."""
        logger = get_logger("test_progress")
        progress = ProgressLogger("Test", total=100, logger=logger)

        with patch.object(logger, "info") as mock_info:
            progress.complete()
            mock_info.assert_called_once()
            assert "Completed" in mock_info.call_args[0][0]


class TestSetupCLILogging:
    """Tests for setup_cli_logging function."""

    def test_setup_cli_logging_verbose_0(self):
        """Test setup_cli_logging with verbose=0 (INFO level)."""
        setup_cli_logging(verbose=0)
        logger = logging.getLogger("docs2synth")
        assert logger.level == logging.INFO

    def test_setup_cli_logging_verbose_1(self):
        """Test setup_cli_logging with verbose=1 (DEBUG level)."""
        setup_cli_logging(verbose=1)
        logger = logging.getLogger("docs2synth")
        assert logger.level == logging.DEBUG

    def test_setup_cli_logging_verbose_2(self):
        """Test setup_cli_logging with verbose=2 (DEBUG with details)."""
        setup_cli_logging(verbose=2)
        logger = logging.getLogger("docs2synth")
        assert logger.level == logging.DEBUG

    def test_setup_cli_logging_quiets_third_party(self):
        """Test setup_cli_logging quiets third-party loggers at verbose < 2."""
        setup_cli_logging(verbose=1)
        assert logging.getLogger("urllib3").level == logging.WARNING


class TestFormatStrings:
    """Tests for format string constants."""

    def test_format_strings_defined(self):
        """Test that format strings are properly defined."""
        assert isinstance(SIMPLE_FORMAT, str)
        assert isinstance(DEFAULT_FORMAT, str)
        assert isinstance(DETAILED_FORMAT, str)

        # Simple should be simpler than default
        assert len(SIMPLE_FORMAT) < len(DEFAULT_FORMAT)
        # Detailed should have more info than default
        assert (
            "filename" in DETAILED_FORMAT.lower() or "lineno" in DETAILED_FORMAT.lower()
        )
