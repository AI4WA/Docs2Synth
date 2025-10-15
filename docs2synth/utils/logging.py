"""Logging utilities for Docs2Synth.

This module provides centralized logging configuration and utilities for
consistent logging across the package.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

# Default format always includes filename and line number
LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)


def setup_logging_from_config(config: Any = None) -> None:
    """Set up logging using configuration from config.yml.

    Always logs to both console and file with line numbers included.

    Args:
        config: Config object. If None, loads from default config.

    Example:
        >>> from docs2synth.utils import get_config, setup_logging_from_config
        >>> config = get_config()
        >>> setup_logging_from_config(config)
    """
    if config is None:
        from .config import get_config

        config = get_config()

    # Get logging configuration
    log_level = config.get("logging.level", "INFO")
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Get root logger
    root_logger = logging.getLogger("docs2synth")
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.propagate = False

    # Create formatter with line numbers
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (always enabled)
    log_file_path = config.get("logging.file.path", "./logs/docs2synth.log")
    if not isinstance(log_file_path, (str, Path)):
        log_file_path = "./logs/docs2synth.log"
    max_bytes = config.get("logging.file.max_bytes", 10485760)  # 10MB default
    try:
        max_bytes = int(max_bytes)
    except Exception:
        max_bytes = 10485760
    backup_count = config.get("logging.file.backup_count", 5)
    try:
        backup_count = int(backup_count)
    except Exception:
        backup_count = 5

    # Create log directory if it doesn't exist
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Use RotatingFileHandler for automatic log rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Configure third-party loggers
    third_party_level = config.get("logging.third_party.level", "WARNING")
    if isinstance(third_party_level, str):
        third_party_level = getattr(logging, third_party_level.upper(), logging.WARNING)
    if not isinstance(third_party_level, int):
        third_party_level = logging.WARNING

    third_party_loggers = config.get(
        "logging.third_party.loggers",
        ["urllib3", "requests", "transformers", "torch", "tensorflow", "PIL"],
    )
    if not isinstance(third_party_loggers, (list, tuple)):
        third_party_loggers = [
            "urllib3",
            "requests",
            "transformers",
            "torch",
            "tensorflow",
            "PIL",
        ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(third_party_level)


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path = "./logs/docs2synth.log",
) -> None:
    """Set up logging configuration for the package.

    Always logs to both console and file with line numbers included.

    Args:
        level: Logging level (e.g., logging.INFO, "DEBUG")
        log_file: File path to write logs to (default: ./logs/docs2synth.log)

    Example:
        >>> setup_logging(level="DEBUG", log_file="docs2synth.log")
        >>> logging.info("Application started")
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Get root logger
    root_logger = logging.getLogger("docs2synth")
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.propagate = False

    # Create formatter with line numbers
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (always enabled)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(f"docs2synth.{name}")


class LoggerContext:
    """Context manager for temporarily changing log level.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LoggerContext(logger, logging.DEBUG):
        ...     logger.debug("This will be logged")
        >>> logger.debug("This won't be logged if level was higher")
    """

    def __init__(self, logger: logging.Logger, level: int):
        """Initialize context manager.

        Args:
            logger: Logger to modify
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level

    def __enter__(self) -> logging.Logger:
        """Enter context and change log level."""
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore original log level."""
        self.logger.setLevel(self.old_level)


def configure_third_party_loggers(level: int = logging.WARNING) -> None:
    """Configure logging for common third-party libraries.

    Reduces noise from verbose third-party libraries by setting
    their log level to WARNING or higher.

    Args:
        level: Logging level for third-party loggers

    Example:
        >>> configure_third_party_loggers(logging.ERROR)
    """
    noisy_loggers = [
        "urllib3",
        "requests",
        "transformers",
        "torch",
        "tensorflow",
        "PIL",
        "matplotlib",
        "openai",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(level)


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator to log function calls with arguments.

    Args:
        logger: Logger instance to use
        level: Logging level for the message

    Example:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        ... def process_data(data):
        ...     return len(data)
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            logger.log(level, f"Calling {func_name}({signature})")

            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func_name} returned {result!r}")
                return result
            except Exception as e:
                logger.exception(f"{func_name} raised {type(e).__name__}: {e}")
                raise

        return wrapper

    return decorator


class ProgressLogger:
    """Logger for tracking progress of long-running operations.

    Example:
        >>> progress = ProgressLogger("Processing documents", total=100)
        >>> for i in range(100):
        ...     progress.update(i + 1)
        ...     # do work
        >>> progress.complete()
    """

    def __init__(
        self,
        name: str,
        total: int,
        logger: logging.Logger | None = None,
        log_interval: int = 10,
    ):
        """Initialize progress logger.

        Args:
            name: Name of the operation
            total: Total number of items to process
            logger: Logger to use (defaults to root Docs2Synth logger)
            log_interval: Log every N% progress
        """
        self.name = name
        self.total = total
        self.logger = logger or logging.getLogger("docs2synth")
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_percent = 0

    def update(self, current: int | None = None) -> None:
        """Update progress.

        Args:
            current: Current count (if None, increments by 1)
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1

        percent = int((self.current / self.total) * 100)

        # Log at intervals
        if percent >= self.last_logged_percent + self.log_interval:
            self.logger.info(f"{self.name}: {self.current}/{self.total} ({percent}%)")
            self.last_logged_percent = percent

    def complete(self) -> None:
        """Mark operation as complete."""
        self.logger.info(f"{self.name}: Completed ({self.total}/{self.total})")


def setup_cli_logging(verbose: int = 0, config: Any = None) -> None:
    """Set up logging for CLI commands.

    Always logs to both console and file with line numbers included.
    The verbose flag controls the log level.

    Args:
        verbose: Verbosity level (0=INFO, 1+=DEBUG)
        config: Config object. If None, loads from default config.

    Example:
        >>> # In CLI
        >>> setup_cli_logging(verbose=args.verbose, config=config)
    """
    # Load config if not provided
    if config is None:
        from .config import get_config

        config = get_config()

    # Determine log level based on verbose flag (robust to non-int verbose)
    try:
        log_level = logging.DEBUG if int(verbose) > 0 else logging.INFO
    except Exception:
        log_level = logging.INFO

    # Get root logger
    root_logger = logging.getLogger("docs2synth")
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.propagate = False

    # Create formatter with line numbers
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (always enabled)
    log_file_path = config.get("logging.file.path", "./logs/docs2synth.log")
    if not isinstance(log_file_path, (str, Path)):
        log_file_path = "./logs/docs2synth.log"
    max_bytes = config.get("logging.file.max_bytes", 10485760)  # 10MB default
    try:
        max_bytes = int(max_bytes)
    except Exception:
        max_bytes = 10485760
    backup_count = config.get("logging.file.backup_count", 5)
    try:
        backup_count = int(backup_count)
    except Exception:
        backup_count = 5

    # Create log directory if it doesn't exist
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Use RotatingFileHandler for automatic log rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Configure third-party loggers to reduce noise
    third_party_level = config.get("logging.third_party.level", "WARNING")
    if isinstance(third_party_level, str):
        third_party_level = getattr(logging, third_party_level.upper(), logging.WARNING)
    if not isinstance(third_party_level, int):
        third_party_level = logging.WARNING

    third_party_loggers = config.get(
        "logging.third_party.loggers",
        ["urllib3", "requests", "transformers", "torch", "tensorflow", "PIL"],
    )
    if not isinstance(third_party_loggers, (list, tuple)):
        third_party_loggers = [
            "urllib3",
            "requests",
            "transformers",
            "torch",
            "tensorflow",
            "PIL",
        ]
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(third_party_level)
