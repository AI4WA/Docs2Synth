"""Logging utilities for Docs2Synth.

This module provides centralized logging configuration and utilities for
consistent logging across the package.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)


def setup_logging(
    level: int | str = logging.INFO,
    format_string: str = DEFAULT_FORMAT,
    log_file: str | Path | None = None,
    console: bool = True,
) -> None:
    """Set up logging configuration for the package.

    Args:
        level: Logging level (e.g., logging.INFO, "DEBUG")
        format_string: Format string for log messages
        log_file: Optional file path to write logs to
        console: Whether to log to console (stdout)

    Example:
        >>> setup_logging(level="DEBUG", log_file="docs2synth.log")
        >>> logging.info("Application started")
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Get root logger
    root_logger = logging.getLogger("Docs2Synth")
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
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
    return logging.getLogger(f"Docs2Synth.{name}")


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
        self.logger = logger or logging.getLogger("Docs2Synth")
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


def setup_cli_logging(verbose: int = 0) -> None:
    """Set up logging for CLI commands.

    Args:
        verbose: Verbosity level (0=INFO, 1=DEBUG, 2+=DEBUG with details)

    Example:
        >>> # In CLI
        >>> setup_cli_logging(verbose=args.verbose)
    """
    if verbose == 0:
        level = logging.INFO
        format_string = SIMPLE_FORMAT
    elif verbose == 1:
        level = logging.DEBUG
        format_string = DEFAULT_FORMAT
    else:
        level = logging.DEBUG
        format_string = DETAILED_FORMAT

    setup_logging(level=level, format_string=format_string, console=True)

    # Quiet third-party loggers
    if verbose < 2:
        configure_third_party_loggers(logging.WARNING)
