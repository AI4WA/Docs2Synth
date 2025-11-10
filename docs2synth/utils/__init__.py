"""Utility functions and helpers for Docs2Synth.

This module provides common utilities including:
- Logging configuration and utilities
- Performance timing tools
- PDF image management utilities
"""

from __future__ import annotations

from docs2synth.utils.config import Config, get_config, load_config, set_config
from docs2synth.utils.logging import (
    LoggerContext,
    ProgressLogger,
    configure_third_party_loggers,
    get_logger,
    log_function_call,
    setup_cli_logging,
    setup_logging,
    setup_logging_from_config,
)
from docs2synth.utils.pdf_images import (
    convert_pdf_to_images,
    convert_pdfs_in_directory,
    get_pdf_images,
    get_pdf_images_dir,
    is_pdf_images_dir,
    save_pdf_images,
)
from docs2synth.utils.text import DEFAULT_MAX_CONTEXT_LENGTH, truncate_context
from docs2synth.utils.timer import Timer, format_time, timeit, timer

__all__ = [
    # Config
    "Config",
    "get_config",
    "load_config",
    "set_config",
    # Logging
    "setup_logging",
    "setup_logging_from_config",
    "get_logger",
    "LoggerContext",
    "configure_third_party_loggers",
    "log_function_call",
    "ProgressLogger",
    "setup_cli_logging",
    # Timing
    "timer",
    "timeit",
    "Timer",
    "format_time",
    # PDF Images
    "get_pdf_images",
    "get_pdf_images_dir",
    "is_pdf_images_dir",
    "save_pdf_images",
    "convert_pdf_to_images",
    "convert_pdfs_in_directory",
    # Text Processing
    "truncate_context",
    "DEFAULT_MAX_CONTEXT_LENGTH",
]
