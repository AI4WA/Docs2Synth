"""Utility functions and helpers for Docs2Synth.

This module provides common utilities including:
- Logging configuration and utilities
- Performance timing tools
"""

from __future__ import annotations

from Docs2Synth.utils.logging import (
    LoggerContext,
    ProgressLogger,
    configure_third_party_loggers,
    get_logger,
    log_function_call,
    setup_cli_logging,
    setup_logging,
)
from Docs2Synth.utils.timer import (
    Timer,
    format_time,
    timer,
    timeit,
)

__all__ = [
    # Logging
    "setup_logging",
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
]
