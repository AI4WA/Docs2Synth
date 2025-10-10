"""Timing utilities for performance monitoring.

This module provides decorators and context managers for tracking execution time
of functions and code blocks.
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


@contextmanager
def timer(
    name: str = "operation", log_level: int = logging.INFO
) -> Generator[dict[str, Any], None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name of the operation being timed
        log_level: Logging level for the timing message

    Yields:
        Dict with timing information (updated after block completes)

    Example:
        >>> with timer("data processing"):
        ...     process_data()
        INFO: data processing completed in 2.34s
    """
    timing_info = {"name": name, "elapsed": 0.0}
    start_time = time.perf_counter()

    try:
        yield timing_info
    finally:
        elapsed = time.perf_counter() - start_time
        timing_info["elapsed"] = elapsed
        logger.log(log_level, f"{name} completed in {elapsed:.2f}s")


def timeit(func: Callable | None = None, *, log_level: int = logging.INFO) -> Callable:
    """Decorator for timing function execution.

    Can be used with or without arguments.

    Args:
        func: Function to be timed (when used without arguments)
        log_level: Logging level for the timing message

    Returns:
        Decorated function

    Example:
        >>> @timeit
        ... def slow_function():
        ...     time.sleep(1)
        ...
        >>> slow_function()
        INFO: slow_function completed in 1.00s

        >>> @timeit(log_level=logging.DEBUG)
        ... def another_function():
        ...     pass
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f.__name__
            start_time = time.perf_counter()

            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                logger.log(log_level, f"{func_name} completed in {elapsed:.2f}s")

        return wrapper

    # Handle both @timeit and @timeit() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


class Timer:
    """Reusable timer class for tracking multiple operations.

    Example:
        >>> timer = Timer()
        >>> timer.start("operation1")
        >>> # do work
        >>> timer.stop("operation1")
        >>> print(f"Elapsed: {timer.elapsed('operation1'):.2f}s")
    """

    def __init__(self) -> None:
        """Initialize the timer."""
        self._timers: dict[str, float] = {}
        self._results: dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing an operation.

        Args:
            name: Name of the operation
        """
        self._timers[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing an operation and return elapsed time.

        Args:
            name: Name of the operation

        Returns:
            Elapsed time in seconds

        Raises:
            KeyError: If timer was not started
        """
        if name not in self._timers:
            raise KeyError(f"Timer '{name}' was not started")

        elapsed = time.perf_counter() - self._timers[name]
        self._results[name] = elapsed
        del self._timers[name]

        return elapsed

    def elapsed(self, name: str) -> float:
        """Get elapsed time for a completed operation.

        Args:
            name: Name of the operation

        Returns:
            Elapsed time in seconds

        Raises:
            KeyError: If timer was not completed
        """
        if name not in self._results:
            raise KeyError(f"Timer '{name}' has no recorded results")

        return self._results[name]

    def get_all(self) -> dict[str, float]:
        """Get all recorded timing results.

        Returns:
            Dict mapping operation names to elapsed times
        """
        return self._results.copy()

    def reset(self) -> None:
        """Clear all timing data."""
        self._timers.clear()
        self._results.clear()


def format_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string

    Example:
        >>> format_time(0.123)
        '123.00ms'
        >>> format_time(65.5)
        '1m 5.50s'
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"
