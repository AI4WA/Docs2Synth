"""Tests for timing utilities."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from docs2synth.utils.timer import Timer, format_time, timer, timeit


class TestTimerContextManager:
    """Tests for timer context manager."""

    def test_timer_basic_usage(self):
        """Test timer context manager basic usage."""
        with timer("test operation") as timing_info:
            time.sleep(0.01)

        assert "name" in timing_info
        assert "elapsed" in timing_info
        assert timing_info["name"] == "test operation"
        assert timing_info["elapsed"] > 0

    def test_timer_records_elapsed_time(self):
        """Test timer accurately records elapsed time."""
        with timer("test") as timing_info:
            time.sleep(0.05)

        # Should be approximately 0.05 seconds (with some tolerance)
        assert 0.04 < timing_info["elapsed"] < 0.1

    def test_timer_logs_completion(self):
        """Test timer logs completion message."""
        logger = logging.getLogger("docs2synth.utils.timer")

        with patch.object(logger, "log") as mock_log:
            with timer("test operation", log_level=logging.DEBUG):
                pass

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.DEBUG
            assert "test operation" in call_args[0][1]
            assert "completed" in call_args[0][1]

    def test_timer_executes_even_with_exception(self):
        """Test timer logs even when exception is raised."""
        logger = logging.getLogger("docs2synth.utils.timer")

        with patch.object(logger, "log") as mock_log:
            with pytest.raises(ValueError):
                with timer("failing operation"):
                    raise ValueError("Test error")

            # Should still log despite exception
            mock_log.assert_called_once()

    def test_timer_custom_log_level(self):
        """Test timer with custom log level."""
        logger = logging.getLogger("docs2synth.utils.timer")

        with patch.object(logger, "log") as mock_log:
            with timer("test", log_level=logging.WARNING):
                pass

            assert mock_log.call_args[0][0] == logging.WARNING


class TestTimeitDecorator:
    """Tests for timeit decorator."""

    def test_timeit_basic_usage(self):
        """Test timeit decorator basic usage."""

        @timeit
        def sample_function():
            time.sleep(0.01)
            return "result"

        result = sample_function()
        assert result == "result"

    def test_timeit_with_arguments(self):
        """Test timeit decorator with function arguments."""

        @timeit
        def add_numbers(a, b):
            return a + b

        result = add_numbers(2, 3)
        assert result == 5

    def test_timeit_with_kwargs(self):
        """Test timeit decorator with keyword arguments."""

        @timeit
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World"

    def test_timeit_logs_execution(self):
        """Test timeit logs function execution."""
        logger = logging.getLogger("docs2synth.utils.timer")

        @timeit
        def sample_function():
            pass

        with patch.object(logger, "log") as mock_log:
            sample_function()
            mock_log.assert_called_once()
            assert "sample_function" in mock_log.call_args[0][1]

    def test_timeit_with_log_level_argument(self):
        """Test timeit with custom log level as argument."""
        logger = logging.getLogger("docs2synth.utils.timer")

        @timeit(log_level=logging.DEBUG)
        def sample_function():
            pass

        with patch.object(logger, "log") as mock_log:
            sample_function()
            assert mock_log.call_args[0][0] == logging.DEBUG

    def test_timeit_preserves_function_metadata(self):
        """Test timeit preserves function name and docstring."""

        @timeit
        def documented_function():
            """This is a docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a docstring."

    def test_timeit_handles_exceptions(self):
        """Test timeit still logs when function raises exception."""
        logger = logging.getLogger("docs2synth.utils.timer")

        @timeit
        def failing_function():
            raise ValueError("Test error")

        with patch.object(logger, "log") as mock_log:
            with pytest.raises(ValueError):
                failing_function()
            # Should still log the timing
            mock_log.assert_called_once()


class TestTimerClass:
    """Tests for Timer class."""

    def test_timer_initialization(self):
        """Test Timer class initialization."""
        t = Timer()
        assert isinstance(t._timers, dict)
        assert isinstance(t._results, dict)
        assert len(t._timers) == 0
        assert len(t._results) == 0

    def test_timer_start(self):
        """Test Timer.start method."""
        t = Timer()
        t.start("operation1")
        assert "operation1" in t._timers

    def test_timer_stop_returns_elapsed_time(self):
        """Test Timer.stop returns elapsed time."""
        t = Timer()
        t.start("operation1")
        time.sleep(0.01)
        elapsed = t.stop("operation1")
        assert elapsed > 0
        assert elapsed < 1  # Should be much less than 1 second

    def test_timer_stop_removes_from_active_timers(self):
        """Test Timer.stop removes timer from active timers."""
        t = Timer()
        t.start("operation1")
        t.stop("operation1")
        assert "operation1" not in t._timers

    def test_timer_stop_stores_result(self):
        """Test Timer.stop stores result."""
        t = Timer()
        t.start("operation1")
        elapsed = t.stop("operation1")
        assert "operation1" in t._results
        assert t._results["operation1"] == elapsed

    def test_timer_stop_not_started_raises_error(self):
        """Test Timer.stop raises error if timer not started."""
        t = Timer()
        with pytest.raises(KeyError, match="Timer 'operation1' was not started"):
            t.stop("operation1")

    def test_timer_elapsed(self):
        """Test Timer.elapsed returns stored elapsed time."""
        t = Timer()
        t.start("operation1")
        time.sleep(0.01)
        elapsed = t.stop("operation1")
        assert t.elapsed("operation1") == elapsed

    def test_timer_elapsed_not_completed_raises_error(self):
        """Test Timer.elapsed raises error if timer not completed."""
        t = Timer()
        with pytest.raises(KeyError, match="Timer 'operation1' has no recorded results"):
            t.elapsed("operation1")

    def test_timer_get_all(self):
        """Test Timer.get_all returns all results."""
        t = Timer()
        t.start("op1")
        time.sleep(0.01)
        t.stop("op1")

        t.start("op2")
        time.sleep(0.01)
        t.stop("op2")

        results = t.get_all()
        assert "op1" in results
        assert "op2" in results
        assert results["op1"] > 0
        assert results["op2"] > 0

    def test_timer_get_all_returns_copy(self):
        """Test Timer.get_all returns a copy of results."""
        t = Timer()
        t.start("op1")
        t.stop("op1")

        results1 = t.get_all()
        results1["new_key"] = 123

        results2 = t.get_all()
        assert "new_key" not in results2

    def test_timer_reset(self):
        """Test Timer.reset clears all data."""
        t = Timer()
        t.start("op1")
        t.stop("op1")
        t.start("op2")

        t.reset()

        assert len(t._timers) == 0
        assert len(t._results) == 0

    def test_timer_multiple_operations(self):
        """Test Timer with multiple concurrent operations."""
        t = Timer()
        t.start("op1")
        time.sleep(0.01)
        t.start("op2")
        time.sleep(0.01)

        elapsed1 = t.stop("op1")
        elapsed2 = t.stop("op2")

        # op1 should have more elapsed time since it started first
        assert elapsed1 > elapsed2
        assert "op1" in t._results
        assert "op2" in t._results


class TestFormatTime:
    """Tests for format_time function."""

    def test_format_time_microseconds(self):
        """Test format_time with microseconds."""
        result = format_time(0.0001)  # 0.1ms = 100µs
        assert "µs" in result
        assert result == "100.00µs"

    def test_format_time_milliseconds(self):
        """Test format_time with milliseconds."""
        result = format_time(0.123)  # 123ms
        assert "ms" in result
        assert result == "123.00ms"

    def test_format_time_seconds(self):
        """Test format_time with seconds."""
        result = format_time(5.5)
        assert result == "5.50s"

    def test_format_time_minutes(self):
        """Test format_time with minutes."""
        result = format_time(65.5)  # 1 minute 5.5 seconds
        assert "1m" in result
        assert "5.50s" in result

    def test_format_time_hours(self):
        """Test format_time with hours."""
        result = format_time(3665)  # 1 hour, 1 minute, 5 seconds
        assert "1h" in result
        assert "1m" in result
        assert "5.00s" in result

    def test_format_time_edge_cases(self):
        """Test format_time edge cases."""
        # Just under 1ms
        result = format_time(0.0009)
        assert "µs" in result

        # Just under 1 second
        result = format_time(0.999)
        assert "ms" in result

        # Just under 1 minute
        result = format_time(59.9)
        assert result == "59.90s"

        # Just under 1 hour
        result = format_time(3599)  # 59 minutes 59 seconds
        assert "59m" in result

    def test_format_time_zero(self):
        """Test format_time with zero."""
        result = format_time(0)
        assert "µs" in result

    def test_format_time_very_small(self):
        """Test format_time with very small values."""
        result = format_time(0.000001)  # 1µs
        assert "µs" in result
