"""Tests for retriever metrics."""

from docs2synth.retriever.metrics import (
    ANLS_THRESHOLD,
    _levenshtein_distance,
    calculate_anls,
)


def test_calculate_anls_perfect_match() -> None:
    """Test ANLS with perfect match returns 1.0."""
    assert calculate_anls("hello world", "hello world") == 1.0
    assert calculate_anls("test", "test") == 1.0


def test_calculate_anls_case_insensitive() -> None:
    """Test ANLS is case-insensitive."""
    assert calculate_anls("Hello World", "hello world") == 1.0
    assert calculate_anls("TEST", "test") == 1.0
    assert calculate_anls("MiXeD CaSe", "mixed case") == 1.0


def test_calculate_anls_whitespace_normalization() -> None:
    """Test ANLS normalizes whitespace."""
    assert calculate_anls("  hello  ", "hello") == 1.0
    assert calculate_anls("hello", "  hello  ") == 1.0
    assert calculate_anls("  test  ", "  test  ") == 1.0


def test_calculate_anls_empty_strings() -> None:
    """Test ANLS with empty strings."""
    # Both empty
    assert calculate_anls("", "") == 1.0
    # Only pred empty
    assert calculate_anls("", "hello") == 0.0
    # Only gt empty (pred is not empty, should return 0.0)
    assert calculate_anls("hello", "") == 0.0


def test_calculate_anls_completely_different() -> None:
    """Test ANLS with completely different strings returns 0.0."""
    score = calculate_anls("hello", "world")
    # With default threshold 0.5, distance is 4/5=0.8 > 0.5, so score is 0
    assert score == 0.0


def test_calculate_anls_similar_strings() -> None:
    """Test ANLS with similar strings."""
    # One character difference
    score = calculate_anls("hello", "hallo")
    assert 0.0 < score <= 1.0

    # Two character difference
    score = calculate_anls("hello", "hillo")
    assert 0.0 < score <= 1.0


def test_calculate_anls_threshold() -> None:
    """Test ANLS respects custom threshold."""
    # With higher threshold, more leniency
    score_high = calculate_anls("hello", "world", threshold=1.0)
    score_default = calculate_anls("hello", "world", threshold=0.5)

    # Higher threshold should give non-zero score
    assert score_high > score_default


def test_calculate_anls_partial_match() -> None:
    """Test ANLS with partial matches."""
    # Substring
    score = calculate_anls("hello", "hello world")
    assert 0.0 <= score <= 1.0

    # Prefix match
    score = calculate_anls("test", "testing")
    assert 0.0 < score < 1.0


def test_calculate_anls_default_threshold() -> None:
    """Test ANLS uses ANLS_THRESHOLD by default."""
    # This should be equivalent to explicit threshold
    score1 = calculate_anls("hello", "hallo")
    score2 = calculate_anls("hello", "hallo", threshold=ANLS_THRESHOLD)
    assert score1 == score2


def test_levenshtein_distance_identical() -> None:
    """Test Levenshtein distance for identical strings."""
    assert _levenshtein_distance("hello", "hello") == 0
    assert _levenshtein_distance("test", "test") == 0
    assert _levenshtein_distance("", "") == 0


def test_levenshtein_distance_empty_strings() -> None:
    """Test Levenshtein distance with empty strings."""
    assert _levenshtein_distance("", "hello") == 5
    assert _levenshtein_distance("hello", "") == 5
    assert _levenshtein_distance("test", "") == 4


def test_levenshtein_distance_single_substitution() -> None:
    """Test Levenshtein distance with single character substitution."""
    assert _levenshtein_distance("hello", "hallo") == 1
    assert _levenshtein_distance("test", "text") == 1


def test_levenshtein_distance_single_insertion() -> None:
    """Test Levenshtein distance with single insertion."""
    assert _levenshtein_distance("hello", "helloo") == 1
    assert _levenshtein_distance("test", "tests") == 1


def test_levenshtein_distance_single_deletion() -> None:
    """Test Levenshtein distance with single deletion."""
    assert _levenshtein_distance("hello", "helo") == 1
    assert _levenshtein_distance("test", "tes") == 1


def test_levenshtein_distance_multiple_operations() -> None:
    """Test Levenshtein distance with multiple operations."""
    assert _levenshtein_distance("kitten", "sitting") == 3
    assert _levenshtein_distance("saturday", "sunday") == 3


def test_levenshtein_distance_completely_different() -> None:
    """Test Levenshtein distance with completely different strings."""
    distance = _levenshtein_distance("abc", "xyz")
    assert distance == 3  # Three substitutions


def test_levenshtein_distance_order_invariant() -> None:
    """Test Levenshtein distance is symmetric."""
    d1 = _levenshtein_distance("hello", "world")
    d2 = _levenshtein_distance("world", "hello")
    assert d1 == d2

    d1 = _levenshtein_distance("test", "testing")
    d2 = _levenshtein_distance("testing", "test")
    assert d1 == d2


def test_anls_threshold_constant() -> None:
    """Test ANLS_THRESHOLD is defined and has reasonable value."""
    assert isinstance(ANLS_THRESHOLD, float)
    assert 0.0 <= ANLS_THRESHOLD <= 1.0
    assert ANLS_THRESHOLD == 0.5


def test_calculate_anls_normalized_distance() -> None:
    """Test ANLS calculation and normalization."""
    # Distance of 1 in string of length 5: normalized = 1/5 = 0.2
    # ANLS score = 1 - 0.2 = 0.8
    score = calculate_anls("hello", "hallo", threshold=1.0)
    assert abs(score - 0.8) < 0.01


def test_calculate_anls_threshold_boundary() -> None:
    """Test ANLS behavior at threshold boundary."""
    # Create strings with known normalized distance
    # "hello" vs "hallo": distance=1, max_len=5, normalized=0.2
    # With threshold=0.3, normalized_distance(0.2) <= threshold, should get score
    score = calculate_anls("hello", "hallo", threshold=0.3)
    assert score > 0.0

    # With threshold=0.1, normalized_distance(0.2) > threshold, should get 0
    score = calculate_anls("hello", "hallo", threshold=0.1)
    assert score == 0.0
