"""Tests for retriever dataset utilities."""

from docs2synth.preprocess.schema import QAPair
from docs2synth.retriever.dataset import (
    _extract_verifier_response,
    is_qa_verified,
)


def test_extract_verifier_response_with_response_key() -> None:
    """Test extracting verifier response with Response key."""
    verifier_result = {"Response": "Yes"}
    response = _extract_verifier_response(verifier_result)
    assert response == "yes"

    verifier_result = {"Response": "No"}
    response = _extract_verifier_response(verifier_result)
    assert response == "no"


def test_extract_verifier_response_with_lowercase_key() -> None:
    """Test extracting verifier response with lowercase response key."""
    verifier_result = {"response": "yes"}
    response = _extract_verifier_response(verifier_result)
    assert response == "yes"

    verifier_result = {"response": "no"}
    response = _extract_verifier_response(verifier_result)
    assert response == "no"


def test_extract_verifier_response_case_insensitive() -> None:
    """Test verifier response extraction is case-insensitive."""
    verifier_result = {"Response": "YES"}
    response = _extract_verifier_response(verifier_result)
    assert response == "yes"

    verifier_result = {"Response": "NO"}
    response = _extract_verifier_response(verifier_result)
    assert response == "no"


def test_extract_verifier_response_not_dict() -> None:
    """Test extracting from non-dict returns None."""
    assert _extract_verifier_response("yes") is None
    assert _extract_verifier_response(123) is None
    assert _extract_verifier_response(None) is None
    assert _extract_verifier_response([]) is None


def test_extract_verifier_response_missing_key() -> None:
    """Test extracting when response key is missing."""
    verifier_result = {"other_key": "value"}
    response = _extract_verifier_response(verifier_result)
    # Should fallback to string search
    assert response is None or isinstance(response, str)


def test_extract_verifier_response_fallback_yes() -> None:
    """Test fallback extraction for 'yes' in string representation."""
    verifier_result = {"some_field": "the answer is yes"}
    response = _extract_verifier_response(verifier_result)
    assert response == "yes"


def test_extract_verifier_response_fallback_no() -> None:
    """Test fallback extraction for 'no' in string representation."""
    verifier_result = {"some_field": "no answer"}
    response = _extract_verifier_response(verifier_result)
    assert response == "no"


def test_extract_verifier_response_no_takes_precedence() -> None:
    """Test that 'no' takes precedence over 'yes' in fallback."""
    # String contains both 'yes' and 'no', 'no' should win
    verifier_result = {"field": "yes but actually no"}
    response = _extract_verifier_response(verifier_result)
    assert response == "no"


def test_is_qa_verified_all_yes_require_all() -> None:
    """Test verification when all verifiers say yes (require_all=True)."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={
            "verifier1": {"Response": "yes"},
            "verifier2": {"Response": "yes"},
        },
    )
    assert is_qa_verified(qa_pair, require_all=True) is True


def test_is_qa_verified_one_no_require_all() -> None:
    """Test verification fails when one verifier says no (require_all=True)."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={
            "verifier1": {"Response": "yes"},
            "verifier2": {"Response": "no"},
        },
    )
    assert is_qa_verified(qa_pair, require_all=True) is False


def test_is_qa_verified_all_no_require_all() -> None:
    """Test verification fails when all verifiers say no (require_all=True)."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={
            "verifier1": {"Response": "no"},
            "verifier2": {"Response": "no"},
        },
    )
    assert is_qa_verified(qa_pair, require_all=True) is False


def test_is_qa_verified_at_least_one_yes_require_any() -> None:
    """Test verification passes with at least one yes (require_all=False)."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={
            "verifier1": {"Response": "yes"},
            "verifier2": {"Response": "no"},
        },
    )
    assert is_qa_verified(qa_pair, require_all=False) is True


def test_is_qa_verified_all_no_require_any() -> None:
    """Test verification fails when all say no (require_all=False)."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={
            "verifier1": {"Response": "no"},
            "verifier2": {"Response": "no"},
        },
    )
    assert is_qa_verified(qa_pair, require_all=False) is False


def test_is_qa_verified_no_verification_data() -> None:
    """Test verification fails when no verification data exists."""
    qa_pair = QAPair(question="Q?", answer="A")
    assert is_qa_verified(qa_pair, require_all=True) is False
    assert is_qa_verified(qa_pair, require_all=False) is False


def test_is_qa_verified_empty_verification_dict() -> None:
    """Test verification fails with empty verification dict."""
    qa_pair = QAPair(question="Q?", answer="A", verification={})
    assert is_qa_verified(qa_pair, require_all=True) is False
    assert is_qa_verified(qa_pair, require_all=False) is False


def test_is_qa_verified_single_verifier_yes() -> None:
    """Test verification with single verifier saying yes."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={"verifier1": {"Response": "yes"}},
    )
    assert is_qa_verified(qa_pair, require_all=True) is True
    assert is_qa_verified(qa_pair, require_all=False) is True


def test_is_qa_verified_single_verifier_no() -> None:
    """Test verification with single verifier saying no."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={"verifier1": {"Response": "no"}},
    )
    assert is_qa_verified(qa_pair, require_all=True) is False
    assert is_qa_verified(qa_pair, require_all=False) is False


def test_is_qa_verified_multiple_yes_require_any() -> None:
    """Test verification with multiple yes (require_all=False)."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={
            "verifier1": {"Response": "yes"},
            "verifier2": {"Response": "yes"},
            "verifier3": {"Response": "yes"},
        },
    )
    assert is_qa_verified(qa_pair, require_all=False) is True


def test_is_qa_verified_early_exit_on_no() -> None:
    """Test early exit when finding 'no' with require_all=True."""
    # This tests the optimization path - should return False immediately on first "no"
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={
            "verifier1": {"Response": "no"},  # This should trigger early return
            "verifier2": {"Response": "yes"},
            "verifier3": {"Response": "yes"},
        },
    )
    assert is_qa_verified(qa_pair, require_all=True) is False


def test_is_qa_verified_with_invalid_verifier_response() -> None:
    """Test verification with invalid/unparseable verifier responses."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={
            "verifier1": {"Response": "yes"},
            "verifier2": {"Response": "invalid"},  # Invalid response
        },
    )
    # Invalid responses are not counted as yes
    result = is_qa_verified(qa_pair, require_all=True)
    assert result is False


def test_is_qa_verified_none_response() -> None:
    """Test verification when verifier result is None."""
    qa_pair = QAPair(
        question="Q?",
        answer="A",
        verification={"verifier1": None},
    )
    assert is_qa_verified(qa_pair, require_all=True) is False
    assert is_qa_verified(qa_pair, require_all=False) is False
