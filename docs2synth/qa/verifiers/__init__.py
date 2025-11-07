"""QA verification strategies.

This module provides different strategies for verifying QA pairs.
Each strategy implements a specific approach to verification.
"""

from docs2synth.qa.verifiers.base import BaseQAVerifier
from docs2synth.qa.verifiers.correctness import CorrectnessVerifier
from docs2synth.qa.verifiers.factory import QAVerifierFactory
from docs2synth.qa.verifiers.meaningful import MeaningfulVerifier

__all__ = [
    "BaseQAVerifier",
    "CorrectnessVerifier",
    "MeaningfulVerifier",
    "QAVerifierFactory",
]
