"""Question‐Answer pair generation utilities.

This subpackage provides different strategies for generating questions from document content.
Each strategy implements a specific approach to question generation.
It also provides verifiers for validating QA pairs.
"""

from docs2synth.qa.config import (
    QAGenerationConfig,
    QAStrategyConfig,
    QAVerificationConfig,
    QAVerifierConfig,
)
from docs2synth.qa.generators import (
    BaseQAGenerator,
    LayoutAwareQAGenerator,
    LogicalAwareQAGenerator,
    SemanticQAGenerator,
)
from docs2synth.qa.strategies import QAGeneratorFactory
from docs2synth.qa.verifiers import (
    BaseQAVerifier,
    CorrectnessVerifier,
    MeaningfulVerifier,
    QAVerifierFactory,
)

__all__ = [
    # Generators
    "BaseQAGenerator",
    "SemanticQAGenerator",
    "LayoutAwareQAGenerator",
    "LogicalAwareQAGenerator",
    "QAGeneratorFactory",
    # Config
    "QAStrategyConfig",
    "QAGenerationConfig",
    "QAVerifierConfig",
    "QAVerificationConfig",
    # Verifiers
    "BaseQAVerifier",
    "CorrectnessVerifier",
    "MeaningfulVerifier",
    "QAVerifierFactory",
]
