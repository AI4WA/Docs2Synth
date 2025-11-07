"""Question‐Answer pair generation utilities.

This subpackage provides different strategies for generating questions from document content.
Each strategy implements a specific approach to question generation.
"""

from docs2synth.qa.config import QAGenerationConfig, QAStrategyConfig
from docs2synth.qa.generators import (
    BaseQAGenerator,
    LayoutAwareQAGenerator,
    LogicalAwareQAGenerator,
    SemanticQAGenerator,
)
from docs2synth.qa.strategies import QAGeneratorFactory

__all__ = [
    "BaseQAGenerator",
    "SemanticQAGenerator",
    "LayoutAwareQAGenerator",
    "LogicalAwareQAGenerator",
    "QAGeneratorFactory",
    "QAStrategyConfig",
    "QAGenerationConfig",
]
