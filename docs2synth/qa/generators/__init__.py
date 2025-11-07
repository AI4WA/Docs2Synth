"""QA generation strategies.

This module provides different strategies for generating questions from document content.
Each strategy implements a specific approach to question generation.
"""

from docs2synth.qa.generators.base import BaseQAGenerator
from docs2synth.qa.generators.layout_aware import LayoutAwareQAGenerator
from docs2synth.qa.generators.logical_aware import LogicalAwareQAGenerator
from docs2synth.qa.generators.semantic import SemanticQAGenerator

__all__ = [
    "BaseQAGenerator",
    "SemanticQAGenerator",
    "LayoutAwareQAGenerator",
    "LogicalAwareQAGenerator",
]
