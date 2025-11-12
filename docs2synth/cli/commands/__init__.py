"""CLI command modules."""

from docs2synth.cli.commands.agent import agent_group
from docs2synth.cli.commands.annotation import annotate_command
from docs2synth.cli.commands.datasets import datasets
from docs2synth.cli.commands.preprocess import preprocess
from docs2synth.cli.commands.qa import qa_group
from docs2synth.cli.commands.retriever import retriever_group
from docs2synth.cli.commands.verify import verify_group

__all__ = [
    "agent_group",
    "annotate_command",
    "datasets",
    "preprocess",
    "qa_group",
    "retriever_group",
    "verify_group",
]
