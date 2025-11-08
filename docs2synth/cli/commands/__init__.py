"""CLI command modules."""

from docs2synth.cli.commands.agent import agent_group
from docs2synth.cli.commands.datasets import datasets
from docs2synth.cli.commands.preprocess import preprocess
from docs2synth.cli.commands.qa import qa_group
from docs2synth.cli.commands.verify import verify_group

__all__ = ["agent_group", "datasets", "preprocess", "qa_group", "verify_group"]
