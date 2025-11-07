"""Basic import tests to ensure the package structure is sound."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "docs2synth",
        "docs2synth.qa",
        "docs2synth.retriever",
        "docs2synth.agent",
    ],
)
def test_import_module(module_name):
    assert importlib.import_module(module_name) is not None
