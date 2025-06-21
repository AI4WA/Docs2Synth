"""Basic import tests to ensure the package structure is sound."""

import importlib
import pytest

@pytest.mark.parametrize("module_name", [
    "Docs2Synth",
    "Docs2Synth.qa",
    "Docs2Synth.retriever",
])
def test_import_module(module_name):
    assert importlib.import_module(module_name) is not None 