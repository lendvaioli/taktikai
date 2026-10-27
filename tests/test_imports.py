import importlib

import pytest

MODULES = [
    "video",
    "detection",
    "tracking",
    "pitch",
    "events",
    "clips",
    "metrics",
    "report",
    "evals",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_with_docstring(name):
    module = importlib.import_module(f"match_tagger.{name}")
    assert module.__doc__ and module.__doc__.strip()
