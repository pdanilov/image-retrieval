"""The diagnostics in `scripts/` must at least import.

They sit outside the package and nothing else exercises them, so a refactor that renames
something they use breaks them silently -- which happened once, when `REPOS` became
`SPECS`. Importing each one catches that class of break without running any of them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


def test_there_are_scripts_to_check():
    assert SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_script_imports(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
