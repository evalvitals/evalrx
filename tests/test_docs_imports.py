"""Every `from evalrx... import ...` in the docs' Python examples resolves.

The examples mostly need a model, so they can't run here; but a renamed or
removed name is the drift readers hit first, and that much is checkable.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
IMPORT = re.compile(r"^\s*from (evalrx[\w.]*) import (\([^)]*\)|[^\n]+)", re.M)


def _imports() -> list[tuple[str, str, str]]:
    found = []
    for doc in DOCS:
        for block in BLOCK.findall(doc.read_text(encoding="utf-8")):
            for match in IMPORT.finditer(block):
                statement = f"from {match.group(1)} import {match.group(2)}"
                try:
                    node = ast.parse(statement).body[0]
                except SyntaxError:
                    continue
                assert isinstance(node, ast.ImportFrom)
                for alias in node.names:
                    found.append((doc.relative_to(ROOT).as_posix(), node.module or "", alias.name))
    return sorted(set(found))


@pytest.mark.parametrize(("doc", "module", "name"), _imports())
def test_documented_import_resolves(doc: str, module: str, name: str) -> None:
    try:
        imported = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("evalrx"):
            pytest.skip(f"optional dependency {exc.name} is not installed")
        raise
    assert hasattr(imported, name) or importlib.util.find_spec(f"{module}.{name}"), (
        f"{doc}: `from {module} import {name}` no longer resolves"
    )
