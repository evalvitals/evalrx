"""The template scaffold writes a run.py that compiles against today's API."""

from __future__ import annotations

import ast
import importlib
import py_compile

from evalrx.eval_agent import scaffold_from_description


def test_template_scaffold_imports_only_names_that_exist(tmp_path):
    out = scaffold_from_description(
        "My VLM confuses left and right in spatial questions", output_dir=tmp_path / "exp",
    )
    run_py = out / "run.py"
    assert {p.name for p in out.iterdir()} >= {"run.py", "Dockerfile", "docker-compose.yml"}
    py_compile.compile(str(run_py), doraise=True)
    missing = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(ast.parse(run_py.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("evalrx")
        for alias in node.names
        if not hasattr(importlib.import_module(node.module or ""), alias.name)
    ]
    assert missing == []
