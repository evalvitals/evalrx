"""Reading and checking the code an experiment writer produces.

Pulling files out of an LLM reply, summarising a file for the next one's
prompt, and the static checks run before anything executes: syntax, a main
guard, resolvable cross-file imports, and no model load on an artifacts-only
run.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def extract_code_block(text: str) -> str:
    """Extract the first ```python … ``` block, or the raw text if none found."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_single_file_code(content: str, expected_name: str) -> str:
    """Extract Python code from LLM response for a single named file."""
    m = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(
        rf"```(?:filename:)?{re.escape(expected_name)}\s*\n(.*?)```",
        content, re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    stripped = content.strip()
    if stripped and stripped.startswith(
        ("import ", "from ", "#", "def ", "class ", '"""')
    ):
        return stripped
    return ""


def extract_files(content: str) -> dict[str, str]:
    """Extract multi-file blocks from LLM output.

    Recognises both:
        ```filename:foo.py
        ...code...
        ```
    and:
        ```python  # foo.py
        ...code...
        ```
    """
    files: dict[str, str] = {}
    # Primary format: ```filename:xxx.py
    for m in re.finditer(
        r"```filename:([^\s`]+\.py)\s*\n(.*?)```", content, re.DOTALL
    ):
        fname = m.group(1).strip()
        code = m.group(2).strip()
        if fname and code:
            files[fname] = code
    if files:
        return files
    # Fallback: single ```python block → treat as main.py
    block = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
    if block:
        return {"main.py": block.group(1).strip()}
    return files


def format_files(files: dict[str, str]) -> str:
    parts = [f"```filename:{fname}\n{files[fname]}\n```" for fname in sorted(files)]
    return "\n\n".join(parts)


def build_code_summary(filename: str, code: str) -> dict[str, Any]:
    """Build a CodeMem-style AST summary (mirrors ARC's implementation)."""
    summary: dict[str, Any] = {
        "filename": filename,
        "classes": [],
        "functions": [],
        "imports": [],
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        summary["parse_error"] = True
        return summary

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                {"name": n.name, "args": [a.arg for a in n.args.args if a.arg != "self"]}
                for n in node.body
                if isinstance(n, ast.FunctionDef)
            ]
            summary["classes"].append({
                "name": node.name,
                "bases": [ast.unparse(b) for b in node.bases],
                "methods": methods,
            })
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            summary["functions"].append({
                "name": node.name,
                "args": [a.arg for a in node.args.args],
            })
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                summary["imports"].append(ast.unparse(node))
            except Exception:
                pass

    return summary


def parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from LLM response."""
    def _as_dict(val: Any) -> dict[str, Any] | None:
        return val if isinstance(val, dict) else None

    try:
        return _as_dict(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return _as_dict(json.loads(m.group(1)))
        except (json.JSONDecodeError, ValueError):
            pass
    m = re.search(
        r"\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}",
        text, re.DOTALL,
    )
    if m:
        try:
            return _as_dict(json.loads(m.group(0)))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def is_valid_blueprint(blueprint: dict[str, Any]) -> bool:
    files = blueprint.get("files", [])
    if not files or not isinstance(files, list):
        return False
    has_order = sum(
        1 for f in files if isinstance(f, dict) and "generation_order" in f
    )
    return has_order >= 2


def parse_error_location(
    stderr: str, files: dict[str, str]
) -> tuple[str, int, str] | None:
    """Parse Python traceback to find (filename, line_number, error_message)."""
    known_files = set(files.keys())
    tb_pattern = re.compile(r'File "(?:[^"]*[/\\])?([^"]+\.py)", line (\d+)')
    matches = list(tb_pattern.finditer(stderr))
    if not matches:
        return None
    for m in reversed(matches):
        fname = m.group(1)
        lineno = int(m.group(2))
        if fname in known_files:
            lines = stderr.strip().split("\n")
            error_msg = lines[-1] if lines else "Unknown error"
            return fname, lineno, error_msg
    return None


def hard_validate(
    files: dict[str, str], *, forbid_model_load: bool
) -> tuple[list[str], list[str]]:
    """Return (critical_issues, warnings)."""
    critical: list[str] = []
    warnings: list[str] = []

    # 1. Syntax errors — always critical
    for fname, code in files.items():
        if not fname.endswith(".py"):
            continue
        try:
            ast.parse(code)
        except SyntaxError as exc:
            critical.append(
                f"[{fname}] SyntaxError at line {exc.lineno}: {exc.msg}"
            )

    # 2. main.py MUST have `if __name__ == "__main__":` block (BUG-R41-04)
    main_code = files.get("main.py", "")
    if main_code:
        try:
            main_tree = ast.parse(main_code)
            has_main_guard = any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"
                for node in ast.walk(main_tree)
            )
            if not has_main_guard:
                critical.append(
                    '[main.py] Missing `if __name__ == "__main__":` block — '
                    "script will define functions/classes but never execute. "
                    "Add a main guard that calls the experiment entry point."
                )
        except SyntaxError:
            pass  # already caught above

    # 3. verdict line must be reachable in main.py (warning only)
    if main_code and "verdict:" not in main_code:
        warnings.append(
            "[main.py] No 'verdict:' output found — "
            "the script must print 'verdict: 1.0' or 'verdict: 0.0'"
        )

    # 4. Cross-file import consistency
    known_modules = {
        fname.replace(".py", "")
        for fname in files
        if fname.endswith(".py")
    }
    for fname, code in files.items():
        if not fname.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod_top = node.module.split(".")[0]
                if mod_top in known_modules:
                    target_file = f"{mod_top}.py"
                    if target_file in files and node.names:
                        try:
                            target_tree = ast.parse(files[target_file])
                        except SyntaxError:
                            continue
                        exported = {
                            n.name
                            for n in ast.walk(target_tree)
                            if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                        }
                        for alias in node.names:
                            name = alias.name
                            if name != "*" and name not in exported:
                                critical.append(
                                    f"[{fname}] ImportError: '{name}' not "
                                    f"defined in '{target_file}' — will crash"
                                )

    if forbid_model_load and contains_forbidden_model_load(files):
        critical.append(
            "[project] Model reconstruction is forbidden in this artifact-only M5 "
            "sandbox. Remove evalrx.load/compose/from_pretrained/model-client "
            "construction and analyze the stored case artifacts; if those cannot "
            "test the hypothesis, emit inconclusive: 1.0 without a verdict."
        )

    return critical, warnings


def contains_forbidden_model_load(files: dict[str, str]) -> bool:
    """Detect heavyweight/network model construction in generated M5 code."""
    for name, code in files.items():
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        evalrx_aliases = {"evalrx"}
        direct_loaders: set[str] = set()
        model_classes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "evalrx":
                        evalrx_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    local = alias.asname or alias.name
                    if module == "evalrx" and alias.name in {"load", "compose"}:
                        direct_loaders.add(local)
                    if module.startswith(("transformers", "evalrx.models")) and (
                        alias.name.startswith("AutoModel") or alias.name == "GeminiModel"
                    ):
                        model_classes.add(local)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                if fn.attr.startswith("from_pretrained"):
                    return True
                if (
                    fn.attr in {"load", "compose"}
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id in evalrx_aliases
                ):
                    return True
            if isinstance(fn, ast.Name) and fn.id in direct_loaders | model_classes:
                return True
    return False
