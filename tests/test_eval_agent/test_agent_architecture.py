from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_AGENT = ROOT / "evalrx" / "eval_agent"
AGENT_RUNTIME = ROOT / "evalrx" / "agent_runtime"
ANALYSIS = ROOT / "evalrx" / "analysis"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class_names(path: Path) -> set[str]:
    return {node.name for node in ast.walk(_tree(path)) if isinstance(node, ast.ClassDef)}


def _function_names(path: Path) -> set[str]:
    return {node.name for node in ast.walk(_tree(path)) if isinstance(node, ast.FunctionDef)}


def test_cli_agent_is_compatibility_facade_only():
    path = EVAL_AGENT / "cli_agent.py"
    source = path.read_text(encoding="utf-8")

    assert len(source.splitlines()) <= 90
    assert _class_names(path) == set()
    assert _function_names(path) == set()
    assert "Compatibility facade" in source


def test_provider_and_model_implementations_live_in_their_packages():
    facade = (EVAL_AGENT / "cli_agent.py").read_text(encoding="utf-8")
    assert "class ClaudeCodeAgent" not in facade
    assert "class CodexAgent" not in facade
    assert "class AgyModel" not in facade
    assert "class ClaudeModel" not in facade

    assert "ClaudeCodeAgent" in _class_names(AGENT_RUNTIME / "providers" / "claude_code.py")
    assert "CodexAgent" in _class_names(AGENT_RUNTIME / "providers" / "codex.py")
    assert "AgyModel" in _class_names(AGENT_RUNTIME / "judges" / "agy.py")
    assert "ClaudeModel" in _class_names(AGENT_RUNTIME / "judges" / "claude.py")


def test_codegen_runner_is_the_stage_cli_invocation_boundary():
    production_paths = [
        *(ROOT / "evalrx" / "analysis").glob("*.py"),
        *(EVAL_AGENT / "stages").glob("*.py"),
        *(EVAL_AGENT / "agentic").glob("*.py"),
        EVAL_AGENT / "nl_runner.py",
    ]

    offenders: list[str] = []
    for path in production_paths:
        source = path.read_text(encoding="utf-8")
        if "create_cli_agent" in source or ".run(prompt" in source:
            if "CodegenRunner" not in source:
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_skill_policy_is_not_embedded_in_explorer_or_provider_adapters():
    explorer = (ROOT / "evalrx" / "analysis" / "explorer.py").read_text(
        encoding="utf-8"
    )
    prompt_module = (ANALYSIS / "prompts" / "explorer.py").read_text(encoding="utf-8")

    assert "def _skills_hint" not in explorer
    assert "def _skills_hint" not in prompt_module
    assert "def skills_hint" in (AGENT_RUNTIME / "skills" / "prompt_policy.py").read_text(
        encoding="utf-8"
    )


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _runtime_imports(path: Path) -> list[str]:
    """Module names imported outside of ``if TYPE_CHECKING:`` blocks.

    Type-only imports never execute, so they don't create a real runtime
    dependency between packages — only a static-typing convenience.
    """
    names: list[str] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking_guard(child):
                continue
            if isinstance(child, ast.ImportFrom) and child.module:
                names.append(child.module)
            elif isinstance(child, ast.Import):
                names.extend(alias.name for alias in child.names)
            visit(child)

    visit(_tree(path))
    return names


def test_analysis_and_agent_runtime_never_import_eval_agent():
    """Locks the Phase 1 dependency inversion in place: evalrx.analysis and
    evalrx.agent_runtime must be usable standalone, so neither may depend
    on evalrx.eval_agent at runtime (TYPE_CHECKING-only imports are fine —
    see _runtime_imports)."""
    offenders: list[str] = []
    for pkg in (ANALYSIS, AGENT_RUNTIME):
        for path in pkg.rglob("*.py"):
            for module in _runtime_imports(path):
                if module == "evalrx.eval_agent" or module.startswith(
                    "evalrx.eval_agent."
                ):
                    offenders.append(f"{path.relative_to(ROOT)}: {module}")

    assert offenders == []


def test_stage_prompt_templates_live_in_prompt_modules():
    eval_agent_prompt_modules = {
        "agentic.py",
        "case_discovery.py",
        "diagnosis.py",
        "experiment_writer.py",
        "fix_agent.py",
        "hypothesis_tester.py",
        "nl_runner.py",
        "probe_agent.py",
        "probe_generator.py",
        "whitebox_probe_generator.py",
    }
    assert eval_agent_prompt_modules <= {p.name for p in (EVAL_AGENT / "prompts").glob("*.py")}

    # M2 stats prompts + the explorer prompt live in evalrx.analysis (standalone).
    analysis_prompt_modules = {
        "stats_agent.py",
        "stats_tool_generator.py",
        "hypothesis_agent.py",
        "explorer.py",
    }
    assert analysis_prompt_modules <= {p.name for p in (ANALYSIS / "prompts").glob("*.py")}

    offenders: list[str] = []
    for path in [
        *(EVAL_AGENT / "stages").glob("*.py"),
        *(EVAL_AGENT / "agentic").glob("*.py"),
        *ANALYSIS.glob("*.py"),
    ]:
        tree = _tree(path)
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                text = value.value
                if len(text) > 250 and any(
                    marker in text
                    for marker in ("You are", "Return ONLY", "Reply with ONLY")
                ):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert offenders == []
