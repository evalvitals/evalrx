"""ExperimentWriter — multi-phase LLM or CLI agent writes diagnostic code.

Mirrors the multi-phase CodeAgent from ``researchclaw/pipeline/code_agent.py``,
adapted for evalrx's hypothesis-testing context.

Six execution phases (LLM path):

Phase 1 · Blueprint    LLM produces a YAML spec with file list, per-file
                       pseudocode, dependency order, and output contract.
Phase 2 · Sequential   Generate each file following blueprint order;
                       each prior file is summarised via AST (CodeMem)
                       and injected as context for the next.
Phase 3 · Hard-validate AST-parse all files; critical issues (SyntaxError,
                       missing main guard, unresolvable cross-file imports)
                       trigger targeted repair; warnings are logged only.
Phase 4 · Exec-fix     Run project via ``sandbox.run_project()``.  Feed
                       stderr back to LLM for targeted single-file repair;
                       retry up to ``exec_fix_max_iterations`` times.
Phase 5 · Tree-search  Explore multiple blueprint variants, score by
                       sandbox metrics, select best  (off by default).
Phase 6 · Review       Coder-reviewer dialog; revert if run degrades
                       (off by default, ``review_max_rounds=0``).

CLI agent path (``provider != "llm"``)
    The CLI agent writes + self-repairs; we collect ``experiment.py``
    (or first .py file) and run once via ``sandbox.run_project()``.
    No exec-fix loop — the agent self-repairs during its own run.

Backward compatibility
    The public ``write_and_run(hypothesis, model_context, cases_json, sandbox)``
    signature is unchanged.  With default config the LLM path runs Phase 2
    (single-pass write) → Phase 3 → Phase 4, identical to the previous 3-phase
    implementation.  ``result.code`` always equals ``result.files["main.py"]``
    (or the single-file code) so existing callers are unaffected.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalrx.agent_runtime.sandbox import ExperimentSandbox, SandboxResult
from evalrx.eval_agent.prompts.experiment_writer import (
    _BLUEPRINT_SYSTEM,
    _BLUEPRINT_USER,
    _GENERATE_FILE_SYSTEM,
    _GENERATE_FILE_USER,
    _REPAIR_SYSTEM,
    _REPAIR_USER,
    _REVIEW_SYSTEM,
    _WRITE_SYSTEM,
    _WRITE_USER,
    build_cli_prompt,
)

if TYPE_CHECKING:
    from evalrx.agent_runtime.sandbox import SandboxProtocol
    from evalrx.core.model import Model
    from evalrx.eval_agent.hypothesis import Hypothesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentWriterConfig:
    """Controls which phases run and their limits.

    All new fields default to off / zero so that the existing 3-phase
    behaviour (write → validate → exec-fix) is preserved by default.

    To use a CLI coding agent for M4 instead of the single-pass LLM::

        from evalrx.agent_runtime.cli_types import CliAgentConfig
        cfg = ExperimentWriterConfig(
            cli_agent=CliAgentConfig(provider="claude_code", model="sonnet"),
        )

    To enable the full multi-phase CodeAgent flow::

        cfg = ExperimentWriterConfig(
            blueprint_enabled=True,
            sequential_generation=True,
            review_max_rounds=1,
        )
    """

    # Phase 1: blueprint planning (LLM outlines file structure first)
    blueprint_enabled: bool = False

    # Phase 2: sequential file generation following the blueprint
    sequential_generation: bool = False

    # Phase 3: AST validation + repair
    hard_validation: bool = True
    hard_validation_max_repairs: int = 3

    # Phase 4: exec-fix loop
    exec_fix_max_iterations: int = 3
    exec_fix_timeout_sec: int = 60

    # Phase 5: solution tree search (off by default — multiplies cost)
    tree_search_enabled: bool = False
    tree_search_candidates: int = 3
    tree_search_max_depth: int = 2

    # Phase 6: review dialog (off by default)
    review_max_rounds: int = 0

    # CLI agent dispatch — None → LLM path (backward-compatible default)
    cli_agent: Any = None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ExperimentWriterResult:
    """Output of :class:`ExperimentWriter.write_and_run`.

    Attributes:
        code:              ``files["main.py"]`` or the single-file code.
                           Kept for backward compatibility.
        files:             Multi-file output dict.  ``{}`` in single-file mode.
        blueprint:         YAML spec produced in Phase 1; empty if disabled.
        metrics:           ``{name: float}`` parsed from stdout.
        verdict:           ``1.0`` = SUPPORTED, ``0.0`` = REFUTED,
                           ``None`` = no verdict printed (INCONCLUSIVE).
        stdout / stderr:   Raw subprocess output from the last run.
        returncode:        Exit code of the last sandbox run.
        timed_out:         ``True`` when the process was killed.
        total_llm_calls:   Number of LLM calls made across all phases.
        total_sandbox_runs: Number of sandbox runs (initial + retries).
        validation_log:    Ordered list of events for debugging.
        cli_raw_output:    The CLI agent's narration while writing the code — for
                           ``stream-json`` providers this is the full rendered
                           coding trajectory (tool calls + results).  Empty on
                           the LLM path (use ``validation_log`` there).
        cli_usage:         Token/cost usage reported by the CLI agent, or
                           ``None`` on the LLM path / when unavailable.
        provider:          Which backend wrote the code (``"llm"`` or the CLI
                           provider name, e.g. ``"claude_code"``).
        workdir:           Path to the working directory the agent operated in,
                           so callers can snapshot it before sandbox cleanup.
    """

    code: str = ""
    files: dict[str, str] = field(default_factory=dict)
    blueprint: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    verdict: float | None = None
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    timed_out: bool = False
    total_llm_calls: int = 0
    total_sandbox_runs: int = 0
    validation_log: list[str] = field(default_factory=list)
    cli_raw_output: str = ""
    cli_usage: dict | None = None
    provider: str = "llm"
    workdir: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# ---------------------------------------------------------------------------
# Solution node (tree search)
# ---------------------------------------------------------------------------


@dataclass
class SolutionNode:
    """One candidate in the tree-search phase."""

    node_id: str
    files: dict[str, str]
    parent_id: str | None = None
    depth: int = 0
    runs_ok: bool = False
    returncode: int = -1
    evaluated: bool = False
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    generation_method: str = "initial"


# ---------------------------------------------------------------------------
# Model context helper
# ---------------------------------------------------------------------------


def build_model_context(
    model: "Model", *, allow_reconstruction: bool = True
) -> dict[str, Any]:
    """Extract the information needed to reconstruct *model* inside a subprocess.

    Returns a dict with:
        ``import_expr``   — Python import line(s)
        ``load_expr``     — expression that produces the model object
        ``capabilities``  — sorted list of capability name strings
    """
    caps = sorted(str(c.name) for c in getattr(model, "capabilities", frozenset()))

    if not allow_reconstruction:
        return {
            "import_expr": (
                "# The host model is already resident and is intentionally not "
                "reconstructed in this child process."
            ),
            "load_expr": "None",
            "capabilities": ["STORED_OUTPUTS_ONLY"],
            "access_mode": "artifacts_only",
            "access_note": (
                "The model is unavailable inside the sandbox. Use only each case's "
                "stored observed output, expected answer, label, and metadata. Do not "
                "call evalrx.load, compose, from_pretrained, any model API, or the "
                "network. If the hypothesis requires a fresh model intervention, print "
                "inconclusive: 1.0 as the final line and do not print a verdict."
            ),
        }

    spec = getattr(model, "spec", None)
    if spec is not None and getattr(spec, "key", None):
        runtime = getattr(model, "runtime", None)
        device = getattr(runtime, "device", None)
        dtype = getattr(runtime, "dtype", None)
        # Infer want from capabilities so the sandbox loads with the same features.
        want: list[str] = []
        for cap in getattr(model, "capabilities", frozenset()):
            cname = str(getattr(cap, "name", cap)).lower()
            if cname in ("attention", "hidden_states", "logits"):
                want.append(cname)
        kwargs_parts = []
        if device:
            kwargs_parts.append(f"device={device!r}")
        if dtype:
            kwargs_parts.append(f"dtype={dtype!r}")
        if want:
            kwargs_parts.append(f"want={want!r}")
        kwargs_str = (", " + ", ".join(kwargs_parts)) if kwargs_parts else ""
        return {
            "import_expr": "import evalrx",
            "load_expr": f"evalrx.load({spec.key!r}{kwargs_str})",
            "capabilities": caps,
        }

    if type(model).__name__ == "GeminiModel":
        model_id = getattr(model, "model_id", "gemini-2.5-flash")
        return {
            "import_expr": "from evalrx.models.blackbox.gemini import GeminiModel",
            "load_expr": f"GeminiModel(model_id={model_id!r})",
            "capabilities": caps,
        }

    return {
        "import_expr": "import evalrx",
        "load_expr": f"# WARNING: cannot reconstruct {type(model).__name__!r} automatically",
        "capabilities": caps,
    }


# ---------------------------------------------------------------------------
# ExperimentWriter
# ---------------------------------------------------------------------------


class ExperimentWriter:
    """Multi-phase LLM or CLI agent that writes and executes diagnostic code.

    When ``config.cli_agent`` is None or has ``provider="llm"`` (default):
        Phase 1 — Blueprint planning (optional, ``blueprint_enabled=True``)
        Phase 2 — Sequential file generation  (optional, ``sequential_generation=True``)
                  or single-pass write (default)
        Phase 3 — Hard validation + targeted repair
        Phase 4 — Exec-fix loop: run → stderr → targeted file repair → retry
        Phase 5 — Tree search (optional, ``tree_search_enabled=True``)
        Phase 6 — Reviewer dialog (optional, ``review_max_rounds > 0``)

    When ``config.cli_agent.provider`` is a CLI provider:
        Phase 1 — CLI agent writes ``experiment.py`` autonomously
        Phase 2 — One ``sandbox.run_project()`` to collect metrics (no loop)

    Args:
        judge:  Any :class:`~evalrx.core.model.Model` with ``Capability.GENERATE``.
        config: :class:`ExperimentWriterConfig` controlling phases and limits.
    """

    def __init__(
        self,
        judge: "Model",
        config: ExperimentWriterConfig = ExperimentWriterConfig(),
    ) -> None:
        self._judge = judge
        self._cfg = config
        self._calls = 0
        self._runs = 0
        self._log: list[str] = []
        self._forbid_model_load = False
        # Lazily created write workdir for multi-file output
        self._workdir: Path | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write_and_run(
        self,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
        sandbox: "ExperimentSandbox | SandboxProtocol",
    ) -> ExperimentWriterResult:
        """Generate, validate, and execute a diagnostic experiment.

        Args:
            hypothesis:    The claim to test.
            model_context: Output of :func:`build_model_context`.
            cases_json:    JSON-serialised failure cases.
            sandbox:       Sandbox used for execution.

        Returns:
            :class:`ExperimentWriterResult` with metrics and verdict.
        """
        self._calls = 0
        self._runs = 0
        self._log = []
        self._forbid_model_load = model_context.get("access_mode") == "artifacts_only"
        self._log_event("ExperimentWriter.write_and_run() started")

        # CLI agent dispatch (unchanged from previous implementation)
        cli_cfg = self._cfg.cli_agent
        if cli_cfg is None:
            from evalrx.agent_runtime.cli_types import CliAgentConfig
            cli_cfg = CliAgentConfig()
        if cli_cfg.provider != "llm":
            return self._cli_write_and_run(
                hypothesis, model_context, cases_json, sandbox, cli_cfg
            )

        # LLM path: Phase 1 — blueprint
        blueprint_yaml = ""
        blueprint_dict: dict[str, Any] | None = None
        if self._cfg.blueprint_enabled:
            blueprint_yaml, blueprint_dict = self._phase1_blueprint(
                hypothesis, model_context, cases_json
            )

        # Phase 2 — code generation (sequential or single-pass)
        files: dict[str, str] = {}
        if (
            self._cfg.sequential_generation
            and blueprint_dict is not None
            and self._is_valid_blueprint(blueprint_dict)
        ):
            files = self._phase2_sequential(
                hypothesis, model_context, cases_json, blueprint_yaml, blueprint_dict
            )
        else:
            if self._cfg.sequential_generation and blueprint_dict is None:
                self._log_event(
                    "  Sequential generation requested but blueprint invalid — "
                    "falling back to single-pass write"
                )
            code = self._phase2_write(hypothesis, model_context, cases_json, blueprint_yaml)
            if not code.strip():
                self._log_event("Phase 2 produced empty script — aborting")
                return ExperimentWriterResult(
                    blueprint=blueprint_yaml,
                    validation_log=list(self._log),
                    total_llm_calls=self._calls,
                    total_sandbox_runs=self._runs,
                )
            files = {"main.py": code}

        # Phase 3 — hard validation
        if self._cfg.hard_validation and files:
            files = self._phase3_hard_validate(files, hypothesis, model_context, cases_json)

        if self._forbid_model_load and self._contains_forbidden_model_load(files):
            message = (
                "generated M4 code attempted to reconstruct or download a second model; "
                "execution blocked to protect the resident host model and GPU memory"
            )
            self._log_event(f"Phase 3: {message}")
            main_code = files.get("main.py") or (next(iter(files.values())) if files else "")
            return ExperimentWriterResult(
                code=main_code,
                files=files,
                stderr=message,
                returncode=-1,
                validation_log=list(self._log),
                total_llm_calls=self._calls,
                total_sandbox_runs=self._runs,
            )

        # Phase 4 — exec-fix loop (tree search or plain)
        last_result: SandboxResult
        best_files = files
        nodes_explored = 0

        if self._cfg.tree_search_enabled:
            best_node, nodes_explored = self._phase5_tree_search(
                files, sandbox, hypothesis, model_context, cases_json
            )
            best_files = best_node.files
            last_result = self._run_files(best_files, sandbox)
        else:
            best_files, last_result = self._phase4_exec_fix(files, sandbox, hypothesis)

        # Phase 6 — review dialog
        review_rounds = 0
        if self._cfg.review_max_rounds > 0 and last_result.ok:
            best_files, last_result, review_rounds = self._phase6_review(
                best_files, last_result, sandbox
            )

        # Build result
        main_code = best_files.get("main.py") or (next(iter(best_files.values())) if best_files else "")
        verdict = last_result.metrics.get("verdict")

        self._log_event(
            f"write_and_run() done — {self._calls} LLM calls, "
            f"{self._runs} sandbox runs, verdict={verdict}"
        )

        return ExperimentWriterResult(
            code=main_code,
            files=best_files,
            blueprint=blueprint_yaml,
            metrics=last_result.metrics,
            verdict=verdict,
            stdout=last_result.stdout,
            stderr=last_result.stderr,
            returncode=last_result.returncode,
            timed_out=last_result.timed_out,
            total_llm_calls=self._calls,
            total_sandbox_runs=self._runs,
            validation_log=list(self._log),
            provider="llm",
            workdir=str(getattr(sandbox, "workdir", "")),
        )

    # ------------------------------------------------------------------
    # Phase 1: Blueprint planning
    # ------------------------------------------------------------------

    def _phase1_blueprint(
        self,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Generate YAML blueprint; return (raw_yaml, parsed_dict_or_None)."""
        self._log_event("Phase 1: Blueprint planning")
        system = _BLUEPRINT_SYSTEM.format(timeout_sec=self._cfg.exec_fix_timeout_sec)
        user = _BLUEPRINT_USER.format(
            statement=hypothesis.statement,
            failure_mode=hypothesis.predicted_failure_mode,
            import_expr=model_context.get("import_expr", "import evalrx"),
            load_expr=model_context.get("load_expr", "# model"),
            capabilities=", ".join(model_context.get("capabilities", [])) or "GENERATE",
            model_access_note=model_context.get("access_note", ""),
            cases_json_snippet=cases_json[:1500],
        )
        raw = self._llm_call(system, user)
        # Extract YAML block
        m = re.search(r"```ya?ml\s*\n(.*?)```", raw, re.DOTALL)
        blueprint_yaml = m.group(1).strip() if m else raw.strip()
        self._log_event(f"  Blueprint spec: {len(blueprint_yaml)} chars")

        blueprint_dict = self._parse_blueprint(blueprint_yaml)
        if blueprint_dict:
            n_files = len(blueprint_dict.get("files", []))
            self._log_event(f"  Parsed blueprint: {n_files} files")
        else:
            self._log_event("  WARNING: Could not parse blueprint YAML")
        return blueprint_yaml, blueprint_dict

    def _parse_blueprint(self, yaml_text: str) -> dict[str, Any] | None:
        """Parse blueprint YAML, sanitising Python type annotations first (BUG-178)."""
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            self._log_event("  pyyaml not available — blueprint parsing skipped")
            return None

        sanitized_lines = []
        for line in yaml_text.split("\n"):
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                sanitized_lines.append(line)
                continue
            if stripped.startswith(("- ", "---", "...")):
                if stripped.startswith("- ") and ":" in stripped[2:]:
                    inner = stripped[2:]
                else:
                    sanitized_lines.append(line)
                    continue
            elif ":" in stripped:
                inner = stripped
            else:
                sanitized_lines.append(line)
                continue

            m = re.search(r":\s", inner)
            if not m:
                sanitized_lines.append(line)
                continue
            val_part = inner[m.end():].strip()
            if not val_part:
                sanitized_lines.append(line)
                continue
            if val_part.startswith(("'", "|", ">")):
                sanitized_lines.append(line)
                continue
            needs_quoting = False
            if val_part.startswith('"'):
                if not val_part.endswith('"') or val_part.count('"') % 2 != 0:
                    needs_quoting = True
                else:
                    sanitized_lines.append(line)
                    continue
            elif ":" in val_part or "->" in val_part:
                needs_quoting = True

            if needs_quoting:
                clean = val_part.strip('"').replace('"', '\\"')
                comment_idx = clean.find("  #")
                if comment_idx >= 0:
                    clean = clean[:comment_idx].rstrip()
                indent = line[:len(line) - len(stripped)]
                prefix = stripped[:len(stripped) - len(inner)]
                key_sep = inner[:m.end()]
                sanitized_lines.append(f'{indent}{prefix}{key_sep}"{clean}"')
            else:
                sanitized_lines.append(line)

        sanitized = "\n".join(sanitized_lines)
        for attempt_text in (sanitized, yaml_text):
            try:
                data = yaml.safe_load(attempt_text)
                if isinstance(data, dict) and "files" in data:
                    return data
            except Exception as exc:
                self._log_event(f"  Blueprint YAML parse error: {exc}")
        return None

    @staticmethod
    def _is_valid_blueprint(blueprint: dict[str, Any]) -> bool:
        files = blueprint.get("files", [])
        if not files or not isinstance(files, list):
            return False
        has_order = sum(
            1 for f in files if isinstance(f, dict) and "generation_order" in f
        )
        return has_order >= 2

    # ------------------------------------------------------------------
    # Phase 2a: Sequential file generation
    # ------------------------------------------------------------------

    def _phase2_sequential(
        self,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
        blueprint_yaml: str,
        blueprint: dict[str, Any],
    ) -> dict[str, str]:
        """Generate files one-by-one following blueprint dependency order."""
        self._log_event("Phase 2: Sequential generation (blueprint-guided)")

        generated: dict[str, str] = {}
        code_memory: dict[str, dict[str, Any]] = {}

        file_specs = [f for f in blueprint.get("files", []) if isinstance(f, dict)]
        for i, fs in enumerate(file_specs):
            if "generation_order" not in fs:
                fs["generation_order"] = i + 1
        file_specs.sort(key=lambda f: f.get("generation_order", 99))

        for file_spec in file_specs:
            file_name = file_spec.get("name", "")
            if not file_name:
                continue
            self._log_event(
                f"  Generating {file_name} (order={file_spec.get('generation_order')})"
            )

            deps = file_spec.get("dependencies", [])
            dep_summaries = ""
            dep_code = ""
            for dep in deps:
                if isinstance(dep, str):
                    if dep in code_memory:
                        dep_summaries += (
                            f"\n### {dep} (summary)\n"
                            + json.dumps(code_memory[dep], indent=2)
                            + "\n"
                        )
                    if dep in generated:
                        dep_code += (
                            f"\n### {dep}\n```python\n{generated[dep]}\n```\n"
                        )
            dep_summaries = dep_summaries or "(no dependencies yet)"
            dep_code = dep_code or "(no dependencies yet)"

            system = _GENERATE_FILE_SYSTEM
            user = _GENERATE_FILE_USER.format(
                file_name=file_name,
                file_spec=json.dumps(file_spec, indent=2, default=str),
                blueprint=blueprint_yaml[:3000],
                dep_summaries=dep_summaries,
                dep_code=dep_code,
                model_access_note=model_context.get("access_note", ""),
            )
            raw = self._llm_call(system, user)
            code = self._extract_single_file_code(raw, file_name)
            if not code:
                self._log_event(f"  WARNING: Empty code for {file_name}")
                continue

            generated[file_name] = code
            code_memory[file_name] = self._build_code_summary(file_name, code)
            self._log_event(
                f"  {file_name}: {len(code.splitlines())} lines, "
                f"{len(code_memory[file_name].get('classes', []))} classes"
            )

        if "main.py" not in generated and generated:
            first_key = next(iter(generated))
            self._log_event(
                f"  WARNING: No main.py generated — promoting '{first_key}'"
            )
            generated["main.py"] = generated.pop(first_key)

        self._log_event(f"  Sequential generation complete: {len(generated)} files")
        return generated

    # ------------------------------------------------------------------
    # Phase 2b: Single-pass write (default path)
    # ------------------------------------------------------------------

    def _phase2_write(
        self,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
        blueprint_yaml: str = "",
    ) -> str:
        self._log_event("Phase 2: Writing diagnostic script")
        blueprint_context = (
            f"## Blueprint (follow this structure)\n```yaml\n{blueprint_yaml}\n```\n"
            if blueprint_yaml else ""
        )
        system = _WRITE_SYSTEM.format(timeout_sec=self._cfg.exec_fix_timeout_sec)
        user = _WRITE_USER.format(
            statement=hypothesis.statement,
            failure_mode=hypothesis.predicted_failure_mode,
            import_expr=model_context.get("import_expr", "import evalrx"),
            load_expr=model_context.get("load_expr", "# model"),
            capabilities=", ".join(model_context.get("capabilities", [])) or "GENERATE",
            model_access_note=model_context.get("access_note", ""),
            blueprint_context=blueprint_context,
            cases_json=cases_json,
        )
        raw = self._llm_call(system, user)
        code = self._extract_code_block(raw)
        self._log_event(f"  Script: {len(code)} chars")
        return code

    # ------------------------------------------------------------------
    # Phase 3: Hard validation
    # ------------------------------------------------------------------

    def _phase3_hard_validate(
        self,
        files: dict[str, str],
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
    ) -> dict[str, str]:
        """AST-based validation; repair critical issues; log warnings."""
        self._log_event("Phase 3: Hard validation")

        for attempt in range(self._cfg.hard_validation_max_repairs + 1):
            critical, warnings = self._hard_validate(files)

            for w in warnings:
                self._log_event(f"  WARNING: {w}")

            if not critical:
                self._log_event(
                    f"  Hard validation passed ({len(warnings)} warning(s), "
                    f"attempt {attempt})"
                )
                return files

            self._log_event(
                f"  {len(critical)} CRITICAL issue(s) "
                f"(attempt {attempt}/{self._cfg.hard_validation_max_repairs})"
            )
            for c in critical:
                self._log_event(f"  CRITICAL: {c}")

            if attempt >= self._cfg.hard_validation_max_repairs:
                self._log_event("  Max repairs reached — proceeding with warnings")
                return files

            files = self._repair_critical_issues(files, critical)

        return files

    def _hard_validate(
        self, files: dict[str, str]
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

        if self._forbid_model_load and self._contains_forbidden_model_load(files):
            critical.append(
                "[project] Model reconstruction is forbidden in this artifact-only M4 "
                "sandbox. Remove evalrx.load/compose/from_pretrained/model-client "
                "construction and analyze the stored case artifacts; if those cannot "
                "test the hypothesis, emit inconclusive: 1.0 without a verdict."
            )

        return critical, warnings

    @staticmethod
    def _contains_forbidden_model_load(files: dict[str, str]) -> bool:
        """Detect heavyweight/network model construction in generated M4 code."""
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

    def _repair_critical_issues(
        self,
        files: dict[str, str],
        critical_issues: list[str],
    ) -> dict[str, str]:
        """Ask LLM to fix critical validation issues."""
        self._log_event("  Targeted repair for critical issues")

        affected: set[str] = set()
        for issue in critical_issues:
            m = re.match(r"\[([^\]]+\.py)\]", issue)
            if m:
                affected.add(m.group(1))
            else:
                affected.update(f for f in files if f.endswith(".py"))
        if not affected:
            affected.update(f for f in files if f.endswith(".py"))

        files_ctx = self._format_files(files)
        issues_text = "\n".join(f"- {issue}" for issue in critical_issues)

        prompt = (
            "Your code has CRITICAL issues that will cause runtime failures. "
            "Fix ALL of them.\n\n"
            f"## Critical Issues\n{issues_text}\n\n"
            f"## Current Code\n{files_ctx}\n\n"
            "## Rules\n"
            "1. Fix every critical issue listed\n"
            "2. main.py MUST have `if __name__ == \"__main__\":` block\n"
            "3. The last printed line MUST be `verdict: 1.0` or `verdict: 0.0`\n"
            "4. Output ALL files in ```filename:xxx.py``` format\n"
        )
        raw = self._llm_call(
            "You are an expert Python debugging engineer. Fix the code issues listed.",
            prompt,
        )
        fixed = self._extract_files(raw)
        if fixed:
            merged = dict(files)
            merged.update(fixed)
            self._log_event(f"  Repair updated {len(fixed)} file(s): {', '.join(sorted(fixed))}")
            return merged
        # Fall back to single-file repair if multi-file extraction failed
        if "main.py" in files:
            single = self._extract_code_block(raw)
            if single.strip():
                result = dict(files)
                result["main.py"] = single
                self._log_event("  Repair updated main.py (single-file fallback)")
                return result
        self._log_event("  WARNING: Repair produced no extractable files")
        return files

    # ------------------------------------------------------------------
    # Phase 4: Exec-fix loop
    # ------------------------------------------------------------------

    def _phase4_exec_fix(
        self,
        files: dict[str, str],
        sandbox: Any,
        hypothesis: "Hypothesis",
    ) -> tuple[dict[str, str], SandboxResult]:
        """Run → fix → retry loop using targeted file repair."""
        last_result: SandboxResult | None = None

        for i in range(self._cfg.exec_fix_max_iterations):
            result = self._run_files(files, sandbox)
            last_result = result

            if result.ok:
                self._log_event(f"  Exec-fix iter {i}: OK ({result.elapsed_sec:.1f}s)")
                break

            if result.timed_out:
                self._log_event(f"  Exec-fix iter {i}: TIMEOUT — no further repair")
                break

            self._log_event(
                f"  Exec-fix iter {i}: crashed (rc={result.returncode}), "
                f"stderr={len(result.stderr)} chars"
            )
            if i < self._cfg.exec_fix_max_iterations - 1:
                files = self._fix_runtime_error(files, result)

        if last_result is None:
            last_result = SandboxResult(
                returncode=-1, stdout="", stderr="no runs performed", elapsed_sec=0.0
            )

        return files, last_result

    def _fix_runtime_error(
        self, files: dict[str, str], result: SandboxResult
    ) -> dict[str, str]:
        """Targeted or full-file repair after a sandbox crash (E-05 pattern)."""
        stderr_tail = result.stderr[-3000:]
        error_loc = self._parse_error_location(stderr_tail, files)

        if error_loc:
            fname, lineno, error_msg = error_loc
            self._log_event(f"  Targeted repair: {fname}:{lineno} — {error_msg[:80]}")
            fixed = self._targeted_file_repair(files, fname, lineno, error_msg, stderr_tail)
            if fixed:
                return fixed

        # Fallback: full-file repair (single-file or multi-file)
        if len(files) == 1:
            code = next(iter(files.values()))
            repaired = self._llm_call(
                _REPAIR_SYSTEM,
                _REPAIR_USER.format(error=stderr_tail[-2000:], code=code),
            )
            repaired_code = self._extract_code_block(repaired)
            if repaired_code.strip():
                fname = next(iter(files))
                return {fname: repaired_code}
        else:
            files_ctx = self._format_files(files)
            prompt = (
                f"## Error\n```\n{stderr_tail}\n```\n\n"
                f"## Code\n{files_ctx}\n\n"
                "Fix the error. Output ALL files in ```filename:xxx.py``` format."
            )
            raw = self._llm_call(
                "You are a Python debugging expert. Fix the runtime error.", prompt
            )
            fixed = self._extract_files(raw)
            if fixed:
                merged = dict(files)
                merged.update(fixed)
                return merged

        return files

    @staticmethod
    def _parse_error_location(
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

    def _targeted_file_repair(
        self,
        files: dict[str, str],
        target_file: str,
        error_line: int,
        error_msg: str,
        full_stderr: str,
    ) -> dict[str, str] | None:
        """Repair a single file with a ±30-line context window around the error."""
        if target_file not in files:
            return None
        code = files[target_file]
        code_lines = code.split("\n")
        total_lines = len(code_lines)
        window = 30
        start = max(0, error_line - window - 1)
        end = min(total_lines, error_line + window)
        numbered = "\n".join(
            f"{start + i + 1:4d} | {line}"
            for i, line in enumerate(code_lines[start:end])
        )

        dep_summaries = ""
        for fname, fcode in files.items():
            if fname != target_file and fname.endswith(".py"):
                summary = self._build_code_summary(fname, fcode)
                dep_summaries += (
                    f"\n### {fname}: "
                    f"{len(summary.get('classes', []))} classes, "
                    f"{len(summary.get('functions', []))} functions\n"
                )
                for cls in summary.get("classes", []):
                    methods = ", ".join(m["name"] for m in cls.get("methods", []))
                    dep_summaries += (
                        f"  class {cls['name']}"
                        f"({', '.join(cls.get('bases', []))}): [{methods}]\n"
                    )

        prompt = (
            f"Fix the runtime error in `{target_file}` at line {error_line}.\n\n"
            f"## Error\n```\n{error_msg}\n```\n\n"
            f"## Traceback (last 1500 chars)\n```\n{full_stderr[-1500:]}\n```\n\n"
            f"## {target_file} (lines {start + 1}-{end})\n"
            f"```python\n{numbered}\n```\n\n"
            f"## Other files\n{dep_summaries}\n\n"
            f"## Full {target_file} ({total_lines} lines)\n"
            f"```python\n{code}\n```\n\n"
            f"Output the COMPLETE fixed `{target_file}` in "
            f"```filename:{target_file}``` format."
        )
        raw = self._llm_call(
            "You are a debugging expert. Fix the specific runtime error shown. "
            "Output the COMPLETE fixed file.",
            prompt,
        )
        fixed = self._extract_files(raw)
        if not fixed:
            m = re.search(r"```(?:python|filename:\S+)\s*\n(.*?)```", raw, re.DOTALL)
            if m:
                fixed = {target_file: m.group(1).strip()}

        if fixed and target_file in fixed:
            merged = dict(files)
            merged.update(fixed)
            self._log_event(
                f"  Targeted repair applied to {target_file} "
                f"({len(fixed[target_file].splitlines())} lines)"
            )
            return merged
        return None

    # ------------------------------------------------------------------
    # Phase 5: Tree search (optional)
    # ------------------------------------------------------------------

    def _phase5_tree_search(
        self,
        initial_files: dict[str, str],
        sandbox: Any,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
    ) -> tuple[SolutionNode, int]:
        """Explore multiple candidates; return best node and count explored."""
        self._log_event("Phase 5: Tree search")
        all_nodes: list[SolutionNode] = []

        # Generate initial candidates from blueprint variants
        n_cand = max(self._cfg.tree_search_candidates, 1)
        for k in range(n_cand):
            self._log_event(f"  Generating candidate {k + 1}/{n_cand}")
            if k == 0:
                cand_files = initial_files
            else:
                # Re-generate with slight variation
                code = self._phase2_write(hypothesis, model_context, cases_json)
                cand_files = {"main.py": code} if code.strip() else initial_files
            node = SolutionNode(
                node_id=f"gen-{k}",
                files=cand_files,
                depth=0,
                generation_method="initial",
            )
            all_nodes.append(node)

        # Evaluate-fix-branch loop
        for depth in range(self._cfg.tree_search_max_depth):
            for node in all_nodes:
                if not node.evaluated:
                    self._evaluate_node(node, sandbox)

            all_nodes.sort(key=lambda n: n.score, reverse=True)
            self._log_event(
                f"  Depth {depth}: {len(all_nodes)} nodes, "
                f"best={all_nodes[0].node_id} score={all_nodes[0].score:.2f}"
            )

            if all_nodes[0].runs_ok:
                break

            new_nodes: list[SolutionNode] = []
            for node in all_nodes[:2]:
                if not node.runs_ok:
                    fixed = self._fix_runtime_error(
                        node.files,
                        SandboxResult(
                            returncode=node.returncode,
                            stdout=node.stdout,
                            stderr=node.stderr,
                            elapsed_sec=0.0,
                        ),
                    )
                    new_nodes.append(SolutionNode(
                        node_id=f"{node.node_id}-fix{depth}",
                        files=fixed,
                        parent_id=node.node_id,
                        depth=depth + 1,
                        generation_method="fix",
                    ))
            all_nodes.extend(new_nodes)

        # Final evaluation
        for node in all_nodes:
            if not node.evaluated:
                self._evaluate_node(node, sandbox)

        all_nodes.sort(key=lambda n: n.score, reverse=True)
        best = all_nodes[0]
        self._log_event(
            f"  Tree search complete: best={best.node_id} "
            f"score={best.score:.2f}, explored {len(all_nodes)} nodes"
        )
        return best, len(all_nodes)

    def _evaluate_node(self, node: SolutionNode, sandbox: Any) -> None:
        result = self._run_files(node.files, sandbox)
        node.evaluated = True
        node.returncode = result.returncode
        node.stdout = result.stdout
        node.stderr = result.stderr
        node.runs_ok = result.ok
        node.metrics = dict(result.metrics)
        node.score = self._score_node(node)

    @staticmethod
    def _score_node(node: SolutionNode) -> float:
        score = 0.0
        if node.runs_ok:
            score += 1.0
        if node.stdout and len(node.stdout) > 100:
            score += 0.3
        if node.metrics:
            score += 0.5
            if "verdict" in node.metrics:
                score += 0.5
        if node.stderr and "Error" in node.stderr:
            score -= 0.2
        return max(score, 0.0)

    # ------------------------------------------------------------------
    # Phase 6: Review dialog (optional)
    # ------------------------------------------------------------------

    def _phase6_review(
        self,
        files: dict[str, str],
        last_result: SandboxResult,
        sandbox: Any,
    ) -> tuple[dict[str, str], SandboxResult, int]:
        """Coder-reviewer dialog; revert if re-run degrades."""
        self._log_event("Phase 6: Review dialog")
        rounds = 0

        for r in range(self._cfg.review_max_rounds):
            rounds += 1
            files_ctx = self._format_files(files)
            raw = self._llm_call(
                _REVIEW_SYSTEM,
                f"## Experiment Code\n{files_ctx}\n\nReturn JSON only.",
            )
            review = self._parse_json(raw)
            if not isinstance(review, dict):
                self._log_event(f"  Review round {r + 1}: could not parse JSON, skipping")
                break

            verdict = review.get("verdict", "APPROVE")
            score = review.get("score", 10)
            critical = review.get("critical_issues", [])
            self._log_event(
                f"  Review round {r + 1}: verdict={verdict}, score={score}, "
                f"critical={len(critical)}"
            )

            if verdict == "APPROVE" or not critical:
                break

            fix_prompt = (
                "A reviewer found critical issues. Fix ALL of them while "
                "preserving hypothesis testing logic.\n\n"
                "## Critical Issues\n"
                + "\n".join(f"- {issue}" for issue in critical)
                + f"\n\n## Current Code\n{files_ctx}\n\n"
                "Output ALL files in ```filename:xxx.py``` format."
            )
            fix_raw = self._llm_call(
                "You are an expert Python engineer. Fix the critical issues listed.",
                fix_prompt,
            )
            fixed = self._extract_files(fix_raw)
            if not fixed:
                break

            candidate = dict(files)
            candidate.update(fixed)
            new_result = self._run_files(candidate, sandbox)
            if new_result.ok and new_result.returncode == 0:
                files = candidate
                last_result = new_result
                self._log_event("  Review fixes applied and verified")
            else:
                self._log_event(
                    f"  Review fixes degraded the run (rc={new_result.returncode}) — reverting"
                )
                break

        return files, last_result, rounds

    # ------------------------------------------------------------------
    # CLI agent path (unchanged from previous implementation)
    # ------------------------------------------------------------------

    def _cli_write_and_run(
        self,
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        cases_json: str,
        sandbox: Any,
        cli_cfg: Any,
    ) -> ExperimentWriterResult:
        self._log_event(f"CLI path: provider={cli_cfg.provider!r}")
        workdir = getattr(sandbox, "workdir", Path(tempfile.mkdtemp(prefix="evalrx_cli_")))

        import json as _json
        (workdir / "cases.json").write_text(cases_json, encoding="utf-8")
        # Summarise which cases have images so the agent prompt is accurate
        try:
            _cases = _json.loads(cases_json)
            _n_images = sum(1 for c in _cases if c.get("image_path"))
        except Exception:
            _n_images = 0

        (workdir / "hypothesis.md").write_text(
            f"# Hypothesis\n\n"
            f"**Statement:** {hypothesis.statement}\n\n"
            f"**Failure mode:** {hypothesis.predicted_failure_mode}\n\n"
            f"**Target model:** {hypothesis.target_model}\n",
            encoding="utf-8",
        )

        prompt = build_cli_prompt(
            hypothesis, model_context, self._cfg.exec_fix_timeout_sec,
            n_images=_n_images,
        )
        from evalrx.agent_runtime.codegen import CodegenRunner

        self._log_event(f"  invoking {cli_cfg.provider!r}")
        cli_result = CodegenRunner(cli_cfg).run(
            prompt=prompt,
            workdir=workdir,
            timeout_sec=cli_cfg.timeout_sec,
        )
        # The agent's stdout is its narration / coding trajectory while it
        # writes and self-repairs the script — keep it for the coding log.
        cli_raw_output = cli_result.raw_output
        cli_usage = cli_result.usage
        self._log_event(
            f"  CLI finished: ok={cli_result.ok}, "
            f"files={list(cli_result.files)}, elapsed={cli_result.elapsed_sec:.1f}s"
        )
        if cli_result.error:
            self._log_event(f"  CLI error: {cli_result.error}")

        if not cli_result.files:
            self._log_event("CLI agent produced no .py files — aborting")
            return ExperimentWriterResult(
                validation_log=list(self._log),
                total_llm_calls=0,
                total_sandbox_runs=0,
                cli_raw_output=cli_raw_output,
                cli_usage=cli_usage,
                provider=cli_cfg.provider,
                workdir=str(workdir),
            )

        code = cli_result.files.get("experiment.py") or next(iter(cli_result.files.values()))
        self._log_event(f"  collected script: {len(code)} chars")

        if self._forbid_model_load and self._contains_forbidden_model_load(
            {"experiment.py": code}
        ):
            message = (
                "generated M4 code attempted to reconstruct or download a second model; "
                "execution blocked to protect the resident host model and GPU memory"
            )
            self._log_event(message)
            return ExperimentWriterResult(
                code=code,
                files={"experiment.py": code},
                stderr=message,
                returncode=-1,
                validation_log=list(self._log),
                cli_raw_output=cli_raw_output,
                cli_usage=cli_usage,
                provider=cli_cfg.provider,
                workdir=str(workdir),
            )

        if self._cfg.hard_validation:
            errors = []
            try:
                ast.parse(code)
            except SyntaxError as exc:
                errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")
            if errors:
                self._log_event(f"  AST warnings: {'; '.join(errors)}")

        result = sandbox.run(code, timeout_sec=self._cfg.exec_fix_timeout_sec)
        self._runs += 1
        self._log_event(
            f"  sandbox run: rc={result.returncode}, "
            f"timed_out={result.timed_out}, metrics={list(result.metrics)}"
        )

        verdict = result.metrics.get("verdict")
        return ExperimentWriterResult(
            code=code,
            files={"experiment.py": code},
            metrics=result.metrics,
            verdict=verdict,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            timed_out=result.timed_out,
            total_llm_calls=0,
            total_sandbox_runs=self._runs,
            validation_log=list(self._log),
            cli_raw_output=cli_raw_output,
            cli_usage=cli_usage,
            provider=cli_cfg.provider,
            workdir=str(workdir),
        )

    @staticmethod
    def _build_cli_prompt(
        hypothesis: "Hypothesis",
        model_context: dict[str, Any],
        timeout_sec: int,
        n_images: int = 0,
    ) -> str:
        """Backward-compatible wrapper for the external CLI-agent prompt."""
        return build_cli_prompt(hypothesis, model_context, timeout_sec, n_images=n_images)

    # ------------------------------------------------------------------
    # Sandbox execution helpers
    # ------------------------------------------------------------------

    def _run_files(self, files: dict[str, str], sandbox: Any) -> SandboxResult:
        """Write files to a temp directory and run via sandbox."""
        self._runs += 1
        project_dir = Path(tempfile.mkdtemp(prefix="evalrx_exp_"))
        try:
            for fname, code in files.items():
                fpath = (project_dir / fname).resolve()
                if not fpath.is_relative_to(project_dir.resolve()):
                    self._log_event(f"  WARNING: Skipping path-traversal filename: {fname}")
                    continue
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(code, encoding="utf-8")

            if hasattr(sandbox, "run_project"):
                result = sandbox.run_project(
                    project_dir,
                    entry_point="main.py",
                    timeout_sec=self._cfg.exec_fix_timeout_sec,
                )
            else:
                # Fallback: run single-file via sandbox.run()
                main_code = files.get("main.py") or next(iter(files.values()), "")
                result = sandbox.run(main_code, timeout_sec=self._cfg.exec_fix_timeout_sec)
        except Exception as exc:  # noqa: BLE001
            result = SandboxResult(
                returncode=1,
                stdout="",
                stderr=f"[ExperimentWriter] sandbox error: {exc}",
                elapsed_sec=0.0,
            )
        finally:
            import shutil as _shutil
            _shutil.rmtree(project_dir, ignore_errors=True)

        self._log_event(
            f"  run_files: rc={result.returncode}, "
            f"timed_out={result.timed_out}, metrics={list(result.metrics)}"
        )
        return result

    # ------------------------------------------------------------------
    # Code extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_block(text: str) -> str:
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

    @staticmethod
    def _extract_single_file_code(content: str, expected_name: str) -> str:
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

    @staticmethod
    def _extract_files(content: str) -> dict[str, str]:
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
        m = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
        if m:
            return {"main.py": m.group(1).strip()}
        return files

    @staticmethod
    def _format_files(files: dict[str, str]) -> str:
        parts = [f"```filename:{fname}\n{files[fname]}\n```" for fname in sorted(files)]
        return "\n\n".join(parts)

    @staticmethod
    def _build_code_summary(filename: str, code: str) -> dict[str, Any]:
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

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
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

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _llm_call(self, system: str, user: str) -> str:
        self._calls += 1
        try:
            prompt = f"{system}\n\n{user}"
            return self._judge.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            self._log_event(f"  LLM call failed: {exc}")
            return ""

    def _log_event(self, msg: str) -> None:
        logger.debug(msg)
        self._log.append(msg)
