"""M1 tier (b) — ProbeGenerator: synthesise a new black-box probe on demand.

When no registered analyzer in the catalog targets the observed failure, this
generator creates a bespoke *probe* — but adapted to M1's reality: a probe needs
the model, which (unlike M2's data-only stats tools) cannot be shipped into a
subprocess.  So the split is:

1. **Host collects** the model's outputs on the cases (the host owns the loaded
   model) into ``m1_probe_input.json``.
2. A **sandboxed generated script** reads that JSON and computes a per-case probe
   metric over the *outputs* (refusal detection, language drift, format/printf
   adherence, length, keyword presence, answer-extraction failure, …), printing a
   strict ``PROBE_RESULT_JSON=`` line.
3. The host wraps the parsed findings into a
   :class:`~evalvitals.core.result.Result` whose ``per_case`` entries flow into
   M2 (stats tools) → M5 exactly like any catalog analyzer's output.

This keeps the M2-tier(b) safety model: generated code never touches the repo
source, never sees the weights, and runs in an
:class:`~evalvitals.agent_runtime.sandbox.ExperimentSandbox` subprocess.

Scope (v1): probes that are **functions over a single forward-pass output**.
Multi-sample (self-consistency), perturbation, and white-box probes need the
model handle and are out of scope here — they belong to an in-process path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalvitals.agent_runtime.sandbox import ExperimentSandbox
from evalvitals.core.result import Result
from evalvitals.eval_agent.prompts.probe_generator import (
    _GENERATE_PROMPT,
    _INPUT_FILENAME,
    _MAX_OUTPUT_CHARS,
    _RESULT_MARKER,
)

if TYPE_CHECKING:
    from evalvitals.agent_runtime.cli_types import CliAgentConfig
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)



@dataclass
class GeneratedProbe:
    """A validated, sandbox-executed probe that can be re-run on new cases.

    Attributes:
        name:   Short identifier (becomes ``generated:<name>``).
        code:   The Python source that was written and executed.
        need:   The natural-language failure pattern it probes for.
        source: Which backend wrote it (``"cli:<provider>"`` or ``"llm"``).
    """

    name: str
    code: str
    need: str = ""
    source: str = ""


class ProbeGenerator:
    """Generate, run, and cache bespoke black-box probes in a sandbox.

    Args:
        judge:       LLM used for the single-pass code-writing path.
        cli_config:  CLI coding-agent config (``provider != "llm"``) used before
                     the judge when present.
        sandbox:     Execution sandbox (fresh temp-dir when ``None``).
        timeout_sec: Hard wall-clock limit per sandbox run.
        max_cases:   Cap on cases whose outputs are collected (cost guard).
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        sandbox: "ExperimentSandbox | None" = None,
        timeout_sec: int = 60,
        max_cases: int = 200,
        run_logger: "Any | None" = None,
    ) -> None:
        self._judge = judge
        self._cli_config = cli_config
        self._timeout_sec = timeout_sec
        self._max_cases = max_cases
        self._sandbox = sandbox or ExperimentSandbox()
        # Optional RunLogger — when set, every code-writing attempt (the prompt,
        # the code produced, the backend used, and the pass/fail outcome) is
        # recorded as a "tool_codegen" event so tool synthesis is fully traceable.
        self.run_logger = run_logger
        self._last_prompt: str = ""
        self._last_raw: str = ""
        self._last_raw_stream: str = ""
        self._last_usage: dict | None = None

    @property
    def available(self) -> bool:
        return self._judge is not None or (
            self._cli_config is not None and self._cli_config.provider != "llm"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        need: str,
        model: "Model",
        cases: "CaseBatch",
        name: str = "custom",
    ) -> "tuple[Result | None, GeneratedProbe | None]":
        """Collect outputs, write+run a probe over them, return (Result, probe)."""
        if not self.available:
            logger.debug("ProbeGenerator: no code-writing backend configured")
            return None, None

        self._collect_outputs(model, cases)
        self._last_prompt = ""
        self._last_raw = ""
        self._last_raw_stream = ""
        self._last_usage = None
        try:
            code, source = self._write_code(need)
        except Exception as exc:
            logger.warning("ProbeGenerator: code writing failed: %s", exc)
            self._emit_codegen(name, need, "", "", ok=False, error=f"code writing failed: {exc}")
            return None, None
        if not code.strip():
            self._emit_codegen(name, need, source, "", ok=False, error="empty code produced")
            return None, None

        result = self._run_code(code, name, model, cases)
        self._emit_codegen(
            name, need, source, code, ok=result is not None,
            error="" if result is not None else "sandbox produced no parseable result",
        )
        if result is None:
            return None, None
        probe = GeneratedProbe(name=name, code=code, need=need, source=source)
        return result, probe

    def _emit_codegen(
        self, name: str, need: str, source: str, code: str, *, ok: bool, error: str = ""
    ) -> None:
        """Record one code-writing attempt to the RunLogger, if attached."""
        if self.run_logger is None:
            return
        extra = ({"cli_usage": self._last_usage}
                 if source.startswith("cli:") and self._last_usage else None)
        try:
            self.run_logger.log_tool_codegen(
                module="m1_probe", name=name, need=need, source=source, ok=ok,
                code=code, prompt=self._last_prompt, raw_output=self._last_raw,
                raw_stream=self._last_raw_stream, error=error,
                extra=extra,
            )
        except Exception as exc:  # logging must never break generation
            logger.debug("ProbeGenerator: log_tool_codegen failed: %s", exc)

    def run_cached(
        self,
        probe: GeneratedProbe,
        model: "Model",
        cases: "CaseBatch",
    ) -> "Result | None":
        """Re-run an already-generated probe on fresh cases (no LLM call)."""
        self._collect_outputs(model, cases)
        return self._run_code(probe.code, probe.name, model, cases)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_outputs(self, model: "Model", cases: "CaseBatch") -> None:
        """Run the model on each case and serialise outputs to the sandbox dir."""
        records: list[dict[str, Any]] = []
        for case in list(cases)[: self._max_cases]:
            inp = getattr(case, "inputs", None)
            try:
                output = str(model.generate(inp)) if inp is not None else ""
            except Exception as exc:  # a probe over partial outputs is still useful
                logger.debug("ProbeGenerator: generate failed for %s: %s", case.id, exc)
                output = ""
            label = getattr(case, "label", None)
            records.append({
                "id": case.id,
                "prompt": str(getattr(inp, "prompt", "")) if inp is not None else "",
                "expected": getattr(case, "expected", None),
                "label": getattr(label, "value", None),
                "output": output[:_MAX_OUTPUT_CHARS],
            })
        path = Path(self._sandbox.workdir) / _INPUT_FILENAME
        path.write_text(json.dumps({"cases": records}, default=str), encoding="utf-8")

    def _write_code(self, need: str) -> tuple[str, str]:
        if self._cli_config is not None and self._cli_config.provider != "llm":
            code = self._write_code_cli(need)
            if code:
                return code, f"cli:{self._cli_config.provider}"
        prompt = self._build_prompt(need, fenced=True)
        self._last_prompt = prompt
        raw = self._judge.generate(prompt)  # type: ignore[union-attr]
        self._last_raw = str(raw)
        return _extract_code(str(raw)), "llm"

    def _write_code_cli(self, need: str) -> str:
        from evalvitals.agent_runtime.codegen import CodegenRunner

        prompt = self._build_prompt(need, fenced=False)
        self._last_prompt = prompt
        result = CodegenRunner(self._cli_config).write_code(  # type: ignore[arg-type]
            prompt,
            workdir=Path(self._sandbox.workdir),
            timeout_sec=self._timeout_sec,
            preferred_filenames=("probe.py",),
        )
        self._last_raw = result.raw_output
        self._last_raw_stream = ""
        if result.raw_stream_path:
            try:
                self._last_raw_stream = (
                    Path(self._sandbox.workdir) / result.raw_stream_path
                ).read_text(encoding="utf-8")
            except OSError:
                pass
        self._last_usage = result.usage
        return result.code

    def _build_prompt(self, need: str, *, fenced: bool) -> str:
        return _GENERATE_PROMPT.format(
            need=need.strip() or "Detect outputs that fail the task.",
            input_filename=_INPUT_FILENAME,
            marker=_RESULT_MARKER,
            fences_hint=" inside a ```python code block" if fenced else
                        ", written to a file named probe.py",
        )

    def _run_code(
        self,
        code: str,
        name: str,
        model: "Model",
        cases: "CaseBatch",
    ) -> "Result | None":
        sandbox_result = self._sandbox.run(code, timeout_sec=self._timeout_sec)
        if not sandbox_result.ok:
            logger.warning(
                "ProbeGenerator: sandbox run failed (rc=%s): %s",
                sandbox_result.returncode, (sandbox_result.stderr or "").strip()[:200],
            )
            return None
        return _parse_result(sandbox_result.stdout, name, model, cases)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned.strip()


def _parse_result(
    stdout: str,
    name: str,
    model: "Model",
    cases: "CaseBatch",
) -> "Result | None":
    """Parse the last ``PROBE_RESULT_JSON=`` line into a Result, or None."""
    marker_line = None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith(_RESULT_MARKER):
            marker_line = s[len(_RESULT_MARKER):]
    if marker_line is None:
        logger.warning("ProbeGenerator: no PROBE_RESULT_JSON line in probe output")
        return None
    try:
        data = json.loads(marker_line)
    except json.JSONDecodeError as exc:
        logger.warning("ProbeGenerator: unparseable PROBE_RESULT_JSON: %s", exc)
        return None

    findings: dict[str, Any] = {}
    raw_findings = data.get("findings")
    if isinstance(raw_findings, dict):
        findings.update(raw_findings)
    per_case = data.get("per_case")
    if isinstance(per_case, list):
        findings["per_case"] = [e for e in per_case if isinstance(e, dict)]

    return Result(
        analyzer=f"generated:{name}",
        model=repr(model),
        cases=cases,
        findings=findings,
        metadata={"generated": True, "probe": name},
    )
