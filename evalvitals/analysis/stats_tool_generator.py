"""M2 tier (b) — StatsToolGenerator: synthesise a new statistical tool on demand.

When no catalog tool in :mod:`~evalvitals.analysis.stats_tools` fits the
data, this generator asks an LLM / CLI coding agent to *write* a small statistics
script, runs it inside an :class:`~evalvitals.agent_runtime.sandbox.ExperimentSandbox`
(subprocess + resource limits + path-traversal protection), and parses a strict
``STATS_RESULT_JSON=`` line from its stdout back into a
:class:`~evalvitals.analysis.stats_tools.StatsToolResult`.

Design (mirrors M4's ExperimentWriter):

- **Generated code never touches the repo source.**  It lives only in the
  sandbox workdir and is executed in a child process; the host never imports it.
- The generated script reads ``m2_stats_input.json`` (the serialised
  :class:`StatsInput`) from its working directory and may ``import
  evalvitals.stats`` to compose existing primitives or write novel statistics
  with numpy.
- **Tier (c) reuse**: a script that runs and parses cleanly is returned as a
  :class:`GeneratedStatsTool`; the caller can re-run it on later cycles via
  :meth:`run_cached` with fresh data — no second LLM call.

Backends (first available wins):

1. ``cli_config`` with ``provider != "llm"`` → a CLI coding agent
   (agy / codex / claude_code …) writes files into the workdir.
2. ``judge`` model → single-pass ``generate()`` returning a fenced code block.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalvitals.agent_runtime.sandbox import ExperimentSandbox
from evalvitals.analysis.prompts.stats_tool_generator import (
    _GENERATE_PROMPT,
    _INPUT_FILENAME,
    _RESULT_MARKER,
)
from evalvitals.analysis.stats_tools import StatsInput, StatsToolResult, describe_data
from evalvitals.stats import compare
from evalvitals.stats.evalue import evalue_bernoulli

if TYPE_CHECKING:
    from evalvitals.agent_runtime.cli_types import CliAgentConfig
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)



@dataclass
class GeneratedStatsTool:
    """A validated, sandbox-executed statistical tool that can be re-run.

    Attributes:
        name:   Short identifier (becomes ``generated:<name>``).
        code:   The Python source that was written and executed.
        need:   The natural-language goal it was generated for.
        source: Which backend wrote it (``"cli:<provider>"`` or ``"llm"``).
    """

    name: str
    code: str
    need: str = ""
    source: str = ""


class StatsToolGenerator:
    """Generate, run, and cache bespoke statistical tools in a sandbox.

    Args:
        judge:       LLM used for the single-pass code-writing path.
        cli_config:  :class:`~evalvitals.agent_runtime.cli_types.CliAgentConfig`
                     with ``provider != "llm"`` to use a CLI coding agent
                     instead of (or before) the judge.
        sandbox:     Execution sandbox.  A fresh temp-dir sandbox is created
                     when ``None``.
        timeout_sec: Hard wall-clock limit per sandbox run — the primary guard.
        max_cpu_seconds / max_memory_bytes: optional resource caps for the
                     default sandbox.  Default ``None`` (no cap) to match M4:
                     generated tools may ``import evalvitals.stats`` which pulls
                     in torch, whose large virtual-memory reservation a
                     ``RLIMIT_AS`` cap would crash.  Ignored when *sandbox* is
                     supplied.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        sandbox: "ExperimentSandbox | None" = None,
        timeout_sec: int = 60,
        max_cpu_seconds: int | None = None,
        max_memory_bytes: int | None = None,
        run_logger: "Any | None" = None,
        alpha: float = 0.05,
    ) -> None:
        self._judge = judge
        self._cli_config = cli_config
        self._timeout_sec = timeout_sec
        # Significance level the HOST uses to reconstruct each generated tool's
        # reject decision from its sufficient statistics (the LLM never decides).
        self._alpha = float(alpha)
        self._sandbox = sandbox or ExperimentSandbox(
            max_cpu_seconds=max_cpu_seconds,
            max_memory_bytes=max_memory_bytes,
        )
        # Optional RunLogger — records each stats-tool code-writing attempt.
        self.run_logger = run_logger
        self._last_prompt: str = ""
        self._last_raw: str = ""
        self._last_raw_stream: str = ""
        self._last_usage: dict | None = None

    @property
    def available(self) -> bool:
        """True when at least one code-writing backend is configured."""
        return self._judge is not None or (
            self._cli_config is not None and self._cli_config.provider != "llm"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        need: str,
        inp: StatsInput,
        name: str = "custom",
    ) -> "tuple[StatsToolResult, GeneratedStatsTool | None]":
        """Write a new stats tool for *need*, run it on *inp*, and return both.

        Returns ``(result, tool)``.  ``tool`` is ``None`` when generation or the
        first execution failed (the result then carries ``ok=False`` + error).
        """
        if not self.available:
            return (
                StatsToolResult(
                    tool=f"generated:{name}", ok=False,
                    error="no code-writing backend (judge or cli_config) configured",
                    summary="codegen unavailable",
                ),
                None,
            )

        self._write_input(inp)
        self._last_prompt = ""
        self._last_raw = ""
        self._last_raw_stream = ""
        self._last_usage = None
        try:
            code, source = self._write_code(need, inp)
        except Exception as exc:
            logger.warning("StatsToolGenerator: code writing failed: %s", exc)
            self._emit_codegen(name, need, "", "", ok=False, error=f"code writing failed: {exc}")
            return (
                StatsToolResult(
                    tool=f"generated:{name}", ok=False,
                    error=f"code writing failed: {exc}", summary="codegen failed",
                ),
                None,
            )

        if not code.strip():
            self._emit_codegen(name, need, source, "", ok=False, error="backend produced no code")
            return (
                StatsToolResult(
                    tool=f"generated:{name}", ok=False,
                    error="backend produced no code", summary="codegen empty",
                ),
                None,
            )

        result = self._run_code(code, name)
        self._emit_codegen(
            name, need, source, code, ok=result.ok,
            error="" if result.ok else (result.error or "tool run failed"),
        )
        tool = (
            GeneratedStatsTool(name=name, code=code, need=need, source=source)
            if result.ok else None
        )
        return result, tool

    def _emit_codegen(
        self, name: str, need: str, source: str, code: str, *, ok: bool, error: str = ""
    ) -> None:
        """Record one stats-tool code-writing attempt to the RunLogger."""
        if self.run_logger is None:
            return
        extra = ({"cli_usage": self._last_usage}
                 if source.startswith("cli:") and self._last_usage else None)
        try:
            self.run_logger.log_tool_codegen(
                module="m2_stats", name=name, need=need, source=source, ok=ok,
                code=code, prompt=self._last_prompt, raw_output=self._last_raw,
                raw_stream=self._last_raw_stream, error=error,
                extra=extra,
            )
        except Exception as exc:  # logging must never break generation
            logger.debug("StatsToolGenerator: log_tool_codegen failed: %s", exc)

    def run_cached(self, tool: GeneratedStatsTool, inp: StatsInput) -> StatsToolResult:
        """Re-run an already-generated tool on fresh *inp* (no LLM call)."""
        self._write_input(inp)
        return self._run_code(tool.code, tool.name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_input(self, inp: StatsInput) -> None:
        payload = {
            "labels": inp.labels,
            "per_case": inp.per_case,
            "scalars": inp.scalars,
            "groups": inp.groups,
        }
        path = Path(self._sandbox.workdir) / _INPUT_FILENAME
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_code(self, need: str, inp: StatsInput) -> tuple[str, str]:
        """Return ``(code, source)`` from the CLI agent or the judge."""
        if self._cli_config is not None and self._cli_config.provider != "llm":
            code = self._write_code_cli(need, inp)
            if code:
                return code, f"cli:{self._cli_config.provider}"
        # Single-pass LLM fallback
        prompt = self._build_prompt(need, inp, fenced=True)
        self._last_prompt = prompt
        raw = self._judge.generate(prompt)  # type: ignore[union-attr]
        self._last_raw = str(raw)
        return _extract_code(str(raw)), "llm"

    def _write_code_cli(self, need: str, inp: StatsInput) -> str:
        from evalvitals.agent_runtime.codegen import CodegenRunner

        prompt = self._build_prompt(need, inp, fenced=False)
        self._last_prompt = prompt
        result = CodegenRunner(self._cli_config).write_code(  # type: ignore[arg-type]
            prompt,
            workdir=Path(self._sandbox.workdir),
            timeout_sec=self._timeout_sec,
            preferred_filenames=("stats_tool.py",),
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
        if not result.code:
            logger.debug("CLI codegen produced no files (%s)", result.error)
        return result.code

    def _build_prompt(self, need: str, inp: StatsInput, *, fenced: bool) -> str:
        return _GENERATE_PROMPT.format(
            need=need.strip() or "Test whether the analyzer signals predict case FAIL.",
            input_filename=_INPUT_FILENAME,
            data_shape=json.dumps(describe_data(inp), indent=2),
            marker=_RESULT_MARKER,
            fences_hint=" inside a ```python code block" if fenced else
                        ", written to a file named stats_tool.py",
        )

    def _run_code(self, code: str, name: str) -> StatsToolResult:
        repo_root = Path(__file__).resolve().parents[3]
        wrapped = (
            "import sys\n"
            f"sys.path.insert(0, {str(repo_root)!r})\n"
            + code
        )
        sandbox_result = self._sandbox.run(wrapped, timeout_sec=self._timeout_sec)
        if not sandbox_result.ok:
            err = (sandbox_result.stderr or "").strip()[:300] or "non-zero exit"
            return StatsToolResult(
                tool=f"generated:{name}", ok=False,
                error=f"sandbox run failed: {err}",
                summary=f"generated:{name} failed to run",
                details={"returncode": sandbox_result.returncode,
                         "timed_out": sandbox_result.timed_out},
            )
        return _parse_result(sandbox_result.stdout, name, self._alpha)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    """Pull the Python source out of an LLM response (fenced or bare)."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned.strip()


def _reconstruct_decision(
    suff: Any, alpha: float
) -> "tuple[bool, float | None, float | None, tuple | None] | None":
    """Recompute ``(reject, e_value, effect, ci)`` HOST-SIDE from a generated
    tool's sufficient statistics, using the validated core — so a generated
    tool can choose WHAT statistic to compute but never WHETHER to reject.

    Returns ``None`` when *suff* is missing or not an adjudicable shape; the
    caller then treats the tool as descriptive (``reject=False``).
    """
    if not isinstance(suff, dict):
        return None
    kind = suff.get("kind")
    try:
        if kind == "paired_binary":
            b, c = int(suff["b"]), int(suff["c"])
            if b < 0 or c < 0:
                return None
            n = b + c
            e_value = evalue_bernoulli(b, n, p0=0.5) if n > 0 else 1.0
            reject = e_value >= 1.0 / alpha
            effect = (b - c) / n if n > 0 else 0.0
            return reject, float(e_value), effect, None
        if kind == "two_group":
            a = [int(x) for x in suff["a"]]
            grp_b = [int(x) for x in suff["b"]]
            if not a or not grp_b:
                return None
            sr = compare(a, grp_b, paired=False, alpha=alpha)
            return bool(sr.reject), sr.e_value, sr.effect, tuple(sr.ci)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _parse_result(stdout: str, name: str, alpha: float = 0.05) -> StatsToolResult:
    """Parse the last ``STATS_RESULT_JSON=`` line of *stdout*.

    The generated script supplies a DESCRIPTION (summary/effect/ci/details) plus
    SUFFICIENT STATISTICS; the host reconstructs the *decision*
    (reject / e-value) from those statistics via the validated core. A
    self-declared ``reject``/``e_value``/``p_value`` in the JSON is IGNORED — the
    LLM proposes evidence, it never adjudicates it. A tool with no adjudicable
    sufficient statistic is descriptive only (``reject=False``), mirroring the
    ``single_rate_evalue`` muzzle, so it can never reach M5's headline.
    """
    marker_line = None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith(_RESULT_MARKER):
            marker_line = s[len(_RESULT_MARKER):]
    if marker_line is None:
        return StatsToolResult(
            tool=f"generated:{name}", ok=False,
            error="no STATS_RESULT_JSON line in output",
            summary=f"generated:{name} produced no result line",
            details={"stdout_tail": stdout.strip()[-300:]},
        )
    try:
        data = json.loads(marker_line)
    except json.JSONDecodeError as exc:
        return StatsToolResult(
            tool=f"generated:{name}", ok=False,
            error=f"unparseable STATS_RESULT_JSON: {exc}",
            summary=f"generated:{name} bad result json",
        )

    ci = data.get("ci")
    ci_tuple = tuple(ci) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
    effect = _as_float(data.get("effect"))
    details = data.get("details") if isinstance(data.get("details"), dict) else {}

    # HOST reconstructs the decision; the script's own reject/e_value/p_value are
    # never trusted as a verdict (closes the codegen p-hacking hole).
    recon = _reconstruct_decision(data.get("sufficient"), alpha)
    if recon is not None:
        reject, e_value, eff_h, ci_h = recon
        if eff_h is not None:
            effect = eff_h
        if ci_h is not None:
            ci_tuple = ci_h
        details = {**details, "host_adjudicated": True}
    else:
        reject, e_value = False, None
        details = {**details, "host_adjudicated": False, "descriptive_only": True}

    return StatsToolResult(
        tool=f"generated:{name}",
        ok=True,
        summary=str(data.get("summary", "")) or f"generated:{name}",
        effect=effect,
        ci=ci_tuple,
        reject=reject,
        e_value=e_value,
        p_value=None,  # diagnostic at most; never a host decision input
        underpowered=bool(data.get("underpowered", False)),
        analysis_key=f"generated:{name}",
        correction_family="e_bh" if e_value is not None else None,
        raw_reject=reject,
        details=details,
    )


def _as_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
