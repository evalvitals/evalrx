"""M1 tier (b), white-box — capture-then-compute probe generation.

Black-box generated probes (:mod:`probe_generator`) only see the model's text
outputs.  Mechanism hypotheses from M3 ("structural tokens act as attention
sinks", "attention never leaves the text region", …) need **internal** evidence
— but the loaded model cannot be shipped into a sandbox.  The split here keeps
the M2-tier(b) safety model intact:

1. **Host captures** (trusted code): one ``model.forward`` per case with
   ``Capability.ATTENTION``; the heavy trace is immediately reduced to a small
   standard bundle — head-averaged attention from the last query position for
   every layer ``(n_layers, seq)`` float16 — plus the token strings and the
   image-token mask, dumped to the sandbox workdir.
2. **Sandboxed compute** (generated code): an LLM/CLI-written numpy script
   reads the dump, computes a per-case scalar/boolean signal for the
   hypothesised mechanism, and prints a strict ``PROBE_RESULT_JSON=`` line.
3. The parsed findings become a normal :class:`~evalvitals.core.result.Result`
   whose ``per_case`` entries flow into M2's stats layer and M5's routing.

Dump layout (documented verbatim in the generation prompt):

    m1_whitebox_manifest.json   {"cases": [{"id", "label", "seq_len",
                                            "n_layers", "tokens": [...]}, ...]}
    m1_whitebox/<case_id>.npz   attn_last:        (n_layers, seq) float16
                                image_token_mask: (seq,) bool

Scope (v1): read-only statistics over the captured attention bundle.  Causal
interventions (activation patching, attention knockout) need in-process
execution with the model handle and are deliberately out of scope.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from evalvitals.agent_runtime.sandbox import ExperimentSandbox
from evalvitals.core.capability import Capability
from evalvitals.core.result import Result
from evalvitals.eval_agent.prompts.whitebox_probe_generator import (
    _DUMP_DIR,
    _GENERATE_PROMPT,
    _MANIFEST,
    _RESULT_MARKER,
)

if TYPE_CHECKING:
    from evalvitals.agent_runtime.cli_types import CliAgentConfig
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)



@dataclass
class GeneratedWhiteboxProbe:
    """A validated, sandbox-executed white-box probe (re-runnable on new cases)."""

    name: str
    code: str
    need: str = ""
    source: str = ""


class WhiteboxProbeGenerator:
    """Generate and run capture-then-compute attention probes.

    Args:
        judge:       LLM for the single-pass code-writing path.
        cli_config:  CLI coding-agent config (used before the judge when set).
        sandbox:     Execution sandbox (fresh temp dir when ``None``).
        timeout_sec: Wall-clock limit per sandbox run.
        max_cases:   Cap on captured cases (one ATTENTION forward each); 0 (the default) = every case.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        sandbox: "ExperimentSandbox | None" = None,
        timeout_sec: int = 60,
        max_cases: int = 0,
        run_logger: "Any | None" = None,
    ) -> None:
        self._judge = judge
        self._cli_config = cli_config
        self._timeout_sec = timeout_sec
        self._max_cases = max_cases
        self._sandbox = sandbox or ExperimentSandbox()
        # Optional RunLogger — records each attention-probe code-writing attempt.
        self.run_logger = run_logger
        self._last_prompt: str = ""
        self._last_raw: str = ""
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
    ) -> "tuple[Result | None, GeneratedWhiteboxProbe | None]":
        """Capture internals, write+run a probe over the dump, return (Result, probe)."""
        if not self.available:
            logger.debug("WhiteboxProbeGenerator: no code-writing backend configured")
            return None, None
        if Capability.ATTENTION not in getattr(model, "capabilities", frozenset()):
            logger.debug("WhiteboxProbeGenerator: model lacks ATTENTION capability")
            return None, None

        n = self._capture(model, cases)
        if n == 0:
            logger.warning("WhiteboxProbeGenerator: no case could be captured")
            return None, None

        self._last_prompt = ""
        self._last_raw = ""
        self._last_usage = None
        try:
            code, source = self._write_code(need)
        except Exception as exc:
            logger.warning("WhiteboxProbeGenerator: code writing failed: %s", exc)
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
        probe = GeneratedWhiteboxProbe(name=name, code=code, need=need, source=source)
        return result, probe

    def _emit_codegen(
        self, name: str, need: str, source: str, code: str, *, ok: bool, error: str = ""
    ) -> None:
        """Record one attention-probe code-writing attempt to the RunLogger."""
        if self.run_logger is None:
            return
        extra = ({"cli_usage": self._last_usage}
                 if source.startswith("cli:") and self._last_usage else None)
        try:
            self.run_logger.log_tool_codegen(
                module="m1_whitebox", name=name, need=need, source=source, ok=ok,
                code=code, prompt=self._last_prompt, raw_output=self._last_raw, error=error,
                extra=extra,
            )
        except Exception as exc:  # logging must never break generation
            logger.debug("WhiteboxProbeGenerator: log_tool_codegen failed: %s", exc)

    def run_cached(
        self,
        probe: GeneratedWhiteboxProbe,
        model: "Model",
        cases: "CaseBatch",
    ) -> "Result | None":
        """Re-capture and re-run an already-generated probe (no LLM call)."""
        if self._capture(model, cases) == 0:
            return None
        return self._run_code(probe.code, probe.name, model, cases)

    # ------------------------------------------------------------------
    # Host capture (trusted)
    # ------------------------------------------------------------------

    def _capture(self, model: "Model", cases: "CaseBatch") -> int:
        """Forward each case, reduce, and dump the attention bundle. Returns n captured."""
        workdir = Path(self._sandbox.workdir)
        dump_dir = workdir / _DUMP_DIR
        dump_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict[str, Any]] = []
        selected = list(cases)
        if self._max_cases > 0:  # 0 = every case
            selected = selected[: self._max_cases]
        for case in selected:
            try:
                trace = model.forward(case.inputs, capture={Capability.ATTENTION})
                attns = trace.require(Capability.ATTENTION)
                rows = np.stack([
                    a.float().mean(dim=0)[-1].cpu().numpy() for a in attns
                ]).astype(np.float16)                       # (n_layers, seq)
                mask = trace.extras.get("image_token_mask")
                mask_np = (
                    mask.cpu().numpy().astype(bool)
                    if mask is not None else np.zeros(rows.shape[1], dtype=bool)
                )
                tokens = list(trace.tokens)
            except Exception as exc:  # noqa: BLE001 - skip uncapturable cases
                logger.debug("whitebox capture failed for %s: %s", case.id, exc)
                continue
            finally:
                trace = None  # free the heavy attention tensors promptly

            np.savez_compressed(
                dump_dir / f"{case.id}.npz",
                attn_last=rows, image_token_mask=mask_np,
            )
            label = getattr(case, "label", None)
            manifest.append({
                "id": case.id,
                "label": getattr(label, "value", None),
                "seq_len": int(rows.shape[1]),
                "n_layers": int(rows.shape[0]),
                "tokens": tokens,
            })

        (workdir / _MANIFEST).write_text(
            json.dumps({"cases": manifest}), encoding="utf-8"
        )
        return len(manifest)

    # ------------------------------------------------------------------
    # Code writing + sandbox run (mirrors probe_generator)
    # ------------------------------------------------------------------

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
        self._last_usage = result.usage
        return result.code

    def _build_prompt(self, need: str, *, fenced: bool) -> str:
        return _GENERATE_PROMPT.format(
            need=need.strip() or "Probe the attention pattern behind the failures.",
            manifest=_MANIFEST,
            dump_dir=_DUMP_DIR,
            marker=_RESULT_MARKER,
            fences_hint=" inside a ```python code block" if fenced else
                        ", written to a file named probe.py",
        )

    def _run_code(
        self, code: str, name: str, model: "Model", cases: "CaseBatch",
    ) -> "Result | None":
        sandbox_result = self._sandbox.run(code, timeout_sec=self._timeout_sec)
        if not sandbox_result.ok:
            logger.warning(
                "WhiteboxProbeGenerator: sandbox run failed (rc=%s): %s",
                sandbox_result.returncode, (sandbox_result.stderr or "").strip()[:200],
            )
            return None
        return _parse_result(sandbox_result.stdout, name, model, cases)


# ---------------------------------------------------------------------------
# Module helpers (shared shape with probe_generator)
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned.strip()


def _parse_result(
    stdout: str, name: str, model: "Model", cases: "CaseBatch",
) -> "Result | None":
    marker_line: Optional[str] = None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith(_RESULT_MARKER):
            marker_line = s[len(_RESULT_MARKER):]
    if marker_line is None:
        logger.warning("WhiteboxProbeGenerator: no PROBE_RESULT_JSON line in output")
        return None
    try:
        data = json.loads(marker_line)
    except json.JSONDecodeError as exc:
        logger.warning("WhiteboxProbeGenerator: unparseable PROBE_RESULT_JSON: %s", exc)
        return None

    findings: dict[str, Any] = {}
    if isinstance(data.get("findings"), dict):
        findings.update(data["findings"])
    per_case = data.get("per_case")
    if isinstance(per_case, list):
        findings["per_case"] = [e for e in per_case if isinstance(e, dict)]

    return Result(
        analyzer=f"generated_wb:{name}",
        model=repr(model),
        cases=cases,
        findings=findings,
        metadata={"generated": True, "whitebox": True, "probe": name},
    )
