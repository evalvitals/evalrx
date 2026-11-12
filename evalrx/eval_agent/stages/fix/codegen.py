"""Agent-written (coded) repair pipelines: write, run, repair.

The coding agent (a CLI coder first, the judge as fallback) writes a brand-new
pipeline as Python; it runs sandboxed with bridged model access
(:mod:`.pipeline`). A failed run gets one repair round, and a frozen-model
control records which cases the code solves without the model.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.eval_agent.prompts.fix_agent import _L2_CODE_PROMPT, _REPAIR_PROMPT_BODY
from evalrx.eval_agent.stages.fix.descriptions import code_description
from evalrx.eval_agent.stages.fix.pipeline import (
    CodedPipelineResult,
    run_coded_pipeline,
    score_outputs,
)
from evalrx.eval_agent.stages.fix.recorder import FixRecorder
from evalrx.eval_agent.stages.fix.tiers import FixTier
from evalrx.eval_agent.stages.fix.tools import TEXT_ONLY_CATALOG_NOTE, catalog_text
from evalrx.eval_agent.stages.fix.types import FixCandidate
from evalrx.eval_agent.stages.probe_generator import _extract_code

if TYPE_CHECKING:
    from evalrx.agent_runtime.cli_types import CliAgentConfig
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model
    from evalrx.eval_agent.run_context import Trial

logger = logging.getLogger(__name__)


def code_calls_name(code: str, name: str) -> bool:
    """Return whether executable source directly calls a named model bridge."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


def code_copies_example(code: str, examples: str, *, min_words: int = 8) -> bool:
    """Reject generated programs that memorize an example ID or prompt phrase."""
    try:
        constants = [
            node.value
            for node in ast.walk(ast.parse(str(code or "")))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
    except SyntaxError:
        return False
    literal_text = "\n".join(constants).lower()
    ids = re.findall(r"^###\s+(?:FAIL|PASS)\s+case\s+(\S+)", examples, re.MULTILINE)
    if any(case_id.lower() in literal_text for case_id in ids):
        return True
    prompts = re.findall(r"^PROMPT:\s*(.+)$", examples, re.MULTILINE)
    literal_words = re.findall(r"[a-z0-9]+", literal_text)
    literal_ngrams = {
        tuple(literal_words[i : i + min_words])
        for i in range(max(0, len(literal_words) - min_words + 1))
    }
    for prompt in prompts:
        words = re.findall(r"[a-z0-9]+", prompt.lower())
        if any(
            tuple(words[i : i + min_words]) in literal_ngrams
            for i in range(max(0, len(words) - min_words + 1))
        ):
            return True
    return False


def code_redefines_model_bridge(code: str) -> bool:
    """Reject user pipelines that shadow host-provided model bridge functions."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return False
    bridge_names = {"model_generate", "model_attend"}
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in bridge_names
        for node in ast.walk(tree)
    )


class PipelineCoder:
    """Writes, runs and repairs coded repair pipelines for one fix agent.

    Owns the coder's scratch state (the shared sandbox, the last raw stream /
    usage / repair prompt); nothing outside this class reads it.
    """

    def __init__(
        self,
        *,
        judge: "Model | None",
        cli_config: "CliAgentConfig | None",
        allow_codegen: bool,
        score: "Callable[[FailureCase, str], Optional[bool]]",
        recorder: FixRecorder,
        run_context: "Any | None" = None,
        sandbox: "Any | None" = None,
        exec_timeout_sec: int = 600,
        concurrency: int = 1,
        max_repair_rounds: int = 1,
        prewritten_code: str = "",
    ) -> None:
        self._judge = judge
        self._cli_config = cli_config
        self._allow_codegen = allow_codegen
        self.score = score
        self.recorder = recorder
        self.run_context = run_context
        self.sandbox = sandbox
        self.exec_timeout_sec = exec_timeout_sec
        self.concurrency = concurrency
        self.max_repair_rounds = max_repair_rounds
        self._prewritten_code = str(prewritten_code or "")
        self._last_repair_prompt = ""
        self._last_raw_stream = ""
        self._last_usage: dict | None = None

    @property
    def available(self) -> bool:
        """True when there is a code-writing backend (CLI coder or judge)."""
        return self._allow_codegen and (
            self._judge is not None
            or (self._cli_config is not None and self._cli_config.provider != "llm")
        )

    def selection_guidance(self, prior_text: str = "") -> str:
        """Tell codegen whether this is discovery, revision, or one-shot use."""
        if prior_text:
            return (
                "- This is a FEEDBACK-DRIVEN EXPLORE revision. Use the prior "
                "aggregate fixed/broken counts and implementation below as "
                "training feedback. Raw case IDs, prompts, and answers are "
                "deliberately withheld: revise the general mechanism, never "
                "match a benchmark item or phrase. Keep baseline_output outside "
                "a label-free, generally applicable gate. If the prior attempt "
                "repaired zero cases, abandon its override mechanism instead of "
                "merely retuning it."
            )
        if self.max_repair_rounds > 1:
            return (
                "- This is EXPLORE round 1, used to learn which task subtypes "
                "benefit before a later candidate is frozen. Treat "
                "baseline_output as the baseline answer and keep it on ties, "
                "but you may use a controlled 2-of-3 alternative consensus "
                "(two DISTINCT enhanced calls agreeing on the same different "
                "answer) so the paired helped/hurt feedback is informative. "
                "Never hard-code answers or compute the benchmark task "
                "outside the model."
            )
        return (
            "- Treat baseline_output (the ORIGINAL direct answer) as the "
            "safety baseline. Keep it unless all 3 independent enhanced/"
            "reasoned passes agree on a different answer and none supports "
            "the baseline. Do not use unconditional majority replacement."
        )

    def propose(
        self,
        hyp_lines: str,
        examples: str,
        model: "Model",
        prior_text: str = "",
        *,
        max_tier: FixTier,
        context_block: str = "",
        catalog: "str | None" = None,
        text_only: bool = False,
    ) -> "list[FixCandidate]":
        """The coding agent writes a brand-new pipeline (CLI first, judge fallback).

        Allocates its trial up front (not after dedup, unlike declarative
        candidates) because the CLI agent needs a workdir *now* to write
        ``pipeline.py`` into — coded candidates dedup on code text, which is
        ~never identical run-to-run, so the orphan-folder risk this would
        otherwise create is negligible.
        """
        from evalrx.core.capability import Capability
        from evalrx.eval_agent.stages.fix.pipeline import (
            CASES_FILENAME,
            RESULT_MARKER,
        )

        trial = (
            self.run_context.new_trial("fixes", "coded_pipeline")
            if self.run_context is not None
            else None
        )
        attend_available = (
            max_tier >= FixTier.L3A_INTERNALS_READ
            and Capability.ATTENTION in getattr(model, "capabilities", frozenset())
        )
        attend_hint = (
            (
                "\n- A function  model_attend(case_id, prompt=None) -> "
                '{"grid": [[float,...],...], "shape": [H, W]}  is ALSO defined: '
                "the model's attention heatmap over image patches (read-only "
                "internals). Use it e.g. to find where the model looks, then "
                "crop_region there and re-ask. This forward pass counts as ONE "
                "of the per-case model-call budget below."
            )
            if attend_available
            else ""
        )
        code, source, prompt, raw = "", "", "", ""
        if catalog is None:
            catalog = TEXT_ONLY_CATALOG_NOTE if text_only else catalog_text()
        selection_guidance = self.selection_guidance(prior_text)
        # Explore rounds learn from a 2-of-3 consensus; a one-shot candidate
        # must have all three enhanced passes agree. The prompt states the
        # same number the host bridge enforces.
        min_support = 2 if self.max_repair_rounds > 1 else 3
        base = dict(
            hypotheses=hyp_lines,
            examples=examples,
            catalog=catalog,
            cases_file=CASES_FILENAME,
            marker=RESULT_MARKER,
            attend_hint=attend_hint,
            context=context_block,
            selection_guidance=selection_guidance,
            min_support=min_support,
        )
        if self._prewritten_code.strip():
            code = self._prewritten_code
            source = "prewritten"
            prompt = "Frozen prewritten coded pipeline supplied by the caller."
        elif self._cli_config is not None and self._cli_config.provider != "llm":
            prompt = (
                _L2_CODE_PROMPT.format(fences_hint=", written to a file named pipeline.py", **base)
                + prior_text
            )
            code, raw = self._write_with_cli(prompt, trial)
            source = f"cli:{self._cli_config.provider}"
            if code_copies_example(code, examples) or code_redefines_model_bridge(code):
                logger.warning("FixAgent: generated code violated anti-memorization/bridge rules; dropped")
                code = ""
        if not code.strip() and self._judge is not None:
            self._last_raw_stream = ""
            prompt = (
                _L2_CODE_PROMPT.format(fences_hint=" inside a ```python code block", **base)
                + prior_text
            )
            try:
                raw = str(self._judge.generate(prompt))
            except Exception as exc:
                logger.warning("FixAgent: code-writing judge call failed: %s", exc)
                raw = ""
            code = _extract_code(raw)
            # Syntax gate: a judge that answered in prose must not become a
            # "coded pipeline" candidate (the CLI path returns real files).
            if code.strip():
                import ast

                try:
                    ast.parse(code)
                except SyntaxError:
                    logger.warning("FixAgent: judge code failed to parse; dropped")
                    code = ""
            source = "judge"
        if code_copies_example(code, examples) or code_redefines_model_bridge(code):
            logger.warning("FixAgent: generated code violated anti-memorization/bridge rules; dropped")
            code = ""
        self.recorder.codegen(
            "coded_pipeline", prompt, source, code, raw, ok=bool(code.strip()), trial=trial,
            usage=self._last_usage, raw_stream=self._last_raw_stream,
        )
        if not code.strip():
            return []
        # Classify by the intervention the generated program actually uses,
        # not by an optional bridge merely being advertised in its prompt.
        # This prevents black-box multi-call/image-tool scaffolds from being
        # reported as L3a just because the model happens to expose attention.
        enable_attend = attend_available and code_calls_name(code, "model_attend")
        tier = FixTier.L3A_INTERNALS_READ if enable_attend else FixTier.L2_SCAFFOLD
        return [
            FixCandidate(
                tier=tier,
                name="coded_pipeline",
                kind="code",
                description=code_description(code),
                payload={
                    "code": code,
                    "enable_attend": enable_attend,
                    "text_only": bool(text_only),
                    # Prompt instructions are advisory; the host bridge also
                    # enforces the selection rule without seeing gold labels
                    # (anchored on each case's recorded baseline_output).
                    "consensus_min_support": min_support,
                    "max_calls_per_case": 4,
                },
                source=source,
                trial=trial,
            )
        ]

    def run(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        *,
        max_tokens_floor: "int | None" = None,
    ) -> "CodedPipelineResult":
        """Run a coded candidate; a failed run gets ONE coder repair round.

        The execution error (strict-bridge message, timeout, traceback tail) is
        fed back verbatim — the coder fixes its own contract violation instead
        of the candidate silently dying.  The final error, if any, is stashed
        in ``payload["exec_error"]`` for honest escalation accounting.
        """
        workdir = self._workdir(candidate.trial)
        result = run_coded_pipeline(
            candidate.payload["code"],
            model,
            data,
            workdir=workdir,
            timeout_sec=self.exec_timeout_sec,
            enable_attend=bool(candidate.payload.get("enable_attend")),
            max_tokens_floor=max_tokens_floor,
            concurrency=self.concurrency,
            consensus_min_support=int(candidate.payload.get("consensus_min_support", 0)),
            max_calls_per_case=int(candidate.payload.get("max_calls_per_case", 0)),
            call_logger=self.recorder.bridge_exchange,
        )
        if not result.ok and self.available:
            logger.warning("FixAgent: coded pipeline failed (%s) — one repair round", result.error)
            repaired, source, raw = self._repair(candidate, result.error)
            self.recorder.codegen(
                "coded_pipeline_repair",
                self._last_repair_prompt,
                source,
                repaired,
                raw,
                ok=bool(repaired.strip()),
                trial=candidate.trial,
                usage=self._last_usage,
                raw_stream=self._last_raw_stream,
            )
            if repaired.strip():
                candidate.payload["code"] = repaired
                result = run_coded_pipeline(
                    repaired,
                    model,
                    data,
                    workdir=workdir,
                    timeout_sec=self.exec_timeout_sec,
                    enable_attend=bool(candidate.payload.get("enable_attend")),
                    max_tokens_floor=max_tokens_floor,
                    concurrency=self.concurrency,
                    consensus_min_support=int(
                        candidate.payload.get("consensus_min_support", 0)
                    ),
                    max_calls_per_case=int(candidate.payload.get("max_calls_per_case", 0)),
                    call_logger=self.recorder.bridge_exchange,
                )
        candidate.payload["exec_error"] = "" if result.ok else result.error
        candidate.payload["selection_guard"] = {
            "min_support": int(candidate.payload.get("consensus_min_support", 0)),
            "n_guarded": result.n_guarded,
            "guarded_ids": result.guarded_ids,
            # Audit trail for the anchor semantics: how many cases were
            # anchored on their recorded baseline because the pipeline made
            # no direct call, how many plain direct calls were answered from
            # the record, and which cases had nothing to anchor on.
            "n_anchored_from_recorded": result.n_anchored_from_recorded,
            "n_replayed": result.n_replayed,
            "unanchored_ids": result.unanchored_ids,
        }
        if result.unanchored_ids:
            logger.warning(
                "FixAgent: %d case(s) excluded from the coded pipeline result — no "
                "recorded baseline and no direct model_generate(case_id) call to "
                "anchor the selection guard on", len(result.unanchored_ids),
            )
        if candidate.trial is not None:
            # Persist the audit trail next to this attempt's prompt and code:
            # the payload itself is never logged, and "did the guard anchor on
            # the record, did it revert anything" is the first question a
            # reviewer asks of a coded round.
            candidate.trial.write(
                "coded_pipeline_result.json",
                json.dumps({
                    "ok": result.ok,
                    "exec_error": "" if result.ok else result.error,
                    "n_calls": result.n_calls,
                    "n_outputs": len(result.outputs),
                    **candidate.payload["selection_guard"],
                }, indent=1),
            )
        if not result.ok:
            logger.warning("FixAgent: coded pipeline produced no result: %s", result.error)
        return result

    def frozen_model_control(self, candidate: FixCandidate, data: "CaseBatch") -> None:
        """Re-run a coded candidate with the model frozen; record what it still
        gets right in ``payload["frozen_model_control"]``.

        Validation reads ``solved`` and drops those failing cases from the
        paired test: with every model call answered by the model's own recorded
        output, a repair can only have come from the code. A control that does
        not complete (the pipeline crashes or spins without a live model) is
        evidence the pipeline needs the model — nothing is discounted, and the
        reason is kept.
        """
        from pathlib import Path

        from evalrx.eval_agent.stages.fix.pipeline import frozen_model_control

        workdir = Path(self._workdir(candidate.trial)) / "frozen_model_control"
        ctrl = frozen_model_control(
            candidate.payload["code"], data, workdir=workdir,
            timeout_sec=self.exec_timeout_sec,
            consensus_min_support=int(candidate.payload.get("consensus_min_support", 0)),
            max_calls_per_case=int(candidate.payload.get("max_calls_per_case", 0)),
        )
        solved: "list[str]" = []
        if ctrl.ok:
            for cid, ok in score_outputs(ctrl, data, self.score).items():
                if ok is True:
                    solved.append(cid)
        candidate.payload["frozen_model_control"] = {
            "ok": ctrl.ok,
            "error": ctrl.error,
            "n_calls": ctrl.n_calls,
            "solved": solved,
        }
        if candidate.trial is not None:
            candidate.trial.write(
                "frozen_model_control.json",
                json.dumps(candidate.payload["frozen_model_control"], indent=1),
            )
        if solved:
            logger.info(
                "FixAgent: frozen-model control — %s still solves %d/%d case(s) with the "
                "model held at its recorded answers", candidate.name, len(solved), len(data),
            )

    def _repair(self, candidate: FixCandidate, error: str) -> "tuple[str, str, str]":
        """Ask the coder to fix its failed pipeline; returns (code, source, raw)."""
        from pathlib import Path

        from evalrx.eval_agent.stages.fix.pipeline import (
            CASES_FILENAME,
            RESULT_MARKER,
        )

        attend_clause = (
            " and model_attend(case_id, prompt=None)"
            if candidate.payload.get("enable_attend")
            else ""
        )
        base = _REPAIR_PROMPT_BODY.format(
            error=error[:600],
            code=str(candidate.payload.get("code", ""))[:4000],
            attend_clause=attend_clause,
            catalog=(TEXT_ONLY_CATALOG_NOTE if candidate.payload.get("text_only")
                     else catalog_text()),
            cases_file=CASES_FILENAME,
            marker=RESULT_MARKER,
            # The repair round restates the selection contract in full: a
            # repair that only echoes the execution error used to re-violate
            # the selection rule it was never told about.
            min_support=int(candidate.payload.get("consensus_min_support", 0) or 0),
            selection_guidance=self.selection_guidance(),
        )
        code, source, raw = "", "", ""
        if self._cli_config is not None and self._cli_config.provider != "llm":
            # The execution host consumes/removes pipeline.py. Restore the
            # user's source before asking the CLI coder to edit it, otherwise
            # an agent can mistake fix_pipeline_exec.py for the target.
            Path(self._workdir(candidate.trial), "pipeline.py").write_text(
                str(candidate.payload.get("code", "")), encoding="utf-8"
            )
            self._last_repair_prompt = (
                base + "\nWrite the corrected code to a file named pipeline.py."
            )
            code, raw = self._write_with_cli(self._last_repair_prompt, candidate.trial)
            source = f"cli:{self._cli_config.provider}"
        if not code.strip() and self._judge is not None:
            self._last_raw_stream = ""
            self._last_repair_prompt = (
                base + "\nReturn ONLY the corrected Python code inside a ```python code block."
            )
            try:
                raw = str(self._judge.generate(self._last_repair_prompt))
            except Exception as exc:
                logger.warning("FixAgent: repair judge call failed: %s", exc)
                raw = ""
            code = _extract_code(raw)
            if code.strip():
                import ast

                try:
                    ast.parse(code)
                except SyntaxError:
                    code = ""
            source = "judge"
        if code_redefines_model_bridge(code):
            logger.warning("FixAgent: repaired code redefined a host model bridge; dropped")
            code = ""
        return code, source, raw

    def _write_with_cli(self, prompt: str, trial: "Trial | None" = None) -> "tuple[str, str]":
        from pathlib import Path

        from evalrx.agent_runtime.codegen import CodegenRunner

        workdir = Path(self._workdir(trial))
        result = CodegenRunner(self._cli_config).write_code(  # type: ignore[arg-type]
            prompt,
            workdir=workdir,
            timeout_sec=self._cli_config.timeout_sec,  # type: ignore[union-attr]
            preferred_filenames=("pipeline.py",),
        )
        self._last_usage = result.usage
        self._last_raw_stream = ""
        if result.raw_stream_path:
            try:
                self._last_raw_stream = (workdir / result.raw_stream_path).read_text(encoding="utf-8")
            except OSError:
                pass
        return result.code, result.raw_output

    def _workdir(self, trial: "Trial | None" = None) -> str:
        """Sandbox workdir for a coded run.

        With a *trial* (a ``RunContext`` is in play), each candidate gets its
        own durable ``<trial>/workspace/`` — no cross-attempt overwriting.
        Without one (legacy / no ``RunContext``), falls back to a single
        shared sandbox for the agent's lifetime, exactly as before.
        """
        if trial is not None:
            return str(trial.workspace)
        if self.sandbox is None:
            from evalrx.agent_runtime.sandbox import ExperimentSandbox

            workdir = (
                self.run_context.new_workdir("fix") if self.run_context is not None else None
            )
            self.sandbox = ExperimentSandbox(workdir=workdir)
        return str(self.sandbox.workdir)
