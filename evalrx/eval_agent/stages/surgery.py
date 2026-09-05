"""M5 — SurgeryAgent: perform targeted interventions to verify hypotheses.

Given a hypothesis from the diagnosis agent, the surgery agent operates on the
model/data to check whether the hypothesized cause actually predicts the
observed failures.  Three strategies are available (in priority order):

1. **verify_fn** (injected) — caller supplies the full verification logic.
2. **analyzer_params** (param sweep) — re-run named analyzers with modified
   parameters and surface before/after findings for comparison.
3. **Default** (label correlation) — look for per-case signals already present
   in the analysis results (e.g. ``has_loop``, ``n_ignored``), split cases into
   "signal" vs "no-signal" groups, and compare FAIL rates using a 10 % gap
   threshold.  Cases in the signal group can be filtered out to produce
   ``new_data`` for the next M1 cycle.

Usage::

    agent = SurgeryAgent()
    iv = agent.operate(hypothesis, model, results, data)
    print(iv.status, iv.fixed, iv.evidence)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from evalrx.core.case import CaseBatch, Label
from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus

if TYPE_CHECKING:
    from evalrx.core.model import Model
    from evalrx.core.result import Result


@dataclass
class InterventionResult:
    """Output of :class:`SurgeryAgent.operate`.

    Attributes:
        hypothesis:          The hypothesis under test.
        status:              Updated :class:`~evalrx.eval_agent.hypothesis.HypothesisStatus`.
        fixed:               ``True`` when the intervention completely separates failing from
                             passing cases (signal group always fails, control group never
                             fails).  Signals the loop that the problem is resolved.
        evidence:            Supporting statistics or sweep findings.
        new_data:            When *status* is SUPPORTED, the cases **not** in the signal
                             group — the refined subset for the next M1 cycle.
        confidence_score:    Float in [0, 1] measuring overall evidence strength.  Derived
                             from the combination of signal-vs-control gap, sample adequacy,
                             and control-group cleanliness.  A binary ``fixed`` flag alone
                             cannot convey how strongly supported the finding is — e.g.
                             a 10 % gap on 3 cases is very different from a 90 % gap on 300.
        evidence_dimensions: Breakdown of ``confidence_score`` by sub-criterion so the loop
                             and callers can make finer-grained decisions (e.g. raise a
                             INCONCLUSIVE when ``sample_adequacy`` is low even if the gap
                             looks large).
    """

    hypothesis: Hypothesis
    status: HypothesisStatus
    fixed: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    new_data: CaseBatch | None = None
    confidence_score: float = 0.0
    evidence_dimensions: dict[str, float] = field(default_factory=dict)
    # Rich experiment payload from the experiment-writer strategy (strategy 3):
    # generated source files, run stdout/stderr, blueprint, verdict, the agent's
    # intermediate thinking and the working directory.  Kept separate from
    # ``evidence`` (which is logged inline) so RunLogger.log_experiment can
    # persist the heavy parts to disk instead of bloating the JSONL event.
    # ``None`` for the passive strategies (label correlation / param sweep).
    experiment: dict[str, Any] | None = None


# Keys that carry diagnostic meaning in per-case finding entries.
# Excluded from the signal scan because they are indices, not boolean flags.
def _serialize_cases(data: CaseBatch, image_dir: "Any | None" = None) -> str:
    """Serialize *data* to a compact JSON string embeddable in a script.

    Includes the GOLD (``expected``) and the graded GENERATION (``observed``).
    Without them a whole class of hypothesis is untestable in the sandbox rather
    than merely hard: M1 and M3 keep proposing "these labels are grading
    artifacts" on free-response sets, and the judge cannot check a grading claim
    with no gold to grade against.  Observed live on qwen3.5-2b/minervamath,
    where the experiment said so itself --

        "cases.json carries only {prompt, label, id} -- no gold answer, no
         stored output.  So `gold_in_output` cannot be recomputed directly.
         So the claim is tested on its MECHANISM instead"

    -- substituted a resampling proxy, and returned SUPPORTED for a hypothesis
    that regrading the full batch refutes outright (0 of 114 PASS labels wrong).

    This is NOT the sandbox that runs a candidate fix.  That one is
    :func:`~evalrx.eval_agent.stages.fix_pipeline.cases_payload`, which is
    id+prompt only on purpose, so a fix cannot be written against the answers.
    The split matters: a DIAGNOSTIC needs the gold to diagnose grading, a REPAIR
    must not see it. Widening this function does not widen that one.

    When *image_dir* is provided, PIL images are saved as JPEG files there and
    the path is included as ``image_path`` so the generated diagnostic script
    (and codex) can load them.
    """
    from pathlib import Path as _Path

    records = []
    for case in data:
        rec: dict[str, Any] = {"prompt": str(case.inputs)}
        if getattr(case, "label", None) is not None:
            rec["label"] = str(case.label)
        if getattr(case, "id", None):
            rec["id"] = case.id
        expected = getattr(case, "expected", None)
        if expected is not None:
            rec["expected"] = expected if isinstance(
                expected, (str, int, float, bool)) else str(expected)
        observed = getattr(case, "observed", None)
        if observed is not None:
            rec["observed"] = str(observed)
        meta = getattr(case, "metadata", {}) or {}
        if meta:
            rec["metadata"] = {k: v for k, v in meta.items()
                               if isinstance(v, (str, int, float, bool, type(None)))}

        # Save image to disk so codex / the diagnostic script can read it
        image = getattr(case.inputs, "image", None) if hasattr(case, "inputs") else None
        if image is not None and image_dir is not None:
            try:
                img_dir = _Path(image_dir)
                img_dir.mkdir(parents=True, exist_ok=True)
                case_id = case.id or f"case_{len(records)}"
                img_path = img_dir / f"{case_id}.jpg"
                # PIL image
                image.convert("RGB").save(str(img_path), format="JPEG")
                rec["image_path"] = str(img_path)
            except Exception:
                pass  # non-PIL or unsavable — skip silently

        records.append(rec)
    return json.dumps(records, indent=2)


_NON_SIGNAL_KEYS = frozenset(
    {"sample_id", "id", "step", "first_error_step", "action", "judge_raw"}
)

# Minimum cases per group before sample_adequacy reaches 1.0
_ADEQUACY_SATURATION = 30


def _compute_confidence(
    fail_signal: float,
    fail_control: float,
    n_signal: int,
    n_control: int,
) -> tuple[float, dict[str, float]]:
    """Compute overall confidence and per-dimension breakdown.

    Three dimensions, each in [0, 1]:
    - ``evidence_gap``:     normalised fail-rate difference (how much worse signal group is).
    - ``sample_adequacy``:  whether both groups are large enough to trust the gap.
    - ``control_cleanliness``: how low the control fail-rate is (ideally 0).

    ``confidence_score`` = geometric mean of all three so that a near-zero on
    any single dimension collapses the overall score — a 90 % gap on 2 cases
    should not produce high confidence.
    """
    evidence_gap = max(0.0, min(1.0, (fail_signal - fail_control) / 1.0))
    n_min = min(n_signal, n_control) if n_control > 0 else n_signal
    sample_adequacy = min(1.0, n_min / _ADEQUACY_SATURATION)
    control_cleanliness = 1.0 - fail_control

    dims = {
        "evidence_gap": round(evidence_gap, 3),
        "sample_adequacy": round(sample_adequacy, 3),
        "control_cleanliness": round(control_cleanliness, 3),
    }
    # Geometric mean: penalises any dimension being near zero
    product = evidence_gap * sample_adequacy * control_cleanliness
    confidence = round(product ** (1.0 / 3.0), 3) if product > 0 else 0.0
    return confidence, dims


def _extract_per_case_signals(results: dict[str, "Result"]) -> dict[str, bool]:
    """Collect a per-case boolean signal from all per_case finding entries.

    A case is marked as having a signal if any finding entry for it contains a
    truthy numeric or boolean value (e.g. ``has_loop=True``, ``n_ignored=2``).
    """
    signal: dict[str, bool] = {}
    for result in results.values():
        for entry in result.findings.get("per_case", []):
            cid = entry.get("sample_id") or entry.get("id", "")
            if not cid:
                continue
            hit = any(
                isinstance(v, (int, float, bool)) and bool(v)
                for k, v in entry.items()
                if k not in _NON_SIGNAL_KEYS
            )
            signal[cid] = signal.get(cid, False) or hit
    return signal


class SurgeryAgent:
    """Verify hypotheses by operating on the model or data.

    Strategy selection (first match wins):

    1. ``verify_fn`` injected → full custom override.
    2. ``analyzer_params`` provided → param sweep (re-run named analyzers).
    3. ``judge`` provided → :class:`~evalrx.eval_agent.experiment_writer.ExperimentWriter`
       writes a targeted Python script, runs it in a sandbox, and interprets
       the ``verdict`` metric printed to stdout.
    4. Default → label-correlation analysis (passive, no code execution).

    Args:
        verify_fn:       Custom verification callable.
        analyzer_params: ``{analyzer_name: {param: value}}`` for param sweep.
        judge:           Any ``Model`` with ``Capability.GENERATE`` used by the
                         experiment writer to write diagnostic code.  When ``None``
                         the writer is disabled and the agent falls back to label
                         correlation.
        sandbox_dir:     Directory for sandbox script files.  A temp dir is
                         created automatically when ``None``.  Sets up ONE
                         shared sandbox for the agent's lifetime — pass this
                         (or omit ``run_context``) to keep that legacy
                         behaviour even when a ``RunContext`` is given.
        writer_config:   :class:`~evalrx.eval_agent.experiment_writer.ExperimentWriterConfig`
                         controlling phases and limits.
        run_context:     Optional :class:`~evalrx.eval_agent.run_context.RunContext`.
                         When set (and *sandbox_dir* is not), each ``operate()``
                         call allocates its own self-contained *trial* under
                         ``experiments/`` (code + sandbox + record.md, see
                         :meth:`RunContext.new_trial`) instead of every M5
                         experiment sharing — and overwriting — one sandbox.
    """

    def __init__(
        self,
        verify_fn: Callable[
            [Hypothesis, "Model", dict[str, "Result"], CaseBatch],
            InterventionResult,
        ]
        | None = None,
        analyzer_params: dict[str, dict[str, Any]] | None = None,
        judge: "Model | None" = None,
        sandbox_dir: "str | None" = None,
        writer_config: "Any | None" = None,
        run_context: "Any | None" = None,
    ) -> None:
        self.verify_fn = verify_fn
        self.analyzer_params = analyzer_params or {}

        self._writer = None
        self._sandbox = None
        self._run_context = run_context
        if judge is not None:
            from evalrx.eval_agent.stages.experiment_writer import (
                ExperimentWriter,
                ExperimentWriterConfig,
            )

            cfg = writer_config if isinstance(writer_config, ExperimentWriterConfig) \
                else ExperimentWriterConfig()
            self._writer = ExperimentWriter(judge=judge, config=cfg)
            if run_context is None or sandbox_dir is not None:
                # No RunContext (or an explicit override): one shared sandbox
                # for the agent's lifetime, exactly as before.
                from evalrx.agent_runtime.sandbox import ExperimentSandbox

                self._sandbox = ExperimentSandbox(workdir=sandbox_dir)
            # Else: each operate() call allocates its own trial + durable,
            # non-cleaned-up sandbox — see _execute_experiment.

    def operate(
        self,
        hypothesis: Hypothesis,
        model: "Model",
        results: dict[str, "Result"],
        data: CaseBatch,
    ) -> InterventionResult:
        """Perform the intervention for *hypothesis* and return the outcome."""
        if self.verify_fn is not None:
            return self.verify_fn(hypothesis, model, results, data)
        if self.analyzer_params:
            return self._param_sweep(hypothesis, model, data)
        if self._writer is not None:
            return self._execute_experiment(hypothesis, model, data)
        return self._correlate_with_labels(hypothesis, results, data)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _correlate_with_labels(
        self,
        hypothesis: Hypothesis,
        results: dict[str, "Result"],
        data: CaseBatch,
    ) -> InterventionResult:
        """Default: correlate per-case signals with PASS/FAIL labels."""
        signal = _extract_per_case_signals(results)

        if not signal:
            return InterventionResult(
                hypothesis=hypothesis,
                status=HypothesisStatus.INCONCLUSIVE,
                fixed=False,
                evidence={"reason": "no per-case findings available to correlate"},
            )

        # Index by both the case UUID and the trajectory sample_id so that
        # per-case entries (which use sample_id) can be matched.
        labeled: dict[str, bool] = {}
        for c in data:
            if getattr(c, "label", None) is None:
                continue
            is_fail = c.label == Label.FAIL
            labeled[c.id] = is_fail
            traj = getattr(c, "trajectory", None)
            if traj is not None:
                labeled[getattr(traj, "sample_id", "")] = is_fail

        if not labeled:
            return InterventionResult(
                hypothesis=hypothesis,
                status=HypothesisStatus.INCONCLUSIVE,
                fixed=False,
                evidence={"reason": "no labeled cases to correlate with"},
            )

        with_signal    = [labeled[cid] for cid, hit in signal.items() if hit     and cid in labeled]
        without_signal = [labeled[cid] for cid, hit in signal.items() if not hit and cid in labeled]

        if not with_signal:
            return InterventionResult(
                hypothesis=hypothesis,
                status=HypothesisStatus.REFUTED,
                fixed=False,
                evidence={"reason": "no cases match the hypothesis signal"},
            )

        fail_signal  = sum(with_signal)    / len(with_signal)
        fail_control = sum(without_signal) / len(without_signal) if without_signal else 0.0

        evidence = {
            "n_with_signal":    len(with_signal),
            "n_without_signal": len(without_signal),
            "fail_rate_signal":  round(fail_signal, 3),
            "fail_rate_control": round(fail_control, 3),
        }

        confidence, dims = _compute_confidence(
            fail_signal, fail_control, len(with_signal), len(without_signal)
        )

        if fail_signal > fail_control + 0.10:
            status = HypothesisStatus.SUPPORTED
            fixed  = fail_signal >= 1.0 and fail_control == 0.0
        elif fail_signal < fail_control - 0.05:
            status = HypothesisStatus.REFUTED
            fixed  = False
        else:
            status = HypothesisStatus.INCONCLUSIVE
            fixed  = False

        new_data: CaseBatch | None = None
        if status == HypothesisStatus.SUPPORTED:
            new_data = CaseBatch([c for c in data if not signal.get(c.id, False)])

        return InterventionResult(
            hypothesis=hypothesis,
            status=status,
            fixed=fixed,
            evidence=evidence,
            new_data=new_data,
            confidence_score=confidence,
            evidence_dimensions=dims,
        )

    def _execute_experiment(
        self,
        hypothesis: Hypothesis,
        model: "Model",
        data: CaseBatch,
    ) -> InterventionResult:
        """M5 strategy 3: write + execute a targeted diagnostic script.

        Mirrors ``researchclaw`` Stage-14 diagnosis + repair loop:

        1. :class:`~evalrx.eval_agent.experiment_writer.ExperimentWriter`
           generates a self-contained Python script via the LLM judge.
        2. The script is run in a subprocess sandbox (exec-fix loop).
        3. ``verdict: 1.0`` → SUPPORTED; ``verdict: 0.0`` → REFUTED;
           no verdict or crash → INCONCLUSIVE.
        """
        from evalrx.eval_agent.stages.experiment_writer import build_model_context

        sandbox = self._sandbox
        trial = None
        if sandbox is None and self._run_context is not None:
            from evalrx.agent_runtime.sandbox import ExperimentSandbox

            label = getattr(hypothesis, "predicted_failure_mode", "") or "experiment"
            trial = self._run_context.new_trial("experiments", label)
            # cleanup=False: a successful run's script must stay on disk —
            # it's the whole point of giving this experiment its own folder.
            sandbox = ExperimentSandbox(workdir=str(trial.workspace), cleanup=False)

        # The model under test is already resident in the parent process.  A
        # second evalrx.load() in M5 can duplicate tens of GB of weights
        # and OOM (notably Qwen Omni).  Generated diagnostics therefore operate
        # on frozen case artifacts; fresh interventions remain host-run M4/fix
        # work and an artifact-insufficient M5 result is INCONCLUSIVE.
        model_context = build_model_context(model, allow_reconstruction=False)
        # Save images alongside cases.json so codex can load them
        image_dir = getattr(sandbox, "workdir", None)
        cases_json = _serialize_cases(data, image_dir=image_dir)

        writer_result = self._writer.write_and_run(  # type: ignore[union-attr]
            hypothesis=hypothesis,
            model_context=model_context,
            cases_json=cases_json,
            sandbox=sandbox,  # type: ignore[arg-type]
        )

        evidence: dict[str, Any] = {
            "metrics": writer_result.metrics,
            "returncode": writer_result.returncode,
            "timed_out": writer_result.timed_out,
            "llm_calls": writer_result.total_llm_calls,
            "sandbox_runs": writer_result.total_sandbox_runs,
            "validation_log": writer_result.validation_log,
        }

        # Full experiment payload for RunLogger.log_experiment — the generated
        # script(s), the run's output, the agent's thinking, and the workspace.
        experiment: dict[str, Any] = {
            "module": "m5",
            "provider": writer_result.provider,
            "code": writer_result.code,
            "files": writer_result.files,
            "blueprint": writer_result.blueprint,
            "stdout": writer_result.stdout,
            "stderr": writer_result.stderr,
            "verdict": writer_result.verdict,
            "metrics": writer_result.metrics,
            "returncode": writer_result.returncode,
            "timed_out": writer_result.timed_out,
            "validation_log": writer_result.validation_log,
            "cli_raw_output": writer_result.cli_raw_output,
            "cli_usage": writer_result.cli_usage,
            "llm_calls": writer_result.total_llm_calls,
            "sandbox_runs": writer_result.total_sandbox_runs,
            "model_calls": writer_result.model_calls,
            "workdir": writer_result.workdir,
            # Self-contained trial folder (see RunContext.new_trial) — None
            # when no RunContext is in play (legacy flat layout).
            "trial_root": str(trial.root) if trial is not None else None,
        }

        # Crashed or timed out with no metrics → inconclusive
        if not writer_result.ok and not writer_result.metrics:
            return InterventionResult(
                hypothesis=hypothesis,
                status=HypothesisStatus.INCONCLUSIVE,
                fixed=False,
                evidence={**evidence, "reason": "script did not produce metrics"},
                experiment=experiment,
            )

        verdict = writer_result.verdict
        if verdict is None:
            return InterventionResult(
                hypothesis=hypothesis,
                status=HypothesisStatus.INCONCLUSIVE,
                fixed=False,
                evidence={**evidence, "reason": "no verdict line in output"},
                experiment=experiment,
            )

        if verdict >= 0.5:
            status = HypothesisStatus.SUPPORTED
            # "Fixed" means confidence is very high (verdict == 1.0 and high confidence)
            confidence = writer_result.metrics.get("confidence", verdict)
            fixed = verdict >= 1.0 and confidence >= 0.9
        else:
            status = HypothesisStatus.REFUTED
            fixed = False

        return InterventionResult(
            hypothesis=hypothesis,
            status=status,
            fixed=fixed,
            evidence=evidence,
            experiment=experiment,
        )

    def _param_sweep(
        self,
        hypothesis: Hypothesis,
        model: "Model",
        data: CaseBatch,
    ) -> InterventionResult:
        """Re-run specified analyzers with modified parameters."""
        from evalrx.eval_agent._tools import run_analysis

        sweep: dict[str, Any] = {}
        for analyzer_name, params in self.analyzer_params.items():
            try:
                result = run_analysis(model, analyzer_name, data, **params)
                sweep[analyzer_name] = result.findings
            except Exception as exc:
                sweep[analyzer_name] = {"error": str(exc)}

        return InterventionResult(
            hypothesis=hypothesis,
            status=HypothesisStatus.INCONCLUSIVE,
            fixed=False,
            evidence={"param_sweep": sweep},
        )
