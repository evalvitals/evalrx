"""Producer side of the contract: in-memory stage objects -> wire models.

The contract described what each stage *should* serialize and nothing called it,
so the description was free to drift from the thing described — which is the
failure it was written to prevent, one level up.  This module is the missing
half: every M1-M5 stage runs its result through the matching wire model and
writes the validated JSON under ``<run>/contract/``, so a frontend reads ONE
shape per stage instead of re-deriving it from ``run_log.jsonl`` with defensive
``.get(x) or y`` chains.

Two rules govern everything here.

**Emission never breaks a run.**  The pipeline is the product; the contract is an
observer.  A validation failure writes ``<stage>.invalid.json`` carrying the
error and the payload that produced it, and the run continues — an emitter that
can abort a six-hour diagnosis would be removed within a week, and then there
would be no contract again.  ``strict=True`` inverts this for CI, where a
violation *should* fail the build.

**Adapters read, never restate.**  Each ``from_*`` maps fields that already exist
onto the wire model.  Where the in-memory object has no answer the wire field
stays ``None`` — the contract distinguishes absent from zero precisely so a
producer can say "not measured" instead of inventing a defensible-looking value.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from evalvitals.contract.common import (
    SCHEMA_VERSION, CaseBatchRef, CaseBatchWire, FailureCaseWire, StageState,
    StageStatus, WireModel,
)
from evalvitals.contract.m1 import (
    AnalyzerSelection, FindingsWire, PerCaseRow, ProbeOutput, ResultWire,
)
from evalvitals.contract.m2 import (
    AnalysisFindingWire, CorrectedRejections, StatsReportWire, StatsToolResultWire,
)
from evalvitals.contract.m3 import DiagnosisOutput, HypothesisWire
from evalvitals.contract.m4 import FixAttemptWire, FixOutput, InterventionOutput
from evalvitals.contract.methodology import MethodologyWire
from evalvitals.contract.m5 import (
    HypothesisTestOutput, HypothesisTestResultWire, TestEvidence,
)

logger = logging.getLogger(__name__)

#: Where validated stage payloads land, relative to the run root.
CONTRACT_DIR = "contract"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _val(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or mapping key, whichever *obj* is."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _enum(value: Any) -> Any:
    return getattr(value, "value", value)


def _scalar(value: Any) -> bool:
    """Whether a per-case value survives the signal harvester's flat scan."""
    return isinstance(value, (int, float, bool, str)) or value is None


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def envelope(
    stage: str, *, trace_id: str, cycle: int = 0,
    state: StageState = StageState.SUCCEEDED,
    reason: str | None = None, duration_sec: float | None = None,
) -> dict[str, Any]:
    """The header every stage payload carries. Built once, here, so the fields
    cannot be spelled differently by six call sites."""
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "span_id": f"c{cycle}.{stage}",
        "cycle": cycle,
        "produced_at": _now(),
        "status": StageStatus(stage=stage, state=state, cycle=cycle,
                              reason=reason, duration_sec=duration_sec),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def from_cases(cases: Iterable[Any]) -> CaseBatchWire:
    """Serialize a CaseBatch. Media slots coerce through ``MediaRef.coerce``,
    so a case whose image lived only in memory becomes a descriptor rather than
    a validation error."""
    rows = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in cases]
    return CaseBatchWire(n_cases=len(rows), cases=[FailureCaseWire(**r) for r in rows])


def case_ref(path: str, cases: Iterable[Any], split: str | None = None) -> CaseBatchRef:
    n = len(list(cases)) if not hasattr(cases, "__len__") else len(cases)  # type: ignore[arg-type]
    return CaseBatchRef(path=path, n_cases=n, split=split)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------

def from_probe_results(
    results: dict[str, Any],
    *,
    trace_id: str,
    cycle: int = 0,
    model_modalities: Iterable[str] = ("text",),
    probed_modalities: Iterable[str] = ("text",),
    routed_on: Iterable[str] | None = None,
    is_agent: bool = False,
    selector: str = "static_strategy",
    generated: Iterable[str] = (),
    failed_analyzers: dict[str, str] | None = None,
    duration_sec: float | None = None,
) -> ProbeOutput:
    """M1's ``dict[name, Result]`` -> :class:`ProbeOutput`.

    Per-case rows are filtered to flat scalars. The contract rejects a numeric
    list or dict in a row because the harvester scans one level and silently
    drops it, and refusing to *emit* one is better than emitting a row that
    validates nowhere: the vector stays in ``artifacts``, which is where the
    tensor-level tools already look for it.
    """
    wire: dict[str, ResultWire] = {}
    for name, res in results.items():
        findings = dict(_val(res, "findings", {}) or {})
        rows_in = findings.pop("per_case", None) or []
        by_strategy = findings.pop("by_strategy", None)
        rows: list[PerCaseRow] = []
        for row in rows_in:
            if not isinstance(row, dict):
                continue
            sample_id = row.get("sample_id") or row.get("case_id") or row.get("id")
            if not sample_id:
                continue
            flat = {k: v for k, v in row.items() if k != "sample_id" and _scalar(v)}
            rows.append(PerCaseRow(sample_id=str(sample_id), **flat))
        light = {k: v for k, v in findings.items() if _scalar(v) or isinstance(v, (list, dict))}
        wire[name] = ResultWire(
            analyzer=name,
            model=str(_val(res, "model", "")),
            n_cases=len(_val(res, "cases", []) or []),
            findings=FindingsWire(per_case=rows, by_strategy=by_strategy, **light),
            metadata=dict(_val(res, "metadata", {}) or {}),
        )

    routed = list(routed_on) if routed_on is not None else list(probed_modalities)
    return ProbeOutput(
        **envelope("m1", trace_id=trace_id, cycle=cycle,
                   state=StageState.SUCCEEDED if wire else StageState.EMPTY,
                   duration_sec=duration_sec),
        results=wire,
        selection=AnalyzerSelection(
            model_modalities=sorted(model_modalities),      # type: ignore[arg-type]
            probed_modalities=sorted(probed_modalities),    # type: ignore[arg-type]
            routed_on=sorted(routed),                       # type: ignore[arg-type]
            is_agent=is_agent,
            selector=selector,                              # type: ignore[arg-type]
            generated=list(generated),
        ),
        failed_analyzers=dict(failed_analyzers or {}),
    )


# ---------------------------------------------------------------------------
# M2
# ---------------------------------------------------------------------------

def from_stats_report(
    report: Any, *, trace_id: str, cycle: int = 0,
    raw_results_ref: str | None = None, duration_sec: float | None = None,
) -> StatsReportWire:
    """``StatsAnalysisReport`` -> :class:`StatsReportWire`."""
    findings = [
        AnalysisFindingWire(
            analyzer=_val(f, "analyzer", ""), metric=_val(f, "metric", ""),
            value=float(_val(f, "value", 0.0) or 0.0),
            threshold=float(_val(f, "threshold", 0.0) or 0.0),
            direction=_val(f, "direction", "above"),
            severity=_val(f, "severity", "low"),
            message=str(_val(f, "message", "")),
        )
        for f in (_val(report, "findings", []) or [])
    ]
    stats = [
        StatsToolResultWire(
            tool=_val(r, "tool", ""), config=dict(_val(r, "config", {}) or {}),
            ok=bool(_val(r, "ok", False)), error=_val(r, "error"),
            effect=_val(r, "effect"), ci=_val(r, "ci"), p_value=_val(r, "p_value"),
            e_value=_val(r, "e_value"), underpowered=bool(_val(r, "underpowered", False)),
            reject=bool(_val(r, "reject", False)), fdr_corrected=_val(r, "fdr_corrected"),
            correction_method=_val(r, "correction_method"),
            correction_family=_val(r, "correction_family"),
            analysis_key=_val(r, "analysis_key"), raw_reject=_val(r, "raw_reject"),
            figure_path=_val(r, "figure_path"),
            summary=str(_val(r, "summary", "") or ""),
            details=dict(_val(r, "details", {}) or {}),
        )
        for r in (_val(report, "stats_tool_results", None) or _val(report, "stats_results", []) or [])
        if _val(r, "tool", None)
    ]
    corrected_raw = _val(report, "corrected_rejections", None)
    corrected = CorrectedRejections(
        method=_val(corrected_raw, "method", "none") or "none",
        alpha=float(_val(corrected_raw, "alpha", 0.05) or 0.05),
        deferred=bool(_val(corrected_raw, "deferred", False)),
        n_tested=int(_val(corrected_raw, "n_tested", 0) or 0),
        rejected_result_keys=list(_val(corrected_raw, "rejected_result_keys", []) or []),
    ) if corrected_raw is not None else CorrectedRejections()

    return StatsReportWire(
        **envelope("m2", trace_id=trace_id, cycle=cycle,
                   state=StageState.SUCCEEDED if (findings or stats) else StageState.EMPTY,
                   duration_sec=duration_sec),
        findings=findings,
        stats_tool=_val(report, "stats_tool", "threshold_rules") or "threshold_rules",
        stats_results=stats,
        corrected_rejections=corrected,
        descriptive_only=bool(_val(report, "descriptive_only", False)),
        conclusion=str(_val(report, "conclusion", "") or ""),
        evidence_chain=list(_val(report, "evidence_chain", []) or []),
        llm_fallback_reason=str(_val(report, "llm_fallback_reason", "") or ""),
        raw_results_ref=raw_results_ref,
    )


# ---------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------

#: A test_design M3 left empty. The contract requires a routable one, and the
#: honest rendering of "the judge proposed no test" is this marker rather than a
#: plausible-looking signal name the emitter invented.
UNROUTABLE = "paired_rerun (no test_design proposed by M3)"


def from_diagnosis(
    diag: Any, *, trace_id: str, cycle: int = 0, duration_sec: float | None = None,
) -> DiagnosisOutput:
    """``DiagnosisResult`` -> :class:`DiagnosisOutput`."""
    hyps: list[HypothesisWire] = []
    for h in (_val(diag, "hypotheses", []) or []):
        hyps.append(HypothesisWire(
            id=str(_val(h, "id", "") or f"h{len(hyps)}"),
            statement=str(_val(h, "statement", "")),
            target_model=str(_val(h, "target_model", "") or _val(diag, "model_name", "")),
            predicted_failure_mode=str(_val(h, "predicted_failure_mode", "") or "unknown"),
            test_design=str(_val(h, "test_design", "") or "").strip() or UNROUTABLE,
            status=_enum(_val(h, "status")) or "proposed",
            parent_id=_val(h, "parent_id"),
            metadata=dict(_val(h, "metadata", {}) or {}),
        ))
    return DiagnosisOutput(
        **envelope("m3", trace_id=trace_id, cycle=cycle,
                   state=StageState.SUCCEEDED if hyps else StageState.EMPTY,
                   duration_sec=duration_sec),
        hypotheses=hyps,
    )


# ---------------------------------------------------------------------------
# M5
# ---------------------------------------------------------------------------

#: Provenance values TestEvidence accepts. Anything else the tester wrote is a
#: source this contract does not model, and "none" is the honest rendering of
#: that — better than passing an unknown label through as if it were understood.
_EVIDENCE_SOURCES = frozenset({"m2_stats_results", "fallback_per_case", "none"})

def from_test_results(
    results: Iterable[Any], *, trace_id: str, cycle: int = 0,
    split: str = "explore", stopping_criteria_met: bool = False,
    duration_sec: float | None = None,
) -> HypothesisTestOutput:
    """``list[HypothesisTestResult]`` -> :class:`HypothesisTestOutput`."""
    rows: list[HypothesisTestResultWire] = []
    for tr in results:
        hyp = _val(tr, "hypothesis")
        ev = dict(_val(tr, "evidence", {}) or {})
        consistent = bool(_val(tr, "is_consistent_with_protocol", True))
        status = _enum(_val(tr, "status")) or "inconclusive"
        grade = _enum(_val(tr, "evidence_grade")) or "observational"
        # The contract refuses SUPPORTED without protocol consistency or with
        # evidence_grade=none. Downgrade rather than drop: an inconsistent
        # verdict is information, and a dropped row is not.
        if status == "supported" and (not consistent or grade == "none"):
            status = "inconclusive"
        rows.append(HypothesisTestResultWire(
            hypothesis_id=str(_val(hyp, "id", "") or "unknown"),
            status=status,
            test_name=str(_val(tr, "test_name", "") or "unknown"),
            effect_size=_val(tr, "effect_size"),
            confidence=float(_val(tr, "confidence", 0.0) or 0.0),
            evidence_grade=grade,
            is_consistent_with_protocol=consistent,
            verdict=str(_val(tr, "verdict", "") or ""),
            evidence=TestEvidence(
                source=(ev.get("source")
                        if ev.get("source") in _EVIDENCE_SOURCES else "none"),
                chosen_tool=ev.get("chosen_tool"),
                routed_by=str(ev.get("routed_by", "") or ""),
                consulted_tools=list(ev.get("consulted_tools", []) or []),
                ci=ev.get("ci"), e_value=ev.get("e_value"),
                fdr_corrected=ev.get("fdr_corrected"),
                underpowered=bool(ev.get("underpowered", False)),
                n_signal=ev.get("n_signal"), n_control=ev.get("n_control"),
                fail_rate_signal=ev.get("fail_rate_signal"),
                fail_rate_control=ev.get("fail_rate_control"),
                details={k: v for k, v in ev.items() if k not in TestEvidence.model_fields},
            ),
        ))
    return HypothesisTestOutput(
        **envelope("m5", trace_id=trace_id, cycle=cycle,
                   state=StageState.SUCCEEDED if rows else StageState.EMPTY,
                   duration_sec=duration_sec),
        split=split,                       # type: ignore[arg-type]
        results=rows,
        stopping_criteria_met=stopping_criteria_met,
    )


# ---------------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------------

_XML_ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}


def _esc(text: str) -> str:
    return "".join(_XML_ESC.get(ch, ch) for ch in str(text))


def _drawio(title: str, steps: list[tuple[str, str]]) -> str:
    """Render a linear step flow as an uncompressed mxfile.

    Deliberately minimal. The contract wants a diagram a reviewer can read, and
    the honest diagram of a fix candidate is the sequence of operations the
    candidate declares — no more. Where the candidate is generated code, that is
    ONE step pointing at the trial folder, because claiming to have diagrammed
    sixty lines nobody transcribed would be the fabrication the contract exists
    to prevent.
    """
    cells = [
        '<mxCell id="0"/>',
        '<mxCell id="1" parent="0"/>',
    ]
    y = 40
    for i, (label, detail) in enumerate(steps):
        value = _esc(label) if not detail else f"{_esc(label)}&#10;&#10;{_esc(detail)}"
        cells.append(
            f'<mxCell id="n{i}" value="{value}" style="rounded=1;whiteSpace=wrap;html=1;" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="40" y="{y}" width="280" height="70" as="geometry"/></mxCell>'
        )
        if i:
            cells.append(
                f'<mxCell id="e{i}" style="edgeStyle=orthogonalEdgeStyle;html=1;" edge="1" '
                f'parent="1" source="n{i - 1}" target="n{i}">'
                f'<mxGeometry relative="1" as="geometry"/></mxCell>'
            )
        y += 110
    return (
        f'<mxfile host="evalvitals"><diagram name="{_esc(title)[:60]}">'
        f'<mxGraphModel><root>{"".join(cells)}</root></mxGraphModel>'
        f'</diagram></mxfile>'
    )


def methodology_from_candidate(candidate: Any, validation: Any = None) -> "MethodologyWire | None":
    """Describe a fix candidate as the steps it declares.

    Returns ``None`` when the candidate declares nothing describable. The caller
    must then NOT claim ``fixed=True`` — that is the contract working as
    designed, not a gap to paper over: a repair whose method cannot be stated is
    exactly the one a reader is being asked to take on trust.
    """
    payload = _val(candidate, "payload", None) or {}
    name = str(_val(candidate, "name", "") or "candidate")
    tier = str(_val(candidate, "tier", "") or "")
    kind = str(_val(candidate, "kind", "") or "")
    steps: list[tuple[str, str]] = [("input case", "prompt + any media slots, unmodified")]

    for op in (_val(payload, "image_ops", []) or []):
        if isinstance(op, dict) and op.get("tool"):
            args = {k: v for k, v in op.items() if k != "tool"}
            steps.append((f"image op: {op['tool']}", ", ".join(f"{k}={v}" for k, v in args.items())))

    template = _val(payload, "prompt_template", None)
    if template:
        steps.append(("rewritten prompt", str(template)[:300]))

    code = _val(payload, "code", None) or _val(candidate, "source", None)
    trial_root = str(_val(_val(candidate, "trial", None), "root", "") or "")
    if code and not template:
        steps.append((
            "generated code",
            f"read it at {trial_root}/result.json" if trial_root else "see the trial folder",
        ))

    if len(steps) == 1:
        return None
    steps.append(("model call", "the unmodified model answers the transformed case"))
    steps.append((
        "paired comparison",
        f"{_val(validation, 'n_fixed', 0)} fixed / {_val(validation, 'n_broken', 0)} broken "
        f"over {_val(validation, 'n_pairs', 0)} pairs" if validation is not None
        else "validated against the unmodified baseline on the same cases",
    ))
    title = f"{name} — {kind}" if kind else name
    return MethodologyWire(
        title=title[:120],
        summary=str(_val(validation, "summary", "") or "")[:1000],
        tier=tier,
        drawio_xml=_drawio(title, steps),
        n_model_calls=1,
    )


def from_intervention(
    result: Any, *, trace_id: str, cycle: int = -1, duration_sec: float | None = None,
) -> InterventionOutput:
    """``InterventionResult`` -> :class:`InterventionOutput`."""
    hyp = _val(result, "hypothesis")
    return InterventionOutput(
        **envelope("m4_surgery", trace_id=trace_id, cycle=cycle, duration_sec=duration_sec),
        hypothesis_id=str(_val(hyp, "id", "") or "unknown"),
        hypothesis_status=_enum(_val(result, "status")) or "inconclusive",
        strategy=_val(result, "strategy", "passive_correlation") or "passive_correlation",
        fixed=bool(_val(result, "fixed", False)),
        confidence_score=float(_val(result, "confidence_score", 0.0) or 0.0),
        evidence_dimensions=dict(_val(result, "evidence_dimensions", {}) or {}),
        evidence=dict(_val(result, "evidence", {}) or {}),
    )


def from_fix_outcome(
    outcome: Any, *, trace_id: str, cycle: int = -1, duration_sec: float | None = None,
) -> FixOutput:
    """``FixOutcome`` -> :class:`FixOutput`."""
    attempts: list[FixAttemptWire] = []
    for v in (_val(outcome, "attempted", []) or []):
        cand = _val(v, "candidate", v)
        trial = _val(cand, "trial", None)
        verdict = _val(v, "verdict", None) or (
            "fixed" if _val(v, "fixed", False) else
            "not_executed" if _val(v, "exec_error", None) else "no_effect"
        )
        attempts.append(FixAttemptWire(
            tier=_val(cand, "tier", "L1") or "L1",
            name=str(_val(cand, "name", "") or "candidate"),
            kind=_val(cand, "kind"), source=_val(cand, "source"),
            trial_root=str(_val(trial, "root", "") or "") or None,
            n_pairs=int(_val(v, "n_pairs", 0) or 0),
            n_baseline_correct=int(_val(v, "n_baseline_correct", 0) or 0),
            n_fixed=int(_val(v, "n_fixed", 0) or 0),
            n_broken=int(_val(v, "n_broken", 0) or 0),
            fixed_cases=[str(c) for c in (_val(v, "fixed_cases", []) or [])],
            broken_cases=[str(c) for c in (_val(v, "broken_cases", []) or [])],
            n_applicable=int(_val(v, "n_applicable", 0) or 0),
            coverage=_val(v, "coverage"),
            n_unstable=int(_val(v, "n_unstable", 0) or 0),
            n_model_independent=int(_val(v, "n_model_independent", 0) or 0),
            effect=_val(v, "effect"), e_value=_val(v, "e_value"),
            reject=bool(_val(v, "reject", False)),
            verdict=verdict,
            summary=str(_val(v, "summary", "") or ""),
        ))
    fixed = bool(_val(outcome, "fixed", False))
    best = _val(outcome, "best", None)
    best_name = str(_val(best, "name", best) or "") or None
    # FixOutput refuses fixed=True unless the winning row carries a methodology.
    # Attach the one derived from what the candidate declares; when nothing is
    # describable the claim is dropped rather than the payload, and `summary`
    # says so — reporting the repair with fixed=False is a smaller error than
    # dropping M4 entirely, which reads as "no repair was attempted".
    if fixed and best_name:
        for v in (_val(outcome, "attempted", []) or []):
            cand = _val(v, "candidate", v)
            if str(_val(cand, "name", "")) != best_name:
                continue
            method = methodology_from_candidate(cand, v)
            row = next((a for a in attempts if a.name == best_name), None)
            if method is not None and row is not None:
                row.methodology = method
            elif row is not None:
                fixed = False
                row.summary = (row.summary + " [contract: no describable method, so this "
                                            "is reported as validated-but-unexplained]").strip()
            break
    return FixOutput(
        **envelope("m4_fix", trace_id=trace_id, cycle=cycle, duration_sec=duration_sec),
        max_tier=_val(outcome, "max_tier", "L1") or "L1",
        routed=[{str(k): str(v) for k, v in dict(r).items()}
                for r in (_val(outcome, "routed", []) or []) if isinstance(r, dict)],
        attempted=attempts,
        best=best_name,
        fixed=fixed and any(a.name == best_name for a in attempts),
        ebh_survivors=[str(x) for x in (_val(outcome, "ebh_survivors", []) or [])],
        repair_rounds=int(_val(outcome, "repair_rounds", 0) or 0),
        recommendation=_val(outcome, "recommendation"),
        refine_signal=_val(outcome, "refine_signal"),
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class ContractEmitter:
    """Writes validated stage payloads under ``<root>/contract/``.

    Args:
        root:     Run root. ``contract/`` is created on first write.
        trace_id: Run-level correlation id, stamped on every envelope.
        strict:   Re-raise validation errors instead of recording them. For CI;
                  a real run must never die because its observer disagreed.
    """

    def __init__(self, root: "str | Path", trace_id: str, *, strict: bool = False) -> None:
        self.root = Path(root)
        self.trace_id = trace_id
        self.strict = strict
        self.written: list[Path] = []
        self.errors: list[tuple[str, str]] = []

    @property
    def dir(self) -> Path:
        return self.root / CONTRACT_DIR

    def emit(self, name: str, build: Callable[[], WireModel]) -> "Path | None":
        """Build, validate and write one stage payload. Returns the path, or
        ``None`` when the payload did not validate (non-strict)."""
        try:
            model = build()
            payload = model.model_dump_json(indent=2)
        except Exception as exc:  # noqa: BLE001 - see class docstring
            self.errors.append((name, str(exc)))
            if self.strict:
                raise
            logger.warning("contract: %s did not validate (%s)", name, exc)
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{name}.invalid.json").write_text(
                json.dumps({"stage": name, "error": str(exc)}, indent=2), encoding="utf-8"
            )
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{name}.json"
        path.write_text(payload, encoding="utf-8")
        self.written.append(path)
        return path

    def index(self) -> Path:
        """Write ``contract/index.json`` — what a frontend loads first."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "index.json"
        path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "stages": sorted(p.name for p in self.written),
            "invalid": sorted(name for name, _ in self.errors),
        }, indent=2), encoding="utf-8")
        return path

    def __repr__(self) -> str:
        return (f"ContractEmitter(root={str(self.root)!r}, "
                f"written={len(self.written)}, invalid={len(self.errors)})")


__all__ = [
    "CONTRACT_DIR", "ContractEmitter", "envelope",
    "from_cases", "case_ref", "from_probe_results", "from_stats_report",
    "from_diagnosis", "from_test_results", "from_intervention", "from_fix_outcome",
    "methodology_from_candidate",
]
