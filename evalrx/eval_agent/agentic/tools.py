"""Tool registry for the agentic diagnosis loop.

Each :class:`ToolSpec` wraps one existing stage (M1 probe, M2 stats, M3
diagnosis, M5 test, ...) behind a name/schema the judge can call. The host —
not the judge — enforces call caps, preconditions, and the pre-registration
discipline (no declaring success without a statistically supported,
protocol-consistent hypothesis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from evalrx.eval_agent.agentic.actions import Action
    from evalrx.eval_agent.agentic.board import EvidenceBoard
    from evalrx.eval_agent.loop_reports import VLDiagnoseReport


@dataclass(frozen=True)
class ToolSpec:
    """One tool the judge may call."""

    name: str
    description: str
    params_schema: dict[str, Any]
    handler: "Callable[[EvidenceBoard, dict[str, Any]], ToolOutcome]"
    max_calls: int = 3
    requires: tuple[str, ...] = ()


@dataclass
class ToolOutcome:
    """What happened when a tool ran — the summary is what re-enters the prompt."""

    ok: bool
    summary: str
    payload: Any = None
    error: str | None = None


# Precondition names -> board attribute truthiness checks.
_PRECONDITIONS: dict[str, "Callable[[EvidenceBoard], bool]"] = {
    "probe_findings": lambda b: bool(b.probe_findings),
    "stats_findings": lambda b: bool(b.stats_findings),
    "stats_confirmatory": lambda b: bool(b.stats_confirmatory),
    "hypotheses": lambda b: bool(b.hypotheses),
    "supported_hypothesis": lambda b: b.has_supported_hypothesis(),
}


class ToolRegistry:
    """Holds the tools available to the judge for one agentic run."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def tool_names(self) -> set[str]:
        return set(self._specs)

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def catalog_for_prompt(self, board: "EvidenceBoard") -> str:
        lines = []
        for spec in self._specs.values():
            calls_left = (
                spec.max_calls - board.calls_made(spec.name)
                if spec.max_calls > 0
                else None
            )
            unmet = [r for r in spec.requires if not _PRECONDITIONS[r](board)]
            bits = [f"- {spec.name}: {spec.description}"]
            if calls_left is not None:
                bits.append(f"(calls left: {calls_left})")
            if unmet:
                bits.append(f"[BLOCKED until: {', '.join(unmet)}]")
            lines.append(" ".join(bits))
        return "\n".join(lines)

    def dispatch(self, action: "Action", board: "EvidenceBoard") -> ToolOutcome:
        spec = self._specs.get(action.tool)
        if spec is None:
            return ToolOutcome(False, f"unknown tool {action.tool!r}", error="unknown_tool")

        if spec.max_calls > 0 and board.calls_made(spec.name) >= spec.max_calls:
            return ToolOutcome(
                False,
                f"{spec.name} has reached its call limit ({spec.max_calls})",
                error="max_calls",
            )

        unmet = [r for r in spec.requires if not _PRECONDITIONS[r](board)]
        if unmet:
            return ToolOutcome(
                False,
                f"{spec.name} requires {unmet} first",
                error="precondition",
            )

        if spec.name == "stop" and action.params.get("resolved") and not board.has_supported_hypothesis():
            return ToolOutcome(
                False,
                "cannot declare success: no statistically supported, "
                "protocol-consistent hypothesis yet — call test_hypothesis first",
                error="no_supported_hypothesis",
            )

        return spec.handler(board, action.params)


# ---------------------------------------------------------------------------
# Default registry: wraps the same M1-M5 stages VLDiagnoseLoop calls directly.
# ---------------------------------------------------------------------------


@dataclass
class _RunState:
    """Mutable, per-run state the default tool handlers read and update.

    Mirrors the local variables ``VLDiagnoseLoop.run()`` keeps on its stack —
    an agentic run just spreads the same bookkeeping across tool calls instead
    of a fixed cycle.
    """

    original_data: Any
    data: Any
    cycle: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    all_hypotheses: list[Any] = field(default_factory=list)
    pending_hypotheses: list[Any] = field(default_factory=list)
    all_test_results: list[Any] = field(default_factory=list)
    prior_cycles: list[dict[str, Any]] = field(default_factory=list)
    probe_results: dict[str, Any] = field(default_factory=dict)
    artifact_pngs: list[Any] = field(default_factory=list)
    stats_report: Any = None
    failure_modes: Any = None
    probe_search_result: Any = None
    surgery_outcome: Any = None
    fix_outcome: Any = None


def _summarize_probe(probe_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact, board-friendly summary of one M1 pass (scalars only, no per-case rows)."""
    out = []
    for name, res in probe_results.items():
        findings = res.findings or {}
        scalars = {k: v for k, v in findings.items() if isinstance(v, (int, float, bool, str))}
        out.append({
            "analyzer": name,
            "scalars": scalars,
            "n_per_case": len(findings.get("per_case", []) or []),
        })
    return out


def _summarize_stats(report: Any) -> list[dict[str, Any]]:
    results = getattr(report, "stats_results", None) or []
    return [
        {
            "tool": r.tool, "ok": r.ok, "effect": r.effect,
            "reject": r.reject, "e_value": r.e_value, "summary": r.summary,
        }
        for r in results
    ]


def _current_report(loop: Any, state: _RunState) -> "VLDiagnoseReport":
    """Snapshot the run so far into the same report shape run_m4/run_fix expect."""
    from evalrx.eval_agent.loop_reports import VLDiagnoseReport

    verified = loop.hypothesis_tester.best_hypotheses(state.all_test_results)
    return VLDiagnoseReport(
        cycles=state.cycle,
        resolved=bool(verified),
        stopped_by="in_progress",
        verified_hypotheses=verified,
        final_hypotheses=state.all_hypotheses,
        all_test_results=state.all_test_results,
        final_stats_report=state.stats_report,
        store=loop.store,
        _run_id=getattr(loop, "_run_id", ""),
    )


def _run_probe(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        probe_results, artifact_pngs = loop._do_m1(
            state.cycle, state.data, state.all_hypotheses, state.timings
        )
        if not probe_results:
            return ToolOutcome(False, "M1 produced no probe results", error="empty")
        state.probe_results = probe_results
        state.artifact_pngs = artifact_pngs
        board.probe_findings = _summarize_probe(probe_results)
        summary = f"ran {len(probe_results)} analyzer(s): {sorted(probe_results)}"
        state.cycle += 1
        return ToolOutcome(True, summary, payload=probe_results)

    return handler


def _run_stats(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        confirmatory = bool(params.get("confirmatory", True))
        stats_report = loop._do_m2(
            state.cycle, state.probe_results, state.data, state.artifact_pngs,
            state.timings, confirmatory=confirmatory,
        )
        state.stats_report = stats_report
        board.stats_findings = _summarize_stats(stats_report)
        board.stats_confirmatory = confirmatory and not getattr(
            stats_report, "descriptive_only", False
        )
        return ToolOutcome(
            True,
            f"severity={stats_report.severity}, confirmatory={board.stats_confirmatory}",
            payload=stats_report,
        )

    return handler


def _explore_data(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        from evalrx.analysis.operationalize import per_case_to_records
        from evalrx.analysis.stats_tools import build_stats_input

        question = str(
            params.get("question")
            or "Explore this dataset and surface the patterns that matter."
        )
        stats_input = build_stats_input(state.probe_results, state.data)
        records = per_case_to_records(stats_input.per_case, stats_input.labels)
        if not records:
            return ToolOutcome(False, "no per-case records to explore yet", error="empty")
        report = loop.explorer.explore_records(records, question=question)
        board.explore_takeaways = [t.to_dict() for t in getattr(report, "takeaways", [])]
        summary = (
            f"{len(board.explore_takeaways)} takeaway(s)"
            if report.ok else (report.error or "explore failed")
        )
        return ToolOutcome(bool(report.ok), summary, payload=report)

    return handler


def _cluster_failures(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        from evalrx.analysis.failure_modes import cluster_failures
        from evalrx.analysis.operationalize import per_case_to_records
        from evalrx.analysis.stats_tools import build_stats_input

        stats_input = build_stats_input(state.probe_results, state.data)
        records = per_case_to_records(stats_input.per_case, stats_input.labels)
        if not records:
            return ToolOutcome(False, "no per-case records to cluster yet", error="empty")
        report = cluster_failures(
            records,  # per_case_to_records writes label="fail"/"pass" — matches the defaults
            min_cluster_size=int(params.get("min_cluster_size", 3)),
            max_clusters=int(params.get("max_clusters", 8)),
        )
        state.failure_modes = report
        board.failure_modes = [c.to_dict() for c in report.clusters]
        run_logger = getattr(loop, "run_logger", None)
        if run_logger is not None:
            run_logger.save_artifact_json("failure_modes.json", report.to_dict())
        return ToolOutcome(
            True, f"{len(report.clusters)} failure mode(s) from {report.n_fail_cases} FAIL case(s)",
            payload=report,
        )

    return handler


def _search_probes(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        from evalrx.eval_agent.stages.probe_search_agent import ProbeSearchAgent

        seed_pool = state.data
        if not len(seed_pool):
            return ToolOutcome(False, "no seed cases available to probe", error="empty")
        budget = int(params.get("budget", 10))
        agent = ProbeSearchAgent(judge=loop.judge, protocol=loop.protocol, budget=budget)
        result = agent.run(loop.model, seed_pool)
        state.probe_search_result = result
        board.probe_search_findings = [
            {
                "prompt": c.inputs.prompt, "expected": c.expected, "observed": str(c.observed),
            }
            for c in result.failure_cases
        ]
        run_logger = getattr(loop, "run_logger", None)
        if run_logger is not None:
            run_logger.save_artifact_json("probe_search_result.json", result.to_dict())
        summary = (
            f"{result.n_simulations} simulation(s) (macro={result.n_macro}, "
            f"micro={result.n_micro}), {len(result.failure_cases)} new failure(s) "
            f"found (error_rate={result.error_rate:.0%})"
        )
        return ToolOutcome(bool(result.n_simulations), summary, payload=result)

    return handler


def _propose_hypotheses(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        diag = loop._do_m3(
            state.cycle, state.stats_report, state.prior_cycles, state.timings,
            failure_modes=state.failure_modes,
        )
        if diag is None or not diag.hypotheses:
            return ToolOutcome(False, "M3 produced no hypotheses", error="empty")
        for h in diag.hypotheses:
            loop.store.add_hypothesis(h)
        state.all_hypotheses.extend(diag.hypotheses)
        state.pending_hypotheses = list(diag.hypotheses)
        board.hypotheses = [
            {
                "statement": h.statement,
                "failure_mode": h.predicted_failure_mode,
                "status": h.status.value if h.status else "proposed",
            }
            for h in state.all_hypotheses
        ]
        state.prior_cycles.append({
            "cycle": state.cycle,
            "severity": getattr(state.stats_report, "severity", "none"),
            "hypotheses": [
                {
                    "statement": h.statement,
                    "failure_mode": h.predicted_failure_mode,
                    "status": h.status.value if h.status else "pending",
                }
                for h in diag.hypotheses
            ],
        })
        return ToolOutcome(True, f"proposed {len(diag.hypotheses)} hypothesis(es)", payload=diag)

    return handler


def _test_hypothesis(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        if not state.pending_hypotheses:
            return ToolOutcome(
                False, "no untested hypotheses — call propose_hypotheses first", error="empty"
            )
        loop._finalize_confirmatory_stats(state.stats_report)
        board.stats_confirmatory = True
        test_results = loop._do_m5(
            state.cycle, state.pending_hypotheses, state.stats_report, state.data, state.timings
        )
        for tr in test_results:
            tr.hypothesis.status = tr.status
        state.all_test_results.extend(test_results)
        state.pending_hypotheses = []
        board.hypotheses = [
            {
                "statement": tr.hypothesis.statement,
                "failure_mode": tr.hypothesis.predicted_failure_mode,
                "status": tr.status.value,
                "is_consistent_with_protocol": tr.is_consistent_with_protocol,
                "effect_size": tr.effect_size,
                "confidence": tr.confidence,
            }
            for tr in state.all_test_results
        ]
        n_supported = sum(1 for tr in test_results if tr.status.value == "supported")
        return ToolOutcome(
            True, f"tested {len(test_results)} hypothesis(es), {n_supported} supported",
            payload=test_results,
        )

    return handler


def _run_surgery(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        report = _current_report(loop, state)
        iv = loop.run_m4(report, state.original_data)
        state.surgery_outcome = iv
        if iv is None:
            return ToolOutcome(False, "surgery produced no result", error="empty")
        return ToolOutcome(True, f"surgery status={iv.status.value} fixed={iv.fixed}", payload=iv)

    return handler


def _run_fix(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        report = _current_report(loop, state)
        outcome = loop.run_fix(report, state.original_data)
        state.fix_outcome = outcome
        if outcome is None:
            return ToolOutcome(False, "fix produced no result", error="empty")
        fixed = bool(getattr(outcome, "fixed", False))
        return ToolOutcome(True, f"fix fixed={fixed}", payload=outcome)

    return handler


def _stop(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        resolved = bool(params.get("resolved", False))
        reason = str(params.get("reason") or "")
        return ToolOutcome(True, f"stop(resolved={resolved}): {reason}".strip())

    return handler


def build_default_registry(loop: Any, state: _RunState) -> ToolRegistry:
    """Build the default M1-M5 tool registry for one :class:`AgenticDiagnoseLoop` run."""
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="run_probe",
        description="Run M1 analyzer probing to gather per-case findings on the model.",
        params_schema={"type": "object", "properties": {}},
        handler=_run_probe(loop, state),
        max_calls=3,
    ))
    registry.register(ToolSpec(
        name="run_stats",
        description=(
            "Run M2 statistical analysis over the probe findings gathered so far. "
            "params.confirmatory (bool, default true): apply e-BH FDR correction now "
            "vs. defer it (descriptive-only pass)."
        ),
        params_schema={
            "type": "object",
            "properties": {"confirmatory": {"type": "boolean"}},
        },
        handler=_run_stats(loop, state),
        max_calls=3,
        requires=("probe_findings",),
    ))
    if getattr(loop, "explorer", None) is not None:
        registry.register(ToolSpec(
            name="explore_data",
            description=(
                "Run free-form exploratory data analysis over the per-case signals "
                "gathered so far. params.question (str, optional): what to look for."
            ),
            params_schema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
            },
            handler=_explore_data(loop, state),
            max_calls=2,
            requires=("probe_findings",),
        ))
    registry.register(ToolSpec(
        name="cluster_failures",
        description=(
            "Cluster FAIL cases into interpretable failure modes from the per-case "
            "signals gathered so far — surfaces patterns propose_hypotheses can draw "
            "on. params.min_cluster_size (int, default 3), params.max_clusters (int, default 8)."
        ),
        params_schema={
            "type": "object",
            "properties": {
                "min_cluster_size": {"type": "integer"},
                "max_clusters": {"type": "integer"},
            },
        },
        handler=_cluster_failures(loop, state),
        max_calls=2,
        requires=("probe_findings",),
    ))
    registry.register(ToolSpec(
        name="search_probes",
        description=(
            "Run a hierarchical Macro/Micro MCTS probe search (ProbeLLM-style) that "
            "synthesizes and evaluates NEW test cases beyond the loaded dataset, "
            "seeded from it — surfaces failures the original data never showed. "
            "params.budget (int, default 10): total new probes to simulate."
        ),
        params_schema={"type": "object", "properties": {"budget": {"type": "integer"}}},
        handler=_search_probes(loop, state),
        max_calls=2,
    ))
    registry.register(ToolSpec(
        name="propose_hypotheses",
        description="Run M3 to propose falsifiable hypotheses from the stats findings so far.",
        params_schema={"type": "object", "properties": {}},
        handler=_propose_hypotheses(loop, state),
        max_calls=4,
        requires=("stats_findings",),
    ))
    registry.register(ToolSpec(
        name="test_hypothesis",
        description="Run M5 to statistically test the most recently proposed hypotheses.",
        params_schema={"type": "object", "properties": {}},
        handler=_test_hypothesis(loop, state),
        max_calls=4,
        requires=("hypotheses",),
    ))
    registry.register(ToolSpec(
        name="run_surgery",
        description="Propose a mechanistic fix (M4) for the best verified hypothesis.",
        params_schema={"type": "object", "properties": {}},
        handler=_run_surgery(loop, state),
        max_calls=1,
        requires=("supported_hypothesis",),
    ))
    registry.register(ToolSpec(
        name="run_fix",
        description="Run the tiered, validated fix module (L1-L4) for the best verified hypothesis.",
        params_schema={"type": "object", "properties": {}},
        handler=_run_fix(loop, state),
        max_calls=1,
        requires=("supported_hypothesis",),
    ))
    registry.register(ToolSpec(
        name="stop",
        description=(
            "End the investigation. params.resolved (bool): true only once a "
            "hypothesis has been tested and is statistically supported and "
            "protocol-consistent — otherwise it is rejected. params.reason (str)."
        ),
        params_schema={
            "type": "object",
            "properties": {"resolved": {"type": "boolean"}, "reason": {"type": "string"}},
        },
        handler=_stop(loop, state),
        # A rejected stop (premature resolved=true, or resolved=false to give up
        # and try again) does no real work, so it shouldn't burn down a scarce
        # quota the way a real M1-M5 stage call does — cap generously.
        max_calls=5,
    ))
    return registry
