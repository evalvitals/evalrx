"""Tool registry for the agentic diagnosis loop.

Each :class:`ToolSpec` wraps one existing stage (M1 probe, M2 stats, M3
diagnosis, M4 test, ...) behind a name/schema the judge can call. The host —
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
    from evalrx.eval_agent.loop_reports import AutoDiagnoseReport


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
# Default registry: wraps the same M1-M4 stages VLDiagnoseLoop calls directly.
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
    judge: Any = None  # the policy's decision judge (also drives probe search)
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
    diagnosis_failed: bool = False


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


def _current_report(loop: Any, state: _RunState) -> "AutoDiagnoseReport":
    """Snapshot the run so far into the same report shape run_m5/run_fix expect."""
    from evalrx.eval_agent.loop_reports import AutoDiagnoseReport

    verified = loop.hypothesis_tester.best_hypotheses(state.all_test_results)
    return AutoDiagnoseReport(
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


#: Full M1 passes (``run_probe`` with no probe named) allowed per run.
_MAX_UNNAMED_PROBE_RUNS = 3


def _probe_overrides(loop: Any) -> "dict[str, Any]":
    """Constructed analyzers the host configured on the loop's probe agent."""
    return dict(getattr(getattr(loop, "probe_agent", None), "_overrides", None) or {})


def _requested_probes(params: "dict[str, Any]") -> "list[tuple[str, dict[str, Any]]] | str":
    """Read ``run_probe`` arguments as ``[(name, params), ...]``.

    Accepts ``{"name": ..., "params": {...}}`` for one probe and
    ``{"probes": [{"name": ..., "params": {...}}, ...]}`` for several. Returns
    an error string when the shape is wrong, an empty list when the call names
    no probe (the automatic-selection form).
    """
    entries: list[Any] = []
    if params.get("probes") is not None:
        if not isinstance(params["probes"], list):
            return "params.probes must be a list of {name, params} objects"
        entries.extend(params["probes"])
    if params.get("name") is not None:
        entries.append({"name": params["name"], "params": params.get("params")})
    out: list[tuple[str, dict[str, Any]]] = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            return "each probe needs a name"
        args = entry.get("params") or {}
        if not isinstance(args, dict):
            return f"params for {entry['name']!r} must be an object"
        out.append((str(entry["name"]).strip(), dict(args)))
    return out


def _list_probes(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        from evalrx.probing import ProbeRouter

        modes = params.get("failure_modes") or []
        if isinstance(modes, str):
            modes = [modes]
        if not modes:
            # Default to what the investigation currently suspects.
            modes = [h.get("failure_mode") for h in board.hypotheses if h.get("failure_mode")]
        families = params.get("families")
        if isinstance(families, str):
            families = [families]
        router = ProbeRouter(
            selector=getattr(getattr(loop, "probe_agent", None), "selector", None),
            overrides=_probe_overrides(loop),
        )
        route = router.route(
            loop.model, state.data,
            failure_modes=modes or None,
            families=families or None,
            access=params.get("access"),
            allow_interventions=bool(params.get("allow_interventions", True)),
            only_matching=bool(params.get("only_matching", False)),
            max_probes=int(params.get("max_probes", 12)),
        )
        board.probe_catalog = route.to_prompt()
        summary = f"{len(route.selected)} runnable probe(s)"
        if route.unavailable:
            summary += f", {len(route.unavailable)} requested but unavailable"
        if not route.selected:
            return ToolOutcome(False, summary + "; widen the filters", payload=route, error="empty")
        return ToolOutcome(True, summary + ": " + ", ".join(route.names), payload=route)

    return handler


def _run_probe(loop: Any, state: _RunState) -> "Callable[[EvidenceBoard, dict], ToolOutcome]":
    def handler(board: "EvidenceBoard", params: dict[str, Any]) -> ToolOutcome:
        requested = _requested_probes(params or {})
        if isinstance(requested, str):
            return ToolOutcome(False, requested, error="bad_params")
        if not requested:
            # No probe named: M1 selects (judge-guided or static), as before,
            # and under the cap a full pass always had. The tool's own cap is
            # higher only to leave room for the narrower named calls.
            unnamed = sum(
                1 for a in board.action_log
                if a.get("tool") == "run_probe" and a.get("ok")
                and not _requested_probes(a.get("params") or {})
            )
            if unnamed >= _MAX_UNNAMED_PROBE_RUNS:
                return ToolOutcome(
                    False,
                    f"run_probe without a probe name has reached its limit "
                    f"({_MAX_UNNAMED_PROBE_RUNS}); name the probes to run (see list_probes)",
                    error="max_calls",
                )
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

        from evalrx.probing import ProbeCallError, availability, build_probe, spec_for

        overrides = _probe_overrides(loop)
        names: list[str] = []
        call_params: dict[str, dict[str, Any]] = {}
        for name, args in requested:
            # Validate every call before any model work: one bad argument must
            # not cost the probes named beside it, nor a slot of the call cap.
            try:
                build_probe(name, args, overrides=overrides)
            except ProbeCallError as exc:
                return ToolOutcome(False, str(exc), error=exc.code)
            verdict = availability(spec_for(name), loop.model, state.data, overrides=overrides)
            if not verdict.available:
                return ToolOutcome(
                    False, f"{name} cannot run here: {'; '.join(verdict.missing)}",
                    error="unavailable",
                )
            names.append(name)
            if args:
                call_params[name] = args
        names = list(dict.fromkeys(names))
        try:
            probe_results, artifact_pngs = loop._do_m1(
                state.cycle, state.data, state.all_hypotheses, state.timings,
                analyzers=names, analyzer_params=call_params or None,
            )
        except ProbeCallError as exc:
            return ToolOutcome(False, str(exc), error=exc.code)
        if not probe_results:
            return ToolOutcome(False, f"no result from {names}", error="empty")
        # Named calls ADD evidence: the judge is assembling a picture one probe
        # at a time, so an earlier probe's findings must survive a later call.
        state.probe_results = {**state.probe_results, **probe_results}
        state.artifact_pngs = list(state.artifact_pngs) + list(artifact_pngs)
        board.probe_findings = _summarize_probe(state.probe_results)
        state.cycle += 1
        missing = [n for n in names if n not in probe_results]
        summary = f"ran {sorted(probe_results)}"
        if missing:
            summary += f"; no result from {missing} (failed at runtime or filtered)"
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
        question = str(
            params.get("question")
            or "Explore this dataset and surface the patterns that matter."
        )
        # Reuse host adjudication, artifact capture, and M3 context publication.
        report = loop._do_explore(
            state.cycle, state.probe_results, state.data, state.timings, question=question,
        )
        if report is None:
            return ToolOutcome(False, "exploration produced no report", error="empty")
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
        agent = ProbeSearchAgent(
            judge=state.judge, protocol=loop.protocol, budget=budget,
            run_logger=getattr(loop, "run_logger", None),
        )
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
            failure_modes=state.failure_modes, cases=state.data,
        )
        state.diagnosis_failed = diag is None
        if diag is None:
            return ToolOutcome(False, "M3 failed after bounded retries; inspect diagnosis errors",
                               error="diagnosis_error")
        if not diag.hypotheses:
            return ToolOutcome(False, "M3 produced no hypotheses; gather new evidence", error="empty")
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
        test_results = loop._do_m4(
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
        iv = loop.run_m5(report, state.original_data)
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
    """Build the default M1-M4 tool registry for one :class:`JudgePolicy` run."""
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="list_probes",
        description=(
            "List the probes that can run on this model and data, grouped by family, "
            "with their arguments and why each is ranked; probes you asked for that "
            "cannot run are listed with what they are missing. Runs no model. "
            "params.failure_modes (list[str], default: the current hypotheses' modes), "
            "params.families (list[str]), params.access ('black_box' | 'white_box'), "
            "params.allow_interventions (bool, default true), "
            "params.only_matching (bool, default false), params.max_probes (int, default 12)."
        ),
        params_schema={
            "type": "object",
            "properties": {
                "failure_modes": {"type": "array", "items": {"type": "string"}},
                "families": {"type": "array", "items": {"type": "string"}},
                "access": {"type": "string", "enum": ["black_box", "white_box"]},
                "allow_interventions": {"type": "boolean"},
                "only_matching": {"type": "boolean"},
                "max_probes": {"type": "integer"},
            },
        },
        handler=_list_probes(loop, state),
        max_calls=4,
    ))
    registry.register(ToolSpec(
        name="run_probe",
        description=(
            "Gather per-case findings on the model (M1). With no params, the probe "
            f"agent selects the analyzers (at most {_MAX_UNNAMED_PROBE_RUNS} times). "
            "To call specific probes, pass params.name "
            "(str) and params.params (object), or params.probes "
            "(list of {name, params}); see list_probes for names and arguments. "
            "Named calls add to the findings gathered so far."
        ),
        params_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object"},
                "probes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "params": {"type": "object"},
                        },
                    },
                },
            },
        },
        handler=_run_probe(loop, state),
        # Named calls are narrower than a full M1 pass, so the cap allows a
        # few of them; the run's max_actions still bounds the total.
        max_calls=6,
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
        description="Run M4 to statistically test the most recently proposed hypotheses.",
        params_schema={"type": "object", "properties": {}},
        handler=_test_hypothesis(loop, state),
        max_calls=4,
        requires=("hypotheses",),
    ))
    registry.register(ToolSpec(
        name="run_surgery",
        description="Propose a mechanistic fix (M5) for the best verified hypothesis.",
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
        # quota the way a real M1-M4 stage call does — cap generously.
        max_calls=5,
    ))
    return registry
