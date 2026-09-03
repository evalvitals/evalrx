"""ToolRegistry mechanics: preconditions, call caps, and the stop-gate that
enforces the pre-registration discipline (no declaring success without a
tested, supported, protocol-consistent hypothesis) — all host-enforced, not
prompt-enforced.
"""

from __future__ import annotations

from evalrx.core.result import Result
from evalrx.eval_agent.agentic.actions import Action
from evalrx.eval_agent.agentic.board import EvidenceBoard
from evalrx.eval_agent.agentic.tools import (
    ToolOutcome,
    ToolRegistry,
    ToolSpec,
    _summarize_probe,
    _summarize_stats,
)


def _act(tool: str, params: dict | None = None) -> Action:
    return Action(tool=tool, params=params or {}, rationale="test")


def test_dispatch_rejects_unknown_tool():
    registry = ToolRegistry()
    board = EvidenceBoard()
    outcome = registry.dispatch(_act("nope"), board)
    assert outcome.ok is False
    assert outcome.error == "unknown_tool"


def test_dispatch_enforces_call_cap():
    calls = []
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="probe", description="d", params_schema={},
        handler=lambda b, p: calls.append(1) or ToolOutcome(True, "ok"),
        max_calls=1,
    ))
    board = EvidenceBoard()

    first = registry.dispatch(_act("probe"), board)
    assert first.ok is True
    board.action_log.append({"step": 0, "tool": "probe", "params": {}, "ok": True, "summary": "ok"})

    second = registry.dispatch(_act("probe"), board)
    assert second.ok is False
    assert second.error == "max_calls"
    assert len(calls) == 1  # handler not invoked the second time


def test_dispatch_enforces_unmet_precondition():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="run_stats", description="d", params_schema={},
        handler=lambda b, p: ToolOutcome(True, "ok"),
        requires=("probe_findings",),
    ))
    board = EvidenceBoard()

    outcome = registry.dispatch(_act("run_stats"), board)
    assert outcome.ok is False
    assert outcome.error == "precondition"

    board.probe_findings = [{"analyzer": "x"}]
    outcome2 = registry.dispatch(_act("run_stats"), board)
    assert outcome2.ok is True


def test_stop_resolved_true_is_rejected_without_a_supported_hypothesis():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="stop", description="d", params_schema={},
        handler=lambda b, p: ToolOutcome(True, "stopped"),
    ))
    board = EvidenceBoard()

    outcome = registry.dispatch(_act("stop", {"resolved": True}), board)
    assert outcome.ok is False
    assert outcome.error == "no_supported_hypothesis"


def test_stop_resolved_true_succeeds_with_a_supported_consistent_hypothesis():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="stop", description="d", params_schema={},
        handler=lambda b, p: ToolOutcome(True, "stopped"),
    ))
    board = EvidenceBoard()
    board.hypotheses = [
        {"statement": "s", "status": "supported", "is_consistent_with_protocol": True}
    ]

    outcome = registry.dispatch(_act("stop", {"resolved": True}), board)
    assert outcome.ok is True


def test_stop_resolved_false_is_never_gated():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="stop", description="d", params_schema={},
        handler=lambda b, p: ToolOutcome(True, "giving up"),
    ))
    board = EvidenceBoard()

    outcome = registry.dispatch(_act("stop", {"resolved": False}), board)
    assert outcome.ok is True


def test_catalog_for_prompt_shows_blocked_and_calls_left():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="run_stats", description="run M2", params_schema={},
        handler=lambda b, p: ToolOutcome(True, "ok"),
        max_calls=2, requires=("probe_findings",),
    ))
    board = EvidenceBoard()

    catalog = registry.catalog_for_prompt(board)
    assert "run_stats" in catalog
    assert "calls left: 2" in catalog
    assert "BLOCKED until: probe_findings" in catalog

    board.probe_findings = [{"analyzer": "x"}]
    catalog2 = registry.catalog_for_prompt(board)
    assert "BLOCKED" not in catalog2


def test_evidence_board_has_supported_hypothesis_requires_protocol_consistency():
    board = EvidenceBoard()
    board.hypotheses = [{"statement": "s", "status": "supported", "is_consistent_with_protocol": False}]
    assert board.has_supported_hypothesis() is False

    board.hypotheses = [{"statement": "s", "status": "supported", "is_consistent_with_protocol": True}]
    assert board.has_supported_hypothesis() is True


def test_summarize_probe_extracts_scalars_and_per_case_count():
    res = Result(
        analyzer="attention", model="m",
        findings={
            "mean_attention_flag": 1.0,
            "per_case": [{"sample_id": "a", "attention_flag": True}],
        },
    )
    summary = _summarize_probe({"attention": res})
    assert summary == [{
        "analyzer": "attention",
        "scalars": {"mean_attention_flag": 1.0},
        "n_per_case": 1,
    }]


def test_summarize_stats_extracts_tool_verdicts():
    from evalrx.analysis.stats_tools import StatsToolResult

    report = type("R", (), {"stats_results": [
        StatsToolResult(tool="mcnemar_evalue", ok=True, summary="s", effect=0.2, reject=True, e_value=12.0),
    ]})()
    summary = _summarize_stats(report)
    assert summary == [{
        "tool": "mcnemar_evalue", "ok": True, "effect": 0.2, "reject": True,
        "e_value": 12.0, "summary": "s",
    }]
