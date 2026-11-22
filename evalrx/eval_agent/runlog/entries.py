"""What each logged event records: one pure function per event kind.

Each takes the object a stage produced and returns the JSON-safe entry the
logger appends for it. None of them touches a file or the logger's state;
`RunLogger` saves any media first, stamps the time, and appends the result.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sized
from pathlib import Path
from typing import Any, Callable

from evalrx.eval_agent.hypothesis import hypothesis_id


def json_safe(value: Any) -> Any:
    """*value* as plain JSON data: dataclasses and ``to_dict`` objects unpacked."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            value = value.to_dict()
        except Exception:  # noqa: BLE001
            pass
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _case_snapshot(case: Any) -> "dict[str, Any]":
    """Make a small, renderer-safe baseline record for an evidence example."""
    if hasattr(case, "to_dict"):
        value = case.to_dict()
    elif isinstance(case, dict):
        value = dict(case)
    else:
        value = {"id": str(getattr(case, "id", ""))}
    inputs = value.get("inputs") if isinstance(value.get("inputs"), dict) else {}
    return {
        "id": str(value.get("id") or value.get("case_id") or ""),
        "input": inputs.get("prompt") or value.get("prompt") or value.get("instruction") or "",
        "baseline_output": value.get("observed", value.get("output")),
        "expected": value.get("expected"),
        "outcome": value.get("label") or value.get("status") or "unknown",
    }


def _iter_cases(cases: Any) -> "list[Any]":
    """Accept CaseBatch, a plain sequence, or a generator without assumptions."""
    if cases is None:
        return []
    value = getattr(cases, "cases", cases)
    try:
        return list(value)
    except TypeError:
        return []


def _probe_examples(results: "dict[str, Any]", cases: Any) -> "list[dict[str, Any]]":
    """Persist two real, bounded M1 walkthroughs beside aggregate findings.

    A probe only becomes a before/after comparison when its analyzer explicitly
    records both outputs.  Otherwise this records an honest *baseline case +
    check result* example; downstream UI must not call it an intervention.
    """
    snapshots: dict[str, dict[str, Any]] = {}
    for case in _iter_cases(cases):
        snapshot = _case_snapshot(case)
        if snapshot["id"]:
            snapshots[snapshot["id"]] = snapshot
    output: list[dict[str, Any]] = []
    used: set[str] = set()
    for name, result in results.items():
        findings = getattr(result, "findings", {}) or {}
        rows = findings.get("per_case") or []
        if not isinstance(rows, list):
            continue
        rows = sorted(
            (row for row in rows if isinstance(row, dict)),
            key=lambda row: 0 if str(snapshots.get(str(row.get("sample_id") or row.get("case_id") or ""), {}).get("outcome", "")).lower() == "fail" else 1,
        )
        for row in rows:
            case_id = str(row.get("sample_id") or row.get("case_id") or "")
            case_snapshot = snapshots.get(case_id)
            if not case_snapshot or case_id in used:
                continue
            checked = {
                str(key).replace("_", " "): value for key, value in row.items()
                if key not in {"sample_id", "case_id"} and isinstance(value, (str, int, float, bool))
            }
            if not checked:
                continue
            used.add(case_id)
            output.append({
                "id": f"m1-{name}-{case_id}", "kind": "case_measurement", "case_id": case_id,
                "probe_title": str(name).replace("_", " ").title(),
                **case_snapshot, "check_result": checked,
                "plain_reading": "This one case illustrates the recorded check. The aggregate M1 result uses all measured cases.",
                "evidence_scope": "one recorded case within M1",
            })
            break
        if len(output) >= 2:
            break
    return output


def probe_entry(
    cycle: int, results: dict[str, Any], *, result_docs: dict[str, Any], artifact_paths: dict[str, str],
    inline_artifacts: dict[str, Any], n_model_calls: int, cases: Any, schema: Any,
    failed_analyzers: "dict[str, str] | None", judge_prompt: "str | None", judge_raw: "str | None",
    duration_sec: "float | None",
) -> dict[str, Any]:
    """One M1 probe pass: every analyzer's findings and complete result, inline."""
    entry: dict[str, Any] = {
        "cycle": cycle,
        "analyzers": list(results),
        "findings": {name: r.findings for name, r in results.items()},
        "results": result_docs,
        "artifact_paths": artifact_paths,
        "artifacts": inline_artifacts,
        "n_model_calls": n_model_calls,
    }
    examples = _probe_examples(results, cases)
    if examples:
        entry["examples"] = examples
    if failed_analyzers:
        entry["failed_analyzers"] = dict(failed_analyzers)
    if schema is not None:
        entry["selection_rationale"] = getattr(schema, "rationale", "")
        selected = getattr(schema, "selected_analyzers", None)
        if selected is not None:
            entry["selected_analyzers"] = list(selected)
    if judge_prompt:
        entry["judge_prompt"] = judge_prompt
    if judge_raw:
        entry["judge_response"] = judge_raw
    if duration_sec is not None:
        entry["duration_sec"] = round(duration_sec, 3)
    return entry


def analysis_entry(
    cycle: int, report: Any, *, portable_path: "Callable[[Any], str]", duration_sec: "float | None",
) -> dict[str, Any]:
    """One M2 analysis: severity, findings, and whichever statistical outputs the report carries."""
    entry: dict[str, Any] = {
        "cycle": cycle,
        "severity": report.severity,
        "n_findings": len(report.findings),
        "findings": [str(f) for f in report.findings],
        "narrative": report.narrative,
        "descriptive_only": bool(getattr(report, "descriptive_only", False)),
    }
    stats_tool = getattr(report, "stats_tool", None)
    if stats_tool:
        entry["stats_tool"] = stats_tool
    fallback_reason = getattr(report, "llm_fallback_reason", None)
    if fallback_reason:
        entry["llm_fallback_reason"] = fallback_reason
    conclusion = getattr(report, "conclusion", None)
    if conclusion:
        entry["conclusion"] = conclusion
    evidence_chain = getattr(report, "evidence_chain", None)
    if evidence_chain:
        entry["evidence_chain"] = list(evidence_chain)
    stats_plan = getattr(report, "stats_plan", None)
    if stats_plan:
        entry["stats_plan"] = stats_plan
    stats_results = getattr(report, "stats_results", None)
    if stats_results:
        entry["stats_results"] = [r.to_dict() for r in stats_results]
    corrected = getattr(report, "corrected_rejections", None)
    if corrected:
        entry["corrected_rejections"] = corrected
    figures = getattr(report, "figures", None)
    if figures:
        entry["figures"] = [portable_path(f) for f in figures]
    llm_prompt = getattr(report, "llm_prompt", None)
    llm_raw = getattr(report, "llm_raw", None)
    if llm_prompt:
        entry["judge_prompt"] = llm_prompt
    if llm_raw:
        entry["judge_response"] = llm_raw
    if duration_sec is not None:
        entry["duration_sec"] = round(duration_sec, 3)
    return entry


def explore_entry(
    cycle: int, report: Any, *, ok: bool, charts: list[Any], rendered: list[str],
    workspace: "dict[str, Any] | None", out_dir: Any, duration_sec: "float | None",
) -> dict[str, Any]:
    """One explore step: counts, the first observations and caveats, and the inlined workspace."""
    tables = getattr(report, "tables", None) or {}
    adjudication = dict(getattr(report, "adjudication", None) or {}) if report is not None else {}
    entry: dict[str, Any] = {
        "cycle": cycle, "ok": ok,
        "n_observations": len(getattr(report, "observations", None) or []) if report is not None else 0,
        "n_charts": len(charts), "n_charts_rendered": len(rendered),
        "n_tables": len(tables) if isinstance(tables, dict) else len(list(tables or [])),
        "n_candidate_signals": len(getattr(report, "candidate_signals", None) or []) if report is not None else 0,
        "n_hypotheses": len(getattr(report, "hypotheses", None) or []) if report is not None else 0,
        "adjudication": {
            k: adjudication[k] for k in (
                "method", "alpha", "split", "n_host_adjudicated", "n_rejected",
                "n_in_family", "n_descriptive_only",
            ) if k in adjudication
        },
        "observations": [str(o) for o in (getattr(report, "observations", None) or [])[:12]] if report is not None else [],
        "caveats": [str(c) for c in (getattr(report, "caveats", None) or [])[:8]] if report is not None else [],
        "figures": rendered,
        "workspace_snapshot": workspace,
        "attempts": int(getattr(report, "attempts", 0) or 0) if report is not None else 0,
    }
    error = str(getattr(report, "error", "") or "") if report is not None else "explorer produced no report"
    if error:
        entry["error"] = error
    if out_dir is not None and workspace is None:
        entry["out_dir"] = str(out_dir)
        report_path = Path(out_dir) / "exploratory_report.json"
        if report_path.exists():
            entry["report_path"] = str(report_path)
    if report is not None and getattr(report, "code", None):
        entry["code"] = str(report.code)
    if report is not None and getattr(report, "raw_outputs", None):
        entry["raw_outputs"] = [str(r) for r in report.raw_outputs]
    if duration_sec is not None:
        entry["duration_sec"] = round(duration_sec, 3)
    return entry


def diagnosis_entry(
    cycle: int, diag: Any, *, duration_sec: "float | None", explore_figures: "list[str] | None",
) -> dict[str, Any]:
    """One M3 diagnosis: the accepted hypotheses, what was proposed, and the critic's review."""
    entry: dict[str, Any] = {
        "cycle": cycle,
        "model_name": diag.model_name,
        "n_hypotheses": len(diag.hypotheses),
        "hypotheses": [
            {
                # Join key for M4/M5 entries logged against this same
                # hypothesis later (log_surgery/log_experiment/log_fix) —
                # see evalrx.eval_agent.hypothesis.hypothesis_id.
                "id": hypothesis_id(h),
                "statement": h.statement, "plain_statement": h.plain_statement,
                "failure_mode": h.predicted_failure_mode,
                "status": h.status.value if h.status else None,
                "test_design": h.test_design,
                "critic": (h.metadata or {}).get("critic"),
                "critic_reason": (h.metadata or {}).get("critic_reason"),
            }
            for h in diag.hypotheses
        ],
        "raw_judge_output": diag.raw_judge_output,
        "n_critic_kept": int(getattr(diag, "n_critic_kept", 0) or 0),
        "n_critic_rejected": int(getattr(diag, "n_critic_rejected", 0) or 0),
    }
    critic_raw = getattr(diag, "critic_raw_output", "") or ""
    if critic_raw:
        entry["critic_raw_output"] = critic_raw
        entry["critic_prompt"] = getattr(diag, "critic_prompt", "") or ""
    proposed = list(getattr(diag, "proposed_hypotheses", None) or [])
    if proposed:
        entry["proposed_hypotheses"] = [
            {"statement": h.statement, "plain_statement": h.plain_statement,
             "failure_mode": h.predicted_failure_mode, "test_design": h.test_design}
            for h in proposed
        ]
    review_decisions = list(getattr(diag, "review_decisions", None) or [])
    if review_decisions:
        entry["review"] = {
            "n_kept": sum(d.get("decision") == "keep" for d in review_decisions),
            "n_rejected": sum(d.get("decision") == "reject" for d in review_decisions),
            "decisions": review_decisions,
        }
    referenced = getattr(diag, "referenced_charts", None)
    if referenced:
        entry["referenced_charts"] = list(referenced)
    if getattr(diag, "explore_context_used", False):
        entry["explore_context_used"] = True
    if getattr(diag, "failure_modes_used", False):
        entry["failure_modes_used"] = True
    if explore_figures:
        entry["explore_figures"] = list(explore_figures)
    m3_prompt = getattr(diag, "prompt", None) or ""
    if m3_prompt:
        entry["judge_prompt"] = m3_prompt
    review_prompt = getattr(diag, "review_prompt", None) or ""
    review_raw = getattr(diag, "review_raw", None) or ""
    if review_prompt or review_raw:
        entry["review_prompt"] = review_prompt
        entry["review_response"] = review_raw
    if duration_sec is not None:
        entry["duration_sec"] = round(duration_sec, 3)
    return entry


def surgery_entry(
    cycle: int, hypothesis: Any, iv: Any, *, stage: str, validation_cases: Any,
    judge_prompt: "str | None", judge_raw: "str | None", duration_sec: "float | None",
) -> dict[str, Any]:
    """One M4 verdict or M5 intervention on a hypothesis."""
    entry: dict[str, Any] = {
        "cycle": cycle,
        "module": stage.lower(),

        # Joins this verdict back to its M3 hypotheses[] entry (same id).
        "hypothesis_id": hypothesis_id(hypothesis),
        "hypothesis": hypothesis.statement,
        "failure_mode": hypothesis.predicted_failure_mode,
        "status": iv.status.value, "fixed": iv.fixed,
        "confidence_score": iv.confidence_score,
        "evidence_dimensions": iv.evidence_dimensions,
        "evidence": iv.evidence,
        "n_refocused_cases": len(iv.new_data) if iv.new_data else None,
    }
    if stage == "M4":
        candidates = _iter_cases(validation_cases)
        if candidates:
            snapshots = [_case_snapshot(case) for case in candidates]
            snapshot = next(
                (item for item in snapshots if str(item.get("outcome", "")).lower() == "fail"),
                snapshots[0],
            )
            entry["validation_examples"] = [{
                "id": f"m4-{snapshot.get('id')}", "kind": "validation_case",
                "case_id": snapshot.get("id"), **snapshot,
                "plain_reading": "This is one case in the independent validation pool. The verdict is determined from the full pool, not this case alone.",
                "evidence_scope": "one case in the independent validation pool",
            }]
    if judge_prompt:
        entry["judge_prompt"] = judge_prompt
    if judge_raw:
        entry["judge_response"] = judge_raw
    if duration_sec is not None:
        entry["duration_sec"] = round(duration_sec, 3)
    return entry


def experiment_entry(
    cycle: int, hypothesis: Any, iv: Any, *, module: str, exp: dict[str, Any], files: dict[str, Any],
    workspace: "dict[str, Any] | None",
) -> dict[str, Any]:
    """One experiment the agent wrote and ran: its code, output and verdict as strings."""
    entry: dict[str, Any] = {
        "cycle": cycle, "module": module,
        # Joins this experiment back to its M3 hypotheses[] entry (same id).
        "hypothesis_id": hypothesis_id(hypothesis),
        "hypothesis": hypothesis.statement,
        "failure_mode": hypothesis.predicted_failure_mode,
        "status": iv.status.value if iv.status else None,
        "fixed": iv.fixed,
        "provider": exp.get("provider"), "verdict": exp.get("verdict"),
        "metrics": exp.get("metrics"), "returncode": exp.get("returncode"),
        "timed_out": exp.get("timed_out"), "cli_usage": exp.get("cli_usage"),
        "llm_calls": exp.get("llm_calls"), "sandbox_runs": exp.get("sandbox_runs"),
        "code": files,
        "stdout": exp.get("stdout"), "stderr": exp.get("stderr"),
        "blueprint": exp.get("blueprint"), "cli_raw_output": exp.get("cli_raw_output"),
        "validation_log": list(exp.get("validation_log") or []) or None,
        "workspace_snapshot": workspace,
        "trial_root": exp.get("trial_root"),
    }
    return entry


def diagnose_report_entry(
    report: Any, cases: Sized, discovery: "list[dict[str, Any]] | None",
) -> dict[str, Any]:
    """The post-diagnosis report: every hypothesis, every M4 verdict, the discovery rows."""
    hypotheses = [
        {
            "statement": h.statement,
            "plain_statement": getattr(h, "plain_statement", ""),
            "failure_mode": h.predicted_failure_mode,
            "status": h.status.value if h.status else None,
        }
        for h in getattr(report, "final_hypotheses", [])
    ]
    m4_results = [
        {
            "hypothesis": tr.hypothesis.statement,
            "failure_mode": tr.hypothesis.predicted_failure_mode,
            "status": tr.status.value,
            "effect_size": tr.effect_size,
            "confidence": tr.confidence,
            "protocol_consistent": tr.is_consistent_with_protocol,
            "verdict": tr.verdict,
            "evidence": tr.evidence,
        }
        for tr in getattr(report, "all_test_results", [])
    ]
    return {
        "cycles": report.cycles,
        "stopped_by": getattr(report, "stopped_by", None),
        "resolved": getattr(report, "resolved", None),
        "n_cases": len(cases),
        "n_hypotheses": len(hypotheses),
        "n_verified": len(getattr(report, "verified_hypotheses", [])),
        "hypotheses": hypotheses,
        "m4_results": m4_results,
        "discovery": list(discovery or []),
    }


def loop_end_entry(
    report: Any, *, tokens_used: "int | None", timings: "dict[str, float] | None",
) -> dict[str, Any]:
    """How a loop ended: cycles, timings, and the hypotheses it finished with."""
    entry: dict[str, Any] = {"cycles": report.cycles}
    if tokens_used is not None:
        entry["tokens_used"] = tokens_used
    if timings:
        entry["timings_sec"] = {k: round(v, 3) for k, v in timings.items()}
        entry["total_duration_sec"] = round(sum(timings.values()), 3)
    if hasattr(report, "resolved"):
        entry["resolved"] = report.resolved
        hyps = getattr(report, "final_hypotheses", [])
        entry["n_hypotheses"] = len(hyps)
        entry["final_hypotheses"] = [
            {"statement": h.statement, "plain_statement": h.plain_statement,
             "failure_mode": h.predicted_failure_mode, "status": h.status.value if h.status else None}
            for h in hyps
        ]
    if hasattr(report, "stopped_by"):
        entry["stopped_by"] = report.stopped_by
        all_hyps = getattr(report, "final_hypotheses", [])
        verified = getattr(report, "verified_hypotheses", [])
        entry["n_hypotheses"] = len(all_hyps)
        entry["n_verified"] = len(verified)
        entry["verified_hypotheses"] = [
            {"statement": tr.hypothesis.statement,
             "failure_mode": tr.hypothesis.predicted_failure_mode,
             "status": tr.status.value,
             "confidence": tr.confidence,
             "protocol_consistent": tr.is_consistent_with_protocol,
             "verdict": getattr(tr, "verdict", None)}
            for tr in verified
        ]
    return entry
