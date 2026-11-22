"""The run log mirrored into Langfuse: one function per logged event.

`RunLogger` writes an event to its JSON document and then calls the matching
function here with the tracer it owns. Nothing in this module touches the
log's files; a run with tracing off calls the same functions on a tracer that
records nothing.
"""

from __future__ import annotations

from typing import Any

#: How many of an analyzer's target-model calls are mirrored per probe.
_MAX_CALLS_PER_PROBE = 50


def trace_run_start(tracer: Any, entry: dict[str, Any]) -> None:
    """Open the trace for a run that just logged its start."""
    model_name = str(entry.get("model") or "Target Model")
    proto = entry.get("protocol") or {}
    proto_desc = proto.get("description", "") if isinstance(proto, dict) else str(proto)
    bench_name = str(entry.get("benchmark_name") or proto_desc or "Benchmark")
    tracer.start_trace(
        model=model_name, benchmark=bench_name,
        n_cases=int(entry.get("n_cases", 0) or 0), metadata=entry,
    )


def _trace_model_calls(tracer: Any, analyzer: str, model_calls: list[dict[str, Any]], span: Any) -> int:
    """Mirror *analyzer*'s target-model calls under *span*; how many it made."""
    calls = [c for c in model_calls if c.get("analyzer") == analyzer]
    for call in calls[:_MAX_CALLS_PER_PROBE]:
        tracer.log_generation(
            name=f"{analyzer} · {call.get('method')} #{call.get('call_index')}",
            model="target_model", prompt=call.get("inputs"),
            completion=call.get("error") or call.get("output"),
            span_id=span,
            metadata={"duration_sec": call.get("duration_sec"), "error": call.get("error")},
        )
    return len(calls)


def trace_probe(
    tracer: Any, cycle: int, results: dict[str, Any], entry: dict[str, Any], *,
    artifact_paths: dict[str, str], figures: list[Any], judge_prompt: "str | None",
    judge_raw: "str | None", duration_sec: "float | None", model_calls: list[dict[str, Any]],
) -> None:
    """One M1 span, a child span per analyzer, and the target-model calls each made."""
    m1_span = tracer.start_span(
        name=f"M1: Multi-Dimensional Checkup (Cycle {cycle})",
        stage="M1",
        input_data={"analyzers": list(results.keys()), "selected_analyzers": entry.get("selected_analyzers", [])},
        metadata={"duration_sec": duration_sec, "artifacts": {
            "artifact_paths": artifact_paths, "figures": [str(p) for p in figures],
        }},
    )
    if judge_prompt or judge_raw:
        tracer.log_generation(
            name="M1 Analyzer Selection", model="judge",
            prompt=judge_prompt or "", completion=judge_raw or "", span_id=m1_span,
        )
    for name, r in results.items():
        findings = getattr(r, "findings", {}) or {}
        per_case = findings.get("per_case") or []
        probe_span = tracer.start_span(
            name=f"Probe: {name}", stage=f"M1_{name}", input_data={"probe": name},
            parent_id=m1_span,
            metadata={
                "n_scored": len(per_case) if per_case else (findings.get("n_cases") or findings.get("n_scored")),
            },
        )
        n_calls = _trace_model_calls(tracer, name, model_calls, probe_span)
        tracer.end_span(probe_span, output_data={"findings": findings, "n_model_calls": n_calls})
    orphaned = {c["analyzer"] for c in model_calls} - set(results)
    for name in orphaned:
        failed_span = tracer.start_span(
            name=f"Probe: {name} (failed)", stage=f"M1_{name}", input_data={"probe": name},
            parent_id=m1_span, metadata={"note": "analyzer raised before producing a Result"},
        )
        n_calls = _trace_model_calls(tracer, name, model_calls, failed_span)
        tracer.end_span(failed_span, status="failed", output_data={"n_model_calls": n_calls})
    tracer.end_span(m1_span, output_data={"n_probes": len(results)})


def trace_analysis(tracer: Any, cycle: int, report: Any, *, duration_sec: "float | None") -> None:
    """One M2 span with the judge exchange and an effect score per signal."""
    stats_tool = getattr(report, "stats_tool", None)
    figures = getattr(report, "figures", None)
    llm_prompt, llm_raw = getattr(report, "llm_prompt", None), getattr(report, "llm_raw", None)
    stats_results = getattr(report, "stats_results", None)
    conclusion = getattr(report, "conclusion", None)
    m2_span = tracer.start_span(
        name=f"M2: Screening & Confirmatory Signals (Cycle {cycle})", stage="M2",
        input_data={"severity": report.severity, "n_findings": len(report.findings)},
        metadata={"duration_sec": duration_sec, "stats_tool": stats_tool,
                  "artifacts": {"figures": [str(f) for f in (figures or [])]}},
    )
    if llm_prompt or llm_raw:
        tracer.log_generation(
            name="M2 Statistical Screening Analysis", model="judge",
            prompt=llm_prompt or "", completion=llm_raw or "", span_id=m2_span,
        )
    for s in (stats_results or []):
        s_dict = s.to_dict() if hasattr(s, "to_dict") else (s if isinstance(s, dict) else {})
        sig_name = s_dict.get("config", {}).get("signal") or s_dict.get("tool") or "signal"
        eff = s_dict.get("effect")
        if eff is not None:
            tracer.log_score(
                name=f"m2_effect_{sig_name}", value=float(eff),
                comment=f"p={s_dict.get('p_value')}", span_id=m2_span,
            )
    tracer.end_span(m2_span, output_data={"conclusion": conclusion or ""})


def trace_explore(
    tracer: Any, cycle: int, report: Any, entry: dict[str, Any], *,
    ok: bool, out_dir: Any, rendered: list[str], duration_sec: "float | None",
) -> None:
    """One span for the explore step with the coder agent's outputs."""
    exp_span = tracer.start_span(
        name=f"Explore: Free-form EDA (Cycle {cycle})", stage="EXPLORE",
        input_data={"ok": ok, "attempts": entry.get("attempts", 0)},
        metadata={"duration_sec": duration_sec, "artifacts": {
            "out_dir": str(out_dir) if out_dir is not None else None,
            "report_path": entry.get("report_path"), "figures": rendered,
        }},
    )
    if report is not None:
        for i, raw in enumerate(getattr(report, "raw_outputs", None) or []):
            tracer.log_generation(
                name=f"Explore Coder Agent (attempt {i + 1})", model="coder_agent",
                prompt=None, completion=str(raw), span_id=exp_span,
            )
        if getattr(report, "code", None):
            tracer.log_generation(
                name="Explore Analysis Code (analysis.py)", model="coder_agent",
                prompt=None, completion=str(report.code), span_id=exp_span,
            )
    tracer.end_span(exp_span, output_data={
        "n_observations": entry.get("n_observations", 0),
        "n_candidate_signals": entry.get("n_candidate_signals", 0),
    })


def trace_diagnosis(
    tracer: Any, cycle: int, diag: Any, entry: dict[str, Any], *, duration_sec: "float | None",
) -> None:
    """One M3 span with the diagnosis exchange and the adversarial review."""
    m3_prompt = getattr(diag, "prompt", None) or ""
    review_prompt = getattr(diag, "review_prompt", None) or ""
    review_raw = getattr(diag, "review_raw", None) or ""
    review_decisions = list(getattr(diag, "review_decisions", None) or [])
    proposed = list(getattr(diag, "proposed_hypotheses", None) or [])
    m3_span = tracer.start_span(
        name=f"M3: Root-Cause Diagnosis (Cycle {cycle})", stage="M3",
        input_data={"model_name": diag.model_name, "n_hypotheses": len(diag.hypotheses)},
        metadata={"duration_sec": duration_sec},
    )
    tracer.log_generation(
        name="AI Doctor Diagnostician",
        model=str(tracer.trace_metadata.get("judge") or "diagnosis_judge"),
        prompt=m3_prompt, completion=diag.raw_judge_output or "", span_id=m3_span,
        metadata={"hypotheses": [h.statement for h in diag.hypotheses]},
    )
    if review_prompt or review_raw:
        tracer.log_generation(
            name="M3 Adversarial Evidence Review",
            model=str(tracer.trace_metadata.get("judge") or "diagnosis_judge"),
            prompt=review_prompt, completion=review_raw, span_id=m3_span,
            metadata={"decisions": review_decisions},
        )
    tracer.end_span(m3_span, output_data={
        "n_hypotheses": len(diag.hypotheses),
        "n_proposed": len(proposed) or len(diag.hypotheses),
        "review": entry.get("review"),
    })


def trace_surgery(
    tracer: Any, hypothesis: Any, iv: Any, *, is_m4: bool, judge_prompt: "str | None", judge_raw: "str | None",
) -> None:
    """One span per verdict (M4) or intervention (M5) with its confidence score."""
    stage_title = "M4 Adjudication" if is_m4 else "M5 Intervention"
    surg_span = tracer.start_span(
        name=f"{stage_title}: {hypothesis.statement[:60]}",
        stage="M4" if is_m4 else "M5_SURGERY",
        input_data={"hypothesis": hypothesis.statement, "failure_mode": hypothesis.predicted_failure_mode},
        metadata={"status": iv.status.value, "fixed": iv.fixed, "confidence_score": iv.confidence_score},
    )
    if judge_prompt or judge_raw:
        tracer.log_generation(
            name=f"{stage_title}: Protocol Consistency Judge", model="judge",
            prompt=judge_prompt or "", completion=judge_raw or "", span_id=surg_span,
        )
    if iv.confidence_score is not None:
        tracer.log_score(
            name="adjudication_confidence", value=float(iv.confidence_score),
            comment=f"status={iv.status.value}, fixed={iv.fixed}", span_id=surg_span,
        )
    tracer.end_span(surg_span, output_data={"evidence": iv.evidence or {}})


def trace_experiment(
    tracer: Any, hypothesis: Any, exp: dict[str, Any], entry: dict[str, Any], *, stage: str, module: str,
) -> None:
    """One span for an experiment the agent wrote and ran."""
    exp_span = tracer.start_span(
        name=f"{stage}: Experiment — {hypothesis.statement[:60]}",
        stage=f"{stage}_EXPERIMENT",
        input_data={"hypothesis": hypothesis.statement, "failure_mode": hypothesis.predicted_failure_mode},
        metadata={"provider": exp.get("provider"), "returncode": exp.get("returncode")},
    )
    cli_raw = exp.get("cli_raw_output")
    if cli_raw:
        tracer.log_generation(
            name=f"{module.upper()} Coder Agent", model=str(exp.get("provider") or "coder_agent"),
            prompt=None, completion=str(cli_raw), span_id=exp_span,
        )
    vlog = exp.get("validation_log")
    if vlog:
        tracer.log_generation(
            name=f"{module.upper()} Validation Log", model=str(exp.get("provider") or "coder_agent"),
            prompt=None, completion="\n".join(str(x) for x in vlog), span_id=exp_span,
        )
    tracer.end_span(exp_span, output_data={
        "status": entry.get("status"), "fixed": entry.get("fixed"), "verdict": entry.get("verdict"),
    })


def trace_fix(tracer: Any, outcome: dict[str, Any], best: dict[str, Any]) -> None:
    """One span for the fix step with the winner's gain and patch."""
    fix_span = tracer.start_span(
        name="M5: Targeted Repair & Confirmation", stage="M5_FIX",
        input_data={"candidates_evaluated": len(outcome.get("attempted", []))},
    )
    if best.get("effect") is not None:
        tracer.log_score(
            name="repair_net_accuracy_gain", value=float(best.get("effect", 0.0)),
            comment=f"cured={best.get('n_fixed')}, broken={best.get('n_broken')}",
            span_id=fix_span,
        )
    if (best.get("payload") or {}).get("prompt_template"):
        tracer.log_generation(
            name="Winning Repair Patch", model="evalrx_repair",
            prompt="Repair Candidate Search",
            completion=best["payload"]["prompt_template"], span_id=fix_span,
        )
    tracer.end_span(fix_span, output_data={"selected": best.get("name")})


def trace_agent_decision(
    tracer: Any, step: int, *, action: str, params: "dict[str, Any] | None", rationale: str, valid: bool,
    repair_attempts: int, fallback_used: bool, judge_prompt: "str | None", judge_raw: "str | None",
) -> None:
    """One span per decision the agentic loop's judge made."""
    decision_span = tracer.start_span(
        name=f"Agent Decision: step {step}", stage="AGENT_DECISION",
        input_data={"action": action, "params": params or {}},
        metadata={"valid": valid, "repair_attempts": repair_attempts, "fallback_used": fallback_used},
    )
    if judge_prompt or judge_raw:
        tracer.log_generation(
            name="Agent Decision Judge", model="judge",
            prompt=judge_prompt or "", completion=judge_raw or "", span_id=decision_span,
        )
    tracer.end_span(decision_span, output_data={"action": action, "rationale": rationale})


def trace_stage_skipped(tracer: Any, stage: str, reason_code: str, detail: str) -> None:
    """A zero-length span marking a stage the run skipped."""
    span = tracer.start_span(
        name=f"{stage}: skipped", stage=stage,
        input_data={"reason_code": reason_code}, metadata={"detail": detail},
    )
    tracer.end_span(span, output_data={"reason_code": reason_code}, status="skipped")


def trace_tool_codegen(
    tracer: Any, *, module: str, name: str, need: str, source: str, ok: bool, error: str,
    prompt: str, raw_output: str, raw_stream: str, code: str,
) -> None:
    """One span per tool the agent wrote."""
    cg_span = tracer.start_span(
        name=f"Tool Codegen: {module}/{name}", stage=f"CODEGEN_{module}",
        input_data={"need": need, "source": source},
        metadata={"ok": ok, "error": error or None},
    )
    if prompt or raw_output:
        tracer.log_generation(
            name=f"Tool Synthesis: {name}", model=source,
            prompt=prompt or "", completion=raw_stream or raw_output or code, span_id=cg_span,
        )
    tracer.end_span(cg_span, output_data={"ok": ok, "code_chars": len(code or "")})
