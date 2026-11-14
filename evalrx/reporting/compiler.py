"""Compile a loaded run into the plain-language summary a report opens with."""

from __future__ import annotations

from typing import Any

from evalrx.reporting.model import ReaderFinding, ReaderReport


def compile_reader_report(data: dict[str, Any]) -> ReaderReport:
    """Compile a conservative, plain-language summary from a loaded run.

    This function deliberately does not infer causality from a correlation.  It
    makes the verification state explicit so the renderer cannot accidentally
    present a promising pattern as an established root cause.
    """
    run = data.get("run") or {}
    m1 = data.get("m1") or {}
    m2 = data.get("m2") or {}
    m3 = data.get("m3") or {}
    m4 = data.get("m4") or {}
    m5_fix = data.get("m5_fix") or {}
    explore = m2.get("explore") or {}
    n_cases = int(run.get("n_cases") or 0)
    model = str(run.get("model") or "the model")
    benchmark = str(run.get("benchmark_name") or "this benchmark")
    question = _reader_question(run.get("protocol"))

    m4_ran = bool(m4.get("ran"))
    hypotheses = m3.get("hypotheses") or []
    repair_ran = bool(m5_fix.get("ran"))
    repair_effect = (m5_fix.get("best") or {}).get("effect")
    if m4_ran:
        headline = "A possible explanation was checked on new cases."
        confidence = "Independently checked"
    elif hypotheses:
        headline = "We found a lead that still needs an independent check."
        confidence = "Needs independent checking"
    else:
        headline = "We found patterns, not a proven cause."
        confidence = "Early evidence"
    if repair_ran and isinstance(repair_effect, (int, float)) and repair_effect > 0:
        headline = "A repair was tried and improved this run."

    what_we_did = [
        f"{model} was evaluated on {n_cases:,} {benchmark} cases." if n_cases else f"{model} was evaluated on {benchmark}.",
        f"{len(m1.get('analyzers') or [])} behavior checks were compared between successful and unsuccessful cases.",
    ]

    findings = _reader_findings(explore, m2, verified=m4_ran)
    if not findings:
        findings = [ReaderFinding(
            title="No clear pattern",
            summary="This run did not produce a finding safe to summarize.",
            why_it_matters="",
            evidence_level="No conclusion yet",
        )]

    open_questions = [
        "Whether the leading pattern explains the outcome on new cases."
    ] if not m4_ran else []
    next_steps = [
        "Test the leading pattern on new cases."
    ] if not m4_ran else []

    answer = (
        "The findings below are leads, not proven causes."
        if not m4_ran else "This report separates observed patterns from independently checked results."
    )
    return ReaderReport(
        headline=headline,
        question=question,
        answer=answer,
        confidence=confidence,
        what_we_did=what_we_did,
        key_findings=findings,
        open_questions=open_questions,
        next_steps=next_steps,
    )


def _reader_findings(explore: dict[str, Any], m2: dict[str, Any], *, verified: bool) -> list[ReaderFinding]:
    signals = [signal for signal in (explore.get("candidate_signals") or []) if isinstance(signal, dict)]
    findings: list[ReaderFinding] = []
    for signal in signals[:2]:
        name = str(signal.get("name") or "")
        title, summary, why = _plain_signal(name, signal)
        level = "Independently checked" if verified else "Pattern seen in this run"
        findings.append(ReaderFinding(
            title=title,
            summary=summary,
            why_it_matters=why,
            evidence_level=level,
            limitation="This pattern alone does not prove what caused the mistake." if not verified else "Interpret alongside the independent-check details.",
        ))
    if findings:
        return findings
    for observation in (explore.get("observations") or [])[:2]:
        if isinstance(observation, str) and observation.strip():
            findings.append(ReaderFinding(
                title="A pattern worth checking",
                summary=observation,
                why_it_matters="",
                evidence_level="Pattern seen in this run",
                limitation="It is not proof of a root cause.",
            ))
    return findings


def _plain_signal(name: str, signal: dict[str, Any]) -> tuple[str, str, str]:
    """Return only text supplied by an arbitrary task's structured artifacts.

    The renderer owns the evidence-status wording.  It must not encode domain
    assumptions (for example, multiple-choice or audio-specific explanations)
    in order to work for every evaluator that emits candidate signals.
    """
    display = str(signal.get("display_name") or name.replace("_", " ").strip().title() or "Observed pattern")
    summary = str(
        signal.get("reader_summary")
        or signal.get("summary")
        or signal.get("rationale")
        or "This behavior differed between the outcome groups in this run."
    )
    return display, summary, ""


def _reader_question(protocol: Any) -> str:
    """Extract a compact task question without assuming a protocol schema."""
    if isinstance(protocol, dict):
        for key in ("question", "description", "name"):
            value = protocol.get(key)
            if value:
                return str(value)
    if protocol:
        return str(protocol)
    return "What patterns are linked to the observed outcomes?"
