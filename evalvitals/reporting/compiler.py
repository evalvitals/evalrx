"""Compile raw dashboard artifacts into a claim-first diagnostic report."""

from __future__ import annotations

from typing import Any

from evalvitals.reporting.model import (
    Claim,
    DiagnosticReport,
    Evidence,
    ReaderFinding,
    ReaderReport,
    ReportStep,
)
from evalvitals.viz.labels import display_name


def compile_diagnostic_report(
    story: dict[str, Any] | None,
    explore_report: dict[str, Any] | None,
) -> DiagnosticReport:
    """Build a semantic report from a loop story plus the Step-1 explore report.

    This compiler is deterministic.  Agent-authored fields such as ``claims`` or
    ``chart_readings`` are preserved when present, but host-confirmed signal
    verdicts and loop outcomes remain the authoritative backbone.
    """
    story = story or {}
    explore_report = explore_report or {}
    signals = _candidate_signals(explore_report)
    evidence = _compile_evidence(explore_report, story)
    claims = _compile_claims(explore_report, story, signals)

    # Analysis-phase (descriptive) mode: M2 ran with the e-BH validity verdict
    # DEFERRED to the confirm phase, so the dashboard must NOT show
    # supported/not-supported claims — only distributions, charts, and proposed
    # hypotheses. Demote every signal claim to descriptive until confirmation.
    descriptive_only = _is_descriptive_only(story)
    caveats = [str(c) for c in (explore_report.get("caveats") or [])]
    if descriptive_only:
        claims = [_demote_to_descriptive(c) for c in claims]
        caveats.insert(0, _DESCRIPTIVE_BANNER)

    answer = _answer(explore_report, claims) if not descriptive_only else _DESCRIPTIVE_ANSWER
    confidence = _confidence(claims)

    return DiagnosticReport(
        question=_fallback_question(explore_report, story),
        answer=answer,
        confidence=confidence,
        claims=claims,
        evidence=evidence,
        timeline=_compile_timeline(story, explore_report),
        visual_decisions=list(explore_report.get("visual_plan") or []),
        chart_readings=list(explore_report.get("chart_readings") or []),
        dashboard_storyboard=_dashboard_storyboard(explore_report, story, claims),
        critique=_critique(explore_report, signals),
        caveats=caveats,
        next_actions=_next_actions(explore_report, claims),
    )


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
    m5 = data.get("m5") or {}
    m4_fix = data.get("m4_fix") or {}
    explore = m2.get("explore") or {}
    n_cases = int(run.get("n_cases") or 0)
    model = str(run.get("model") or "the model")
    benchmark = str(run.get("benchmark_name") or "this benchmark")
    question = _reader_question(run.get("protocol"))

    m5_ran = bool(m5.get("ran"))
    hypotheses = m3.get("hypotheses") or []
    repair_ran = bool(m4_fix.get("ran"))
    repair_effect = (m4_fix.get("confirm") or {}).get("effect")
    if m5_ran:
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

    findings = _reader_findings(explore, m2, verified=m5_ran)
    if not findings:
        findings = [ReaderFinding(
            title="No clear pattern",
            summary="This run did not produce a finding safe to summarize.",
            why_it_matters="",
            evidence_level="No conclusion yet",
        )]

    open_questions = [
        "Whether the leading pattern explains the outcome on new cases."
    ] if not m5_ran else []
    next_steps = [
        "Test the leading pattern on new cases."
    ] if not m5_ran else []

    answer = (
        "The findings below are leads, not proven causes."
        if not m5_ran else "This report separates observed patterns from independently checked results."
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


_DESCRIPTIVE_BANNER = (
    "ANALYSIS PHASE (descriptive): distributions, charts, and proposed hypotheses "
    "only — signal/hypothesis VALIDITY (e-BH + M5) is deferred to the confirm phase "
    "and is not shown here."
)
_DESCRIPTIVE_ANSWER = (
    "Analysis phase — candidate signals and proposed hypotheses are shown without a "
    "validity verdict; run the confirm phase to adjudicate them."
)


def _is_descriptive_only(story: dict[str, Any]) -> bool:
    """True when every M2 analysis in the story deferred its validity verdict and
    no confirmation (M5/surgery) has run yet — i.e. the loaded run is analysis-only.

    A single confirmatory M2 (``descriptive_only=False``, logged by the confirm
    phase or the all-in-one loop) or any recorded surgery flips this off, so the
    full validity rendering returns once the run is confirmed."""
    analyses = story.get("analyses") or []
    if not analyses:
        return False
    if not all(a.get("descriptive_only") for a in analyses):
        return False
    return not (story.get("surgeries") or [])


def _demote_to_descriptive(claim: Claim) -> Claim:
    """Strip a claim's validity verdict (supported/inconclusive/refuted) to
    descriptive for the analysis-phase view, preserving its text and evidence."""
    if claim.status == "descriptive":
        return claim
    note = "Association shown descriptively; validity is deferred to the confirm phase."
    interp = (claim.interpretation + " " if claim.interpretation else "") + note
    return Claim(
        id=claim.id,
        text=claim.text,
        status="descriptive",
        evidence_ids=list(claim.evidence_ids),
        counter_evidence_ids=list(claim.counter_evidence_ids),
        interpretation=interp.strip(),
        do_not_infer=claim.do_not_infer,
        downstream=list(claim.downstream),
    )


_GENERIC_QUESTION = "What distinguishes failures from passes?"


def _fallback_question(explore_report: dict[str, Any], story: dict[str, Any]) -> str:
    """The explore report's own question wins; a loop/agentic run with no
    explore_report artifact falls back to the protocol description logged at
    run_start — otherwise a run with no explore report shows a question with
    no connection to what it actually investigated."""
    question = str(explore_report.get("question") or "").strip()
    if question:
        return question
    protocol = (story.get("run_start") or {}).get("protocol") or {}
    return str(protocol.get("description") or "").strip() or _GENERIC_QUESTION


def _dashboard_storyboard(
    explore_report: dict[str, Any],
    story: dict[str, Any],
    claims: list[Claim],
) -> list[dict[str, Any]]:
    raw = explore_report.get("dashboard_storyboard") or explore_report.get("ui_panels")
    if isinstance(raw, list) and all(isinstance(p, dict) for p in raw):
        return [dict(p) for p in raw]

    supported = next((c.text for c in claims if c.status == "supported"), "")
    observations = [str(x) for x in (explore_report.get("observations") or [])[:3]]
    readings = [
        str(r.get("reading"))
        for r in (explore_report.get("chart_readings") or [])
        if isinstance(r, dict) and r.get("reading")
    ][:3]
    hypotheses = []
    for diag in story.get("diagnoses") or []:
        for h in diag.get("hypotheses") or []:
            hypotheses.append(str(h.get("statement") or h))

    return [
        {
            "id": "problem_setting",
            "title": "Problem Setting",
            "stages": ["M1"],
            "summary": _fallback_question(explore_report, story),
            "items": observations,
            "artifact_refs": ["data_profile", "candidate_signals"],
        },
        {
            "id": "analysis",
            "title": "Analysis",
            "stages": ["M2"],
            "summary": supported or str(explore_report.get("conclusion") or ""),
            "items": readings,
            "artifact_refs": ["candidate_signals", "charts", "chart_readings"],
        },
        {
            "id": "hypotheses_artifacts",
            "title": "Hypotheses & Artifacts",
            "stages": ["M3", "M4", "M5"],
            "summary": "Hypotheses and downstream decisions generated from the analysis.",
            "items": hypotheses[:5],
            "artifact_refs": ["diagnoses", "surgeries", "fixes"],
        },
    ]


def _candidate_signals(explore_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        s for s in (explore_report.get("candidate_signals") or [])
        if isinstance(s, dict)
    ]


def _compile_evidence(explore_report: dict[str, Any], story: dict[str, Any]) -> list[Evidence]:
    evidence: list[Evidence] = []
    for idx, signal in enumerate(_candidate_signals(explore_report), start=1):
        name = str(signal.get("name") or f"signal_{idx}")
        evidence.append(Evidence(
            id=f"signal:{name}",
            kind="confirmed_signal",
            title=str(signal.get("display_name") or display_name(name)),
            summary=_signal_summary(signal),
            artifact=signal,
        ))
    for idx, chart in enumerate(explore_report.get("charts") or [], start=1):
        if not isinstance(chart, dict):
            continue
        raw_title = str(chart.get("display_name") or chart.get("title") or chart.get("name") or f"Chart {idx}")
        title = display_name(raw_title)
        evidence.append(Evidence(
            id=f"chart:{_slug(title)}",
            kind="chart",
            title=title,
            summary=str(chart.get("description") or chart.get("kind") or ""),
            artifact=chart,
        ))
    for idx, analysis in enumerate(story.get("analyses") or [], start=1):
        summary = str(analysis.get("conclusion") or analysis.get("narrative") or "")
        evidence.append(Evidence(
            id=f"analysis:{idx}",
            kind="m2_analysis",
            title=f"M2 analysis cycle {analysis.get('cycle', idx)}",
            summary=summary,
            artifact=analysis,
        ))
    return evidence


def _compile_claims(
    explore_report: dict[str, Any],
    story: dict[str, Any],
    signals: list[dict[str, Any]],
) -> list[Claim]:
    explicit = [
        _claim_from_agent(c, i)
        for i, c in enumerate(explore_report.get("claims") or [], start=1)
        if isinstance(c, dict)
    ]
    if explicit:
        return explicit

    claims: list[Claim] = []
    for idx, signal in enumerate(signals, start=1):
        name = str(signal.get("name") or f"signal_{idx}")
        status = _signal_status(signal)
        downstream = _downstream_for_signal(name, story)
        claims.append(Claim(
            id=f"C{idx}",
            text=_claim_text(signal),
            status=status,
            evidence_ids=[f"signal:{name}", *_chart_evidence_for_signal(name, explore_report)],
            interpretation=_signal_interpretation(signal),
            do_not_infer=_do_not_infer(signal),
            downstream=downstream,
        ))
    if not claims:
        claims.append(Claim(
            id="C0",
            text="No candidate signal was confirmed in the loaded report.",
            status="inconclusive",
            interpretation="The run may still contain exploratory observations, but no supported "
            "claim can be made from the available confirmation layer.",
            do_not_infer="Do not treat exploratory plots as confirmed root causes.",
        ))
    return _sort_claims(claims)


def _claim_from_agent(raw: dict[str, Any], idx: int) -> Claim:
    status = str(raw.get("status") or "descriptive").lower()
    if status not in {"supported", "inconclusive", "refuted", "descriptive"}:
        status = "descriptive"
    return Claim(
        id=str(raw.get("id") or f"C{idx}"),
        text=str(raw.get("text") or raw.get("claim") or ""),
        status=status,  # type: ignore[arg-type]
        evidence_ids=[str(x) for x in (raw.get("evidence_ids") or [])],
        counter_evidence_ids=[str(x) for x in (raw.get("counter_evidence_ids") or [])],
        interpretation=str(raw.get("interpretation") or ""),
        do_not_infer=str(raw.get("do_not_infer") or ""),
        downstream=[str(x) for x in (raw.get("downstream") or [])],
    )


def _compile_timeline(story: dict[str, Any], explore_report: dict[str, Any]) -> list[ReportStep]:
    steps = [
        ReportStep(
            stage="Explore",
            title="Agent explored patterns and proposed visuals/signals",
            summary=f"{len(explore_report.get('observations') or [])} observation(s), "
            f"{len(explore_report.get('charts') or [])} chart(s), "
            f"{len(explore_report.get('candidate_signals') or [])} signal(s).",
        )
    ]
    analyses = story.get("analyses") or []
    if analyses:
        steps.append(ReportStep(
            stage="M2",
            title="Host confirmed signal associations",
            summary=str(analyses[-1].get("conclusion") or analyses[-1].get("narrative") or ""),
            artifact_ids=[f"analysis:{len(analyses)}"],
        ))
    diagnoses = story.get("diagnoses") or []
    if diagnoses:
        n_h = sum(len(d.get("hypotheses") or []) for d in diagnoses)
        steps.append(ReportStep(
            stage="M3",
            title="Agent formed falsifiable hypotheses",
            summary=f"{n_h} hypothesis/hypotheses proposed from confirmed and exploratory context.",
        ))
    surgeries = story.get("surgeries") or []
    if surgeries:
        steps.append(ReportStep(
            stage="M5/M4",
            title="Hypotheses were tested by interventions",
            summary=f"{len(surgeries)} test/intervention event(s) recorded.",
        ))
    fixes = story.get("fixes") or []
    if fixes:
        steps.append(ReportStep(
            stage="Fix",
            title="Fix candidates were adjudicated",
            summary=f"{len(fixes)} fix event(s) recorded.",
        ))
    return steps


def _answer(explore_report: dict[str, Any], claims: list[Claim]) -> str:
    supported = [c.text for c in claims if c.status == "supported"]
    if supported:
        leaky = sum(1 for c in claims if "sanity check" in c.do_not_infer.lower())
        suffix = f" ({leaky} {_plural(leaky, 'sanity check')} demoted.)" if leaky else ""
        return supported[0] + suffix
    conclusion = str(explore_report.get("conclusion") or "").strip()
    if conclusion:
        return conclusion
    return "No supported diagnostic claim is available in the loaded report."


def _confidence(claims: list[Claim]) -> str:
    if any(c.status == "supported" for c in claims):
        return "medium"
    if any(c.status == "descriptive" for c in claims):
        return "low"
    return "unknown"


def _signal_status(signal: dict[str, Any]) -> str:
    if _is_leaky_signal(signal):
        return "descriptive"
    if signal.get("reject") is True:
        return "supported"
    if signal.get("reject") is False:
        return "inconclusive"
    return "descriptive"


def _signal_summary(signal: dict[str, Any]) -> str:
    parts = []
    if signal.get("effect") is not None:
        parts.append(f"effect={_fmt(signal.get('effect'))}")
    ci = signal.get("ci")
    if isinstance(ci, list | tuple) and len(ci) == 2:
        parts.append(f"CI={_fmt(ci[0])}..{_fmt(ci[1])}")
    if signal.get("e_value") is not None:
        parts.append(f"e={_fmt(signal.get('e_value'))}")
    verdict = "supported" if signal.get("reject") is True else "not supported"
    return f"{verdict}" + (f" ({', '.join(parts)})" if parts else "")


def _claim_text(signal: dict[str, Any]) -> str:
    name = str(signal.get("display_name") or display_name(signal.get("name") or "signal"))
    if _is_leaky_signal(signal):
        return f"{name} tracks the failure label and is treated as descriptive plumbing."
    if signal.get("reject") is True:
        return f"{name} is associated with FAIL cases on the confirmation split."
    if signal.get("reject") is False:
        return f"{name} did not produce a supported FAIL/PASS association."
    return f"{name} is an exploratory signal without a host-confirmed verdict."


def _signal_interpretation(signal: dict[str, Any]) -> str:
    if _is_leaky_signal(signal):
        return "This is useful for auditing the pipeline, not for explaining the failure."
    if signal.get("reject") is True:
        return "Use this as a confirmed association that can motivate M3 hypotheses."
    if signal.get("reject") is False:
        return "Treat this as a negative or underpowered result unless new data changes it."
    return "Treat this as descriptive only."


def _do_not_infer(signal: dict[str, Any]) -> str:
    if _is_leaky_signal(signal):
        return "Sanity check only: do not rank this as a root cause."
    return "Do not infer causality from association without M5/M4 support."


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _downstream_for_signal(name: str, story: dict[str, Any]) -> list[str]:
    out = []
    needle = name.lower()
    for diag in story.get("diagnoses") or []:
        refs = " ".join(str(x) for x in (diag.get("referenced_charts") or [])).lower()
        hyps = diag.get("hypotheses") or []
        if needle in refs or any(needle in str(h).lower() for h in hyps):
            out.append(f"M3 cycle {diag.get('cycle')}: referenced in diagnosis context")
    return out


def _chart_evidence_for_signal(name: str, explore_report: dict[str, Any]) -> list[str]:
    out = []
    lname = name.lower()
    for chart in explore_report.get("charts") or []:
        if not isinstance(chart, dict):
            continue
        title = str(chart.get("title") or chart.get("name") or "")
        blob = " ".join(str(chart.get(k, "")) for k in ("name", "title", "data")).lower()
        if lname in blob or any(part and part in blob for part in lname.split("_")):
            out.append(f"chart:{_slug(title)}")
    return out[:3]


def _critique(explore_report: dict[str, Any], signals: list[dict[str, Any]]) -> list[str]:
    raw = explore_report.get("critique")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    notes = [str(c) for c in (explore_report.get("caveats") or [])]
    if any(_is_leaky_signal(s) for s in signals):
        notes.append("One or more signals are sanity checks and are demoted to descriptive evidence.")
    if not notes:
        notes.append("No explicit critique was recorded; inspect raw artifacts before over-claiming.")
    return notes


def _next_actions(explore_report: dict[str, Any], claims: list[Claim]) -> list[str]:
    actions = [str(x) for x in (explore_report.get("recommended_confirmatory_tests") or [])]
    if any(c.status == "supported" for c in claims):
        actions.append("Inspect the linked M3/M5 outcomes before treating supported signals as causes.")
    if not actions:
        actions.append("Re-run with a larger held-out split or richer probes if no claim is supported.")
    return actions


def _sort_claims(claims: list[Claim]) -> list[Claim]:
    order = {"supported": 0, "inconclusive": 1, "refuted": 2, "descriptive": 3}
    return sorted(claims, key=lambda c: (order.get(c.status, 9), c.id))


def _is_leaky_signal(signal: dict[str, Any]) -> bool:
    name = str(signal.get("name", "")).lower()
    if name.startswith("probe1") or "false_detection" in name:
        return True
    eff = signal.get("effect")
    ci = signal.get("ci") or [None, None]
    try:
        lo, hi = float(ci[0]), float(ci[1])
        return eff is not None and abs(float(eff)) >= 0.999 and (hi - lo) <= 1e-6
    except (TypeError, ValueError, IndexError):
        return False


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return text or "artifact"
