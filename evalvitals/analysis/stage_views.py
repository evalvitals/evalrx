"""Stage-focused views used by the unified EvalVitals workbench.

These renderers deliberately accept only a normalized :class:`RunView` or a
stage's persisted records.  They do not know whether the source was an upload,
an attached directory, or a live diagnostic-loop log.
"""

from __future__ import annotations

from typing import Any

from evalvitals.analysis.run_view import RunView


def render_measurement(st: Any, view: RunView) -> None:
    """Render raw M1 analyzer output before it is reduced to M2 signals."""
    story = view.story or {}
    probes = list(story.get("probes") or [])
    if not probes:
        st.info("No M1 measurement event was persisted for this run.")
        return
    st.caption("Raw analyzer outputs remain available here; the Evidence view uses their derived signals.")
    for index, probe in enumerate(probes, start=1):
        analyzers = probe.get("selected_analyzers") or probe.get("analyzers") or []
        findings = probe.get("findings") or {}
        with st.expander(f"Measurement pass {index} · {len(analyzers)} analyzer(s)", expanded=index == 1):
            if analyzers:
                st.markdown("**Analyzers:** " + ", ".join(map(str, analyzers)))
            if findings:
                st.json(findings)
            paths = probe.get("result_paths") or probe.get("artifact_paths") or []
            if paths:
                st.caption("Artifacts: " + ", ".join(map(str, paths)))
            if not findings and not paths:
                st.json(probe)


def hypotheses(story: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten M3 output across diagnostic cycles without losing the source log."""
    out: list[dict[str, Any]] = []
    for diagnosis in story.get("diagnoses") or []:
        out.extend(item for item in diagnosis.get("hypotheses") or [] if isinstance(item, dict))
    return out


def render_hypotheses(st: Any, view: RunView) -> None:
    """Render M3 hypotheses with outcomes from either M5 or M4 beside them."""
    story = view.story or {}
    items = hypotheses(story)
    surgeries = list(story.get("surgeries") or [])
    if not items:
        st.info("No M3 hypotheses were persisted for this run.")
        return
    for index, hypothesis in enumerate(items, start=1):
        statement = str(
            hypothesis.get("plain_statement") or hypothesis.get("statement")
            or hypothesis.get("hypothesis") or "Untitled hypothesis"
        )
        exact = str(hypothesis.get("statement") or hypothesis.get("hypothesis") or "")
        matched = [event for event in surgeries if str(event.get("hypothesis") or "").strip() == exact.strip()]
        with st.container(border=True):
            st.markdown(f"**H{index}. {statement}**")
            if hypothesis.get("failure_mode"):
                st.caption(f"Failure mode: {hypothesis['failure_mode']}")
            if hypothesis.get("test_design"):
                st.markdown(f"How to check: {hypothesis['test_design']}")
            if matched:
                outcomes = ", ".join(
                    f"{str(event.get('module', '?')).upper()}: {event.get('status', 'recorded')}"
                    for event in matched
                )
                st.caption("Recorded outcomes — " + outcomes)
            if exact and exact != statement:
                with st.expander("Technical wording", expanded=False):
                    st.write(exact)


def render_stage_events(st: Any, events: list[dict[str, Any]], empty: str) -> None:
    """Small generic event view for M5 validation and M4 intervention records."""
    if not events:
        st.info(empty)
        return
    for index, event in enumerate(events, start=1):
        status = str(event.get("status") or event.get("fixed") or "recorded")
        headline = str(event.get("hypothesis") or event.get("statement") or event.get("module") or f"Record {index}")
        with st.container(border=True):
            st.markdown(f"**{headline}**")
            st.caption(f"Status: {status}")
            evidence = event.get("evidence") or event.get("evidence_dimensions")
            if evidence:
                st.json(evidence)
            with st.expander("Raw event", expanded=False):
                st.json(event)
