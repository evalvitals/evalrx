"""Prompt contract text for exploratory visualization and dashboard outputs."""

VISUALIZATION_OUTPUT_CONTRACT = """\
Before writing plotting code, emit a visual_plan. For every deterministic chart,
write its plotted data as CSV under tables/ and report a chart spec with
{name, display_name, kind, data, x, y, title}. The host renders chart specs from CSV only.
Also emit chart_readings that describe what a human should see and what the
chart cannot prove.
"""

DASHBOARD_STORYBOARD_SYSTEM_PROMPT = """\
You are generating an EvalRX diagnostic report/dashboard, not a raw artifact
browser. Structure the output as three reader-facing panels:

1. Problem Setting (M1 + run context)
   - What data/cases were provided?
   - What is FAIL vs PASS?
   - Which analyzers/probes/signals are available?
   - What split or confirmation policy will be used?

2. Analysis (M2)
   - For each analysis method, produce: method, evidence/chart, takeaway.
   - Lead with held-out confirmatory findings, not exploratory observations.
   - Demote probe-derived or label-reconstructing fields to sanity checks.
   - Every chart needs a human title and a one-sentence reading.

3. Hypotheses & Artifacts (M3-M5)
   - Convert M2 findings into falsifiable M3 hypotheses.
   - Link hypotheses to cited evidence.
   - Summarize M4/M5 intervention/fix outcomes when present.
   - Keep raw artifacts available for inspection, but do not make them the main
     reading path.

Never assume raw field names are user-facing language. Include display_name for
signals/charts and keep raw identifiers only as drill-down provenance.

Emit the panels as `dashboard_storyboard` in the run JSON. The host dashboard is
a stable renderer; the run-specific narrative must come from this storyboard.
"""
