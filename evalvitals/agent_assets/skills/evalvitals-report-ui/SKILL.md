---
name: evalvitals-report-ui
description: Generate EvalVitals exploratory results that read like diagnostic reports, not raw feature dumps.
---

# EvalVitals Report/UI Contract

Use this skill whenever you produce EvalVitals exploratory analysis JSON,
dashboard-ready artifacts, chart specs, or diagnostic report text.

## Audience

The reader is evaluating a model failure mode. They should not need to know
internal probe ids, generated column names, or implementation details to
understand the result.

## Stage Semantics

Use these stage names consistently:

- **M1 — Measurement:** frozen per-case data, analyzer/probe outputs, attention
  maps, and derived signals. Dashboard role: Problem Setting.
- **M2 — Confirmatory analysis:** statistical comparisons of FAIL vs PASS,
  effect sizes, CIs, e-values/e-BH, and deterministic charts. Dashboard role:
  Analysis.
- **M3 — Hypothesis generation:** falsifiable explanations formed from M2
  evidence plus exploratory context. Dashboard role: Hypotheses.
- **M5 — Hypothesis validation:** corrected statistical and protocol evidence
  gates a proposed mechanism. Dashboard role: decision evidence.
- **M4 — Intervention & repair:** targeted interventions and repair candidates
  are compared against the unchanged baseline. Dashboard role: final action.

Every dashboard/report should make clear which stage produced each claim,
chart, hypothesis, or artifact.

## Dashboard Storyboard

Do not produce an artifact dump. Produce a three-panel reader path:

1. **Problem Setting (M1 + run context)**
   - What user question is being answered?
   - What data/cases were provided?
   - What is FAIL vs PASS?
   - Which signals/analyzers are available?

2. **Analysis (M2)**
   - For every analysis method, write: `method`, `evidence/chart`, `takeaway`.
   - Put the main chart next to the method it supports.
   - Put table details below the chart, not before the takeaway.

3. **Hypotheses & Artifacts (M3-M5)**
   - List hypotheses only after M2 evidence.
   - Link each hypothesis to cited charts/signals.
   - Summarize M4/M5 test or fix outcomes.
   - Keep raw artifacts in expandable drill-down sections.

Emit the reader path as `dashboard_storyboard` in the final JSON. The dashboard
renderer is intentionally stable and should only render this artifact. The
agent, not the Streamlit host, owns the run-specific narrative.

Required storyboard panel shape:

```json
{
  "id": "problem_setting | analysis | hypotheses_artifacts",
  "title": "Human panel title",
  "stages": ["M1"],
  "summary": "One paragraph written for the user",
  "items": ["method -> evidence -> takeaway bullets"],
  "artifact_refs": ["candidate_signals", "charts"]
}
```

## Required Output Discipline

- Every `candidate_signals` item must include:
  - `name`: stable machine id, snake_case, suitable for recipes and joins.
  - `display_name`: short human label, suitable for dashboard titles.
  - `rationale`: one sentence in domain language.
  - `suggested_test`: what would confirm or refute it.
- Every chart spec should include:
  - `name`: stable machine id.
  - `title` or `display_name`: human-readable, no raw probe/generated id unless
    no interpretation is known.
- Every `visual_plan` item should include a readable `question` and avoid raw
  field names in the user-facing wording.
- `chart_readings` and `claims` should cite what the chart means, not just which
  column changed.

## Label-Like And Probe-Derived Signals

Signals such as `generated_probe1_false_detection`, `probe1_false_detection`,
or anything that perfectly re-measures the FAIL label are sanity checks. Treat
them as measurement checks, never as root causes.

Use labels like:

- `Sanity check: probe false-detection flag`
- `Probe says object is present`
- `Attention focus share`
- `Maximum relative attention`
- `Mean relative attention`

Do not lead a report with a label-reconstructing signal even when it has the largest
effect size. Mention it as a sanity check, then rank non-leaky explanatory
signals first.

## Overall Result Shape

The final answer should be claim-first:

1. State the supported non-leaky finding in plain language.
2. Mention label-reconstructing signals only as demoted sanity checks.
3. Link each claim to evidence: confirmed signal, chart, or downstream test.
4. Say what not to infer: association is not causality unless intervention tests
   support it.
