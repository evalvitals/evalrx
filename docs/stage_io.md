# Stage Input/Output Reference

A single reference for what goes into and comes out of each stage of the
M1→M5 diagnosis pipeline. [Architecture](architecture.md#eval_agent-automated-diagnosis-pipeline)
covers *why* the pipeline is shaped this way and how the three loops
(`AutoDiagnoseLoop`, `VLDiagnoseLoop`, `AgenticDiagnoseLoop`) orchestrate these
same stages differently; [Exploratory Analysis (M2/M3)](m2_analysis.md) and
[Intervention & Verification (M4/M5)](intervention.md) cover usage. This page
covers the shape of the data crossing each boundary, and — since that shape
is exactly what a viewer has to render — [what to put on screen for it](#ui-reference--building-a-viewer-on-this-pipeline).
Building a new UI on this pipeline? Read the stage you're rendering below,
then [UI Reference](#ui-reference--building-a-viewer-on-this-pipeline) at the
bottom for the existing dashboard's page layout, tab-to-stage mapping, and
the conventions it uses (source: `evalvitals/analysis/dashboard_app.py`).

## Pipeline data flow

```text
CaseBatch (labeled FailureCases)
   │
   ▼
M1  ProbeAgent.probe(model, data)              → dict[str, Result]
   │
   ├─(optional) ExploratoryAnalysisAgent.explore_records(per-case table)
   │            → ExploreContext for M3 + explore/ files for the dashboard
   │              (descriptive; never enters M2's family, M5, or the fix gate)
   ▼
M2  AnalysisModule.analyze(results)            → AnalysisReport
    StatsAnalysisAgent.analyze(results, data)  → StatsAnalysisReport (superset)
   │
   ▼
M3  DiagnosisAgent.diagnose(report)            → DiagnosisResult (.hypotheses)
   │
   ├──────────────────────────────┐
   ▼                              ▼
M4  SurgeryAgent.operate(...)     M5  HypothesisTester.test(...)
    → InterventionResult              → list[HypothesisTestResult]
    (verify why, no repair)           (statistical + protocol verdict;
                                        gates whether the loop stops)
   │
   ▼ (post-loop, on the best verified hypothesis)
M4  FixAgent.propose_and_validate(...) → FixOutcome
    (propose + validate candidate repairs, tiered L1→L4)
```

`VLDiagnoseLoop` runs M1→M2→M3→M5 in a cycle and calls M4 (`SurgeryAgent` then
`FixAgent`) once, post-loop, on the best verified hypothesis. `AutoDiagnoseLoop`
runs M1→M2→M3→M4 (`SurgeryAgent`) every cycle instead of M5. Both consume and
produce the same per-stage types described below — only the wiring differs.

**Pre-M1 (optional):** `ProbeSearchAgent.run(model, seed_pool)` synthesizes
*new* test cases rather than analyzing an already-collected dataset —
`result.failure_cases` is a `CaseBatch` that can seed or extend the data M1
then probes. See [m2_analysis.md](m2_analysis.md#probe-search--hierarchical-mcts-failure-discovery-vlm).

## Quick reference

| Stage | Class | Method | Input | Output |
|---|---|---|---|---|
| Pre-M1 | `ProbeSearchAgent` | `run(model, seed_pool)` | `Model` + `CaseBatch` (seed pool) | `ProbeSearchResult` (`.failure_cases: CaseBatch`) |
| M1 | `ProbeAgent` | `probe(model, data, protocol=, prior_hypotheses=, hint_failure_modes=)` | `Model` + `CaseBatch` | `dict[str, Result]` |
| M2 (loop) | `AnalysisModule` | `analyze(results, model_name)` | `dict[str, Result]` | `AnalysisReport` |
| M2 (loop, confirmatory) | `StatsAnalysisAgent` | `analyze(results, model_name, protocol=, data=)` | `dict[str, Result]` + `CaseBatch` | `StatsAnalysisReport` |
| M2 (standalone) | `ExploratoryAnalysisAgent` | `explore_path(path, question=)` | file/dir path or in-memory records | `exploratory_report.json` dict (takeaways, charts) |
| explore (loop, optional) | `ExploratoryAnalysisAgent` via `VLDiagnoseLoop(explorer=)` | `explore_records(records, question=, outcome_col="label")` | M1's per-case table (`build_stats_input` → `per_case_to_records`) — the same rows M2 sees | `ExploreContext` (observations / rendered charts / caveats) for M3; `explore/{exploratory_report.json,tables/,figures/}` on disk; an `explore` run-log event |
| M3 (loop) | `DiagnosisAgent` | `diagnose(analysis, prior_cycles=, explore_context=, failure_modes=)` | `AnalysisReport` (or `StatsAnalysisReport`) | `DiagnosisResult` (`.hypotheses: list[Hypothesis]`) |
| M3 (standalone) | `HypothesisAgent` | `propose(report_dict)` | M2's report dict | `list[Hypothesis]` |
| M4 (verify) | `SurgeryAgent` | `operate(hypothesis, model, results, data)` | one `Hypothesis` + `dict[str, Result]` + `CaseBatch` | `InterventionResult` |
| M4 (fix) | `FixAgent` | `propose_and_validate(model, data, hypotheses, prior_attempts=)` | `Model` + `CaseBatch` + `list[Hypothesis]` | `FixOutcome` |
| M5 | `HypothesisTester` | `test(hypotheses, stats_report, data, protocol=)` | `list[Hypothesis]` + `StatsAnalysisReport` + `CaseBatch` | `list[HypothesisTestResult]` |
| Loop | `AutoDiagnoseLoop` / `VLDiagnoseLoop` / `AgenticDiagnoseLoop` | `run(cases)` | `CaseBatch` | `AutoDiagnoseReport` (all three loops return this one class — see below) |

## Shared input type — `CaseBatch`

Every stage that touches raw data consumes or produces a `CaseBatch` — a
sequence of `FailureCase`:

```python
FailureCase(
    inputs: Inputs,          # prompt (+ optional image/audio/video)
    expected: Any = None,    # gold / expected behaviour
    observed: Any = None,    # what the model produced, if run
    label: Label = UNKNOWN,  # PASS / FAIL / UNKNOWN
    tags: set[str] = set(),  # free-form failure-taxonomy tags
    id: str = <auto-uuid>,
    metadata: dict = {},
)
```

`as_casebatch(str | FailureCase | Inputs | list | CaseBatch)` normalizes any
of these into a `CaseBatch`, so stage inputs accept several shapes in
practice even though the type hints say `CaseBatch`.

---

## Pre-M1 — `ProbeSearchAgent` (optional)

Synthesizes and evaluates *new* test cases (VLM QA in v1) instead of
analyzing data you already collected.

**Input:** a `Model` and a seed `CaseBatch` (VLM cases: image + question +
expected answer) plus a search `budget` (number of simulations).

**Output:** `ProbeSearchResult` —
`n_simulations`, `n_macro`, `n_micro`, `error_rate`, and
`failure_cases: CaseBatch` (the newly discovered failing cases — feed this
into M1, or straight into `cluster_failures`).

**UI:** not rendered by the current dashboard — it's a data-generation step
that runs *before* a diagnosis run exists, not part of one. If a new UI
wants to expose it, treat it as its own small flow (pick a seed pool, set a
budget, run, then hand the resulting `CaseBatch` into a normal M1 run) —
nothing in the five-tab layout below assumes it happened.

## M1 — `ProbeAgent` (analyzer selection + execution)

**Input:**
- `model: Model` — the model under diagnosis.
- `data: CaseBatch` — cases to run analyzers on.
- `protocol: ExperimentProtocol | None` — NL description of what to
  investigate; enables LLM-guided analyzer selection when a judge is set.
- `prior_hypotheses: list[Hypothesis] | None` — M3 hypotheses from earlier
  cycles, for focused follow-up probing.
- `hint_failure_modes: list[str] | None` — failure-mode tags used by the
  static (no-judge) fallback selector, `StrategyProbe`.

**Output:** `dict[str, Result]` — one `Result` per analyzer that ran, keyed by
analyzer name (e.g. `{"attention": AttentionResult(...), "pope": Result(...)}`).
Each `Result` carries `findings` (light, JSON-safe dict) and `artifacts`
(heavy tensors/arrays). `ProbeAgent.last_schema` records which analyzers ran
and why (selection rationale), separate from the return value.

Analyzer selection itself is two-tiered: `StrategyProbe.detect_kind()` picks
VLM / Agent / LLM based on capabilities, `StrategyProbe.select()` ranks
compatible analyzers for that kind (or an LLM judge picks them directly from
the protocol description); `WhiteboxProbeGenerator`/`ProbeGenerator` write a
bespoke probe when no standard analyzer covers the failure mode.

**UI:** In the existing dashboard, M1's raw output (`dict[str, Result]`) is
never shown directly — only the *derived* per-case feature table M2 builds
from it reaches the screen, on **Tab 1 — Problem Setting**: case counts
(total / FAIL / PASS / explore-confirm split), the reconstructed per-case
signal columns (name + non-null coverage, as a small table), and a "stage
map" chip strip showing which of M1–M5 this particular run actually reached
(`_render_stage_map`, `_render_problem_setting`). If a run has no signal
table to reconstruct, this tab falls back to the raw `data_profile` column
schema instead. **This is a known gap, not a pattern to keep** — the new UI
is explicitly required to also show M1's own results (each analyzer's
`findings` + rendered `artifacts`); see
[Requirements for the new UI](#requirements-for-the-new-ui) below.

## M2 — analysis (threshold rules + statistics)

Two implementations, both taking M1's `dict[str, Result]`:

### `AnalysisModule` (base — always available, no judge required)

**Input:** `results: dict[str, Result]` from M1, `model_name: str`.

**Output:** `AnalysisReport` —
`findings: list[AnalysisFinding]` (flagged threshold violations, sorted
high-severity first), `severity: "high"|"medium"|"low"|"none"`,
`narrative: str` (human-readable summary forwarded to M3), and
`raw_results` (the M1 dict, passed through for M4/M5's per-case signal
extraction).

### `StatsAnalysisAgent` (confirmatory — superset of `AnalysisReport`)

**Input:** `results: dict[str, Result]`, `model_name`,
`protocol: ExperimentProtocol | None`, `data: CaseBatch | None` (required to
run the statistical tool layer — needs per-case PASS/FAIL labels; falls back
to threshold rules only when absent/unlabeled).

**Output:** `StatsAnalysisReport` — everything `AnalysisReport` has, plus:
`conclusion` (NL summary), `evidence_chain` (step-by-step derivation),
`stats_tool` (which statistics path ran: `threshold_rules` / `llm_guided` /
`selected_tools`), `stats_results: list[StatsToolResult]` (per-tool verdicts
from the catalog: `signal_label_assoc`, `mcnemar_evalue`, `bootstrap_diff`,
`friedman`, `rank_corr`, `single_rate_evalue`), `stats_plan` (which tools
were selected and why), `corrected_rejections` (e-BH FDR correction across
all tool e-values), `figures` (paths to generated plots), and
`llm_fallback_reason` (non-empty only when the LLM-guided narrative path was
attempted and raised — `""` otherwise, whether it was never attempted or
succeeded).

### Standalone `ExploratoryAnalysisAgent` (M2, no-code CLI path)

A different, purely-descriptive tool for the `evalvitals explore` CLI /
`evalvitals.explore()` — not loop-internal. See
[m2_analysis.md](m2_analysis.md) for full detail.

**Input:** `path: str | Path` (a `.json`/`.jsonl` file or directory tree of
results — any shape; a coding agent figures out the schema) or in-memory
`list[dict]`, plus a natural-language `question`.

**Output:** `exploratory_report.json` — `takeaways` (with `plain_title` +
technical `title`), `observations`, `candidate_signals`, chart/table
references — persisted alongside `analysis.py` (the generated analysis
code), `records.json` (the tidy table it built), and `figures/`/`tables/`.

**UI (Tab 2 — Exploratory Analysis):** for the artifact-based rendering path
(`_render_standalone_analysis` — see
[UI reference](#ui-reference--building-a-viewer-on-this-pipeline) for the
two rendering modes), the tab shows: an
optional data-structure/raw-data browser, then each `takeaway` as a card —
badge number, headline (`plain_title` shown first, technical `title`
secondary), its referenced charts/tables (`chart_names`/`table_names`
looked up by name — missing ones render an explicit "referenced evidence not
found" notice rather than silently dropping the claim), then full analysis
text + source data in an expander. `observations` and `caveats` sit above
the takeaways in collapsed expanders. Charts/tables the report produced but
no takeaway referenced ("orphans") render at the bottom under their own
section so nothing generated is ever hidden. **Business logic to preserve:**
this tab is *purely descriptive* — no "supported"/"rejected" language
anywhere, even when `stats_results` carries a `reject` flag; that verdict
framing is reserved for Tab 4, because in-sample findings and out-of-sample
verdicts must never look the same to the reader.

## M3 — hypothesis generation

### `DiagnosisAgent` (loop-internal)

**Input:** `analysis: AnalysisReport | StatsAnalysisReport | dict[str, Result]`
(M2's output — dict form is accepted for backward compatibility),
`prior_cycles: list[dict] | None` (summaries of earlier M1→M4 cycles, so the
judge avoids re-proposing tested hypotheses), `explore_context`,
`failure_modes` (optional, descriptive-only context from clustering).

**Output:** `DiagnosisResult` —
`hypotheses: list[Hypothesis]` (proposed, falsifiable, for M4/M5),
`findings_summary` (the findings dict shown to the judge),
`raw_judge_output` (verbatim LLM response), `referenced_charts` (provenance).

### Standalone `HypothesisAgent` (M3, no-code CLI path)

**Input:** the M2 `exploratory_report.json` dict (or `ExploratoryReport`
object).

**Output:** `list[Hypothesis]`, each with `statement` (technical claim),
`plain_statement` (jargon-free rewrite), `basis` (which M2 takeaway grounds
it), `test_design` (what evidence would confirm/refute it). Proposal only —
generating a hypothesis is not testing one.

**UI (Tab 3 — Hypotheses):** one card per hypothesis — headline is
`plain_statement` (falls back to `statement`), technical `statement` shown
as a secondary line only when it differs from the plain one, then `basis`
("based on: …") and `test_design` ("how this could be checked: …").
`candidate_signals` (M2's raw signal list, not yet a hypothesis) and any
`recommended_confirmatory_tests` are demoted into a collapsed "possible
follow-ups, not validated hypotheses" expander below the cards — kept
visually subordinate so they can't be mistaken for the vetted hypothesis
list. **Business logic to preserve:** identical rendering whether or not a
downstream confirm phase ran — this tab is always the pure proposal view; a
verdict badge is never attached here (see Tab 4).

## M4 — intervention (verify, then optionally fix)

### `SurgeryAgent.operate` — verify *why* something fails

**Input:** one `hypothesis: Hypothesis` (from M3), `model: Model`,
`results: dict[str, Result]` (M1's per-analyzer results, for per-case signal
extraction), `data: CaseBatch`.

**Output:** `InterventionResult` —
`status: HypothesisStatus` (SUPPORTED / REFUTED / INCONCLUSIVE),
`fixed: bool` (intervention completely separates failing from passing
cases), `evidence` (supporting statistics), `new_data: CaseBatch | None`
(when SUPPORTED, cases **not** in the signal group — refined subset for the
next M1 cycle), `confidence_score` + `evidence_dimensions` (breakdown),
`experiment` (rich payload when the `ExperimentWriter` strategy ran:
generated files, stdout/stderr, blueprint, verdict).

Four strategies are tried in order (first match wins): caller-supplied
`verify_fn`, `analyzer_params` re-run, `ExperimentWriter` (when `judge` is
set), or passive label correlation.

### `FixAgent.propose_and_validate` — propose + validate repairs (post-loop)

**Input:** `model: Model`, `data: CaseBatch`,
`hypotheses: list[Hypothesis]` (the verified ones from M5),
`prior_attempts: list[FixValidation] | None` (carried over across tier
escalation).

**Output:** `FixOutcome` —
`routed` (which `FixTier` each hypothesis was routed to and why),
`attempted: list[FixValidation]` (every candidate tried), `best:
FixValidation | None` (the winning candidate, paired McNemar + e-value
validated against the unmodified baseline), `fixed: bool`,
`recommendation: dict | None` (e.g. `{"recommend_tier": "L3a", "reason": ...}`
when nothing validated — never auto-escalated), `ebh_survivors` (candidates
whose e-value survives e-BH correction across the tested family),
`repair_rounds` (feedback-driven propose→validate rounds actually run).

Tiers (`FixTier`, an input the caller bounds): L1 prompt rewrite, L2 scaffold
(pipeline around the unchanged model), L3a internals-read, L3b
internals-write, L4 parameter space (recorded; LoRA additionally executed
when `finetune_pool=` is given).

Each `FixValidation` in `attempted` can now also come back
`verdict="model_independent"` (a new tier alongside `fixed` / `partial` /
`unsafe` / `regressed` / `no_effect` / `not_executed`): an L2 candidate is
re-run under `frozen_model_control` — every bridged model call answered with
the case's recorded baseline output — and any FAILING case that comes out
right anyway was solved by the pipeline's own computation, not a repair of
the model. Those cases are excluded from the paired test
(`n_model_independent`) rather than counted as `fixed`.

**UI (Tab 5 — Fix):** greyed "not available" placeholder
(`_render_unavailable_panel`: title, what happened, how to get it) when no
`fix_report.json` sits next to the exploratory report — this pipeline phase
is genuinely optional and the dashboard never fakes a result for it. When
present: a "Surgery context — M5 confirmation" table (one row per tested
hypothesis: statement, M5 status, confidence, evidence grade, held-out
verdict) sourced from `InterventionResult`/`HypothesisTestResult` flattened
into `fix_report["m5_results"]`; then a deterministic narrative digest built
from `attempted` (winner 🏆 with tier/name/repaired/broke/e-value, which
candidates survived e-BH across the family, an L1-2-vs-L3 prompt-vs-internals
contrast when the data shows one); then the full `attempted` candidates
table (sorted by tier, e-value descending, winner marked); then
`recommendation` when nothing validated. **Business logic to preserve:**
"repaired"/"broke" are paired flips against the *unmodified baseline on the
same cases* — never present a raw pass-rate delta as if it were this; a
candidate only earns `fixed` when both its own McNemar+e-value rejects AND
it survives e-BH across every candidate tried in that run (best-of-N
correction) — `ebh_survivors` is the source of truth for the latter, not
just `reject` on its own row.

## M5 — `HypothesisTester` (statistical + protocol verification)

**Input:** `hypotheses: list[Hypothesis]` (from M3),
`stats_report: StatsAnalysisReport` (M2's report — supplies `raw_results`
for per-case signal extraction and any e-BH-corrected `stats_results`),
`data: CaseBatch` (must carry labels for fail-rate comparison),
`protocol: ExperimentProtocol | None` (consistency check target; `None` →
all hypotheses assumed consistent).

**Output:** `list[HypothesisTestResult]`, one per hypothesis, each with:
`status: HypothesisStatus`, `test_name` (e.g. `"fail_rate_comparison"`),
`effect_size` (signal-group minus control-group fail rate),
`is_consistent_with_protocol: bool`, `confidence` (geometric mean of
evidence gap / sample adequacy / control cleanliness),
`evidence_grade` (`"intervention"` causal > `"observational"`
correlational > `"none"`), `verdict` (NL one-liner), `evidence` (stats +
group sizes).

A hypothesis is `SUPPORTED` only when both the statistical test and the
protocol-consistency check hold — this is the gate `HypothesisTester.
stopping_criteria_met(results)` checks before the loop stops.

**UI (Tab 4 — Validation results):** greyed "not available" placeholder when
no `confirm_report.json` sits next to the exploratory report (same pattern
as Tab 5 — this phase re-tests M3's hypotheses on a held-out split the
explorer never touched, e.g. `test_hypotheses.py`, and is optional). When
present: 4 headline metrics (validate-split case count, FAIL count in that
split, signals adjudicated, held-out rejections), a caption naming the
adjudication method/alpha/split, a "signal recipes on the held-out split"
table (per-signal: status, held-out verdict "REJECT H0"/"not rejected", fail
rate flagged vs. unflagged, effect, CI, n), then one hypothesis card per
`hypothesis_verdicts` entry — the same card component Tab 3 uses, plus a
verdict badge (color-coded by verdict) and the judge's reasoning line.
**Business logic to preserve:** the caption explicitly says a REJECT *here*
is a real held-out verdict, unlike Tab 2's in-sample screen — thresholds
were frozen on the explore half before this split was ever touched, so this
tab is where "supported"/"rejected" language is finally allowed to appear
attached to a specific claim.

## Loop-level output

All three loops — `AutoDiagnoseLoop`, `VLDiagnoseLoop`, `AgenticDiagnoseLoop`
— return the **same class**: `AutoDiagnoseReport` (`VLDiagnoseReport` is a
back-compat alias for the identical class, so `isinstance` checks against
either name succeed for a report from any loop). Each loop populates the
subset of fields relevant to what it ran; fields the loop has no equivalent
concept for stay at their default (`None` / empty):

| Field | Populated by | Meaning |
|---|---|---|
| `cycles` | all | M1→M4/M5 cycles executed (or agentic decision steps). |
| `resolved: bool` | all | Diagnosis considered closed — M4 confirmed a fix (legacy loop), or `bool(verified_hypotheses)` (current/agentic loops). |
| `stopped_by: str \| None` | VL / agentic | Why the loop stopped: `"criteria_met"` / `"max_cycles"` / `"budget"` / `"no_hypotheses"` / `"no_probe_results"` / `"analysis_complete"`, or agentic's `"agent_stop"` / `"max_actions"` / `"time_budget"` / `"invalid_actions"`. `None` for the legacy loop. |
| `final_hypotheses: list[Hypothesis]` | all | All M3 proposals across every cycle. (`all_hypotheses` is a read-only alias, kept for existing callers.) |
| `verified_hypotheses: list[HypothesisTestResult]` | VL / agentic (M5) | SUPPORTED + protocol-consistent, highest confidence first — feed into `run_m4`. |
| `all_test_results: list[HypothesisTestResult]` | VL / agentic (M5) | All M5 test results across every cycle. |
| `final_results: dict[str, Result]` | legacy | Raw analyzer results from the last M1 probe. |
| `final_analysis: AnalysisReport \| None` | all | Last M2 report. Auto-populated from `final_stats_report` when only that was set (`StatsAnalysisReport` is an `AnalysisReport` subclass), so this field works regardless of which loop produced the report. |
| `final_stats_report: StatsAnalysisReport \| None` | VL / agentic | Last M2 report (same object as `final_analysis` when set). |
| `fix_proposal` / `fix_outcome` | post-loop | Populated by `run_m4` / `run_fix` after `run()` returns. |
| `store` | all | Accumulated results and hypotheses. |

See [RunContext](architecture.md#runcontext-single-owner-of-a-runs-output-directory)
for where each stage's artifacts land on disk when a run directory is
attached, and [run_log.jsonl schema](architecture.md#eval_agent-automated-diagnosis-pipeline)
for the structured event each stage emits per cycle.

---

## UI reference — building a viewer on this pipeline

There is already a working viewer for this exact data: `evalvitals dashboard`
(Streamlit, `evalvitals/analysis/dashboard_app.py`) and the upload/explore
web workbench (`evalvitals web`, same renderer — see
[m2_analysis.md](m2_analysis.md) and the `deco_hallu_explore` example's
[web upload workbench](../examples/m2_m3/deco_hallu_explore/README.md)).
Read this section as "what to reproduce" if you're building a new UI, and
the function names as where to go read the exact rendering logic.

### Requirements for the new UI

Two explicit requirements on top of what's documented below:

1. **Show the raw data exactly, not just a derived view of it.** The
   existing pattern for this is `_render_raw_data_browser` — it loads
   `records.json` verbatim (the tidy table M2 built, before any
   analysis/aggregation) into a searchable, scrollable table. Today that's a
   collapsed expander tucked inside Tab 2; **for the new UI this should be a
   first-class, easy-to-find view of the actual rows**, not a buried
   afterthought — a reader has to be able to go from "the analysis says X"
   to "here is the literal row that's about" without hunting.
2. **Show M1's results too, not just M2 through M5.** M1 produces
   `dict[str, Result]` — one entry per analyzer that ran, each carrying
   `findings` (light JSON: scores, flagged tokens, contingency tables, …)
   and `artifacts` (heavy: attention maps, heatmaps, embeddings). **This is
   the one stage the existing dashboard does not show at all** (see the M1
   section's UI note above — its raw output was judged "too low-level" and
   only a derived per-case table reaches Tab 1). That gap is explicitly
   in scope for the new UI: surface each analyzer's `findings` (a JSON/table
   view keyed by analyzer name is enough to start) and, where a `Result`
   provides one, its rendered artifact (e.g. the attention/spatial overlay
   PNGs described in [Result image overlays](architecture.md#result-image-overlays) —
   `RunContext`'s `figures/`/`artifacts/` subdirectories are where these
   already land on disk per run, see
   [RunContext](architecture.md#runcontext-single-owner-of-a-runs-output-directory)).

**M2, M3, and M5 already have a good reference — reuse their existing
Mode‑A panels rather than redesigning them:** `_render_standalone_analysis`
(Tab 2), `_render_standalone_hypotheses` (Tab 3), and `_render_holdout_panel`
(Tab 4) — see each stage's UI note above for exactly what they show and the
business logic to preserve (plain-language-first, strict
descriptive/confirmatory separation, held-out-verdict framing). M1 (raw
results — see above) and M4 (Fix, `_render_fix_panel`) don't carry the same
explicit endorsement, so treat their current panels as a starting point to
verify against the input/output described above, not a spec to copy blindly.

### Two rendering modes — pick the one that matches your data source

The existing dashboard's entry point branches on what it's pointed at
(`main()`, keyed off `session["kind"]`), and renders it one of two ways.
**If you're building the new UI, decide up front which of these two your
viewer is: a live-run inspector, or a finished-artifact viewer** — they read
different files and use different tab structures; don't try to merge them
into one layout.

**A — Explore-artifact mode (`render_explore_report`)** — reads persisted
JSON artifacts (`exploratory_report.json` + optional sibling
`confirm_report.json` / `fix_report.json`) with no dependency on a live
run's log. This is what the `deco_hallu_explore` web upload workbench uses
end to end (every uploaded `.zip` becomes one `explore` run, rendered
through this exact path) — **it is the reference to copy** if the new UI is
"upload/point at a result directory and view it." **One fixed five-tab
layout for every result**, stages the run didn't reach greyed out rather
than the tab disappearing (`EXPLORE_TAB_LABELS`):

| # | Tab | Fed by | Source function |
|---|---|---|---|
| 1 | Problem Setting | M1 (derived per-case table) + run metadata | `_render_problem_setting` |
| 2 | Exploratory Analysis | M2 (`exploratory_report.json`) | `_render_standalone_analysis` |
| 3 | Hypotheses | M3 (`hypotheses` in the same report) | `_render_standalone_hypotheses` |
| 4 | Validation results | M5 (`confirm_report.json`, optional) | `_render_holdout_panel` |
| 5 | Fix | M4 (`fix_report.json`, optional) | `_render_fix_panel` |

**B — Loop mode (`_render_loop_story`)** — reads a live/finished loop run's
`run_log.jsonl` directly (`session["kind"] == "loop"`), i.e. the actual
per-cycle event stream from `AutoDiagnoseLoop`/`VLDiagnoseLoop`/
`AgenticDiagnoseLoop`, not a compiled report. Tabs are **Problem Setting →
(Agent Trajectory, agentic runs only) → Analysis → Hypotheses** — M5/M4
results are joined *inline* into each hypothesis card
(`_hypotheses_with_outcomes` matches M3 statements to their M4/M5 test
results by text) rather than getting their own tabs, and a descriptive vs.
confirmatory banner (🔍 vs. ✅) sits inside the Analysis/Hypotheses panels
instead of being a separate tab boundary. Use this shape if the new UI needs
to show a run *while it's still going* or wants per-cycle granularity —
Mode A has no concept of "cycle," only a finished report.

Everything documented per-stage above (Tabs 1–5, business logic to
preserve) describes **Mode A** — it's the simpler, more reusable contract
(pure JSON artifacts, no log-parsing) and the one the reference UI
(`deco_hallu_explore`) is built on. Mode B is worth knowing exists so a
"live progress" view isn't accidentally built by reinventing Mode A's log
parsing from scratch — go read `_render_loop_story` and its helpers
directly if that's the one you need.

### Page layout

Above the tabs, a header band always shows an answer-first summary before
any stage-by-stage detail — but the two modes use different functions for
it, so copy the one matching your mode:
- **Mode A** (`_render_report_overview`): title (plain-language question), a
  one-line verdict sentence, an "analysis stage" label (how far this run
  got — explore-only / validated / fixed), a status pill
  (`finished`/`failed`), and a metrics row (`_render_overview_metrics`:
  cases, explore/confirm split sizes, candidate-signal count, chart count —
  whichever apply).
- **Mode B** (`_render_hero_band`): a run-kind kicker (Analysis Phase /
  Diagnostic Loop Run / Agentic Diagnosis Run), a verdict pill + one-line
  conclusion, the protocol description being investigated (if any), and a
  KPI tile row (stages run, hypotheses proposed/verified, case counts,
  duration, and — for agentic runs — actions taken vs. budget).

Both modes put a sidebar alongside the tabs listing every other run/result
found in the attached directories, so the reader can switch between them
without losing their place.

### The "not available" pattern — the most important convention to copy

This is a **Mode A** convention (Mode B has no Tabs 4/5 to begin with — see
above). Tabs 4 and 5 are **genuinely optional** — a run may stop at M3
(proposal only) and never reach M5/M4. Rather than hiding the tab, the
dashboard always renders all five and shows a greyed placeholder
(`_render_unavailable_panel`) for a stage the run didn't reach: a title
("⚪ Validation results — not available for this run"), one line on *what
happened* ("this run stopped at M3: hypotheses were proposed but not
re-tested on held-out data"), and one line on *how to get it* (which command
produces the missing artifact and where it needs to land on disk). This is
why the reader never has to relearn the page between a quick M2/M3 look and
a full M1→M5→fix run — replicate the fixed-tabs-plus-placeholder shape
rather than a tab list that changes with what the run reached.

### Cross-cutting conventions worth copying

- **Plain language first, technical detail second.** Every headline
  (question, takeaway, hypothesis) prefers a `plain_*` field and shows the
  precise technical wording only as a secondary line, and only when it
  actually differs from the plain one. This is a checked invariant upstream
  (host-side jargon checker on M2/M3 output — see
  [m2_analysis.md](m2_analysis.md)), not just a UI nicety; a new viewer can
  rely on `plain_title`/`plain_statement`/`plain_question` being genuinely
  jargon-free rather than re-deriving a summary itself.
- **Descriptive vs. confirmatory framing is never blurred, in either mode.**
  Mode A keeps it apart by tab: M2 (Tab 2, may include in-sample `reject`
  flags) never uses "supported"/"rejected" language; that's reserved for M5
  (Tab 4, held-out). Mode B keeps it apart in-panel instead, with an explicit
  🔍-descriptive vs. ✅-confirmatory banner switched by whether a test phase
  has actually run for that hypothesis (`_story_is_descriptive`). This is the
  single most load-bearing convention in the existing UI — getting it wrong
  makes an exploratory finding read as a validated one.
- **Referenced-but-missing artifacts say so explicitly** rather than
  silently dropping the reference (a takeaway naming a chart that isn't in
  `report["charts"]` renders a visible "referenced evidence not found"
  notice). Artifacts the report produced but nothing referenced ("orphans")
  still render, in their own section, so nothing generated is ever hidden.
- **Paired-comparison numbers are always framed against the unmodified
  baseline**, never as a bare rate — "repaired N / broke N" on the fix
  panel, "fail rate flagged vs. unflagged" on the validation panel.
- Stage completeness for the *current* run is visualized as a small chip
  strip (`_render_stage_map`, shared by both modes) — every M1–M5 stage
  listed, the ones this run actually reached highlighted — so a reader gets
  pipeline-position context before reading any specific stage's content.
