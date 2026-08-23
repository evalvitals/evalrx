# Stage Input/Output Reference

A single reference for what goes into and comes out of each stage of the
M1→M5 diagnosis pipeline. [Architecture](architecture.md#eval_agent-automated-diagnosis-pipeline)
covers *why* the pipeline is shaped this way and how the three loops
(`AutoDiagnoseLoop`, `VLDiagnoseLoop`, `AgenticDiagnoseLoop`) orchestrate these
same stages differently; [Exploratory Analysis (M2/M3)](m2_analysis.md) and
[Intervention & Verification (M4/M5)](intervention.md) cover usage. This page
covers the shape of the data crossing each boundary, and — since that shape
is exactly what a viewer has to render — [what to put on screen for it](#ui-reference-building-a-viewer-on-this-pipeline).
Building a new UI on this pipeline? Read the stage you're rendering below,
then [UI Reference](#ui-reference-building-a-viewer-on-this-pipeline) at the
bottom for the static report's page layout, tab-to-stage mapping, and
rendering conventions (source: `evalvitals/reporting/html_report.py`).
For an implementation hand-off, start with
[Frontend implementation contract](#frontend-implementation-contract): it
defines the files to load, event fields, joins, state derivation, null/error
semantics, and TypeScript-friendly shapes that are not visible from the Python
method signatures alone.

## Pipeline data flow

```text
CaseBatch (labeled FailureCases)
   │
   ▼
M1  ProbeAgent.probe(model, data)              → dict[str, Result]
   │
   ├─(optional) ExploratoryAnalysisAgent.explore_records(per-case table)
   │            → ExploreContext for M3 + explore/ files for the HTML report
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
then probes. See [m2_analysis.md](m2_analysis.md#probe-search-hierarchical-mcts-failure-discovery-vlm).

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

**UI:** not rendered by the current report — it's a data-generation step
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

Analyzer selection ranks by **modality slot**, not by a model-kind label.
`StrategyProbe.routed_slots(model, data)` intersects what the model declares
with what the batch actually fills, and `select()` concatenates the priority
list for each slot in play (agent → video → audio → image → text). LLM / VLM /
ALM / AVLM are four subsets of `{text, image, audio, video}`, so adding a
modality adds one list rather than multiplying a per-kind table.

The batch decides, not the model: an omni model evaluated on an audio benchmark
declares image too, and ranking on the declaration put image analyzers at the
top of an audio run. When the batch fills no media slot at all the model's
declaration is the fallback — no evidence, rather than evidence of absence — and
`AnalyzerSelection` records `model_modalities`, `probed_modalities` and
`routed_on` separately so a reader can see which path was taken.

`detect_kind()` still returns a coarse `ModelKind` (now including `ALM` and
`AVLM`) for display and for the `priority_override` escape hatch; it does not
decide routing. `Analyzer.requires_modalities` gates analyzers whose slot the
batch never fills — the modality counterpart of `requires_trajectories` — and an
LLM judge may pick directly from the protocol description;
`WhiteboxProbeGenerator`/`ProbeGenerator` write a bespoke probe when no standard
analyzer covers the failure mode.

**UI:** In the static report, M1's raw output (`dict[str, Result]`) is
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
[UI reference](#ui-reference-building-a-viewer-on-this-pipeline) for the
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
is genuinely optional and the report never fakes a result for it. When
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

## Frontend implementation contract

**Read `<run>/contract/` first.** Every stage validates its output against
`evalvitals/contract/` on the way out and writes one JSON per stage —
`c0.m1.json`, `c0.m2.json`, `c0.m3.json`, `c0.m5.json`, `m4_surgery.json`,
`m4_fix.json`. That directory is the typed source of truth, and TypeScript
declarations for all of it are generated into `docs/contract/contract.d.ts`:

```ts
import type { ProbeOutput, StatsReportWire } from "./contract/contract";

const m1: ProbeOutput = await (await fetch("contract/c0.m1.json")).json();
m1.selection.routed_on;      // ["audio", "text"] — what the run was actually about
m1.results["mm_shap"];       // keyed by analyzer name
```

Regenerate with `python -m evalvitals.contract.export --out docs/contract`;
CI runs the same command with `--check` so the artifacts cannot go stale.
A stage that failed validation writes `<stage>.invalid.json` carrying the error
instead — visibly broken rather than quietly missing.

Two things the contract will not do for you, both deliberate:

* **Unknown fields must be ignored, never rejected.** Adding a field is
  additive and does not bump `schema_version`; only a change of *meaning* does.
* **Absent, null, empty, zero and false are five different things.** Every
  optional field is explicitly nullable so a reader can tell them apart. Never
  collapse them with a truthiness fallback — `ungrounded_rate_audio: null`
  means no case carried audio, and `0.0` means every case that did stayed
  grounded.

The rest of this section describes the raw run directory, which remains
readable and is what a run produced before contract emission existed offers.

### 1. Choose a source of truth

A normal persisted loop run has this shape (some folders are optional):

```text
<run>/
├── manifest.json
├── run_log.jsonl
├── contract/                 ← typed, validated; prefer this
│   ├── index.json
│   ├── c0.m1.json … c0.m5.json
│   └── m4_surgery.json, m4_fix.json
├── report/
│   ├── summary.json
│   ├── summary.md
│   ├── hypotheses.json
│   ├── m5_results.json
│   └── discovery_cases.json
├── artifacts/
│   ├── c0_<analyzer>.result.json
│   ├── c0_<artifact>.npy|json|png
│   └── c0_m2_<large-field>.json
├── prompts/
│   └── c0_m{1,2,3}_*.prompt.txt|response.txt
├── explore/
│   ├── exploratory_report.json
│   ├── tables/
│   └── figures/
└── fixes/
    ├── outcome.md
    └── <trial>/
        ├── record.md
        ├── result.json
        └── workspace/
```

Use one of these two contracts; do not silently merge conflicting values from
both:

| UI use case | Primary source | Secondary source |
|---|---|---|
| Live progress / per-cycle story | append-only `run_log.jsonl` | linked artifacts as they appear |
| Finished run / stable report | `manifest.json` + `report/` + `artifacts/` | `run_log.jsonl` for provenance and detail |
| Standalone Explore upload | `exploratory_report.json` and optional sibling `confirm_report.json` / `fix_report.json` | the uploaded records |

For a finished run, `manifest.json` is the file inventory. Its current shape is:

```json
{
  "run_id": "auto_fix_v12",
  "root": "/app/example/outputs/auto_fix_v12",
  "generated_at": "2026-08-18T00:00:00+00:00",
  "config": {
    "benchmark": "spatial457",
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "n_cases": 512,
    "confirm_split": 0.5,
    "fix_tier": "L3a",
    "allow_codegen": true
  },
  "files": {
    "report": ["report/summary.json", "report/hypotheses.json"],
    "artifacts": ["artifacts/c0_attention.result.json"],
    "prompts": ["prompts/c0_m2_analysis.prompt.txt"],
    "fixes": ["fixes/outcome.md"],
    "other": ["run_log.jsonl"]
  }
}
```

`config` is intentionally extensible: render recognized values as summary
chips and preserve the remainder in a raw configuration view.

#### Artifact path resolution

Paths recorded by a container may be absolute inside that container, while the
UI is serving an extracted/copy-mounted run elsewhere. Therefore:

1. Treat the directory containing `manifest.json` as `runRoot`.
2. Resolve every relative path against `runRoot`.
3. Do **not** use `manifest.root` as the server filesystem root; it is provenance
   and may say `/app/example/...` even when the UI sees another path.
4. For an absolute path in an event, first strip the recorded `manifest.root`
   prefix and resolve the suffix below `runRoot`; otherwise match its longest
   unambiguous suffix against `manifest.files`.
5. If no in-run match exists, show “artifact unavailable” and the recorded path
   as text. Never let a client-provided path escape `runRoot`.
6. Serve artifacts through a backend route such as
   `/api/runs/:runId/artifacts/:relativePath`; do not expose host paths in URLs.

### 2. Shared serialized objects

All JSON objects are additive: a producer may add fields without increasing
`schema_version`. Frontend decoders should validate the fields they consume and
retain unknown fields for the raw JSON inspector.

#### Failure case

`FailureCase.to_dict()` serializes one case as:

```json
{
  "id": "sample-0042",
  "inputs": {
    "prompt": "Which square is closest to the red circle?",
    "image": "images/0042.png",
    "audio": null,
    "video": null
  },
  "expected": "B",
  "observed": "C",
  "trajectory": null,
  "label": "fail",
  "tags": ["spatial-reasoning"],
  "provenance": {"source": "dataset", "metadata": {}},
  "metadata": {"split": "explore"}
}
```

Field rules:

| Field | Type | UI meaning |
|---|---|---|
| `id` | `string` | stable sample join key; display and search it |
| `inputs.prompt` | `string` | always present |
| `inputs.image/audio/video` | `unknown \| null` | normally a path/URL; an in-memory rich object may degrade to a descriptor such as `<image 640x480>` |
| `expected`, `observed` | `unknown \| null` | render strings directly and objects in a structured/raw view |
| `label` | `"pass" \| "fail" \| "unknown"` | outcome, not a statistical verdict |
| `tags` | `string[]` | filter chips; order is not semantically meaningful |
| `provenance.source` | `"human" \| "dataset" \| "agent"` | where the case originated |
| `metadata` | `Record<string, unknown>` | dataset-specific columns; never assume a fixed schema |
| `trajectory` | `Trajectory \| null` | agent runs only; steps contain `idx`, `role`, `content`, tool call/observation, span metrics, and optional error annotations |

Media descriptors are not media bytes. Only offer a preview when the value can
be resolved to an allowed file/URL; otherwise display the descriptor.

#### M1 result artifact

Each `artifacts/c<cycle>_<analyzer>.result.json` contains the lightweight,
complete serialized `Result`:

```json
{
  "analyzer": "attention",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "findings": {
    "summary_score": 0.41,
    "per_case": [
      {"sample_id": "sample-0042", "attention_entropy": 0.73}
    ]
  },
  "metadata": {"layer": 20},
  "n_cases": 512
}
```

`findings` is analyzer-specific. The UI should support scalar cards, a dynamic
key/value view, and a virtualized table for `findings.per_case`. Join a per-case
row to raw data using `sample_id` ↔ `FailureCase.id`. Heavy arrays/images are
not embedded here; locate them through the `probe.artifact_paths` map or the
manifest inventory.

#### Hypothesis

The full in-memory hypothesis has the following logical shape:

```ts
type HypothesisStatus =
  | "proposed" | "testing" | "supported" | "refuted" | "inconclusive";

interface Hypothesis {
  id?: string;
  statement: string;
  target_model?: string;
  predicted_failure_mode?: string;
  failure_mode?: string; // run-log alias of predicted_failure_mode
  test_design?: string;
  expected_association?: "higher_on_failures" | "lower_on_failures" | string;
  status?: HypothesisStatus | null;
  parent_id?: string | null;
  evidence?: string[];
  metadata?: Record<string, unknown>;
}
```

Not every persisted view contains every field. In particular, a `diagnosis`
event emits only `statement`, `failure_mode`, `status`, and `test_design`.
Normalize `failure_mode` and `predicted_failure_mode` into one view-model field,
but keep the original JSON untouched.

### 3. `run_log.jsonl` envelope and event contract

Each complete line is one independent JSON object. Every current event has:

```ts
interface RunEventBase {
  event: string;
  schema_version: number; // currently 3; branch on it, do not hard-fail on newer
  ts: string;             // ISO-8601 UTC
  trace_id: string;       // run-level correlation id
  span_id?: string;       // e.g. c0.m1, c0.m2, c0.m3, c0.m5, fix
  cycle?: number;         // normal cycles start at 0; post-loop fix uses -1
  [extra: string]: unknown;
}
```

Events are ordered by file position. `ts` is for display and cross-service
correlation, not for re-sorting lines with equal or skewed timestamps. While
tailing a live file, retain an incomplete final line and retry it after the next
chunk; an incomplete line is not a run error.

| Event | Stage | Required/important payload | UI interpretation |
|---|---|---|---|
| `run_start` | run | model/judge/config, `n_cases`, protocol, budgets, version/git/data fingerprint when available | create run header; missing optional provenance is “unknown,” not failure |
| `probe` | M1 | `cycle`, `analyzers`, `findings`, `result_paths`, `artifact_paths`; optional `selected_analyzers`, rationale, `failed_analyzers`, `judge_io`, duration | one analyzer card per selected analyzer; a name in `failed_analyzers` is a local analyzer failure |
| `explore` | descriptive side path | `cycle`, `ok`, counts, observations, caveats, figures; optional error/report path | `ok: false` fails this optional step, but does not by itself fail the diagnosis loop |
| `analysis` | M2 | `cycle`, severity, findings, narrative, `descriptive_only`; optional stats outputs, conclusion, figures, `llm_fallback_reason`, `judge_io` | label descriptive and confirmatory analysis explicitly; a fallback can still be a successful M2 |
| `diagnosis` | M3 | `cycle`, model, `n_hypotheses`, `hypotheses`, raw output; optional referenced charts/context flags, `judge_io` | zero parsed hypotheses is a completed empty/abstained M3 unless an explicit error exists elsewhere |
| `surgery` | M4 or M5 | `cycle`, `module`, hypothesis text, failure mode, status, `fixed`, confidence/evidence, refocused-case count | route by `module`; do not infer M4 vs M5 from the event name alone |
| `experiment` | M4 | `cycle`, `module`, hypothesis, status/fixed; optional provider, metrics, exit/timing, code/output/workspace paths, record | generated verification execution detail; failure of one experiment need not fail the whole run |
| `fix` | post-loop Fix | `cycle: -1`, `max_tier`, selection/final attempts, best, fixed, recommendation/refine signal, records | distinguish EXPLORE selection from held-out FINAL confirmation; see below |
| `loop_end` | diagnosis loop | cycles, resolved/stopped reason, final and verified hypotheses, tokens/timings | diagnosis loop ended; **not necessarily the last event in the run** |
| `agent_decision` | agentic loop | `step`, action/params/rationale, validity/repair/fallback, judge I/O | trajectory node; `valid: false` means host fallback, not automatically run failure |
| `agent_tool` | agentic loop | `step`, tool, `ok`, summary/error, duration | dispatch result; stage-specific event remains the source for stage payload |
| `tool_codegen` | support | cycle/module/tool/need/source/`ok`, error/code paths | one tool-generation attempt, not a stage verdict |
| `tool_registry` | support | cycle/module/tool inventory | diagnostic/debug inventory only |

Large M2 fields (`stats_results`, `stats_tool_results`, `stats_plan`, or
`corrected_rejections`) are inline until they exceed 4096 serialized bytes.
Then their value becomes a pointer:

```json
{"path": "artifacts/c0_m2_stats_results.json", "n_items": 37, "bytes": 18942}
```

Detect an externalized value by shape (`path` plus `bytes`), lazy-load it, and
keep the count visible while loading. Do not treat the pointer itself as one
statistics result.

### 4. Stage and run state derivation

Use these UI states; a single generic “N/A” loses information the user needs:

| State | Meaning | Typical display |
|---|---|---|
| `not_started` | no evidence that the stage was scheduled | muted step |
| `running` | prerequisite exists and the stage has started, but no terminal stage event yet | spinner + elapsed time |
| `succeeded` | terminal event has a usable non-empty result | green/complete |
| `empty` | stage completed correctly with zero items | neutral “No hypotheses/findings produced” |
| `abstained` | evaluator intentionally could not make a supported choice | neutral warning with reason |
| `partial` | some analyzers/candidates succeeded and some failed | amber + per-item errors |
| `failed` | explicit stage-level error/no usable result | red + error and logs |
| `skipped` | run configuration or stopping rule intentionally omitted the stage | grey + reason |
| `unavailable` | persisted run references data that cannot be loaded | grey + recorded path/recovery hint |

Recommended derivation rules:

1. Partition stage events by `(trace_id, cycle, logicalStage)`. Map a `surgery`
   event using its `module`; treat `fix` as the post-loop stage even though its
   cycle is `-1`.
2. The first matching terminal event completes that item; later matching events
   append/replace item detail rather than rewinding an already completed stage.
3. `diagnosis.n_hypotheses === 0` means M3 completed with an empty result. Do not
   show a network/runtime error unless there is explicit error evidence.
4. `probe.failed_analyzers` with at least one successful analyzer means partial;
   all selected analyzers failed means failed.
5. `explore.ok === false` marks only Explore failed. M2/M3 may continue.
6. `analysis.llm_fallback_reason` means the judge path failed and a fallback
   path ran. Show the fallback badge while deriving M2 success from the actual
   analysis payload.
7. A hypothesis status of `inconclusive` is a valid abstention, not an exception.
8. `loop_end` terminates M1→M5 orchestration, but M4 verification and Fix can be
   logged after it. Do not mark the entire run immutable merely because
   `loop_end` appeared. A live transport/process terminal signal or a finalized
   manifest is the run-level completion signal.
9. Missing events in an older/in-progress run mean “not observed”; do not invent
   a failure. Use `schema_version` and run configuration to decide skipped vs.
   not started.

A simple per-cycle reducer can look like:

```ts
function reduceEvent(state: RunView, e: RunEventBase): RunView {
  state.events.push(e); // preserve source order for raw/audit view
  if (e.event === "probe") updateM1(state, e);
  if (e.event === "explore") updateExplore(state, e);
  if (e.event === "analysis") updateM2(state, e);
  if (e.event === "diagnosis") updateM3(state, e);
  if (e.event === "surgery" && e.module === "m4") updateM4(state, e);
  if (e.event === "surgery" && e.module === "m5") updateM5(state, e);
  if (e.event === "fix") updateFix(state, e);
  if (e.event === "loop_end") state.diagnosisLoopEnded = true;
  return state;
}
```

### 5. Join and identity rules

There is no single ID present in every persisted representation, so use this
precedence and expose ambiguous joins instead of hiding them:

| Objects to join | Preferred key | Fallback |
|---|---|---|
| run events | `trace_id` | enclosing run directory; never merge two directories merely because model/config match |
| analyzer per-case row ↔ raw case | `sample_id` ↔ `FailureCase.id` | no fuzzy matching |
| full hypothesis objects | hypothesis `id` | `(cycle, normalized statement)` |
| M3 ↔ M4/M5 run-log records | `(cycle, exact statement)` | normalized whitespace/case only; mark collisions ambiguous |
| fix candidate across selection/final | `trial_root` when present | `(tier, name, list ordinal)`; name alone is not unique across repair rounds |
| agent decision ↔ dispatch result | `step` | file adjacency within the same trace |
| stage artifact ↔ event | exact relative path | unique manifest suffix match |

Do not slugify hypothesis text and assume it is globally unique. If two
hypotheses in one cycle have the same normalized statement, retain both as
separate cards and show the shared outcome as an ambiguous association until a
stable ID is available.

### 6. Fix event: selection is not confirmation

The Fix UI must keep two datasets separate:

- `selection_attempted`: exploratory candidate trials used to choose a repair.
- `attempted`: final/holdout confirmation attempts. These are the attempts that
  may justify the top-level `fixed` claim.

The full useful attempt shape is:

```ts
type FixVerdict =
  | "fixed" | "partial" | "unsafe" | "regressed"
  | "no_effect" | "not_executed" | "model_independent";

interface FixAttempt {
  tier: string;
  name: string;
  kind?: string;
  source?: string;
  payload?: Record<string, unknown>;
  trial_root?: string | null;
  n_pairs: number;
  n_baseline_correct: number;
  n_candidate_correct: number;
  n_fixed: number;
  n_broken: number;
  fixed_cases: string[];
  broken_cases: string[];
  effect: number | null;
  reject: boolean;
  fixed: boolean;
  n_applicable?: number;
  coverage?: number | null;
  n_unstable?: number;
  n_model_independent?: number;
  e_value?: number | null;
  verdict: FixVerdict;
  summary?: string;
}

interface FixEvent extends RunEventBase {
  event: "fix";
  cycle: -1;
  max_tier: string;
  routed: unknown[];
  selection_attempted: Array<Record<string, unknown>>;
  selected_on_explore: string | null;
  attempted: FixAttempt[];
  best: FixAttempt | null;
  fixed: boolean;
  recommendation: Record<string, unknown> | null;
  refine_signal: Record<string, unknown> | null;
  repair_rounds: number;
  ebh_survivors: string[];
  record?: string;
}
```

Render paired changes, not candidate accuracy alone: “fixed 18 / broke 1,”
coverage, effect, rejection/e-value, and verdict. Useful outcome distinctions:

| Payload | Correct UI state |
|---|---|
| `fixed: true`, `best != null` | confirmed fix; show best and final evidence |
| `attempted.length > 0`, no fixed attempt | completed, no validated fix; show all failures and recommendation |
| `selection_attempted.length > 0`, `selected_on_explore == null`, `attempted: []` | abstained during selection; confirmation was not run |
| selected candidate exists, `attempted: []` | selected but unconfirmed/unfinished; never label fixed |
| `verdict: "unsafe"` or `"regressed"` | prominent safety failure, including `broken_cases` |
| `verdict: "model_independent"` | apparent gain was not attributable to the target model |

`exec_error` exists in the in-memory validation type but is not currently
included in `FixOutcome.to_dict()`. The UI must not depend on that field; use the
attempt `summary`, `verdict`, trial record, and captured execution artifacts.

### 7. Missing, null, empty, zero, and false

These values are not interchangeable:

| JSON value | Meaning |
|---|---|
| field absent | producer/version did not emit it, or it was optional; “unknown/not recorded” |
| `null` | producer explicitly knows no value exists, e.g. no best fix or no refocused batch |
| `[]` / `{}` | valid collection with zero entries; often a completed empty result |
| `""` | empty narrative/reason; do not substitute an invented explanation |
| `0` | measured/count value of zero; display it |
| `false` | explicit negative boolean; display it and do not fall back with `value || default` |

In TypeScript, prefer nullish fallback (`value ?? default`) over truthiness
fallback when zero/false are valid. Preserve the original distinction in the
raw view even if the summary UI groups some cases together.

### 8. Rendering contract by stage

Every stage page/card should have three layers: an answer-first summary, a
structured detail view, and the unmodified raw source/artifact link.

| Stage | Summary | Detail/drill-down | Empty/error treatment |
|---|---|---|---|
| Problem / input | model, benchmark/protocol, case/label counts, split | searchable raw cases; prompt/media/expected/observed; trajectory steps | unresolved media is unavailable, not a missing case |
| M1 | analyzers run/succeeded/failed; top scalar findings | analyzer-specific JSON/table; per-case rows; PNG overlay/heatmap; `.npy` download | selected analyzer with error gets its own failed card |
| Explore | observations/candidate-signal/chart counts; descriptive badge | charts/tables/caveats and source rows | show explicit explorer error while allowing later stages |
| M2 | severity, conclusion, descriptive/confirmatory badge | findings/evidence chain, stats plan/results, corrected decisions, figures, judge-I/O audit link | no findings can be a successful empty result; fallback reason is a warning |
| M3 | hypothesis count and short statements | failure mode, test design, parent/evidence where available, cited charts | zero hypotheses is neutral empty/abstained, not red failure |
| M4 verify | mechanism status, confidence, whether intervention changed outcome | evidence dimensions, experiment metrics, generated files/stdout/stderr/workspace record | inconclusive is valid; execution error belongs to the experiment card |
| M5 | supported/refuted/inconclusive count | test, effect, confidence, protocol consistency, evidence grade and evidence | descriptive M2 rejection must never appear as M5 support |
| Fix | confirmed status and best candidate | separate EXPLORE selection and FINAL confirmation tables; fixed/broken case links; code and record | distinguish no validated fix, abstention, unsafe, and unfinished |

For arbitrary dictionaries, render a bounded structured view first and place
the complete JSON in a collapsible raw panel. Long text generated by a model is
untrusted content: escape HTML/Markdown by default, and never execute generated
code or load arbitrary local paths in the browser.

### 9. Live loading, scale, and failures

- Parse JSONL incrementally and deduplicate by file byte offset, not by event
  content; repeated-looking candidate events can be legitimate.
- Debounce visual updates, but append events in source order. A 250–500 ms UI
  refresh is enough for normal stage durations.
- Virtualize `per_case`/raw-case tables and load heavy JSON, `.npy`, images,
  prompts, responses, and workspaces on demand.
- Do not request an artifact before its event line is complete. If a referenced
  file is still being written, show loading and retry with bounded backoff.
- Distinguish HTTP/read failure from JSON parse failure and from a valid empty
  payload. Include the relative path and retry action in the error panel.
- Preserve the last valid UI state when a live read fails; add a stale/disconnected
  badge rather than clearing completed stages.
- Use explicit size limits and confirmation before rendering very large text or
  arrays. `.npy` should normally be summarized/rendered server-side, not parsed
  by the browser.
- Redact credentials and authorization headers from raw config, prompts, tool
  calls, stdout/stderr, and metadata at the server boundary.

### 10. UI acceptance checklist

A UI hand-off is complete when the implementation can demonstrate all of the
following against both a live and a persisted run:

- All M1–M5 stages remain visible even when later stages were skipped.
- Raw cases and raw M1 analyzer output are discoverable without developer tools.
- An M1 per-case row opens the exact raw case via `sample_id`.
- Descriptive M2/Explore evidence is visually distinct from confirmatory M5.
- An empty M3, failed Explore, inconclusive M5, and missing artifact render as
  four different states.
- An externalized M2 payload lazy-loads from its `{path, n_items, bytes}` pointer.
- A `loop_end` followed by M4/Fix updates the existing run instead of creating a
  second run or freezing the first one.
- Fix selection and final confirmation appear in separate sections, and the UI
  never claims “fixed” from selection evidence alone.
- Container-absolute artifact paths resolve inside the selected run root and
  cannot traverse outside it.
- Unknown additive event/config fields do not crash parsing and remain inspectable
  in the raw view.

---

## UI reference — building a viewer on this pipeline

There is already a working viewer for this exact data: `evalvitals report`
generates `report.html`, and `evalvitals serve <run-dir>` opens it locally.
The source of truth is `evalvitals/reporting/html_report.py`. Read this
section as the contract the static report must preserve.

### Requirements for the new UI

Two explicit requirements on top of what's documented below:

1. **Show the raw data exactly, not just a derived view of it.** The
   report exposes `records.json`/per-case artifacts through the Case Studio
   and agent/artifact sections. A reader must be able to go from "the analysis
   says X" to the literal row or case without hunting.
2. **Show M1's results too, not just M2 through M5.** M1 produces
   `dict[str, Result]` — one entry per analyzer that ran, each carrying
   `findings` (light JSON: scores, flagged tokens, contingency tables, …)
   and `artifacts` (heavy: attention maps, heatmaps, embeddings). The
   static report surfaces each analyzer pass in its M1 view. Keep `findings`
   readable as JSON/table and expose rendered
   artifacts where available (e.g. the attention/spatial overlay PNGs
   described in [Result image overlays](architecture.md#result-image-overlays));
   `RunContext`'s `figures/`/`artifacts/` subdirectories are where they land
   on disk per run, see
   [RunContext](architecture.md#runcontext-single-owner-of-a-runs-output-directory)).

**The current reference UI is one compiled HTML artifact.** Exploratory and
full diagnostic runs use the same renderer and stage order. Missing optional
M5/M4 artifacts retain their place and explain that the stage was not recorded;
the tab list never changes with run completeness.

### Page layout

The common header identifies the run, then a tab rail shows every M1–M5 state
in the action order M1 → M2 → M3 → M5 → M4. The rail is orientation, not a
replacement for the stage views: a reader can move from a count/status to the
full evidence, raw event, or artifact without switching applications.

### The "not available" pattern — the most important convention to copy

M5 and M4 are genuinely optional — a run may stop at M3 and never reach
validation or repair. Keep their tabs visible and render a clear stage-local
empty state instead of deleting navigation. This lets a reader compare a
quick M2/M3 analysis with a full M1→M5→M4 run without relearning the layout.

### Cross-cutting conventions worth copying

- **Plain language first, technical detail second.** Every headline
  (question, takeaway, hypothesis) prefers a `plain_*` field and shows the
  precise technical wording only as a secondary line, and only when it
  actually differs from the plain one. This is a checked invariant upstream
  (host-side jargon checker on M2/M3 output — see
  [m2_analysis.md](m2_analysis.md)), not just a UI nicety; a new viewer can
  rely on `plain_title`/`plain_statement`/`plain_question` being genuinely
  jargon-free rather than re-deriving a summary itself.
- **Descriptive vs. confirmatory framing is never blurred.** M2 may show
  exploratory/in-sample signals but never calls them supported; that wording
  is reserved for M5 validation. This is the single most load-bearing UI
  convention — getting it wrong makes an exploratory finding read as a
  validated one.
- **Referenced-but-missing artifacts say so explicitly** rather than
  silently dropping the reference (a takeaway naming a chart that isn't in
  `report["charts"]` renders a visible "referenced evidence not found"
  notice). Artifacts the report produced but nothing referenced ("orphans")
  still render, in their own section, so nothing generated is ever hidden.
- **Paired-comparison numbers are always framed against the unmodified
  baseline**, never as a bare rate — "repaired N / broke N" on the fix
  panel, "fail rate flagged vs. unflagged" on the validation panel.
- Stage completeness for the *current* run is visualized by the shared stage
  rail (`_render_stage_rail`) — every M1–M5 stage is listed, with the stages
  actually reached highlighted, so a reader gets pipeline-position context
  before reading any specific stage's content.
