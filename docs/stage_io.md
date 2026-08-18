# Stage Input/Output Reference

A single reference for what goes into and comes out of each stage of the
M1→M5 diagnosis pipeline. [Architecture](architecture.md#eval_agent-automated-diagnosis-pipeline)
covers *why* the pipeline is shaped this way and how the three loops
(`AutoDiagnoseLoop`, `VLDiagnoseLoop`, `AgenticDiagnoseLoop`) orchestrate these
same stages differently; [Exploratory Analysis (M2/M3)](m2_analysis.md) and
[Intervention & Verification (M4/M5)](intervention.md) cover usage. This page
only covers the shape of the data crossing each boundary.

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
