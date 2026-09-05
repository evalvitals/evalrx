# M2 → M3: Exploratory Analysis & Hypothesis Generation

`evalrx explore` runs two stages over a results directory, no code required:

- **M2 — `ExploratoryAnalysisAgent`**: a local coding agent profiles your data
  and writes/runs one analysis script, producing takeaways, charts, and
  candidate signals. **Purely descriptive** — no support/unsupported verdict,
  no hypothesis testing.
- **M3 — `HypothesisAgent`**: reads the M2 report and proposes 1-3 falsifiable
  hypotheses that could explain the patterns found. **Proposal only** —
  generating a hypothesis is not the same as testing one; nothing here is
  confirmed or refuted.

Confirmatory testing (turning a candidate signal into a validated claim) is a
separate, loop-internal system (`StatsAnalysisAgent`) — see
[Architecture](architecture.md#eval_agent-automated-diagnosis-pipeline).
This page only covers the standalone M2/M3 workflow.

## Install

```bash
pip install -e .                  # core
```

## Quickstart

```bash
evalrx explore /path/to/results \
  -q "Which features distinguish incorrect cases from correct cases?" \
  --out evalrx_explore_output \
  --serve-report
```

`/path/to/results` is a single `.json`/`.jsonl` file or a directory tree;
EvalRX recursively samples records across files. M3 hypotheses are
generated automatically after a successful M2 pass — pass `--no-hypotheses`
to skip that.

Other example questions:

```text
Compare accuracy across model directories.
Does tool usage correlate with failures?
What predicts yield / latency / cost? (any continuous outcome)
```

### Arbitrary folders, any outcome shape

`explore` does not pre-parse your data. The raw file or directory is handed to
the coding agent as-is (`--path` may be one file or many, in any JSON shape —
a flat list, JSONL, or per-file metadata wrapping a nested list of records
under a key like `cases`/`results`/`rows`); the agent's own generated code
loads and organizes it into a tidy table, then classifies the outcome column
(if any) as `binary`, `categorical`, `continuous`, or `none` and adapts the
analysis and chart battery accordingly — you are not limited to pass/fail
logs, and no host-side loader needs to know your data's shape in advance.
Pass `--outcome-col <name>` to point the agent at an explicit target column
(e.g. `yield_pct`) instead of relying on its own name-based detection.

## Output layout

```text
evalrx_explore_output/
  exploratory_report.json   # takeaways, observations, candidate signals,
                             # charts, tables, and M3 hypotheses
  analysis.py                # the generated code that was actually run,
                              # including its own data-loading step
  records.json                # the tidy table the agent built and analyzed
  figures/  tables/           # rendered charts + tabular artifacts
```

Each M3 hypothesis in `exploratory_report.json["hypotheses"]` has:

```json
{"statement": "...", "plain_statement": "the same claim in one jargon-free, everyday sentence",
 "basis": "which M2 takeaway(s) this is grounded in",
 "test_design": "what evidence would confirm or refute it"}
```

Both M2 takeaways and M3 hypotheses carry a plain-language headline
(`takeaways[i].plain_title`, `hypotheses[i].plain_statement`) alongside the
precise technical line (`title`/`statement`) — the report shows the plain
version first and the technical wording underneath as a secondary detail.
This is host-checked (no stats jargon/acronyms/symbols); a violation triggers
one bounded rewrite before the report is returned.

## Report

```bash
evalrx serve evalrx_explore_output --port 8501
```

Reads the saved artifacts (no re-run) in a portable static HTML report across **Problem Setting**,
**Exploratory Analysis** (M2 charts/takeaways), and **Hypotheses** (M3,
proposal-only — no verdict language).

## Python API

The one-call entry point runs M2 + host adjudication + M3 in a single step,
over a path or in-memory records:

```python
import evalrx

result = evalrx.explore(
    "/path/to/results",                 # or a list[dict] of in-memory records
    question="What predicts failure?",
    provider="claude_code",             # or antigravity/codex/...
    out="evalrx_explore_output",    # omit to skip persisting artifacts
)
print(result.ok, result.hypotheses)
```

`evalrx.explore` is a lazy re-export of `evalrx.analysis.explore`;
`out=None` (the default) keeps everything in memory — pass a directory to also
persist `exploratory_report.json`, rendered figures, and tables (the same
artifacts the `evalrx explore` CLI writes).

For direct control over each stage:

```python
from evalrx.analysis import ExploratoryAnalysisAgent, HypothesisAgent
from evalrx.agent_runtime import CliAgentConfig

cli_config = CliAgentConfig(provider="claude_code")  # or antigravity/codex/...

m2 = ExploratoryAnalysisAgent(cli_config=cli_config)
report = m2.explore_path("/path/to/results", question="What predicts failure?")

m3 = HypothesisAgent(cli_config=cli_config)
hypotheses = m3.propose(report.to_dict())
for h in hypotheses:
    print(h.statement, "—", h.test_design)
```

## Failure-Mode Clustering

`evalrx.analysis.cluster_failures` groups FAIL cases into interpretable
clusters — pattern discovery over the raw failing cases themselves, rather
than the per-signal EDA above. No required dependency: a pure-numpy fallback
(hashing vectorizer + cosine-greedy grouping) always works; install the
`[cluster]` extra (`scikit-learn`, `hdbscan`) for TF-IDF + density-based
clustering.

```python
from evalrx.analysis import cluster_failures

report = cluster_failures(records, min_cluster_size=3, max_clusters=8)
for cluster in report.clusters:
    print(cluster.name, cluster.size, cluster.top_terms)
```

*records* is a list of row dicts with an outcome column (`outcome_col`,
default `"label"`) and text/signal columns (auto-detected, or pass
`text_cols`/`signal_cols` explicitly). Pass `judge=` (any `Model` with
`Capability.GENERATE`) to have an LLM name/describe each cluster from its
exemplars instead of the deterministic top-terms naming. `report.method` tells
you which clustering backend actually ran (`"hdbscan"` / `"agglomerative"` /
`"cosine_greedy"` / `"single_cluster"`); `report.as_hypothesis_context()`
renders a compact section for feeding into M3 hypothesis generation — this is
exactly what `AgenticDiagnoseLoop`'s `cluster_failures` tool does
automatically (see [quickstart](quickstart.md#agenticdiagnoseloop--judge-decided-m1-m4-alternative-to-the-fixed-cycle)).

### Failure-aware embedding and boundary-aware naming (opt-in)

Two ProbeLLM-inspired extensions (arXiv 2602.12966), both off by default —
zero behavior change unless you opt in:

```python
report = cluster_failures(
    records,
    expected_col="expected",     # fold a failure-mechanism signal into grouping
    boundary_aware=True,         # contrast each cluster's edge cases with PASS rows
)
```

- **`expected_col`** (or `error_fn=lambda row: "..."` for full control): groups
  reflect *how* the model failed, not just what the prompt was about. When a
  `judge` is also given, one batched call describes each mismatch mechanism
  (e.g. "off-by-one count", "wrong entity substituted"); without a judge, a
  deterministic `"expected=... got=..."` string is used instead.
- **`boundary_aware=True`**: keeps non-FAIL rows as a contrast pool. For each
  cluster, the cases farthest from the centroid are paired with their nearest
  verified non-failure (`FailureMode.boundary_pairs`), and — when `judge` is
  given — the namer is shown these pairs to describe *where* the failure
  boundary sits rather than only what the failing cases share.

## Probe Search — Hierarchical MCTS Failure Discovery (VLM)

`ProbeSearchAgent` (`evalrx.eval_agent`) implements ProbeLLM's other core
idea: instead of clustering failures the model *already* showed you, it
actively synthesizes and evaluates **new** test cases, adaptively balancing
broad topical coverage (**Macro**) against local refinement around cases that
keep failing (**Micro**) via a hierarchical UCB-guided tree search
(`evalrx.analysis.probe_search.ProbeSearch` — standalone, generic over
injected generator/verifier callables).

```python
from evalrx.eval_agent import ClaudeModel, ProbeSearchAgent

agent = ProbeSearchAgent(judge=ClaudeModel(), budget=20)
result = agent.run(model, seed_pool)  # seed_pool: a VLM CaseBatch (image+question+expected)

print(result.n_simulations, result.n_macro, result.n_micro, result.error_rate)
for case in result.failure_cases:
    print(case.inputs.prompt, "->", case.observed, "(expected", case.expected, ")")
```

**Scope (v1, VLM-only):** the bundled generator
(`evalrx.eval_agent.VLMProbeCandidateGenerator`) only *paraphrases*
existing seed questions over the same image — Macro picks the seed least
similar to what's been explored so far, Micro rewords the current search
node's own case — so the seed's `expected` answer always stays valid without
needing a vision-capable judge or a tool that invents new gold answers. This
trades away the paper's full entity/attribute-substitution Micro generator
(which re-verifies a new gold answer via tools) for a safe, dependency-free
default; supply your own `generate_macro`/`generate_micro` callables to
`ProbeSearch` directly for richer generation.

`result.failure_cases` is a plain `CaseBatch` — feed it straight into
`cluster_failures` above for failure-mode synthesis. Inside
`AgenticDiagnoseLoop`, the same capability is exposed as the `search_probes`
tool (host-capped via `max_calls`), reusing the loop's own decision judge and
target model.

## Backend notes

`--backend` selects the local coding-agent CLI: `antigravity` (default),
`claude_code`, `codex`, `opencode`, `gemini_cli`, `kimi_cli`. On `claude_code`
the bundled `nature-figure` Agent Skill styles agent-drawn figures
automatically (`--no-skills` to disable). `tool_calls_*.json` files are
skipped by default (`--include-tool-calls` to include them).

Run `evalrx explore --help` for the full flag list.
