# RunLoggerV2 — design notes

Status: **coexists with `RunLogger`** (`run_logger.py`). Nothing in `RunLogger`
was touched, modified, or deleted to build this. `RunLoggerV2` is an opt-in
alternative. Use `RunContext(..., logger_version="v2")`, or construct it and
pass it as `run_logger=` anywhere `RunLogger` is accepted. V1 remains the
default and must not be removed until real-run parity and UI acceptance pass.

## Why this exists

`RunLogger` grew organically over many stages of the diagnose-loop project
(see `evalrx/eval_agent/run_logger.py`'s own docstring and CLAUDE.md history)
and its current output for one run looks like:

```
run_dir/
  run_log.jsonl          one JSON line per event, EVERY stage interleaved
  model_calls.jsonl       a second, separate JSONL stream
  artifacts/              *.npy, *.json, *.png — all stages mixed together
  prompts/                one *.prompt.txt + *.response.txt PER judge call
  experiments/             *.py, *.txt, record.md per M5 experiment
  tools/                   *.py, *.txt per synthesised tool
  workspace/                copied sandbox files
  fixes/                    record.md + result.json + outputs.jsonl per candidate
```

That's 7+ top-level directories, an unbounded number of small text files (one
pair per judge call, one per tool-codegen attempt, one per experiment), and no
way to look at "just M3" without grep-ing a shared JSONL stream and cross
referencing `cycle`/`event` fields.

This file describes the replacement layout and the reasoning behind each
rule the user asked for.

## The four rules, and how the layout satisfies them

### 1. Few files

One JSON document per pipeline stage. A run that used all five stages
produces exactly:

```
run_dir/
  run.json          run-wide events (see below)
  M1/log.json
  M2/log.json
  M3/log.json
  M4/log.json
  M5/log.json
  M*/artifacts/      only for stages that actually produced binary media
  media/             only if the run had case-level images/audio
  artifacts/          only if save_artifact_json() was called
```

A stage that never ran (e.g. a run with no M4/M5 activity) still gets a
folder + an empty-ish `log.json` — but only once `close()` runs: `close()` is
the one call site that flushes all five stages unconditionally, so a normal
`with RunLoggerV2(...) as logger: ...` run, or any `run_logger.close()` at the
end of a loop, produces all five `M*/log.json` regardless of activity. A run
that crashes before `close()` leaves folders only for stages that actually
logged something — `_append_stage`/`log_model_call` only flush the stage they
just wrote to, and `RunLoggerV2.__init__` doesn't pre-create anything.
`_stage_dir`/`_stage_artifacts_dir` create the `artifacts/` subfolders lazily,
on first use, so an unused stage never costs more than one small JSON file.

### 2. Same-type logging in one JSON

Within one stage's `log.json`, every event of a given kind is one array, one
key:

```json
{
  "probe": [ {...cycle 0...}, {...cycle 1...} ],
  "model_calls": [ {...}, {...}, ... ],
  "tool_codegen": [ {...} ],
  "tool_registry": [ {...} ],
  "stage_skipped": [ {...} ]
}
```

Never one small file per call. This is what let `model_calls` — the highest
fan-out event (self-consistency resampling, counterfactual regeneration —
see `model_instrumentation.py`) — collapse from its own `model_calls.jsonl`
stream into just another key in `M1/log.json`.

### 3. M1..M5 each get their own folder

Everything above. `run.json` at the top level is the one deliberate
exception — it holds events that are not stage content:

| `run.json` key       | what |
|---|---|
| `run_start`           | run config, git commit, versions (a single object, not a list) |
| `cases`                | the baseline `FailureCase` records + media references |
| `report_published`     | cached-report-publication audit records |
| `diagnose_reports`     | hypotheses, M4 verdicts, discovery rows, and summary |
| `manifest`              | final config and compact file index |
| `loop_end`              | one entry per loop-run summary |
| `agent_decisions`        | `AgenticDiagnoseLoop`'s own dispatch-judge turns |
| `agent_tool_calls`        | the dispatch layer's accept/reject outcome per tool call |
| `unrouted`                 | see "Routing", below — never silently dropped |

### 4. Nothing but JSON, except real binary media

Every text-shaped thing `RunLogger` wrote as a sibling file — judge prompts
and responses, generated code, stdout/stderr, the coder agent's raw
narration, human-readable Markdown summaries — is now a **string value
inside the relevant JSON entry**:

| RunLogger (a file + a path field)              | RunLoggerV2 (inline) |
|---|---|
| `prompts/c0_m1_selection.prompt.txt` + `judge_io.prompt_path` | `probe[].judge_prompt` |
| `experiments/c0_m5_main.py` + `code_paths["main.py"]` | `experiment[].code["main.py"]` |
| `experiments/c0_m5_stdout.txt` + `output_paths["stdout"]` | `experiment[].stdout` |
| `fixes/01_L1_cand1/record.md` (human summary)     | not generated — the same fields (`status`, `evidence`, `verdict`, ...) are already in `fix[]`; a renderer builds its own view from them |
| `tools/c0_m1_probe_gen_probe_01_code.py`  | `tool_codegen[].code` |

The one thing that stays a separate file is **real binary media**: numeric
tensors (`.npy`), rendered figures (`.png`), and any audio/video a sandbox
happens to produce. Those live under `M<n>/artifacts/` (or `media/` for
case-level images/audio), referenced by a relative-path string in the JSON —
see `_MEDIA_EXTS` in `run_logger_v2.py`.

A sandbox workspace snapshot (`log_experiment`'s `workdir`) is a directory of
arbitrary files, not a single value — `_inline_workspace()` walks it and, per
file, either inlines the text (source, data, logs) or — for a recognised
media extension — copies it to that stage's `artifacts/` and leaves a path
reference. Nothing is silently skipped: an unrecognised binary file gets a
one-line `"<skipped: ...>"` note instead of being dropped from the record.

`RunContext` gives V2 producers an ephemeral runtime tree outside the run
artifact. Explore, M5 experiments, and fix candidates may execute real code
there. Before finalization, source, prompts, responses, tables, stdout/stderr,
and validation records are inlined into JSON; recognised binary media is
copied into `M<n>/artifacts/`; then the runtime tree is deleted. V2 finalization
does not emit V1's `README.txt`, `summary.md`, trial `.txt`/`.py` files, or
in-run SQLite outbox.

## Model-call fidelity

Every recorded exchange lives in that stage's single `model_calls` array and
contains `role`, `operation`, complete JSON-safe `inputs`, complete `output`,
`error`, `duration_sec` when measured, and correlation metadata such as
`case_id`, candidate, generation kwargs, or tool name. Coverage includes:

- every pre-loop case-discovery request, response/error, generation kwargs,
  case id, and individually measured latency (written as each call completes);
- every in-process M1 analyzer `generate`/`forward`/`logprobs`/`chat` call,
  without V1's 8,000-character truncation when V2 is active;
- selection, statistics, diagnosis/critic, protocol, tool-codegen, fix-judge,
  experiment-writer, and CLI-coder calls for which the stage exposes I/O;
- fresh fix baselines, template candidates, declarative multi-call pipelines,
  guided/detector search, and every coded-pipeline bridge request/reply.

The stage summary keeps its convenient `judge_prompt`/`judge_response` fields;
`model_calls` is the exhaustive query surface.

## Routing: how an event picks its folder

Most events know their stage unambiguously (`log_probe` → M1, `log_analysis`
→ M2, ...). Four kinds of event carry a free-text `module`/`stage` string
instead (`log_surgery`'s M4-vs-M5 split, `log_experiment`, `log_tool_codegen`,
`log_tool_registry`, `log_stage_skipped`) and need to be **resolved**:

```python
_STAGE_RE = re.compile(r"m([1-5])", re.IGNORECASE)
_STAGE_ALIASES = {"fix_pipeline": "M5", "fix": "M5", "explore": "M2"}
```

1. Search the tag for `m[1-5]` (case-insensitive), anywhere in the string —
   covers `"m1_probe"`, `"M4_SURGERY"`, `"codegen_m2_stats"`.
2. Fall back to `_STAGE_ALIASES` for the handful of tags that carry no digit
   at all — as of this writing, only `fix_agent.py`'s
   `log_tool_codegen(module="fix_pipeline", ...)` needs this.
3. If NEITHER matches, the event goes into `run.json["unrouted"]` with a
   `warnings.warn` — never silently discarded. `_STAGE_ALIASES` was built by
   grepping every literal `module=`/`stage=` call site in `evalrx/eval_agent/`
   as of this writing (see the commit that introduced this file); a NEW call
   site added later with a tag that matches neither the regex nor the alias
   table will still surface — loudly, in `unrouted` — rather than vanish.

`log_surgery`'s M4/M5 split is NOT tag-based — it inspects
`"m4_test_name" in iv.evidence`, exactly mirroring `RunLogger.log_surgery`
post the M4/M5 stage-id swap (hypothesis verification = M4, intervention =
M5). Verified directly against `evalrx/eval_agent/run_metadata.py`, which is
where `m4_test_name` is actually set.

## Durability: why every event triggers a full-file rewrite

`RunLogger` appends one line to a JSONL stream per event — cheap, and a crash
mid-run leaves everything written so far intact. `RunLoggerV2` instead keeps
one in-memory dict per stage (+ one for `run.json`) and, on every single
`log_*` call, serialises the WHOLE current dict and writes it via
temp-file-then-`os.replace` (`_atomic_write_json`):

- **Why not just append?** Rule 2 requires one JSON *document* per stage —
  a bucketed object, not a line stream — so there's no way to "append" to it
  without rewriting the document (or accepting that reads mid-run see a
  syntactically invalid partial file).
- **Why not buffer everything and write once at `close()`?** That would lose
  every event on a crash mid-run — the same failure mode
  `test_holdout_cases_logged.py` exists to catch for V1 (an unlogged case id
  is unrecoverable). `model_calls` in particular was built specifically to
  survive "the process dies mid-cycle" (see `model_instrumentation.py`) — an
  in-memory-only V2 would silently regress that guarantee.
- **The trade-off, stated plainly:** each write is `O(current stage document
  size)`, and a stage that logs many small events (M1's `model_calls`, chiefly
  — a self-consistency/coverage_gap analyzer can make dozens of calls per
  case) rewrites a document that keeps growing. For a realistic run (dozens
  to a few hundred model calls, each a KB or two serialized) this is at most
  low tens of MB of total I/O across a whole cycle — negligible next to the
  model-inference time the run is dominated by. If a future run's volume
  makes this a real bottleneck, the fix is a batching/coalescing write behind
  the same `_flush_stage`/`_flush_run` calls, not a redesign of the layout.
- The write is **atomic**: a reader (a human `cat`-ing the file, or a future
  dashboard) either sees the complete state as of event N, or as of N-1 —
  never a truncated document.

## API parity: how this can be a drop-in

Every `RunLogger.log_*` method has a same-named, same-signature counterpart
here, including the two that matter for callers who don't just log — they
depend on the return value or a live attribute:

- `log_probe(...)` returns `list[Path]` (rendered PNG figures) — `loop.py`
  forwards these to the judge as visual context. `RunLoggerV2.log_probe`
  returns the same shape, pointing into `M1/artifacts/`.
- `current_cycle` is a plain read/write attribute `ProbeAgent`/`loop.py` set
  before probing and `InstrumentedModel` reads at call time — present here
  unchanged.
- `save_artifact_json(stem, obj)` — used by `agentic/tools.py` for
  fixed-name lookups (`failure_modes.json`, `probe_search_result.json`) —
  present, writing under a run-global `artifacts/` (not stage-scoped, since
  callers address it by a fixed name, not by cycle).

`RunLoggerV2` keeps its `RunContext` in `_context`, matching V1's integration
boundary. The difference is storage: `new_trial()` and `new_workdir()` allocate
ephemeral execution space for V2, and the logger captures its evidence before
`RunContext.finalize()` removes it.

## Deliberate scope cuts

- **No Markdown summaries** (`record.md`, `outcome.md`). Same information,
  no separate file; a renderer builds a human view from the JSON on demand.
- **Simplified verbose console output.** `RunLogger`'s `verbose=True` renders
  multi-line, stage-specific narration (`_VerboseFormatter`). `RunLoggerV2`'s
  `verbose=True` prints one line per event (`[M3] diagnosis cycle=0`) — this
  change was about file layout, not console UX, and porting that formatter
  verbatim would have doubled the size of this module for no layout benefit.

## What "confirmed working" means so far

### Non-Langfuse parity additions

- Non-numeric dict/list probe artifacts are inlined under
  `M1/log.json["probe"][i]["artifacts"]`, keyed by `analyzer/artifact`.
  Numeric artifacts remain `.npy` files. `results` retains result metadata
  and summaries independently from these artifacts.
- Fix attempts (including EXPLORE selection attempts with a `trial_root`)
  carry a `workspace_snapshot`. Before deleting its temporary execution tree,
  `RunContext.finalize()` also captures the entire remaining tree in
  `run.json["runtime_snapshot"]`, including discarded trials and late files.
  These managed snapshots preserve text without the normal 2 MB cap, inline
  other UTF-8 text such as shell scripts, and copy binary files into M5 artifacts.
  Failed archival prevents cleanup so it can be retried. Media names include
  source path and content hashes to preserve different trials and revisions.
- M4/M5 surgery records include `module`; the reader supplies this field for
  older V2 bundles too. Existing case-study M4 filtering works on both.
- `loop_end.verified_hypotheses` retains `failure_mode`, `status`, `confidence`,
  and `protocol_consistent`, alongside `statement` and `verdict`.
- Every newly written event carries `event`, `schema_version`, `trace_id`,
  `stage`, `span_id`, and a global `event_seq` assigned under the logger lock.
  The reader preserves these identities and sorts new bundles by sequence;
  historical bundles still use timestamp ordering. These local span IDs do
  not add or change Langfuse delivery.
- `EVALRX_VALIDATE_LOG=1` enables warn-only validation of persisted events
  using `log_schema.build_v2_schema()`. It reuses V1's common event fields and
  adds V2 model-call, diagnosis-report, and unrouted records. The published V1
  schema is unchanged; validation requires the optional `jsonschema` package.

Regression tests cover data retention after cleanup, retry after archival
failure, M4 rendering of old/new bundles, verified-summary parity, concurrent
event ordering, and valid/invalid schema inputs. These are local tests, not
a new real-model benchmark or live Langfuse acceptance run.

`tests/test_eval_agent/test_run_logger_v2.py`:

1. Drives every single `log_*` method with the SAME real domain objects
   `test_log_schema.py` uses to conformance-check `RunLogger` (`Result`,
   `StatsAnalysisReport`, `DiagnosisResult`, `InterventionResult`,
   `ExploratoryAnalysisReport`, `Hypothesis`) — not bespoke V2-shaped fakes.
2. Asserts the layout rules directly: exactly one folder per stage, no
   non-JSON/non-media files anywhere under the complete RunContext root, same-type events
   share one array, an unroutable tag lands in `unrouted` rather than
   vanishing, every JSON file left on disk is valid (parses) after a full
   run including after `close()`, no leftover `.tmp*` files.
3. Runs a real `VLDiagnoseLoop.run()` end to end (the same scenario
   `test_holdout_cases_logged.py` uses to catch a real historical bug —
   the held-out confirm split's cases never getting logged) with
   `RunLoggerV2` and also tests the actual `RunContext(logger_version="v2")`
   boundary, ephemeral trial/explore cleanup, full model-call I/O, report
   inlining, artifact copying, and quarantine rewrites.

The reporting compatibility reader in `evalrx/reporting/run_events.py` exposes
V2 documents to the existing dynamic report, HTML report, server, dashboard,
case-study loader, and Langfuse backfill paths without rewriting them.

## Next steps (explicitly not done here)

- Re-run a real `examples/benchmark/*` job after these integration changes and
  check the entire output tree, per-stage/model-call counts, relative artifact
  references, and rendered UI against its matched V1 run.
- Add a published JSON Schema for V2 and measure write amplification on long
  runs; atomic full-document rewrites favor crash readability over throughput.
- Only after real-run and UI acceptance: swap `RunContext.logger` (or examples) to
  default to `RunLoggerV2`, and remove `run_logger.py` + its sibling files
  (`log_schema.py`, `run_log.schema.json`, `model_calls.jsonl` handling in
  `model_instrumentation.py`'s call sites) — a separate, later change.
