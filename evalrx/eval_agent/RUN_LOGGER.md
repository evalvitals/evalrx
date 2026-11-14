# RunLogger — design notes

`RunLogger` (`run_logger.py`) is the run log every diagnosis loop writes
through, and the logger a `RunContext` owns. This file says what it writes and
why the layout is the way it is.

## The layout

```
run_dir/
  run.json          run-wide events (see below)
  M1/log.json
  M2/log.json
  M3/log.json
  M4/log.json
  M5/log.json
  M*/artifacts/     binary media a stage produced
  media/            case-level images / audio
  artifacts/        named JSON written with save_artifact_json()
```

Four rules shaped it.

### 1. Few files

One JSON document per pipeline stage. `close()` flushes all five stages, so a
finished run has all five `M*/log.json`; a run that died earlier has a folder
only for the stages that logged something. `artifacts/` subfolders are created
on first use.

### 2. Same-type logging in one JSON

Within a stage's `log.json`, every event of one kind is one array under one
key:

```json
{
  "probe": [ {"cycle": 0}, {"cycle": 1} ],
  "model_calls": [ {}, {} ],
  "tool_codegen": [ {} ],
  "stage_skipped": [ {} ]
}
```

There is never one small file per call. `model_calls` is the highest fan-out
event (self-consistency resampling, counterfactual regeneration; see
`model_instrumentation.py`) and it is just another key in its stage's document.

### 3. M1..M5 each get their own folder

A reader who cares about M3 opens one folder. `run.json` holds what is not
stage content:

| `run.json` key     | what |
|---|---|
| `run_start`        | run config, git commit, versions (one object, not a list) |
| `cases`            | the baseline `FailureCase` records, their media references and `split` |
| `report_published` | one audit record per published report |
| `diagnose_reports` | hypotheses, M4 verdicts, discovery rows and a summary |
| `manifest`         | final config and a compact file index |
| `loop_end`         | one entry per loop summary |
| `run_end`          | written by `close()`: the run is finished |
| `agent_decisions`  | `AgenticDiagnoseLoop`'s dispatch-judge turns |
| `agent_tool_calls` | the dispatch layer's accept / reject outcome per tool call |
| `unrouted`         | events whose stage could not be resolved (see Routing) |

### 4. Nothing but JSON, except real binary media

Everything text-shaped is a string value inside the relevant entry: judge
prompts and responses (`judge_prompt`, `judge_response`), generated code
(`experiment[].code`, `tool_codegen[].code`), stdout and stderr, the coder
agent's raw narration. No Markdown summary is generated; a renderer builds its
own view from the fields.

What stays a separate file is binary media: tensors (`.npy`), figures
(`.png`), audio or video. Those live under `M<n>/artifacts/` (or `media/` for
case-level media) and the JSON holds a run-relative path (`_MEDIA_EXTS` in
`run_log_files.py`).

A sandbox workspace is a directory of arbitrary files, not one value.
`inline_workspace()` walks it and, per file, inlines the text or copies
recognised media into the stage's `artifacts/` and leaves a path. An
unrecognised binary file gets a one-line `"<skipped: ...>"` note, so nothing
disappears from the record.

`RunContext` gives producers an ephemeral runtime tree outside the run
directory. Explore, M5 experiments and fix candidates execute there. At
finalization the logger inlines what they left and the runtime tree is deleted.

## The event envelope

Every event carries `event`, `stage`, `event_seq`, `span_id`, `trace_id`,
`schema_version` and `ts`. `event_seq` is one global sequence, assigned under
the logger's lock at write time, so the order holds when calls finish
concurrently or the wall clock moves backwards. Readers sort on it.

## Model-call fidelity

Every recorded exchange is one entry in its stage's `model_calls` array: `role`,
`operation`, complete JSON-safe `inputs`, complete `output`, `error`,
`duration_sec` when measured, and correlation metadata such as `case_id`,
candidate, generation kwargs or tool name. It covers:

- every pre-loop case-discovery request, written as each call completes;
- every in-process M1 analyzer `generate` / `forward` / `logprobs` / `chat` call;
- selection, statistics, diagnosis and critic, protocol, tool-codegen,
  fix-judge, experiment-writer and CLI-coder calls;
- fresh fix baselines, template candidates, declarative multi-call pipelines,
  guided search, and every coded-pipeline bridge request and reply.

A stage summary keeps its convenient `judge_prompt` / `judge_response` fields;
`model_calls` is the exhaustive query surface.

## Routing: how an event picks its folder

Most events know their stage (`log_probe` is M1, `log_analysis` is M2).
`log_experiment`, `log_tool_codegen`, `log_tool_registry` and
`log_stage_skipped` carry a free-text `module` / `stage` tag that is resolved:

1. Search the tag for `m[1-5]`, case-insensitive, anywhere in the string:
   `"m1_probe"`, `"M4_SURGERY"`, `"codegen_m2_stats"`.
2. Fall back to `_STAGE_ALIASES` for tags with no digit (`"fix_pipeline"`).
3. If neither matches, the event goes into `run.json["unrouted"]` with a
   `warnings.warn`. A new call site with an unknown tag surfaces loudly instead
   of vanishing.

`log_surgery` is not tag-based: an intervention result whose evidence carries
`m4_test_name` is a hypothesis verification (M4), anything else an intervention
(M5). `run_metadata.py` is where `m4_test_name` is set.

## Durability: why every event rewrites its document

The logger keeps one in-memory dict per stage plus one for `run.json`, and on
every `log_*` call serialises the whole dict it touched and writes it with
temp-file-then-`os.replace` (`atomic_write_json`).

- **Why not append?** Rule 2 wants one JSON document per stage, a bucketed
  object rather than a line stream, and a document cannot be appended to
  without a reader seeing an invalid partial file.
- **Why not write once at `close()`?** A crash mid-run would lose every event.
  `model_calls` exists to survive "the process dies mid-cycle".
- **The cost:** each write is `O(current stage document size)`. A stage that
  logs many small events (M1's `model_calls`) rewrites a growing document. For
  a realistic run this is low tens of MB of I/O over a cycle, negligible next
  to inference time. If a run's volume makes it a bottleneck, the fix is to
  coalesce writes behind `_flush_stage` / `_flush_run`, not to change the layout.
- **Atomic:** a reader sees the complete state as of event N or N-1, never a
  truncated document.

## What callers depend on beyond logging

- `log_probe(...)` returns `list[Path]`, the rendered figures under
  `M1/artifacts/`; the loop forwards them to the judge as visual context.
- `current_cycle` is a read/write attribute the loop sets before probing and
  `InstrumentedModel` reads at call time.
- `save_artifact_json(stem, obj)` writes under the run-global `artifacts/`,
  because callers address those files by a fixed name, not by cycle.

## Console output

`verbose=True` prints one line per event (`[M3] diagnosis cycle=0`).
`narrate=True` prints the aligned live M1-M5 narration instead
(`narration.LoopNarrator`) and takes over from `verbose`.
