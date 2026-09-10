# Command-Line Interface

Installing the package installs one console script, `evalrx`
(`evalrx.cli:main`), with eight subcommands. `evalrx <command> --help` is
always the source of truth for flags; this page is the map of which command
does what and how they chain together.

```bash
evalrx --help
evalrx <command> --help
```

`-v` / `--verbose` (before the subcommand) turns on EvalRX's internal
stage-by-stage narration for any of them.

## The two jobs a subcommand does

| Job | Commands |
|---|---|
| **Produce** a run — turn result logs, a codebase, or a diagnosis loop into an output directory | [`explore`](#evalrx-explore), [`run-codebase`](#evalrx-run-codebase) |
| **View or export** a finished run — everything below reads a run directory, none of them re-run the model | [`serve`](#evalrx-serve), [`report`](#evalrx-report), [`publish-report`](#evalrx-publish-report), [`dashboard`](#evalrx-dashboard-deprecated), [`export-langfuse`](#evalrx-export-langfuse), [`backfill-langfuse`](#evalrx-backfill-langfuse) |

A "run directory" is either an `explore` output (`exploratory_report.json` +
`figures/` + `tables/`) or a loop run (`run.json`/`M1..M5`, or a `logs/`
folder holding those) — every viewing command accepts either shape.

## Which viewer command do I want?

Four commands all read the same run directory and this is the part that
actually confuses people, because two of them sound like the same thing:

| Command | What it produces | When to reach for it |
|---|---|---|
| `evalrx serve` | Starts a local FastAPI server and opens the **dynamic** React report UI (`http://127.0.0.1:8501` by default) — the same interactive interface used throughout this repo's own examples. Nothing is written to disk beyond a small `.evalrx-cache` unless you also ask it to publish. | Default choice. Point it at a run dir, or launch with none and drop a `.zip` on the page. `--runs-root` adds a runs panel to browse several experiments from one server. |
| `evalrx report` | Renders the **same** React UI but bakes it into one portable, self-contained `report.html` — no server, no network. | Sharing a result with someone who won't run a Python server, or archiving a run as a single file. |
| `evalrx publish-report` | Compiles and caches the underlying `ReportData` + layout spec that both `serve` and `report` render, without launching or exporting anything. | Pre-warming the cache for a run before serving it, or CI. |
| `evalrx dashboard` | **Deprecated.** A dependency-free static-HTML server predating the React UI (`evalrx/analysis/dashboard.py`). Kept only for old `explore` outputs that never got a React-compatible `report.html`. | Don't reach for this new — use `serve`. |

`serve`/`report`/`publish-report` all take `--source {auto,local,langfuse}` +
`--trace-id`: point them at a Langfuse trace instead of a local directory.

## `evalrx explore`

Single-shot exploratory analysis (M2 + M3) over a results directory — no
loop, no code. A local CLI coding agent writes and runs the analysis code;
the host adjudicates the statistics and renders figures.

```bash
evalrx explore ./results \
  --backend codex \
  -q "What distinguishes failed cases from successful ones?" \
  --serve-report
```

| Flag | Default | Purpose |
|---|---|---|
| `path` (positional) | — | File or directory of JSON/JSONL results. Required. |
| `-q, --question` | *"Explore this dataset…"* | Natural-language analysis question for the coding agent. |
| `--outcome-col` | auto-detect | Target/outcome column name. Falls back to unsupervised EDA when none is found. |
| `--out` | `evalrx_explore_output` | Output directory for report/code/figures/tables. |
| `--backend` (`--coder-provider`) | `antigravity` | Coding-agent CLI: `antigravity`, `codex`, `claude_code`, `opencode`, `gemini_cli`, `kimi_cli`. Must be installed + authenticated separately. |
| `--model` (`--coder-model`) | backend default | Model name passed to the coding agent. |
| `--max-rows` / `--max-files` | `2000` / `200` | Sampling caps for large inputs. |
| `--timeout-sec` / `--max-attempts` | `120` / `2` | Per-attempt timeout and repair-retry budget. |
| `--serve-report` | off | Launch `serve` on the output when the run finishes. |
| `--port` | auto | Port for `--serve-report`. |
| `--skill DIR` (repeatable) | — | Agent-Skill directory to style agent-authored figures. Implies `--allow-skills`. |
| `--no-skills` | off | Skip the package's bundled skills (e.g. `nature-figure`). |
| `--no-hypotheses` | off | Skip M3 — stop after M2's takeaways. |
| `--holdout-frac` | `0.0` | Fraction held out *before* exploration (outcome-stratified, deterministic). |
| `--holdout-confirm` | off | Re-test frozen recipes/hypotheses on the held-out rows (`confirm_report.json`). Requires `--holdout-frac > 0`. |
| `--judge-model` | `claude-opus-4-8` | Judge grading each hypothesis against the held-out table (only with `--holdout-confirm`). |

`--dashboard` still exists as a deprecated alias for `--serve-report`.

## `evalrx run-codebase`

For when you don't have result logs yet — only an evaluation/inference
codebase. A CLI coding agent runs it inside an isolated copy (your original
directory is never modified), harvests a `records.json`/`.jsonl` output, and
hands it to the same `explore` (M2+M3) pipeline.

```bash
evalrx run-codebase ./my_eval_repo \
  --backend claude_code \
  -q "Where does the model fail and why?" \
  --out evalrx_run_codebase_output
```

| Flag | Default | Purpose |
|---|---|---|
| `path` (positional) | — | Directory containing the codebase to run. Required. |
| `-q, --question` | *"Explore this dataset…"* | Also given to the run agent as task context. |
| `--out` | `evalrx_run_codebase_output` | Output directory for workspace/records/report/figures/tables. |
| `--backend` (`--coder-provider`) | `claude_code` | Same choices as `explore`. Used both to run the codebase and to explore it. |
| `--records-name` | `records.json` | Output-contract filename the run agent must write. |
| `--timeout-sec` / `--max-attempts` | `1200` / `2` | Budget for running the codebase itself. |
| `--no-explore` | off | Only run + harvest; skip the M2/M3 explore step. |
| `--serve-report` / `--port` | off / auto | Same as `explore`. |

## `evalrx serve`

```bash
evalrx serve outputs/qwen3.5-2b/chartqa
evalrx serve                          # start empty, drop a .zip on the page
evalrx serve --runs-root outputs/     # runs panel over every run under outputs/
```

| Flag | Default | Purpose |
|---|---|---|
| `run_dir` (positional, optional) | none | Run directory. Omit to start with an empty drop-zone. |
| `--port` | `8501` | Loopback port. |
| `--no-browser` | off | Don't auto-open a browser (the default in a headless/remote session). |
| `--source` | `auto` | `auto` \| `local` \| `langfuse`. `auto` uses Langfuse only when `--trace-id` is set. |
| `--trace-id` | — | Langfuse trace id (required with `--source langfuse`). |
| `--runs-root` | `run_dir`'s parent | Where the in-page runs panel looks for other experiments. |

## `evalrx report`

Exports the same UI `serve` shows, as one portable HTML file — no server
needed to view it afterward.

```bash
evalrx report outputs/qwen3.5-2b/chartqa --out chartqa_report.html
```

| Flag | Default | Purpose |
|---|---|---|
| `run_dir` (positional) | `outputs` | Run directory holding `run.json`/`M1..M5`, or a `logs/` folder. |
| `--example-dir` | — | Root holding a benchmark/example's `data/` manifest, for legacy runs. |
| `--out, -o` | `<run_dir>/report.html` | Output HTML path. |
| `--embed-media` | `representative` | `representative` \| `all` \| `none` — how much image/audio to inline. |
| `--audio-bitrate` | `48k` | Transcodes inlined lossless audio to mono MP3 via `ffmpeg` (needs `ffmpeg` on `PATH`); `none` inlines the original bytes. |
| `--no-audio` | off | Skip audio transcoding entirely (forces `--embed-media none`). |
| `--source` / `--trace-id` | `local` / — | Same Langfuse switch as `serve`. |

## `evalrx publish-report`

Compiles and caches `ReportData` plus the validated json-render layout that
`serve`/`report` both render — without launching a server or writing an
HTML file. Useful for pre-warming a run's cache, or in CI.

```bash
evalrx publish-report outputs/qwen3.5-2b/chartqa
```

Takes `run_dir` (positional, default `outputs`), `--example-dir`,
`--source`, `--trace-id` — same meanings as `report`.

## `evalrx dashboard` (deprecated)

```bash
evalrx dashboard evalrx_explore_output
```

A dependency-free static-HTML server (`evalrx/analysis/dashboard.py`)
predating the React report UI. It compiles a legacy `report.html`
(`evalrx/reporting/html_report.py`) if one isn't already there and serves it
with the standard-library `http.server` — no `fastapi` needed. Kept for old
`explore` outputs; reach for `serve` for anything new.

Takes `run_dir` (positional) and `--port`.

## `evalrx export-langfuse`

Maps a finished run (M1–M4, fixes, scores) onto Langfuse Traces, Spans, and
Scores — either as a static JSON bundle or a live sync.

```bash
evalrx export-langfuse outputs/qwen3.5-2b/chartqa --out trace.json
evalrx export-langfuse outputs/qwen3.5-2b/chartqa --sync   # push to a live Langfuse server
```

| Flag | Default | Purpose |
|---|---|---|
| `run_dir` (positional) | `outputs` | Run directory. |
| `--out, -o` | `<run_dir>/langfuse_trace.json` | Output JSON path (ignored with `--sync`). |
| `--sync` | off | Push directly to a live Langfuse server instead of writing a file. |

## `evalrx backfill-langfuse`

Queues an already-completed run for reliable Langfuse ingestion (an outbox
pattern, distinct from `export-langfuse --sync`'s direct push).

```bash
evalrx backfill-langfuse outputs/qwen3.5-2b/chartqa --dry-run
```

`run_dir` (positional, required) and `--dry-run` (inspect without writing an
outbox) are the only flags. Prints `trace=… events=… published=… pending=…`.
