# demo_page — one finished run as a single HTML file

A **one-off demo artifact**, not a product feature. `evalvitals dashboard` is
the supported way to browse a run; this exists for the case where a demo needs
a page you can hand someone — no Streamlit, no `pip install`, no `./data` on
their machine — and where the whole loop matters, not only the case book.

## From a fresh clone

The page is rendered *from a finished run*; it does not carry one. Starting
from nothing, in order:

```bash
pip install -e ".[dashboard]"                  # from the repo root
cd examples/m1_m4/mmau_qwen2_audio
python download_mmau.py --limit 1000 --scan-rows 1000   # writes ./data
docker compose up --build                      # the run itself — writes ./outputs
python demo_page/build_page.py                 # writes ./demo_page/index.html
open demo_page/index.html
```

Only the last step is this directory's; the three before it are the example's
own, documented in `../README.md`. The run needs a GPU and a `claude` CLI on
the host (it is the M2/M3/M5 judge) and takes roughly 25 minutes at
`--limit 896`.

**`run.py --smoke-test` is not enough.** It exercises the loop in-process
against synthetic cases and never persists a repair attempt, so there is no M4
to render; the script says so and exits. A page needs a real run.

Defaults are `--run-dir outputs --example-dir . --out demo_page/index.html`.
`--run-dir` takes either the directory holding `run_log.jsonl`, or a parent
holding `logs/` + `explore/` — the layout `run.py` writes.

## What it renders

Seven sections, in run order. Each opens with the one conclusion that stage
reached; the evidence sits underneath, and the bulkier evidence sits behind a
disclosure. Stages a run skipped are rendered dashed and say so, because a
stage that did not run is information.

| Section | Opens with | Evidence under it |
|---|---|---|
| `pre_m1` | whether probe search ran | — |
| `m1` | what the probes found, in one line | the 8 probes, the question each asks, its headline number, and its coverage |
| `m2` | what the screen concluded | the 4 charts that carry it, the tests that survived BH correction, the conclusion verbatim |
| `m3` | the hypothesis, in one line | its full statement and failure mode |
| `m5` | the adjudication | effect, CI, confidence, the verdict string verbatim |
| `m4_surgery` | whether a causal intervention ran | why the `surgery` event's `module` field decides this |
| `m4_fix` | the validated repair and its effect | candidate screening, confirmation stats, the prompt template, the case book |

`explore` is not given its own section: the contract treats it as a stage
between M1 and M2, but for a reader it is the exploratory half of screening, so
it is folded into `m2` — its charts lead that section and its in-sample status
is stated there.

The case book joins each per-case outcome in
`fixes/<attempt>/outputs.jsonl` back to `data/mmau_test_mini.jsonl`, so a case
carries its question, its options, the correct answer and a playable clip —
filterable by repaired / broke / unchanged.

Only cases whose clip is present locally appear. `download_mmau.py --limit N`
decides that: a 120-clip download against an 896-case run yields a case book of
whatever overlaps, and the page states the count rather than implying coverage.

## Requirements

- `evalvitals` importable — `pip install -e .` from the repo root. The script
  itself imports nothing from the package, but the run that feeds it does.
- `ffmpeg` on PATH — clips are re-encoded to 48 kbit/s mono AAC before being
  inlined, which is what keeps the page near 7 MB instead of ~50 MB. Without it
  the script warns and the case book renders without players; `--no-audio`
  skips the step deliberately.
- A populated `./data` (from `download_mmau.py`) and a finished `./outputs`.

## Caveats

- **It is a snapshot.** Nothing re-reads the run; re-run the script after a new
  one. It does not go through `evalvitals.analysis.dashboard.load_run()`, so it
  will not pick up loader changes on its own.
- **It is MMAU-shaped.** The case-book join assumes this example's manifest
  fields (`instruction` / `choices` / `expected` / `audio_path`). Other
  benchmarks need `collect()` adjusted.
- Figures are the run's own matplotlib output, inlined unmodified. Where a
  figure is less legible than the table beside it, the page keeps it in a
  collapsed block for provenance rather than dropping it.
- `.audio_cache/` holds the transcoded clips so repeat builds are fast. Safe to
  delete.
