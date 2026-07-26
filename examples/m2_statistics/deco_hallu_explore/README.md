# deco_hallu Explore — M2/M3 on real M1 output

Demonstrates `evalvitals explore` (M2 exploratory analysis + M3 hypothesis
proposal) on **real M1 data**, not a synthetic demo: the per-case VLM
object-presence probe results already committed at
[`examples/diagnosis_loops/deco_hallu/data/cases/`](../../diagnosis_loops/deco_hallu/data/cases)
(three Qwen3-VL checkpoints — 2b/4b/8b — answering "Is there a {object} in
the image?" for COCO images, `label` = pass/fail).

No GPU or new M1 run is needed — this data was already produced by that
example's `run_m1.py`. The raw per-model files are handed to the M2 agent
as-is; it reads them, figures out the shape itself (each is a dict with
scalar run metadata plus a nested `cases` list), and organizes the three
files into one tidy table before analysing it — no pre-processing script.

## Run it

```bash
cd examples/m2_statistics/deco_hallu_explore && docker compose up
```

Or directly (needs a local coding-agent CLI, e.g. `claude`):

```bash
pip install -e ".[dashboard,viz]"
bash run.sh
```

Env overrides: `CODER_PROVIDER` (default `claude_code`), `CODER_MODEL`,
`OUT_DIR` (default `outputs`), `TIMEOUT_SEC`.

## What a real run found

- **Overall**: 126/606 cases (20.8%) are hallucination failures.
- **`probe_type` is almost fully deterministic**: adversarial probes (asking
  about an absent object) fail 34.4% of the time (n=366); present-object
  probes fail 0.0% (n=240).
- **Model size is non-monotonic**: within adversarial probes, fail rate is
  2b=33.9%, 4b=30.4%, 8b=38.5% — the *largest* model has the *highest* fail
  rate, not the lowest.
- **Object identity strongly modulates failure**: `potted plant` fails 81.8%
  of adversarial trials (n=11) vs `skis` at 9.1% (n=11).

M3 hypotheses proposed from these findings (not validated):

1. High-fail objects (potted plant, handbag, bottle, truck, bowl, backpack)
   share strong scene co-occurrence priors in the training distribution
   (plants near windows, bags near people, ...) that bias the model toward
   confirming presence even when the object is absent from the image.
2. The 8b model's higher adversarial fail rate reflects a difference in
   instruction-tuning/RLHF calibration toward helpfulness (favoring
   affirmative answers) between checkpoints, rather than a genuine capability
   regression with scale.

Open the dashboard to see the charts and hypothesis cards:

```bash
evalvitals dashboard outputs
```

## Attention-enriched variant (continuous per-case signals)

The raw probe files carry only categorical fields, so the analysis above can
never show FAIL-vs-PASS *distributions*. `data_attn_full/` fixes that: the same
606 cases enriched with 7 per-case attention-geometry scalars
(attention_entropy, focus_share, center_offset, edge_mass, top1_share,
max/mean_relative_weight) extracted with the `relative_attention` analyzer over
ALL cases of ALL three checkpoints (no `max_cases` cap; per-case float16
spatial maps kept under `data_attn_full/maps/`). Regenerate with:

```bash
python extract_attention_all.py --device cuda   # GPU; downloads any missing
                                                # COCO val2014 images itself
```

The enriched data ships with the repo, so exploring it needs no GPU:

```bash
bash run_attn.sh        # same env overrides as run.sh (CODER_PROVIDER/CODER_MODEL/...)
```

A real run found: within adversarial probes, focus_share is the strongest
separator (Cohen's d≈1.3, FAIL more peaked/off-center); the same signals
separate FAIL/PASS at every checkpoint (direction is scale-invariant, magnitude
uneven: 2B ≫ 8B > 4B); after collinearity pruning only focus_share,
center_offset, mean_relative_weight and edge_mass carry independent signal.
`data_2b_attn/` is the smaller 2B-only variant whose attention scalars were
transplanted from the diagnosis-loop's frozen M1 state (32/201 coverage —
useful mainly as an informative-missingness case study). `data_attn_2b/` is
the **full-coverage 2B-only slice** of `data_attn_full/` (201/201 cases with
attention scalars, single checkpoint) — use it to run M2/M3 without the
cross-checkpoint dimension, e.g. as a workbench upload (`data_attn_2b.zip`
ships alongside for exactly that).

## Held-out hypothesis pipeline (propose → test → fix → one report)

`run_attn.sh` analyses everything in-sample. The pipeline variant is the
**complete example**: it splits the data by its `split` column and walks the
full propose → held-out-test → repair arc, ending in one five-tab web report.

**Prerequisites**

- `pip install -e ".[dashboard,viz,stats]"` (`stats` powers the
  outcome-driver-analysis skill's regression path);
- a local `claude` CLI on PATH (phase 1 explorer, phase 2 judge, phase 3
  codegen all run through it);
- phases 0-2 need **no GPU** (the enriched data ships with the repo);
- phase 3 (surgery/fix) needs a **GPU** plus the loop example's frozen M1
  state at
  [`../../diagnosis_loops/deco_hallu/outputs/m1_state.pkl`](../../diagnosis_loops/deco_hallu)
  (produce it once with that example's `python run_m1.py --device cuda`).

**Run**

```bash
cd examples/m2_statistics/deco_hallu_explore
bash run_attn_pipeline.sh                      # full arc (phase 3 on GPU)
SKIP_FIX=1 bash run_attn_pipeline.sh           # stop after held-out testing (no GPU)
```

Env overrides: `CODER_MODEL` / `JUDGE_MODEL` (e.g. `claude-opus-4-8`),
`CODER_PROVIDER` (`claude_code`/`antigravity`/`codex`), `OUT_ROOT`
(default `outputs_pipeline`), `DEVICE` (default `cuda`), `TIMEOUT_SEC`,
`SKIP_FIX=1`, `PY` (python interpreter for the host phases).

**Phases**

1. `prepare_splits.py` — explore half (365) / validate half (241);
2. `evalvitals explore` on the explore half only — hypotheses + frozen,
   threshold-explicit recipes;
3. `test_hypotheses.py` — each recipe re-evaluated VERBATIM on the validate
   half (`adjudicate_signals(split_label="held_out")`: a REJECT here is a real
   held-out verdict), then an LLM judge grades every hypothesis
   (supported / partial / refuted / not_testable + surgery routing);
4. `run_surgery.py` — survivors go to the diagnosis loop's M5 confirm → M4 →
   tiered fix (L1→L3b) on the loop example's frozen M1 batch (GPU).

`confirm_report.json` / `fix_report.json` land next to the exploratory report.
Every explore view has the same FIXED five tabs: tab 3 stays the *pure
proposal* (unvalidated hypotheses, as in `run_attn.sh`), **4 Held-out
Verdicts** and **5 Fix** fill in from those artifacts — and grey out as
"not available" when a run stopped earlier (with `SKIP_FIX=1`, tab 5 stays
greyed until phase 3 runs):

```bash
evalvitals dashboard outputs_pipeline/1_explore
```

**What a real pipeline run found** (opus-4.8 end to end): all 6 frozen
attention-peakedness recipes replicated on the held-out half (6/6 REJECT H0);
the judge graded the scale-moderation hypothesis *partial* (its correlational
part held, the mechanism claim overreached) and the object-priors hypothesis
*not_testable*; in phase 3 M5 supported the surviving hypothesis
observationally but the M4 intervention experiment **refuted** its mechanistic
form — and the tiered fix swept 10 candidates to find that a plain L1 prompt
(`scan_then_decide`: scan the image region-by-region before answering) repaired
12 hallucinations and broke 0 (paired McNemar, e=315 → REJECT H0), beating
every L3b internals-write intervention. Correlation survived held-out;
mechanism died under intervention; the cheapest repair won.

## Web upload workbench (`run_web.sh`)

The variants above analyse the directories committed with this example. The
workbench flips the direction: it serves a page where anyone **uploads a
.zip** of results (JSON / JSONL / CSV — a zipped folder works too) and each
upload becomes one `evalvitals explore` run:

```bash
bash run_web.sh                 # serves http://localhost:8500
PORT=8600 CODER_PROVIDER=codex bash run_web.sh
```

Pick the analysis mode — **Explore only** (M2+M3; split 1 : 0 by default) or
**Explore + held-out verification** (default split 0.6 : 0.4, adjustable):
the verdict share is held out BEFORE exploration, then the frozen recipes and
hypotheses are re-tested on it (e-BH + LLM judge), filling the *Held-out
Verdicts* tab. Set the outcome column / question / backend, hit *Start
analysis*, and watch the live log; the analysis runs as a **detached
subprocess** (closing the tab never kills it) and the finished report renders
in place with the same tabs as `evalvitals dashboard`. Every upload lands
under `web_runs/<name>/` (`data/` extracted payload, `output/` report
artifacts, `explore.log`, `job.sh` to re-run it by hand) and past runs stay
selectable in the sidebar. Try it by zipping `data_2b_attn/` or
`data_attn_full/` and uploading that.

This page is also the **unified view over the sibling scripts' results**: any
existing output directory here (`outputs_attn_full`, `outputs_pipeline/
1_explore`, `outputs`) is attached read-only in the same sidebar (📁), and
every result — uploaded or attached, M3-only or full pipeline — renders with
the same fixed five-tab layout, unreached stages greyed out.

Env overrides: `PORT` (default 8500), `WORKSPACE` (default `web_runs`),
`CODER_PROVIDER` / `CODER_MODEL` / `TIMEOUT_SEC` — these only set the form's
defaults; each upload can override them in the UI — and `ATTACH_DIRS`
(space-separated result dirs to list read-only). The generic entry point is
`evalvitals web <workspace> --port N [--attach DIR ...]` (this script is a
thin wrapper).

See [`docs/m2_analysis.md`](../../../docs/m2_analysis.md) for the general
standalone M2/M3 workflow, and
[`examples/diagnosis_loops/deco_hallu/README.md`](../../diagnosis_loops/deco_hallu/README.md)
for the full M1 → M2 → M3 → M5 → Fix loop this data was built for.
