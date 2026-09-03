# deco_hallu — end-to-end VLM failure diagnosis (M1 → M2 → M3 → M5 → Fix)

This example diagnoses a **VLM over-detection hallucination**: for yes/no questions
like *"Is there a {object} in the image?"*, the model sometimes answers **"Yes"** for
an object that is **not** present. It runs the full LAMBDA×M2 pipeline and produces a
dashboard that tells the whole story — *what was analysed → what was found → which
root-cause hypotheses were formed (and how they tested)*.

The run is split into stages so the expensive M1 forward passes are frozen once and
reused:

```
build_cases.py     data/cases/<model>.json        balanced FAIL/PASS batch (offline, once)
run_m1.py          outputs/m1_state.pkl           M1 analyzers (GPU)
run_fused.py       outputs/fused/...              Step 1: explore + held-out confirm (claude)
run_m2-5.py        outputs/logs_m2_5/run_log.jsonl Step 2: M2→M3→M5→Fix (GPU + claude)
evalrx serve outputs                           the report
```

## Prerequisites

- The repo installed into a venv **with the local + viz extras**:
  ```bash
  cd <repo-root>
  pip install -e ".[local,viz]"               # torch + matplotlib
  # (this repo's server uses a uv venv: VIRTUAL_ENV=.venv uv pip install -e . --no-deps,
  #  then `VIRTUAL_ENV=.venv uv pip install matplotlib jsonschema`)
  ```
- A **CUDA GPU** and the VLM weights cached (HF cache) — needed by `run_m1.py` and
  `run_m2-5.py`.
- The **`claude` CLI installed and logged in** — used by the explorer (Step 1), the M3
  judge, and M2/M4/Fix codegen. (Swap to `--backend codex|agy` if you prefer.)

## Quick start (scripts)

Two wrapper scripts run the whole thing for you (they resolve the repo's venv
python and the bundled `nature-figure` skill automatically):

```bash
cd <repo-root>/examples/m1_m4/deco_hallu

./run_all.sh        # from nothing → dashboard  (build_cases → M1 → fused → M2-5 → dashboard)
./run_from_m1.sh    # assumes outputs/m1_state.pkl exists → skips M1 (fused → M2-5 → dashboard)
```

Both accept env overrides, e.g. `MODEL=qwen3-vl-4b-instruct DEVICE=cuda BACKEND=claude
PORT=8501 ./run_all.sh`. Set `DASHBOARD=0` to finish the pipeline without launching
Streamlit (it just prints the dashboard command). The manual steps below are what the
scripts wrap.

### Decoupled flow — analyse → dashboard → *then* confirm + fix

`run_all.sh` / `run_from_m1.sh` run the loop end to end (M2→M3→**M5**→Fix in one
go). If you'd rather **see the analysis and proposed hypotheses on the dashboard
before** spending GPU/claude on confirmation + repair, split it in two:

```bash
cd <repo-root>/examples/m1_m4/deco_hallu

./run_analysis.sh     # fused → run_analysis (M2 stats+charts → M3 PROPOSE) → dashboard
                      #   no GPU; hypotheses are proposed, NOT yet confirmed (no M5/M4/Fix)
./run_confirm_fix.sh  # later: M5 confirm (same hypotheses) → M4 + tiered Fix → dashboard
                      #   GPU + claude; reuses outputs/analysis/ so the SAME hypotheses
                      #   the dashboard showed are the ones confirmed
```

- **`run_analysis.py`** runs `VLDiagnoseLoop.run_analysis()` — M1 (replayed) → M2
  (the *same* rigorous e-BH stats + charts) → M3 (propose) — and stops. It writes
  `outputs/logs_analysis/run_log.jsonl` plus `outputs/analysis/{proposed_hypotheses.json,
  analysis_state.pkl}` (the proposed hypotheses + the exact M2 stats report).
- **`run_confirm_fix.py`** reloads those artifacts and runs `VLDiagnoseLoop.run_confirm()`
  (M5) on the **same** hypotheses, then `run_m4` + `run_fix`, into
  `outputs/logs_confirm_fix/run_log.jsonl`. The dashboard merges both log dirs, so
  after this each proposed hypothesis gains its M5/M4/Fix verdict.

The one-shot `run_m2-5.py` (M2→M3→M5→Fix together) is still here for the
all-in-one path.

## Run it (4 steps)

```bash
# Use the repo's venv python; run from THIS directory (scripts `import run`).
PY=<repo-root>/.venv/bin/python
NF=<repo-root>/evalrx/agent_assets/skills/nature-figure   # bundled figure-styling skill
cd <repo-root>/examples/m1_m4/deco_hallu

# 0) one-time, offline: build the balanced FAIL/PASS case batch
$PY build_cases.py

# 1) M1 — run analyzers once and freeze them            [GPU]
$PY run_m1.py --model qwen3-vl-2b-instruct --device cuda
#    -> outputs/m1_state.pkl

# 2) Step 1 — explore (rich charts) + confirm on a held-out split   [claude, no GPU]
$PY run_fused.py --backend claude --skill "$NF"
#    -> outputs/fused/fused_report.json        (observations, ~6-12 charts, signal verdicts)
#       outputs/fused/confirmed_recipes.json   (signals that survived e-BH on the held-out split)
#       outputs/fused/figures/*.png            (nature-figure-styled charts)

# 3) Step 2 — the repair loop, fed by Step 1            [GPU + claude]
$PY run_m2-5.py \
     --recipes        outputs/fused/confirmed_recipes.json \
     --explore-report outputs/fused/fused_report.json \
     --device cuda
#    -> outputs/logs_m2_5/run_log.jsonl        (M2 stats, M3 hypotheses, M5 tests, Fix)

# 4) View the report
$PY -m evalrx.cli serve outputs             # or: evalrx serve outputs
```

Remote server → open it locally over an SSH tunnel:
```bash
ssh -L 8501:localhost:8501 <user>@<server>     # then browse http://localhost:8501
```

### What the two hand-offs in Step 2 do

- `--recipes confirmed_recipes.json` — the signals **confirmed** in Step 1 are *bridged*
  into the M2 family (so the LAMBDA-discovered composite signals are tested rigorously).
- `--explore-report fused_report.json` — Step 1's **charts + observations** (descriptive,
  UNCONFIRMED) are shown to **M3 only**, to inform *which* hypotheses it proposes. They
  never enter M2 confirmation, M5 testing, or the fix gate.

## What you get (the dashboard)

Point the dashboard at `outputs` (the parent — **not** a sub-dir): it merges
`logs_m1/` + `logs_m2_5/` and finds `fused/fused_report.json`, then renders a connected
report in the **📊 Analysis** tab:

1. **What we analysed** — the exploratory observations and the signals examined.
2. **What we found** — the e-BH adjudication (which signals are *real* on the held-out
   split), the signal effect-size chart, the signals table, the analyst's evidence chain,
   and the exploratory charts + data tables.
3. **Hypotheses formed** — each M3 root-cause hypothesis, tagged with the charts it drew
   on and its downstream **M5/M4 verdict** (supported / inconclusive / refuted / fixed).

The **🔬 Diagnosis flow** and **🗂 Tables** tabs hold the raw event timeline and CSVs.

## Outputs layout

```
outputs/
  m1_state.pkl
  logs_m1/run_log.jsonl                 # M1 probe events
  fused/
    fused_report.json                   # Step 1: observations, charts, signal verdicts
    confirmed_recipes.json              # signals fed to Step 2
    figures/*.png  sandbox/tables/*.csv # rendered charts + their data
  logs_m2_5/run_log.jsonl               # one-shot Step 2: analysis / diagnosis / surgery / fix

  # decoupled flow (run_analysis.sh → run_confirm_fix.sh):
  logs_analysis/run_log.jsonl           # PHASE 1: M1 + M2 stats/charts + M3 proposed hyps
  analysis/proposed_hypotheses.json     # PHASE 1: the proposed hypotheses (human-readable)
  analysis/analysis_state.pkl           # PHASE 1: {hypotheses, M2 stats_report} → reused by PHASE 2
  logs_confirm_fix/run_log.jsonl        # PHASE 2: M5 verdicts + M4 surgery + Fix
```

## Adapt to a new case

The "case definition" lives in three places; the scripts are generic:

1. **Data** — produce your own balanced FAIL/PASS batch at `data/cases/<model>.json`
   (mirror `build_cases.py`). M2/M5 need both groups.
2. **`config.yaml`** — set `model` (the VLM under test), `judge_model`, codegen/fix knobs.
3. **`run.py: build_protocol()`** — write the `description` as **observations only**: the
   wrong-answer pattern and the input conditions. Do **not** name a suspected mechanism
   (attention, language prior, …) — discovering that is the loop's job; stating it leaks
   the answer.

Easiest path: copy this directory, swap `data/`, edit `config.yaml`'s `model`, and rewrite
`build_protocol`'s description. Then run the 4 steps above.

## Notes

- **GPU per stage:** M1 (step 1) and Step 2 load the VLM; Step 1 (`run_fused.py`) does not
  (it reuses the frozen M1 and runs a CLI-agent explorer + deterministic confirmation).
- **Figure styling:** the host-rendered charts are auto-styled with the bundled
  **nature-figure** palette; `--skill "$NF"` additionally lets the explorer apply the skill
  to any figures it draws itself (box/violin/heatmap). Drop `--skill` to skip that.
- **Honest negatives are results:** if no explorer signal survives the held-out split, or
  no fix validates, that is the *correct* rigorous outcome — not an error.
