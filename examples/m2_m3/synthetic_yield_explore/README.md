# Synthetic Yield — M2/M3 on a continuous outcome

Demonstrates `evalvitals explore` (M2 exploratory analysis + M3 hypothesis
proposal) on a **continuous** outcome, not just pass/fail logs. The data is
generated locally (`generate_data.py`, no model or API key needed): 30
synthetic chemical batches with `temperature`, `pressure`, `catalyst`, and a
`yield_pct` outcome, seeded so temperature and catalyst genuinely drive yield.

## Run it

```bash
cd examples/m2_m3/synthetic_yield_explore && docker compose up
```

Or directly (needs a local coding-agent CLI, e.g. `claude`):

```bash
pip install -e ".[dashboard,viz]"
bash run.sh
```

Env overrides: `CODER_PROVIDER` (default `claude_code`), `CODER_MODEL`,
`OUT_DIR` (default `outputs`), `TIMEOUT_SEC`.

## What it demonstrates

- `--outcome-col yield_pct` points explore at a named continuous target that
  no name heuristic would auto-detect.
- M2 adapts its chart battery to a continuous outcome: scatter + trend lines,
  binned mean-outcome curves, ranked correlation magnitude — not a FAIL/PASS
  bar chart.
- M3 proposes mechanism hypotheses from the M2 findings, not restatements of
  them.

## What a real run found

From an actual run (temperature r=0.86 with yield, the strongest association):

- **M2 observations**: yield ranges 63.8-95.9% (mean 82.2%); temperature
  correlates with yield at r=0.86 (the strongest signal found); pressure only
  r=-0.21; catalyst group means range 81.3% (B) to 83.5% (C).
- **M3 hypotheses** (proposed, not validated):
  1. Catalyst C's higher mean yield and tighter spread stem from higher
     selectivity or better thermal/process stability, rather than catalyst C
     batches simply having been run at more favorable temperatures.
  2. Pressure has little effect on yield because the reaction is not
     mass-transfer/gas-solubility limited across the operating window
     tested — pressure's true effect, if any, lies outside the range sampled.

Open the dashboard to see the charts and hypothesis cards:

```bash
evalvitals serve outputs
```

See [`docs/m2_analysis.md`](../../../docs/m2_analysis.md) for the general
standalone M2/M3 workflow.
