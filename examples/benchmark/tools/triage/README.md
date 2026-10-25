# Triage protocol scripts

The drivers and analysis scripts of the scheduler comparison run with `--m1-policy triage_active`
(design and results: `evalsmith/probing/probing_design_mix.md`; the controller:
[`docs/agentic_benchmark.md`](../../../../docs/agentic_benchmark.md), "Triage policy"). They were
written as scratch for the triage17/18 runs on Qwen3.5-2B and ChartQA and are kept here so the
protocol and the tables derived from it can be re-run. Machine-specific values (python, scratch
directory, data and output roots, the judge) come from the environment; see `_common.sh`.

## The protocol

Every arm is one `evalrx.benchmark.run` invocation into the **same run directory**, so the arms
share one observation ledger (an action with the same identity is read back, not re-run) and each
is charged what its evidence cost when first executed. After each arm its report is copied aside
as `probing/<report stem>.<arm>.json`; then `validate_arms.py` runs the benchmark entry once more
with the controller replaced by the stored reports and the runner's `validate()` replaced by one
over the union of every arm's frozen issues: one family, one Bonferroni level, one validation
ledger, every arm credited with its own issues (`validation/validation_report.{json,md}`,
`validation/frozen.json`).

| arm | flags |
|---|---|
| `rules` | `--triage-scheduler rules` |
| `scored` | `--triage-scheduler scored` |
| `scored_x0` | `--triage-scheduler scored --triage-explore-share 0` |
| `scored_d` | `--triage-scheduler scored --triage-explanations judge` (judge: `JUDGE_FLAGS`) |
| `random`, `cheapest` | `--triage-scheduler random` / `cheapest` (controls over the scored candidates) |
| budget tier `b<N>` | `--triage-budget-sec N`; `full` = no cap |

Common to every arm: `--triage-n-fail 8 --triage-max-fail <32 or 128> --triage-max-actions 30
--triage-chunk 8`, `--attn-impl eager`, the frozen ChartQA pool (`--data-dir ... --limit 1000`),
one GPU with nothing else on it (`CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`), weights
offline, a private `TMPDIR`, and no `CLAUDE*` variables in the environment (the claude CLI judge
must not see a parent session).

## Scripts

| script | what it did |
|---|---|
| `_common.sh` | sourced by the run scripts: paths from the environment, the arm table, code snapshots, the arm loop, the union validation |
| `run_triage17.sh` | triage17 (10-21/22): triage16's 12 stored arms validated once more with a validation-only source added (`--triage-validation-source`) |
| `run_triage18.sh` | triage18 phase 1 (10-22/23): the 12 arms (rules / scored / scored_x0 / scored_d x full / b700 / b900) at `--triage-max-fail 128` on a code snapshot (commit 0d01bb6), then their union validation |
| `run_triage18_phase2.sh` | phase 2 (10-23/24): the controls at full budget and the budget tiers 4000 / 5500 / 7100 s for all six schedulers (20 arms) added to the same directory, then the 32-arm union validation; the 12-arm validation kept as `*.12arms.*` |
| `run_triage18_phase3.sh` | phase 3 (10-24): all 32 arms again under the `step_cost` fix (scoring `mvp3-score-5`), phase-2 reports moved to `probing/arms_p2/` and its validation kept as `*.32arms_p2.*` |
| `launch_when_free.sh` | polls for a GPU with under 3 GB in use every 20 s and starts a run script on it |
| `validate_arms.py` | the union validation driver the run scripts call |
| `discovery_curve.py` | budget-matched discovery curve: supported phenomena each arm had found within a charged budget B (the design document's "预算对齐" tables) |
| `recount_u.py` | re-reads a run's stored U evidence under the exact and the tolerant agreement rule (the rules-5 "离线重算" table) |

The phase scripts are idempotent: an arm whose report copy exists is skipped, so a killed run is
resumed by starting the script again. The 700 / 900 s tiers are kept in the phase-3 validation but
not interpreted: at `--triage-max-fail 128` they stop inside the first expansion.

## Data

The ChartQA pool is `examples/benchmark/vlm/_data/triage_pool/chartqa/` (1000 human-authored rows,
seed 5022, frozen from the HF cache with `python -m evalrx.benchmark.tasks.chartqa`), and the
validation-only source `.../triage_pool/chartqa_validation/` is every human row the pool left
(`python -m evalrx.benchmark.tasks.chartqa <dir> --exclude <pool>/manifest.json`). `_data/` is
not committed; `partition_bundle.json` records each manifest's sha1.

## Usage

```bash
export PY=/path/to/venv/bin/python TRIAGE_SCRATCH=/path/to/scratch   # see _common.sh for the rest
nohup bash examples/benchmark/tools/triage/launch_when_free.sh run_triage18_phase3.sh > launcher.log 2>&1 &
python examples/benchmark/tools/triage/discovery_curve.py <run dir> --grid 3000 4000 5000 6000 7095
python -m evalrx.eval_agent.probing.compare <run dir>/probing/triage_report.rules_full.json \
    <run dir>/probing/triage_report.scored.scored_full.json
```
