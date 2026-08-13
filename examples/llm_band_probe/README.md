# LLM band probe — pick a dataset the model can actually be diagnosed on

Two scripts that answer, in order, the two questions that gate every text-model
arc in this repo:

1. **`band_locate.py`** — *which datasets is this model diagnosable on?*
2. **`probe_validate.py`** — *what do the reasoning probes say once it is?*

## Why a band, not a difficulty

M2 contrasts PASS against FAIL. A benchmark the model gets 95% on has almost no
FAIL mass; one it gets 3% on has almost no PASS mass. Either way the contrast is
underpowered no matter how good the probes are, and the resulting p-values
describe sampling noise. The usable range is roughly **30–70%**, and it is a
property of the *(model, dataset)* pair — a set that is a floor for an 8B model
can be saturated for a frontier one, so it has to be measured, not read off a
leaderboard.

`band_locate.py` therefore reports a Wilson interval alongside the point
estimate and classifies with the interval:

| band | condition | why it is unusable / usable |
|---|---|---|
| `saturated` | CI lower bound ≥ 0.70 | too few failures to attribute |
| `floor` | CI upper bound ≤ 0.30 | too few successes to contrast against |
| `USABLE` | accuracy in [0.30, 0.70] | both classes have mass |
| `marginal` | otherwise | the interval straddles a boundary — sample more |

It also reports `no_answer_tag_rate`. Thinking models overrun the token budget
before they emit the answer tag, and without that column a truncation-limited
score reads as a capability score — the same confound `termination_audit`
exists to control.

## Running

```bash
# a served OpenAI-compatible endpoint
export BAND_BASE_URL=http://127.0.0.1:8020/v1
export BAND_MODEL_ID=qwen3.5-9b

python band_locate.py --n 60 --concurrency 16 --out band_results.json
python band_locate.py --only gsm_symbolic_p2,bbh_tracking7 --n 100   # focused re-run
```

Rows are pulled through the HuggingFace datasets-server (`/rows`), so nothing is
downloaded locally, and the sample is drawn from **windows spread across the
whole split** rather than the head — several of these splits are ordered by
subset or difficulty, and a head sample would measure one slice while claiming
to measure the set.

Adapters carry the real field names, which are not guessable: MuSR ships
`choices` as the *string* repr of a list, OlympiadBench wraps `final_answer` in
a list of LaTeX, MuSiQue golds need `answer_aliases`, Bamboogle capitalises
`Question`/`Answer`, and ZebraLogic is graded on the full grid with every cell
**bound to its house** (a bare set of `attr=value` cells would pass a model that
found every value and assigned them all to the wrong houses).

## Then the probes

```bash
python probe_validate.py --dataset <a USABLE one> --n 40
```

`probe_validate.py` collects a real baseline run (so PASS/FAIL labels come from
the model, not from a fixture), then runs the probes **in reading order**:

```
answer_extraction_audit -> termination_audit   # hygiene: are the labels real?
arith_audit                                    # free: slip vs chain break
coverage_verification_gap, perturbation_battery,
self_repair, cot_faithfulness, self_consistency # interventional mechanisms
```

The order is not cosmetic. If the FAIL pool is partly parse failures or
truncations, every mechanism number below is measured on a contaminated pool and
M2 will happily attribute a harness bug to whichever mechanism is under test.

## Notes on specific sources

- `Putnam-AXIOM/putnam-axiom-dataset-ICML-2025` returns **401** through the
  datasets-server and is omitted, despite the paired originals↔variations design
  being the cleanest memorisation axis in the survey.
- LiveCodeBench `execution-v2` and CRUXEval are graded by **exact string** on the
  predicted return value — no sandbox, no execution, which makes them the
  cheapest code path here by a wide margin.
- `MathArena/arxivmath` is mined from that month's arXiv, so its
  contamination resistance is a construction guarantee rather than a hope.

## Sampling, honestly

`fetch_rows` draws `n_windows` (12) offsets spaced across the whole split,
jittered so they are not page-aligned, and takes an equal share from each.

Shuffling the ORDER of 100-row pages is not enough on its own — an earlier
version did exactly that and then stopped as soon as it had enough rows, which
took them all from whichever one or two pages came first. On MMLU-Pro (ordered
by category) that returned a single category; the current sampler returns eight.

What this is: a stratified cluster sample. What it is not: an iid draw. The
Wilson interval is therefore **approximate** — the design effect from clustering
is unmodelled, so treat the band boundaries as guidance, not as a test.
