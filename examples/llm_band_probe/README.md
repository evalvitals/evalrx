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

It also reports `truncated_rate`, taken from the endpoint's own `finish_reason`.
Thinking models overrun the token budget before they emit an answer, and without
that column a truncation-limited score reads as a capability score — the same
confound `termination_audit` exists to control. `no_answer_tag_rate` is the older
proxy for the same thing and is still reported, but it is only meaningful for
specs that actually ask for an `Answer:` tag; a dataset with its own response
format reads 100% tag-less on a run with no truncation at all.

## Do not sweep with greedy decoding

`--temperature` defaults to Qwen's documented thinking-mode sampling
(`0.6 / top_p 0.95 / top_k 20`), not to 0. This is not a style preference.
Measured on a single ZebraLogic 2\*2 item — a puzzle the model solves correctly
inside its chain within a few hundred tokens either way:

| decoding | tokens | `finish_reason` |
|---|---|---|
| `T=0.0` (greedy) | 16384 | **length** — never terminated |
| `T=0.7, top_p=0.95` | 7602 | stop |
| `T=1.0, top_p=0.95, top_k=20` | **1352** | stop |

Under greedy decoding the model finishes reasoning and then loops on
self-verification — *"I will ensure the answer format is exact."*, repeated
verbatim until the budget runs out. Scoring that as a failed puzzle measures the
decoding configuration, and it inflates cost by an order of magnitude at the same
time. `--greedy` restores the old behaviour if you want to reproduce it.

## Two format orders in one prompt is a bug

The first ZebraLogic sweep asked for `House 1: Name=..., Color=...` and then said
"use the attribute names from the puzzle". On a puzzle with no Color attribute
those orders contradict, and the model does not pick one — it oscillates:

> The attribute names are `Name` and `Car models`. Wait, I'll check if I can just
> use `Car`. Let's assume the user wants `Car`. Wait, I'll check the prompt
> again. …

…to the token cap, on a puzzle it had already solved. The format order is now
built per row from that puzzle's own header (`Use these attribute names verbatim:
Name, CarModel`), which is a formatting aid and not a hint — the header is column
names, never values.

The same applies in reverse: `Spec.append_instruction=False` for datasets that
ship their own response-format section. Enigmata tells the model to print a
fenced grid of numbers; appending "put the final answer on its own last line as
'Answer: `<answer>`'" is a second, contradictory order, and a model obeying
either one gets graded against the other.

## Grade the stated answer, not the chain

Specs with `grades_raw_output=True` grade only the region **after the last
`Answer:` marker** (ZebraLogic) or **inside the last fence** (Enigmata). Scanning
the whole generation looks harmless and is not: a 2\*2 puzzle has two possible
assignments and the chain enumerates both, so a subset match against the full
text passes whatever the model finally concluded. The score would approach 100%
while measuring nothing.

## Running

```bash
# a served OpenAI-compatible endpoint
export BAND_BASE_URL=http://127.0.0.1:8020/v1
export BAND_MODEL_ID=qwen3.5-9b

python band_locate.py --n 60 --concurrency 16 --out band_results.json
python band_locate.py --only gsm_symbolic_p2,bbh_tracking7 --n 100   # focused re-run
```

`--only` runs the specs **in the order given** and the results file is rewritten
after each one, so putting the rungs that bracket a ladder first (`t1,t3,t5,…`)
makes a long sweep readable — and abortable — before it finishes.

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

## Scanning a ladder instead of averaging it

Several of these datasets carry their own difficulty axis as a column, and
measuring the pooled split reports a number that describes no part of it.
`Spec.row_filter` selects one rung; `fetch_multiplier` and `n_windows` control
how hard the sampler works to find enough of it.

- **ZebraLogic** ships exactly 40 items for each of 25 grid sizes (verified over
  the full 1000-row split). A grid with H houses and A attributes has `(H!)^A`
  assignments, which spans **0.6 to 17.1 in log10** across those sizes — 2\*2 and
  6\*6 are not the same benchmark. `zebra_t1..t5` rank the sizes by that number
  and cut into fifths, giving five rungs of 200 rows.
- **Enigmata** ships 36 tasks in 7 types, most with 50 items each at `easy` /
  `medium` / `hard`. `enigmata_easy|medium|hard` scan the declared ladder;
  `enigmata_short` is the subset with scalar golds, where grader risk is lowest.

Both pooled specs are kept in the table, marked superseded, because they are what
produced the earlier (meaningless) pooled numbers.

Enigmata's gold formats were also measured across the full split rather than
assumed: 4 tasks (`binario`, `campsite`, `star_battle`, `zebra_logic`) ship
**multi-line** grid golds that a single-line extractor can never score above 0,
and ~20 more ship JSON matrices that a substring grader fails on a stray space.
`_grade_structural` compares parsed structure and reads the fenced whitespace
grid the dataset actually asks for; the multi-line and free-prose tasks are
listed in `_ENIGMATA_UNGRADED` so their exclusion is a decision on the record
rather than an oversight.

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

## PolyMATH and SuperGPQA (added after cross-validating against the Qwen3.5 card)

**PolyMATH** (`Qwen/PolyMath`) is two paired axes in one repo: config = language
(18), split = difficulty (`low`/`medium`/`high`/`top`), 125 problems each. The
same index in two languages is the *same problem with the same gold*
(`medium-en-0` and `medium-zh-0` both answer `$\frac{\pi}{3}$`), so a fixed
`--seed` keeps the language arms aligned item-for-item. Four specs are wired:
the English `medium`/`high`/`top` ladder plus a `zh medium` cross-language arm.
`low` is GSM8K-level and omitted as pre-saturated.

Its golds are mostly **not** plain numbers — measured over 100 rows per tier:

| tier | golds a numeric grader can judge |
|---|---|
| low | 100% |
| medium | 47% |
| high | 68% |
| top | 31% |

The rest are symbolic LaTeX (`\frac{\pi}{3}`, `\lfloor \log_2 n \rfloor + 1`),
so these specs use `_grade_latex`: numeric equality first, then a normalised
LaTeX surface form that collapses `\dfrac`/`\frac`, `\left`/`\right`, spacing
commands and braces. It is **not** a CAS — `0.5` will not match `\frac{1}{2}`.
A low score on `top` is therefore partly the grader, and the spec note says so.

**SuperGPQA** (`m-a-p/SuperGPQA`, 26,529 rows, 285 disciplines) is 10-option
multiple choice keyed by `answer_letter`, which indexes `options` correctly on
every row sampled. It carries `difficulty`, `discipline`, `field`, `subfield`
and `is_calculation`, so the full split can be sliced into sub-benchmarks
without leaving the dataset.
