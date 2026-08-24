# `extract_figure_data.py`

Pull everything a case-study figure needs out of one benchmark run directory — and hand
the result to Claude to draw the figure.

The input is a run root — the directory holding `summary.json` and `logs/`, i.e.
exactly what [`_common/runner.py`](../_common/runner.py) writes to
`<modality>/<family>/outputs/<model>/<dataset>[.<tag>]/`. The output is the same
numbers in three shapes: a readable JSON document, a flat JSONL stream for
plotting code, and a Markdown write-up of the figure.

Standard library only, Python 3.8+. Nothing to install, nothing imported from
`evalvitals` — it reads the run artifacts and only them.

---

## Usage

```bash
python extract_figure_data.py <run-root>
```

```bash
RUN=../vlm/qwen/outputs/qwen3.5-2b/chartqa

# default: one readable nested JSON
python extract_figure_data.py $RUN

# choose the output stem (the extension is appended)
python extract_figure_data.py $RUN -o mydata

# just the Markdown write-up
python extract_figure_data.py $RUN -o report --format md

# all three
python extract_figure_data.py $RUN -o report --format all

# pin a specific case as the illustrative one
python extract_figure_data.py $RUN --example-case chartqa-human-1142

# pass a parent directory to process several runs at once
python extract_figure_data.py ../vlm/qwen/outputs/qwen3.5-2b -o all_runs
```

---

## The three outputs

| `--format` | file | what it is for |
| --- | --- | --- |
| `json` (default) | `.json` | for reading, and the input to the figure. Sectioned, nulls pruned, floats rounded, prompts split into lines |
| `jsonl` | `.jsonl` | for code. One record per line, every field verbatim, no post-processing |
| `md` | `.md` | the figure in prose. Character bars for the bar chart, p-values and other details dropped |
| `all` | all three | |

`both` = `json` + `jsonl` (kept for older invocations).

---

## Having Claude draw the figure

1. Run the extraction to get `mydata.json`.
2. Open a new conversation and upload `mydata.json` + [`FIGURE_PROMPT.md`](FIGURE_PROMPT.md)
   + `reference/qualitative_vlm_L2.pdf`.
3. Ask: **following the spec in FIGURE_PROMPT.md, draw a case-study figure from the
   numbers in mydata.json in the same style as the reference PDF, output an SVG.**

`FIGURE_PROMPT.md` pins the layout coordinates, the palette, which fields go on which
card, and a set of honesty rules — an example case drawn from explore may not be written
up as "fixed", every `qa_flags` entry has to be accounted for on the figure.

`reference/` holds two finished figures to hand over as templates:

| file | |
| --- | --- |
| `qualitative_vlm_L2.pdf` | the target style |
| `casestudy_chartqa.svg` | the VLM example |
| `casestudy_mmau.svg` | the audio example |

---

## What the JSON holds

```
pipeline            the five modules, with their names and subtitles
headline            read this first: every number the figure prints
run                 model / dataset / split sizes / judge / repair-tier cap
m1_probe            analyzers (which families exist, which was selected)
                    probe_questions (what the probes ask + the 43 ▸ 15 funnel)
                    measurement_inventory (every field, and why it was kept or dropped)
                    signal_curve (the bar chart)
m2_statistics       explore and heldout phases, each with a correction-family summary + per-test rows
m3_hypotheses       the proposed hypotheses + the adversarial critic's objections
m5_verdicts         the adjudication on the held-out split
m4_repair_search    the L1–L4 ladder + the candidate list (explore selection and heldout confirmation kept apart)
accepted_repair     the accepted fix: image ops, prompt, decoding settings, steps split out
heldout_validation  before/after accuracy + the paired 2x2
example_case        one illustrative case (question, gold, model output, media path)
qa_flags            automatic consistency checks
_sources            which file each section was read from
```

Every record carries a `source` pointing at the artifact it came from, so any
number in the figure can be traced back.

The module names are fixed, so every figure labels the pipeline the same way:

```
M1  Suspicious Behavior Detection   Run the analyzer probing library, find per-case suspicious behaviors
M2  Statistical Screening           Using plots to explain, statistical tests to decide
M3  Hypothesis Formation            Explore the reason behind the signals
M5  Held-out Verification           Confirm whether the hypothesis is verified over the held-out cases
M4  Validated Repair                Fix the failure with the validated repair ladder
```

---

## `m1_probe.probe_questions`

M1 as a reader sees it: not a list of field names, but **what the probes are asking**.

```json
{
  "n_analyzers": 3,
  "n_measured": 26,
  "n_forwarded": 9,
  "questions": [
    {"question": "Did it produce anything at all?", "analyzers": ["answer_extraction_audit"], ...},
    {"question": "Same question five times, same answer?", ...}
  ],
  "dropped": {"saw_the_answer_key": 11, "never_varied": 5, "partial_coverage": 4},
  "note": "each case is answered five times; the probes never see the answer key"
}
```

Questions are keyed by **field**, not by analyzer — one analyzer often asks two questions,
and one question is often answered by two analyzers. The table is `QUESTION_BY_FIELD` at
the top of the script, with `QUESTION_BY_ANALYZER` as the fallback for unregistered fields.

`n_forwarded` is the number of signals that entered the multiplicity-correction family: it
equals the row count of the M2 forest plot and the denominator of the BH correction.
`measurement_inventory` carries the per-field detail, each row stating why it was kept
(`candidate`) or dropped (`dropped_sees_answer_key`, `dropped_never_varies`,
`dropped_partial_coverage`).

---

## `qa_flags`

Each extraction self-checks and records what it finds in `qa_flags` (also listed
at the end of the Markdown). Currently reported:

- `shared_test_between_hypotheses` — two hypotheses adjudicated on the same test
  and the same effect, i.e. one finding wearing two hats
- `direction_mismatch` — the verdict's expected direction contradicts the sign
  actually observed
- `repair_without_supported_hypothesis` — a repair was accepted although no
  hypothesis reached `supported`
- `multiple_signals_survived` — more than one signal survived correction; the
  figure has to say which one it plots
- `correction_family_size_differs` — the multiplicity family is a different size
  in the two splits
- `signal_binned_for_plotting` — a continuous signal was quartile-binned, so the
  bars are bins, not raw levels
- `example_case_from_explore` — the illustrative case comes from the explore
  split; do not present it as a measured held-out result
- `accepted_fix_breaks_cases` — the fix broke cases that were previously correct

Read them before drawing, so a known problem does not end up printed in a paper.
`FIGURE_PROMPT.md` requires Claude to account for each one.

---

## `samples/`

The output of two real runs, for comparison:

| file | contents |
| --- | --- |
| `chartqa_vlm.json` / `.md` | VLM, ChartQA, 3 probes, 26 ▸ 9, 1 surviving signal, 2 supported hypotheses, L2 repair 49.2% → 79.7% |
| `mmau_audio.json` / `.md` | Audio LLM, MMAU, 8 probes (white-box included), 43 ▸ 15, 8 surviving signals, no supported hypothesis, L2 repair 43.0% → 60.2% |

The two runs have very different shapes, which makes them a useful reference for
judging whether the tool behaved sensibly on your own logs. The two SVGs in `reference/`
were drawn from exactly these two files.

---

## Adding a new analyzer

Three tables at the top of the script; add a row to each as needed, and nothing crashes
if you don't:

- `ANALYZER_FAMILY` sorts the analyzer into black-box / white-box / multimodal, with
  `FAMILY_KEYWORDS` as a keyword fallback
- `QUESTION_BY_FIELD` / `QUESTION_BY_ANALYZER` decide which plain-language question it
  shows up as in M1
- `OUTCOME_DERIVED` lists the field names that are a function of the answer key and must
  be excluded — testing them would let a signal predict the label from the label

---

## Known limits

- Only tested on single-cycle runs. Multi-cycle runs (`c1_`, `c2_` prefixes) are
  currently treated as separate phases rather than folded into "cycle 2".
- `example_case` can only be drawn from the explore split, because held-out
  cases store no media.
- The split is inferred, not read: explore = the cases with a `case_record` in
  the run log, held-out = the rest of the batch.

Contract tests live in
[`tests/test_examples/test_figure_data_extract.py`](../../../tests/test_examples/test_figure_data_extract.py).
