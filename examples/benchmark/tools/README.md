# `extract_figure_data.py`

Pull everything a case-study figure needs out of one benchmark run directory.

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
| `json` (default) | `.json` | for reading. Sectioned, nulls pruned, floats rounded, prompts split into lines |
| `jsonl` | `.jsonl` | for code. One record per line, every field verbatim, no post-processing |
| `md` | `.md` | the figure in prose. Character bars for the bar chart, p-values and other details dropped |
| `all` | all three | |

`both` = `json` + `jsonl` (kept for older invocations).

---

## What the JSON holds

```
headline            read this first: every number the figure prints
run                 model / dataset / split sizes / judge / repair-tier cap
m1_probe            analyzers (which families exist, which was selected) + signal_curve (the bar chart)
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

---

## `samples/`

The output of two real runs, for comparison:

| file | contents |
| --- | --- |
| `chartqa_vlm.json` / `.md` | VLM, ChartQA, 3 black-box analyzers, 1 surviving signal, 2 supported hypotheses, L2 repair 49.2% → 79.7% |
| `mmau_audio.json` / `.md` | Audio LLM, MMAU, 8 analyzers (white-box included), 8 surviving signals, no supported hypothesis, L2 repair 43.0% → 60.2% |

The two runs have very different shapes, which makes them a useful reference for
judging whether the tool behaved sensibly on your own logs.

---

## Adding a new analyzer

`ANALYZER_FAMILY` maps each analyzer to one of black-box / white-box /
multimodal, with `FAMILY_KEYWORDS` as a keyword fallback. When adding an
analyzer, add a row to the table. An unregistered analyzer does not crash the
tool — it is guessed from keywords in its name, and lands in `other` only when
that fails too.

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
