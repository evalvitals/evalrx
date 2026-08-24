# FIGURE_PROMPT — have Claude draw the case-study figure

After running `extract_figure_data.py`, hand Claude the generated `.json` (or `.md`)
together with this file, and it can draw a figure in the same style as
`reference/qualitative_vlm_L2.pdf`.

---

## How to use it

1. Generate the data:

```bash
python extract_figure_data.py ../vlm/qwen/outputs/qwen3.5-2b/chartqa -o mydata --format all
```

2. Open a new conversation and upload three files:

- `mydata.json`
- this `FIGURE_PROMPT.md`
- `reference/qualitative_vlm_L2.pdf` (the style reference)

3. Ask for it:

> Following the spec in FIGURE_PROMPT.md, draw a case-study figure from the
> numbers in mydata.json in the same style as the reference PDF. Output an SVG file.

`reference/casestudy_chartqa.svg` and `reference/casestudy_mmau.svg` are two finished
figures (drawn from the two runs in `samples/`) that can be handed over as templates too.

---

## The spec, for Claude

### Hard requirements

- Output **one SVG file**, `viewBox="0 0 1760 1010"`, white background, font
  `Helvetica Neue, Helvetica, Arial, sans-serif`.
- **Every number must come from the JSON.** A number that is not in the JSON does not
  go on the figure; leave a gap rather than inventing one.
- Check the result: every `rect` inside the canvas, no text crossing the inner edge of
  the card it belongs to.

### Layout

```
y  24– 76   top black pill, centred
y  92       connector line (cyan square at the left end, black square at the right)
y 108–152   two swimlanes: EXPLORE / DISCOVERY (pale blue), HELD-OUT / VALIDATION (pale grey)
y 168–648   five cards
y 668–900   bottom green frame: the accepted repair walked through one real case
```

The five cards sit at fixed x coordinates:

| card | x | width | border |
| --- | --- | --- | --- |
| M1 | 40 | 290 | thin grey + cyan top bar |
| M2 | 342 | 300 | thin grey + cyan top bar |
| M3 | 654 | 216 | thin grey + cyan top bar |
| M5 | 890 | 320 | **heavy black** (under the held-out swimlane) |
| M4 | 1222 | 498 | **blue** |

The swimlane boundary is near x=880: M1/M2/M3 are explore, M5/M4 are held-out.

### Palette

```
text primary #111111   secondary #4A4A4A   muted #8A8880   greyed out #B0AEA6
blue #1E7BE0   pale blue fill #EAF3FE   cyan #4DC3E8
green #1E7B34  pale green fill #E3F5E8
red #C0392B    pale red fill #FBE5E5     bar red #E05A4E
grey fill #F7F6F1  rule #EDEBE4  bar grey #B4B2A9
```

Green = accepted / passed its test; red = made things worse / refuted; grey = tried but
not selected.

### What goes on each card (fields are JSON paths)

**M1 — title from `pipeline[0].name`, subtitle from `pipeline[0].subtitle`**

From `m1_probe`:

1. Three family boxes (`analyzers.families`): `selected` true = pale blue fill, blue
   border, `SELECTED` pill; false = grey fill, grey text, `not used`. The box title is
   followed by that family's probe count.
2. A `WHAT THE PROBES ASK` section: list `probe_questions.questions[].question` one per
   line, each with a small blue bullet. Usually 5–9 lines.
3. A grey box at the bottom holding the funnel: one horizontal bar filled blue in the
   ratio `n_forwarded / n_measured`, under it
   `{n_measured} measurements  ▸  {n_forwarded} forwarded to M2`,
   then a small line spelling out the three `dropped` counts (saw the answer key /
   never varied / partial coverage), and finally `probe_questions.note`.

**M2 — `Statistical Screening`**

From `m2_statistics.heldout` (fall back to `explore` if there is no held-out pass):

1. A `Forest plot` sub-heading, then the forest plot: one row per entry in `tests` with
   `tool == "signal_label_assoc"` that is not `degenerate`, the −0.7…+0.7 axis mapped to
   240px inside the card, a dashed vertical rule at 0.
   `survives_correction` true = filled green dot + green interval line; otherwise a hollow
   grey dot + grey line.
2. Two small lines under the axis: `one row per candidate signal, extra failure rate when
   the signal is high`.
3. A `Statistical Tests` sub-heading with
   `{n survivors} of {family.n_in_correction_family} signals survives Benjamini–Hochberg (BH)`,
   then a grey pill carrying the strongest surviving signal's `effect`, `ci` and `p_value`.

**M3 — `Hypothesis Formation`**

From `m3_hypotheses.proposed`: a `THE THREE HYPOTHESES` sub-heading, then one blue-bordered,
pale-blue box per hypothesis. First line inside is `H{n} {short name}` (`failure_mode`
rewritten as a human-readable phrase), then two or three lines condensing `statement`.
**Do not paste `statement` verbatim** — it is too long; boil it down to one sentence.

**M5 — `Held-out Verification`**

From `m5_verdicts.verdicts`: one row each, a status pill on the left (`SUPPORTED` green on
green / `REFUTED` red on red / `INCONCLUSIVE` grey on grey) and the hypothesis's short name
on the right.

If `qa_flags` contains `repair_without_supported_hypothesis`, add an amber callout box
(fill `#FEF6E7`, border `#E0A33A`) in the lower half of this card saying that no mechanism
was confirmed and the gain below is an empirical fix.

**M4 — `Validated Repair`**

From `m4_repair_search`:

1. The upper half is the L1–L4 ladder: a round tier badge per rung, its name, and the
   outcome text right-aligned. Colour by `ladder[].status`: `accepted` green, `regressed`
   red, `not_selected` blue fill with grey text, `untouched` all grey. The text on the
   right carries `best_effect`.
2. The lower half is `{n} CANDIDATES TRIED ON EXPLORE`, drawing
   `candidates_selected_on_explore` as a horizontal bar chart: zero line left of centre,
   positive to the right and negative to the left. The selected bar is green and heavier,
   negatives are red. Axis ticks along the lines of `−0.05 0 +0.1 +0.2`.

**Bottom green frame — the accepted repair**

From `accepted_repair` + `example_case` + `heldout_validation`:

1. Title `The accepted repair — {name} ({tier} scaffold)`, subtitle carrying the
   `example_case` id, its question and its gold answer.
2. The input goes on the far left: for an image task, embed the image from
   `example_case.media_paths` (inline base64); for an audio task, draw the wav's amplitude
   envelope as vertical bars.
3. The middle chains `accepted_repair.steps` together with arrows, usually as 3–4 grey
   boxes, the last one (the answer) in a green frame. Title the boxes `1. Image operation`
   / `2. Forced read-out` and so on rather than copying the prompt text.
4. The far right says `Over the full held-out testing cases` with two horizontal bars
   comparing `baseline_rate` and `candidate_rate`, and one line underneath:
   `{n_fixed} fixed, {n_broken} broken, gain far beyond chance (p ≪ .001)`.

### The honesty rules, which are not optional

- When `example_case.split` is `explore`, the last box is *the output shape this pipeline
  demands*, not a measured result for that case. Do not write it up as "fixed".
- Every entry in `qa_flags` has to be honoured:
  - `repair_without_supported_hypothesis` → M5 must carry the callout box
  - `multiple_signals_survived` → M2 must say how many survived, never imply there was one
  - `accepted_fix_breaks_cases` → the bottom must print the broken count, not only fixed
  - `example_case_from_explore` → see the rule above
- Never print `p_value` as 0; below 0.001 write `p ≪ .001`.

### Spacing advice per card

M1 carries the most content; if it does not fit, squash the family boxes to a single line
(title + `SELECTED` pill only). If M3 has more than three hypotheses, shrink the type
rather than dropping a hypothesis. The M2 forest plot has exactly
`family.n_in_correction_family` rows, with the row spacing adapting to the card height.
