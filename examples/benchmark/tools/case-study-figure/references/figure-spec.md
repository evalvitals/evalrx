# Figure spec — layout, palette, and where every number comes from

The drawing half of the `case-study-figure` skill. Fields below are paths into the
`figure_data.json` that `extract_figure_data.py` produced.

## Canvas

One SVG, `viewBox="0 0 1760 1010"`, white background, font
`Helvetica Neue, Helvetica, Arial, sans-serif`.

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

Card titles come from `pipeline[]` — `name` as the title, `subtitle` under it — so
every figure labels the pipeline the same way.

## Palette

```
text primary #111111   secondary #4A4A4A   muted #8A8880   greyed out #B0AEA6
blue #1E7BE0   pale blue fill #EAF3FE   cyan #4DC3E8
green #1E7B34  pale green fill #E3F5E8
red #C0392B    pale red fill #FBE5E5     bar red #E05A4E
grey fill #F7F6F1  rule #EDEBE4  bar grey #B4B2A9
amber callout fill #FEF6E7  amber border #E0A33A
```

Green = accepted / passed its test; red = made things worse / refuted; grey = tried
but not selected.

## M1 · Suspicious Behavior Detection

From `m1_probe`:

1. Three family boxes (`analyzers.families`): `selected` true = pale blue fill, blue
   border, `SELECTED` pill; false = grey fill, grey text, `not used`. The box title is
   followed by that family's probe count.
2. A `WHAT THE PROBES ASK` section: list `probe_questions.questions[].question` one
   per line, each with a small blue bullet. Usually 5–9 lines.
3. A grey box at the bottom holding the funnel: one horizontal bar filled blue in the
   ratio `n_forwarded / n_measured`, under it
   `{n_measured} measurements  ▸  {n_forwarded} forwarded to M2`,
   then a small line spelling out the three `dropped` counts (saw the answer key /
   never varied / partial coverage), and finally `probe_questions.note`.

If the card overflows, squash the family boxes to a single line (title + `SELECTED`
pill only) rather than dropping the funnel.

## M2 · Statistical Screening

From `m2_statistics.heldout` (fall back to `explore` if there is no held-out pass):

1. A `Forest plot` sub-heading, then the forest plot: one row per entry in `tests`
   with `tool == "signal_label_assoc"` that is not `degenerate`, the −0.7…+0.7 axis
   mapped to 240px inside the card, a dashed vertical rule at 0.
   `survives_correction` true = filled green dot + green interval line; otherwise a
   hollow grey dot + grey line.
2. Two small lines under the axis: `one row per candidate signal, extra failure rate
   when the signal is high`.
3. A `Statistical Tests` sub-heading with
   `{n survivors} of {family.n_in_correction_family} signals survives Benjamini–Hochberg (BH)`,
   then a grey pill carrying the strongest surviving signal's `effect`, `ci` and
   `p_value`.

The forest plot has exactly `family.n_in_correction_family` rows — the same number as
M1's `n_forwarded` — with row spacing adapting to the card height.

## M3 · Hypothesis Formation

From `m3_hypotheses.proposed`: a `THE THREE HYPOTHESES` sub-heading, then one
blue-bordered, pale-blue box per hypothesis. First line inside is `H{n} {short name}`
(`failure_mode` rewritten as a human-readable phrase), then two or three lines
condensing `statement`. More than three hypotheses: shrink the type, never drop one.

## M5 · Held-out Verification

From `m5_verdicts.verdicts`: one row each, a status pill on the left (`SUPPORTED`
green on green / `REFUTED` red on red / `INCONCLUSIVE` grey on grey) and the
hypothesis's short name on the right.

## M4 · Validated Repair

From `m4_repair_search`:

1. The upper half is the L1–L4 ladder: a round tier badge per rung, its name, and the
   outcome text right-aligned. Colour by `ladder[].status`: `accepted` green,
   `regressed` red, `not_selected` blue fill with grey text, `untouched` all grey. The
   text on the right carries `best_effect`.
2. The lower half is `{n} CANDIDATES TRIED ON EXPLORE`, drawing
   `candidates_selected_on_explore` as a horizontal bar chart: zero line left of
   centre, positive to the right and negative to the left. The selected bar is green
   and heavier, negatives are red. Axis ticks along the lines of `−0.05 0 +0.1 +0.2`.

## Bottom frame — the accepted repair

From `accepted_repair` + `example_case` + `heldout_validation`:

1. Title `The accepted repair — {name} ({tier} scaffold)`, subtitle carrying the
   `example_case` id, its question and its gold answer.
2. The input goes on the far left: for an image task, embed the image from
   `example_case.media_paths` (inline base64); for an audio task, draw the wav's
   amplitude envelope as vertical bars.
3. The middle chains `accepted_repair.steps` together with arrows, usually as 3–4 grey
   boxes, the last one (the answer) in a green frame. Title the boxes
   `1. Image operation` / `2. Forced read-out` and so on rather than copying the
   prompt text.
4. The far right says `Over the full held-out testing cases` with two horizontal bars
   comparing `baseline_rate` and `candidate_rate`, and one line underneath:
   `{n_fixed} fixed, {n_broken} broken, gain far beyond chance (p ≪ .001)`.

## Templates

`casestudy_chartqa.svg` and `casestudy_mmau.svg` were drawn from
`../../samples/chartqa_vlm.json` and `../../samples/mmau_audio.json`. Read a template
beside the JSON that produced it to see how a given field landed on the canvas.
`qualitative_vlm_L2.pdf` is the target style.
