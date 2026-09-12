# Demo video

The README's terminal demo, and the MP4 behind it. Two renderers, one cut:
the looping SVG on the README and the video for slides/social read the same
storyboard through the same `timeline.build`, so they cannot drift.

Nothing is simulated. Every line, count, verdict and duration comes out of a
real run — the storyboard extractor's whole job is to refuse to invent numbers.

```
storyboard.py   real run  →  storyboard.json   (what happened, and when)
timeline.py     storyboard →  timed, coloured lines   (the cut)
theme.py        geometry, palette, pacing              (shared by both renderers)
render_svg.py   → docs/assets/demo/evalrx-run.svg      (README, animated or still)
render_mp4.py   → docs/assets/demo/evalrx-run.mp4      (+ optional GIF)
```

## Re-render what is committed

The README's hero is the Gemma-4-E2B × MMAU run; `chartqa-qwen3.5-2b.json` is
the same pipeline on a run whose repair did **not** clear the bar, kept for
when that is the story worth telling.

```bash
RUN=tools/demo_video/storyboards/mmau-gemma-4-e2b.json
INSTALL='pip install "evalrx[ui,viz,stats]"'
LAUNCH='python run.py --modality alm --model gemma-4-e2b --dataset mmau --held-out'

python tools/demo_video/render_svg.py "$RUN" \
    --out docs/assets/demo/evalrx-run.svg --command "$INSTALL" --command "$LAUNCH"

python tools/demo_video/render_svg.py "$RUN" --at 22 \
    --out docs/assets/demo/evalrx-run-poster.svg \
    --command "$INSTALL" --command "$LAUNCH"

python tools/demo_video/render_mp4.py "$RUN" \
    --out docs/assets/demo/evalrx-run.mp4 \
    --command "$INSTALL" --command "$LAUNCH" --end-card
```

`--ui-shot FILE` (repeatable) appends report-UI screenshots before the title
card. Only pass shots of **this** run's own report — a different run's UI
behind this run's terminal would be a claim neither of them made.

`--at SECONDS` on `render_svg.py` writes a still frame instead of the loop —
that is the poster image, and the same code path the MP4 rasterises.

## Point it at a different run

A live run directory is the preferred input: the narration then comes from the
real `LoopNarrator` fed the logged records, exactly like
`examples/benchmark/tools/replay_narrate.py`.

```bash
python tools/demo_video/storyboard.py --run-dir path/to/outputs/<model>/<dataset> \
    --out tools/demo_video/storyboards/<name>.json
```

With no run directory at hand, an exported report page works too — it embeds
the event stream, so the storyboard is still that run's own numbers, with stage
durations recomputed from event timestamps:

```bash
python tools/demo_video/storyboard.py --report docs/demo/vlm.html \
    --out tools/demo_video/storyboards/<name>.json
```

Third, a saved terminal transcript, when only the console output survives —
lines are parsed straight back into beats:

```bash
python tools/demo_video/storyboard.py \
    --transcript tools/demo_video/storyboards/mmau-gemma-4-e2b.txt \
    --out tools/demo_video/storyboards/mmau-gemma-4-e2b.json
```

Whichever path produced it, the storyboard records which one in
`meta.provenance`, and the committed JSON is small enough to review in a diff.

## Screenshots for the video's closing act

The MP4's last act is the real report UI. Serve a finished run and capture it:

```bash
evalrx serve path/to/run --port 8520
# then screenshot the overview and the M1-M5 flowboard into build/ui/
```

The SVG deliberately stays terminal-only — an inline README image should not
carry a megabyte of screenshots.

## Requirements

`render_svg.py` needs nothing beyond the standard library. `render_mp4.py`
needs Pillow, NumPy and `ffmpeg` on `PATH`; fonts fall back to the DejaVu faces
matplotlib ships, which are the ones guaranteed to carry the narrator's
`✓ ✗ · ─` glyphs.
