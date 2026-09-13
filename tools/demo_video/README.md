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
    --command "$INSTALL" --command "$LAUNCH" \
    --ui-page build/ui/overview.png --ui-url http://localhost:8520 \
    --serve-cmd "evalrx serve path/to/run --port 8520" \
    --serve-out "Serving dynamic diagnostic report at http://127.0.0.1:8520" \
    --end-card
```

`--ui-page FILE` is what turns the report into part of the story: the terminal
types the `evalrx serve` hand-off (`--serve-cmd`, `--serve-out`; the real
command and its real output line), then a browser window cross-fades in and
scrolls the full-page overview shot top to bottom. The scroll duration adapts
to the page height; `--ui-scroll-seconds` overrides it.

`--ui-scroll FILE` (repeatable, in order) chains more browser acts after the
overview: each capture is scrolled top to bottom, and the hand-off to the
next act is a cursor — it glides in, clicks the button that opens the next
page (the sidebar's next stage, the "See everything this step recorded" CTA
at the summary's bottom, or the footer index), and the click's target page
cross-fades in. Which Full record pages to include is an editorial call:
the shipped video deepens only M2 and M5, the two stages whose records
carry the story. The captures must be full-page shots made with
`shoot_ui --scrollset` (see below); the sidecar records where the sticky
stage-list pins and where the clickable buttons sit.

`--ui-shot FILE` (repeatable) appends further report-UI screenshots after
the browser acts — views that don't need scrolling, like the agent audit log.
Each is shown in the same browser window, cross-faded. Only pass shots of
**this** run's own report — a different run's UI behind this run's terminal
would be a claim neither of them made.

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

## Screenshots for the video's UI acts

The MP4's closing acts are the real report UI. Serve a finished run and shoot
it — `shoot_ui` uses the playwright-core the web build already carries, with a
system Chrome, so there is nothing to install:

```bash
evalrx serve path/to/run --port 8520 &
node tools/demo_video/shoot_ui --url http://localhost:8520 --out build/ui/overview.png
```

An exported `report.html` works too (no server needed). The video scrolls each
page like a real browser, so `--scrollset` is what the per-stage captures
want: it writes the full page plus a pinned-sidebar viewport reference and a
small JSON sidecar that the renderer reads to keep the stage-list stuck while
the rest scrolls:

```bash
for s in M1 M2 M3 M4 M5; do
  node tools/demo_video/shoot_ui --url "$URL" --view evidence --stage "$s" \
      --scrollset --height 626 --out "build/ui/evidence-${s}.png"
done
# deepen only the stages whose records carry the story — M2 and M5 here
for s in M2 M5; do
  node tools/demo_video/shoot_ui --url "$URL" --view evidence --stage "$s" --full \
      --scrollset --height 626 --out "build/ui/full-${s}.png"
done
node tools/demo_video/shoot_ui --url "$URL" --view cases --out build/ui/cases.png
node tools/demo_video/shoot_ui --url "$URL" --view debug --out build/ui/debug.png \
    --height 626 --viewport
```

`--view NAME` is overview (default), evidence, cases, debug. `--stage CODE`
on the evidence view opens that stage's page; `--full` clicks the "Full
record" tab. `--viewport` keeps it to the viewport; without it the capture
is the full page (what the scroll acts want). `--height 626` matters for
`--scrollset`: the sidecar's button positions are viewport coordinates, and
626 is the video's browser viewport.

The SVG deliberately stays terminal-only — an inline README image should not
carry a megabyte of screenshots, and `timeline.build`'s serve segment is
opt-in, which the SVG renderer never takes.

The SVG deliberately stays terminal-only — an inline README image should not
carry a megabyte of screenshots, and `timeline.build`'s serve segment is
opt-in, which the SVG renderer never takes.

## Requirements

`render_svg.py` needs nothing beyond the standard library. `render_mp4.py`
needs Pillow, NumPy and `ffmpeg` on `PATH`; fonts fall back to the DejaVu faces
matplotlib ships, which are the ones guaranteed to carry the narrator's
`✓ ✗ · ─` glyphs.
