#!/usr/bin/env python3
"""Render a storyboard as one self-contained animated SVG — or a still frame.

Why SVG and not a GIF: this sits at the top of the README, where an SVG stays
crisp at any width, animates inline on GitHub without a player, and weighs tens
of KB instead of several MB. Everything is CSS keyframes on a single shared
duration, so the loop never drifts and no JavaScript is needed (GitHub strips
it anyway).

``--at SECONDS`` renders the same frame statically instead — the poster image,
and what the MP4 renderer rasterises frame by frame, so the two cuts cannot
drift apart.

    python tools/demo_video/storyboard.py --report docs/demo/vlm.html \
        --out build/board.json
    python tools/demo_video/render_svg.py build/board.json \
        --out docs/assets/demo/evalrx-run.svg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from demo_video import theme as T
    from demo_video import timeline as TL
else:  # pragma: no cover - depends on how the tool is invoked
    from . import theme as T
    from . import timeline as TL

ESCAPE = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}


def esc(text: str) -> str:
    return "".join(ESCAPE.get(ch, ch) for ch in text)


class Anim:
    """Emits either a CSS-animated group or, for a still, the bare content.

    Holding both behind one object is what keeps the looping SVG and the
    rasterised frames honest about showing the same thing.
    """

    def __init__(self, total: float, *, loop: bool, still: float | None) -> None:
        self.total = total
        self.iteration = "infinite" if loop else "1"
        self.still = still
        self.blocks: list[str] = []
        self._n = 0

    # -- timing helpers ---------------------------------------------------
    def _pct(self, seconds: float) -> float:
        return max(0.0, min(100.0, seconds / self.total * 100.0))

    def _name(self, tag: str) -> str:
        self._n += 1
        return f"{tag}{self._n}"

    # -- wrappers ---------------------------------------------------------
    def fade(self, content: str, at: float, *, rise: float = 4.0,
             dur: float = 0.28) -> str:
        if self.still is not None:
            return content if self.still >= at else ""
        name = self._name("f")
        a, b = self._pct(at), self._pct(at + dur)
        self.blocks.append(
            f"@keyframes {name}{{"
            f"0%,{a:.3f}%{{opacity:0;transform:translateY({rise}px)}}"
            f"{b:.3f}%,100%{{opacity:1;transform:translateY(0)}}}}")
        return (f'<g style="animation:{name} {self.total:.2f}s linear '
                f'{self.iteration}">{content}</g>')

    def window(self, content: str, start: float, end: float, *,
               expires: bool = True) -> str:
        """Visible only between *start* and *end*.

        *expires* says whether this element is gone by the end of the loop; the
        reduced-motion stylesheet, which freezes everything on the final frame,
        drops the ones that are.
        """
        if self.still is not None:
            return content if start <= self.still < end else ""
        name = self._name("w")
        a, b = self._pct(start), self._pct(end)
        edge = min(b, a + 0.001)
        self.blocks.append(
            f"@keyframes {name}{{0%,{a:.3f}%{{opacity:0}}"
            f"{edge:.3f}%,{b:.3f}%{{opacity:1}}"
            f"{min(100.0, b + 0.001):.3f}%,100%{{opacity:0}}}}")
        base = ' class="expires" opacity="0"' if expires else ""
        return (f'<g{base} style="animation:{name} {self.total:.2f}s linear '
                f'{self.iteration}">{content}</g>')

    def typed(self, content: str, at: float, x: float, y: float,
              width: float) -> str:
        """Reveal *content* character by character from *x*."""
        chars = max(1, int(round(width / T.CH)))
        if self.still is not None:
            shown = (self.still - at) * T.TYPE_CPS
            if shown <= 0:
                return ""
            if shown >= chars:
                return content
            clip = self._name("c")
            return (f'<clipPath id="{clip}"><rect x="{x - 2:.1f}" '
                    f'y="{y - T.FONT_SIZE:.1f}" width="{shown * T.CH:.1f}" '
                    f'height="{T.LINE_H}"/></clipPath>'
                    f'<g clip-path="url(#{clip})">{content}</g>')
        name, clip = self._name("t"), self._name("c")
        a, b = self._pct(at), self._pct(at + chars / T.TYPE_CPS)
        self.blocks.append(
            f"@keyframes {name}{{0%,{a:.3f}%{{width:0}}"
            f"{b:.3f}%,100%{{width:{width:.1f}px}}}}")
        return (f'<clipPath id="{clip}"><rect x="{x - 2:.1f}" '
                f'y="{y - T.FONT_SIZE:.1f}" height="{T.LINE_H}" '
                f'width="{width:.1f}" '
                f'style="animation:{name} {self.total:.2f}s steps({chars}) '
                f'{self.iteration}"/></clipPath>'
                f'<g clip-path="url(#{clip})">{content}</g>')

    def caret_x(self, at: float, full_x: float, start_x: float) -> float:
        """Where the caret sits — riding the reveal edge while still typing."""
        if self.still is None:
            return full_x
        shown = max(0.0, (self.still - at) * T.TYPE_CPS)
        return min(full_x, start_x + shown * T.CH)

    def css(self) -> str:
        return "\n".join(self.blocks)


def _row_y(row: int) -> float:
    return T.TEXT_Y0 + row * T.LINE_H


def _spans_svg(spans: list[Any], x: float, y: float) -> tuple[str, float]:
    """Place coloured spans left to right by monospace advance."""
    out: list[str] = []
    cursor = x
    for text, fill, weight in spans:
        if text:
            out.append(f'<text x="{cursor:.1f}" y="{y:.1f}" fill="{fill}" '
                       f'font-weight="{weight}">{esc(text)}</text>')
        cursor += len(text) * T.CH
    return "".join(out), cursor


def render(board: dict[str, Any], *, commands: list[str], title: str | None = None,
           loop: bool = True, still: float | None = None) -> str:
    tl = TL.build(board, commands=commands)
    total = tl["total"]
    an = Anim(total, loop=loop, still=still)
    meta = tl["meta"]
    body: list[str] = []

    heading = title or f"evalrx · {meta.get('model')} × {meta.get('dataset')}"
    bar_mid = T.WIN_Y + T.BAR_H / 2

    # ---- backdrop and window chrome ---------------------------------------
    body.append(f'<rect width="{T.W}" height="{T.H}" fill="{T.BG}"/>')
    body.append(f'<rect width="{T.W}" height="{T.H}" fill="url(#glow)"/>')
    body.append(f'<rect x="{T.WIN_X}" y="{T.WIN_Y}" width="{T.WIN_W}" '
                f'height="{T.WIN_H}" rx="{T.WIN_R}" fill="{T.WIN_BG}" '
                f'stroke="{T.BORDER}"/>')
    body.append(f'<path d="M{T.WIN_X} {T.WIN_Y + T.WIN_R}a{T.WIN_R} {T.WIN_R} 0 0 1 '
                f'{T.WIN_R} -{T.WIN_R}h{T.WIN_W - 2 * T.WIN_R}a{T.WIN_R} {T.WIN_R} 0 0 1 '
                f'{T.WIN_R} {T.WIN_R}v{T.BAR_H - T.WIN_R}h-{T.WIN_W}z" fill="{T.WIN_BG2}"/>')
    body.append(f'<line x1="{T.WIN_X}" y1="{T.WIN_Y + T.BAR_H}" x2="{T.WIN_X + T.WIN_W}" '
                f'y2="{T.WIN_Y + T.BAR_H}" stroke="{T.BORDER}"/>')
    for i, colour in enumerate(T.LIGHTS):
        body.append(f'<circle cx="{T.WIN_X + 22 + i * 18}" cy="{bar_mid}" r="5.5" '
                    f'fill="{colour}"/>')
    body.append(f'<text class="chrome" x="{T.WIN_X + 82}" y="{bar_mid + 4.5}" '
                f'fill="{T.FG}">{esc(heading)}</text>')

    # ---- title bar right: M1-M5 pills, each lit when its stage first speaks
    first_seen: dict[str, float] = {}
    for line in tl["lines"]:
        stage = line.get("stage")
        if stage in T.STAGE_ORDER and stage not in first_seen:
            first_seen[stage] = line["t"]

    pill_w, pill_h, pill_gap = 34, 20, 7
    span = len(T.STAGE_ORDER) * pill_w + (len(T.STAGE_ORDER) - 1) * pill_gap
    pills_x = T.WIN_X + T.WIN_W - 20 - span
    for i, stage in enumerate(T.STAGE_ORDER):
        x = pills_x + i * (pill_w + pill_gap)
        colour = T.STAGE_COLOR[stage]
        body.append(f'<rect x="{x}" y="{bar_mid - pill_h / 2}" width="{pill_w}" '
                    f'height="{pill_h}" rx="5" fill="none" stroke="{T.GRID}"/>')
        body.append(f'<text class="pill" x="{x + pill_w / 2}" y="{bar_mid + 4}" '
                    f'text-anchor="middle" fill="{T.GRID}">{stage}</text>')
        at = first_seen.get(stage)
        if at is None:
            continue
        body.append(an.fade(
            f'<rect x="{x}" y="{bar_mid - pill_h / 2}" width="{pill_w}" '
            f'height="{pill_h}" rx="5" fill="{colour}" fill-opacity="0.14" '
            f'stroke="{colour}" stroke-opacity="0.55"/>'
            f'<text class="pill" x="{x + pill_w / 2}" y="{bar_mid + 4}" '
            f'text-anchor="middle" fill="{colour}">{stage}</text>',
            at, rise=0, dur=0.3))

    dot_x = pills_x - 104
    body.append(f'<circle class="pulse" cx="{dot_x}" cy="{bar_mid}" r="3.5" '
                f'fill="{T.GREEN}"/>')
    body.append(f'<text class="chrome" x="{dot_x + 11}" y="{bar_mid + 4}" '
                f'fill="{T.DIM}">REAL RUN</text>')

    # ---- output lines -----------------------------------------------------
    body.append('<g class="mono">')
    for index, line in enumerate(tl["lines"]):
        y = _row_y(line["row"])
        next_t = tl["lines"][index + 1]["t"] if index + 1 < len(tl["lines"]) else total
        spans_svg, end_x = _spans_svg(line["spans"], T.TEXT_X, y)
        if line.get("typing") is not None:
            width = sum(len(text) for text, _, _ in line["spans"]) * T.CH
            body.append(an.typed(spans_svg, line["t"], T.TEXT_X, y, width))
            # The caret belongs to the line being typed and leaves when the
            # next line lands, or every past prompt keeps one.
            caret = an.caret_x(line["t"], end_x, T.TEXT_X)
            body.append(an.window(
                f'<rect class="caret" x="{caret + 2:.1f}" '
                f'y="{y - T.FONT_SIZE + 2:.1f}" width="{T.CH:.1f}" '
                f'height="{T.FONT_SIZE}" fill="{T.BLUE}"/>', line["t"], next_t))
            continue
        body.append(an.fade(spans_svg, line["t"]))
        elapsed = line.get("elapsed_sec")
        if elapsed is not None:
            # The real wall clock at that point in the run, parked in the right
            # gutter: the loop took hours, the replay takes seconds.
            body.append(an.window(
                f'<text class="stamp" x="{T.WIN_X + T.WIN_W - 22}" y="{y:.1f}" '
                f'text-anchor="end" fill="{T.DIM}">{esc(T.clock(elapsed))}</text>',
                line["t"], total, expires=False))
    body.append("</g>")

    # ---- result strip -----------------------------------------------------
    card = tl["card"]
    card_y = T.WIN_Y + T.WIN_H - T.FOOT_H - 104
    tile_w = (T.WIN_W - 44) / max(1, len(card["tiles"]))
    tiles: list[str] = []
    for i, tile in enumerate(card["tiles"]):
        x = T.WIN_X + 22 + i * tile_w
        tiles.append(
            f'<text class="tilev" x="{x:.1f}" y="{card_y + 44}" fill="{T.BRIGHT}">'
            f'{esc(tile["value"])}</text>'
            f'<text class="tilel" x="{x:.1f}" y="{card_y + 66}" fill="{T.DIM}">'
            f'{esc(tile["label"])}</text>')
        if i:
            tiles.append(f'<line x1="{x - 16:.1f}" y1="{card_y + 18}" '
                         f'x2="{x - 16:.1f}" y2="{card_y + 72}" stroke="{T.BORDER}"/>')
    verdict_text, verdict_fill = card["verdict"]
    body.append(an.fade(
        f'<line x1="{T.WIN_X + 22}" y1="{card_y}" x2="{T.WIN_X + T.WIN_W - 22}" '
        f'y2="{card_y}" stroke="{T.BORDER}"/>{"".join(tiles)}'
        f'<text class="verdict" x="{T.WIN_X + 22}" y="{card_y + 94}" '
        f'fill="{verdict_fill}">{esc(verdict_text)}</text>',
        card["t"], rise=10, dur=T.CARD_IN))

    # ---- status bar -------------------------------------------------------
    foot_y = T.WIN_Y + T.WIN_H - T.FOOT_H
    body.append(f'<line x1="{T.WIN_X}" y1="{foot_y}" x2="{T.WIN_X + T.WIN_W}" '
                f'y2="{foot_y}" stroke="{T.BORDER}"/>')
    body.append(f'<text class="chrome" x="{T.WIN_X + 22}" y="{foot_y + 24}" '
                f'fill="{T.DIM}">'
                f'{esc(meta.get("dataset_full") or meta.get("dataset") or "")}'
                f'  ·  n={meta.get("n_cases")}  ·  held-out confirmation</text>')
    right = T.WIN_X + T.WIN_W - 22
    body.append(an.window(
        f'<text class="chrome" x="{right}" y="{foot_y + 24}" text-anchor="end" '
        f'fill="{T.CYAN}">DIAGNOSING…</text>', 0.0, card["t"]))
    body.append(an.fade(
        f'<text class="chrome" x="{right}" y="{foot_y + 24}" text-anchor="end" '
        f'fill="{verdict_fill}">RUN COMPLETE</text>', card["t"], rise=0, dur=0.3))

    style = f"""
    text{{font-family:{T.MONO};font-size:{T.FONT_SIZE}px;white-space:pre}}
    .chrome{{font-family:{T.SANS};font-size:11.5px;letter-spacing:.12em;
             font-weight:600;text-transform:uppercase}}
    .pill{{font-family:{T.SANS};font-size:10.5px;font-weight:700;letter-spacing:.08em}}
    .stamp{{font-size:11px;letter-spacing:.04em}}
    .tilev{{font-family:{T.SANS};font-size:34px;font-weight:700;letter-spacing:-.01em}}
    .tilel{{font-family:{T.SANS};font-size:12px;font-weight:500;letter-spacing:.04em}}
    .verdict{{font-family:{T.SANS};font-size:12.5px;font-weight:700;letter-spacing:.14em}}
    @media (prefers-reduced-motion: reduce){{
      g[style*=animation]{{opacity:1!important;animation:none!important}}
      .expires{{display:none}}
      .caret,.pulse{{animation:none!important}}
    }}
    .caret{{animation:blink 1s steps(2) infinite}}
    .pulse{{animation:pulse 2.2s ease-in-out infinite}}
    @keyframes blink{{50%{{opacity:0}}}}
    @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.25}}}}
    {an.css()}
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {T.W} {T.H}" '
        f'width="{T.W}" height="{T.H}" role="img" '
        f'aria-label="{esc(heading)} — a real EvalRX M1-M5 run, replayed">'
        f"<defs><radialGradient id='glow' cx='50%' cy='0%' r='85%'>"
        f"<stop offset='0%' stop-color='#0d2a26'/>"
        f"<stop offset='100%' stop-color='{T.BG}'/></radialGradient></defs>"
        f"<style>{style}</style>{''.join(body)}</svg>\n")


def default_commands(meta: dict[str, Any]) -> list[str]:
    return [
        'pip install "evalrx[ui,viz,stats]"',
        f'python run.py --model {meta.get("model")} '
        f'--dataset {str(meta.get("dataset", "")).lower()} --held-out',
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storyboard", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--command", action="append", default=None,
                    help="shell line typed before the output (repeatable)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--at", type=float, default=None,
                    help="render a still frame at this second instead of the loop")
    ap.add_argument("--no-loop", action="store_true")
    args = ap.parse_args()

    board = json.loads(args.storyboard.read_text())
    commands = args.command or default_commands(board["meta"])
    svg = render(board, commands=commands, title=args.title,
                 loop=not args.no_loop, still=args.at)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)
    print(f"{args.out}  ({len(svg) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
