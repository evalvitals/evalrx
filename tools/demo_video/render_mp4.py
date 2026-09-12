#!/usr/bin/env python3
"""Rasterise a storyboard into an MP4 (and optionally a GIF).

Same storyboard, same :mod:`theme` geometry and the same
:func:`timeline.build` cut as the README's animated SVG — this renderer only
differs in that it draws with Pillow and hands the frames to ffmpeg, so the
video can carry things an inline SVG should not: real screenshots of the report
UI after the run, and a closing title.

    python tools/demo_video/render_mp4.py build/board.json \
        --out build/evalrx-run.mp4 --ui-shot build/ui/*.png

Needs Pillow and ffmpeg on PATH. Fonts: any monospace TTF found on the box,
falling back to the DejaVu faces matplotlib ships, so a render is reproducible
on a machine with no system fonts to speak of.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from demo_video import theme as T
    from demo_video import timeline as TL
    from demo_video.render_svg import default_commands
else:  # pragma: no cover - depends on how the tool is invoked
    from . import theme as T
    from . import timeline as TL
    from .render_svg import default_commands

MONO_CANDIDATES = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)
SANS_CANDIDATES = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _matplotlib_font(name: str) -> str | None:
    try:
        import matplotlib
    except ImportError:
        return None
    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / name
    return str(path) if path.exists() else None


def _font(kind: str, size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    if kind == "mono":
        fallback = _matplotlib_font("DejaVuSansMono-Bold.ttf" if bold
                                    else "DejaVuSansMono.ttf")
        candidates = MONO_CANDIDATES
    else:
        fallback = _matplotlib_font("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        candidates = SANS_CANDIDATES
    # DejaVu first when available: it is the one face guaranteed to carry the
    # narrator's ✓/✗/·/─ glyphs, and a missing glyph is a visible hole.
    for path in ([fallback] if fallback else []) + list(candidates):
        try:
            return ImageFont.truetype(path, int(round(size)))
        except OSError:
            continue
    return ImageFont.load_default()


def _glow(*, centre: float, spread: float, strength: float) -> Image.Image:
    """A dithered vertical glow — a line-by-line ramp bands on a dark ground."""
    y = np.arange(T.H, dtype=np.float64)[:, None]
    falloff = np.exp(-(((y - centre * T.H) / (spread * T.H)) ** 2))
    top, ground = np.array(_rgb("#0d2a26"), float), np.array(_rgb(T.BG), float)
    ramp = ground + (top - ground) * (strength * falloff)
    frame = np.repeat(ramp[:, None, :], T.W, axis=1)
    rng = np.random.default_rng(7)          # fixed seed: renders stay identical
    frame += rng.uniform(-0.5, 0.5, frame.shape)
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")


def _rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _mix(colour: str, ground: str, alpha: float) -> tuple[int, int, int]:
    """*colour* over *ground* at *alpha* — Pillow has no per-draw opacity."""
    a, b = _rgb(colour), _rgb(ground)
    alpha = max(0.0, min(1.0, alpha))
    return tuple(int(round(b[i] + (a[i] - b[i]) * alpha)) for i in range(3))  # type: ignore


class Renderer:
    def __init__(self, tl: dict[str, Any], *, heading: str) -> None:
        self.tl = tl
        self.heading = heading
        self.mono = _font("mono", T.FONT_SIZE)
        self.mono_bold = _font("mono", T.FONT_SIZE, bold=True)
        self.chrome = _font("sans", 11.5, bold=True)
        self.pill = _font("sans", 10.5, bold=True)
        self.stamp = _font("mono", 11)
        self.tile_value = _font("sans", 34, bold=True)
        self.tile_label = _font("sans", 12)
        self.verdict = _font("sans", 12.5, bold=True)
        self.backdrop = _glow(centre=-0.05, spread=0.42, strength=0.55)
        self.first_seen: dict[str, float] = {}
        for line in tl["lines"]:
            stage = line.get("stage")
            if stage in T.STAGE_ORDER and stage not in self.first_seen:
                self.first_seen[stage] = line["t"]

    # -- chrome -----------------------------------------------------------
    def _window(self, d: ImageDraw.ImageDraw, t: float) -> None:
        d.rounded_rectangle([T.WIN_X, T.WIN_Y, T.WIN_X + T.WIN_W, T.WIN_Y + T.WIN_H],
                            radius=T.WIN_R, fill=_rgb(T.WIN_BG), outline=_rgb(T.BORDER))
        d.rounded_rectangle([T.WIN_X, T.WIN_Y, T.WIN_X + T.WIN_W, T.WIN_Y + T.BAR_H],
                            radius=T.WIN_R, fill=_rgb(T.WIN_BG2))
        d.rectangle([T.WIN_X, T.WIN_Y + T.BAR_H - T.WIN_R,
                     T.WIN_X + T.WIN_W, T.WIN_Y + T.BAR_H], fill=_rgb(T.WIN_BG2))
        d.line([T.WIN_X, T.WIN_Y + T.BAR_H, T.WIN_X + T.WIN_W, T.WIN_Y + T.BAR_H],
               fill=_rgb(T.BORDER))
        mid = T.WIN_Y + T.BAR_H / 2
        for i, colour in enumerate(T.LIGHTS):
            cx = T.WIN_X + 22 + i * 18
            d.ellipse([cx - 5.5, mid - 5.5, cx + 5.5, mid + 5.5], fill=_rgb(colour))
        d.text((T.WIN_X + 82, mid), self.heading.upper(), font=self.chrome,
               fill=_rgb(T.FG), anchor="lm")

        pill_w, pill_h, gap = 34, 20, 7
        span = len(T.STAGE_ORDER) * pill_w + (len(T.STAGE_ORDER) - 1) * gap
        pills_x = T.WIN_X + T.WIN_W - 20 - span
        for i, stage in enumerate(T.STAGE_ORDER):
            x = pills_x + i * (pill_w + gap)
            box = [x, mid - pill_h / 2, x + pill_w, mid + pill_h / 2]
            at = self.first_seen.get(stage)
            lit = at is not None and t >= at
            alpha = min(1.0, (t - at) / 0.3) if lit else 0.0
            colour = T.STAGE_COLOR[stage]
            if lit:
                d.rounded_rectangle(box, radius=5,
                                    fill=_mix(colour, T.WIN_BG2, 0.14 * alpha),
                                    outline=_mix(colour, T.WIN_BG2, 0.55 * alpha))
                fill = _mix(colour, T.WIN_BG2, alpha)
            else:
                d.rounded_rectangle(box, radius=5, outline=_rgb(T.GRID))
                fill = _rgb(T.GRID)
            d.text((x + pill_w / 2, mid + 0.5), stage, font=self.pill, fill=fill,
                   anchor="mm")
        dot_x = pills_x - 104
        pulse = 0.6 + 0.4 * abs(((t / 1.1) % 2.0) - 1.0)
        d.ellipse([dot_x - 3.5, mid - 3.5, dot_x + 3.5, mid + 3.5],
                  fill=_mix(T.GREEN, T.WIN_BG2, pulse))
        d.text((dot_x + 11, mid), "REAL RUN", font=self.chrome, fill=_rgb(T.DIM),
               anchor="lm")

    def _status(self, d: ImageDraw.ImageDraw, t: float) -> None:
        foot = T.WIN_Y + T.WIN_H - T.FOOT_H
        d.line([T.WIN_X, foot, T.WIN_X + T.WIN_W, foot], fill=_rgb(T.BORDER))
        meta = self.tl["meta"]
        left = (f'{meta.get("dataset_full") or meta.get("dataset")}'
                f'  ·  n={meta.get("n_cases")}  ·  held-out confirmation')
        d.text((T.WIN_X + 22, foot + T.FOOT_H / 2), left.upper(), font=self.chrome,
               fill=_rgb(T.DIM), anchor="lm")
        card = self.tl["card"]
        right_x = T.WIN_X + T.WIN_W - 22
        if t < card["t"]:
            d.text((right_x, foot + T.FOOT_H / 2), "DIAGNOSING…", font=self.chrome,
                   fill=_rgb(T.CYAN), anchor="rm")
        else:
            alpha = min(1.0, (t - card["t"]) / 0.3)
            d.text((right_x, foot + T.FOOT_H / 2), "RUN COMPLETE", font=self.chrome,
                   fill=_mix(card["verdict"][1], T.WIN_BG, alpha), anchor="rm")

    # -- body -------------------------------------------------------------
    def _lines(self, d: ImageDraw.ImageDraw, t: float) -> None:
        for index, line in enumerate(self.tl["lines"]):
            if t < line["t"]:
                break
            y = T.TEXT_Y0 + line["row"] * T.LINE_H
            next_t = (self.tl["lines"][index + 1]["t"]
                      if index + 1 < len(self.tl["lines"]) else self.tl["total"])
            if line.get("typing") is not None:
                shown = int((t - line["t"]) * T.TYPE_CPS)
                x = T.TEXT_X
                budget = shown
                for text, fill, weight in line["spans"]:
                    if budget <= 0:
                        break
                    piece = text[:budget]
                    d.text((x, y), piece, fill=_rgb(fill), anchor="ls",
                           font=self.mono_bold if int(weight) >= 500 else self.mono)
                    x += len(text) * T.CH
                    budget -= len(text)
                if t < next_t and int(t * 2) % 2 == 0:
                    caret = min(T.TEXT_X + shown * T.CH,
                                T.TEXT_X + sum(len(s[0]) for s in line["spans"]) * T.CH)
                    d.rectangle([caret + 2, y - T.FONT_SIZE + 2,
                                 caret + 2 + T.CH, y + 2], fill=_rgb(T.BLUE))
                continue
            alpha = min(1.0, (t - line["t"]) / 0.28)
            rise = (1.0 - alpha) * 4.0
            x = T.TEXT_X
            for text, fill, weight in line["spans"]:
                if text:
                    d.text((x, y + rise), text,
                           font=self.mono_bold if int(weight) >= 500 else self.mono,
                           fill=_mix(fill, T.WIN_BG, alpha), anchor="ls")
                x += len(text) * T.CH
            elapsed = line.get("elapsed_sec")
            if elapsed is not None:
                d.text((T.WIN_X + T.WIN_W - 22, y + rise), T.clock(elapsed),
                       font=self.stamp, fill=_mix(T.DIM, T.WIN_BG, alpha), anchor="rs")

    def _card(self, d: ImageDraw.ImageDraw, t: float) -> None:
        card = self.tl["card"]
        if t < card["t"]:
            return
        alpha = min(1.0, (t - card["t"]) / T.CARD_IN)
        y0 = T.WIN_Y + T.WIN_H - T.FOOT_H - 104 + (1.0 - alpha) * 10
        d.line([T.WIN_X + 22, y0, T.WIN_X + T.WIN_W - 22, y0],
               fill=_mix(T.BORDER, T.WIN_BG, alpha))
        tile_w = (T.WIN_W - 44) / 3
        for i, tile in enumerate(card["tiles"]):
            x = T.WIN_X + 22 + i * tile_w
            d.text((x, y0 + 44), tile["value"], font=self.tile_value,
                   fill=_mix(T.BRIGHT, T.WIN_BG, alpha), anchor="ls")
            d.text((x, y0 + 66), tile["label"], font=self.tile_label,
                   fill=_mix(T.DIM, T.WIN_BG, alpha), anchor="ls")
            if i:
                d.line([x - 16, y0 + 18, x - 16, y0 + 72],
                       fill=_mix(T.BORDER, T.WIN_BG, alpha))
        text, colour = card["verdict"]
        d.text((T.WIN_X + 22, y0 + 94), text, font=self.verdict,
               fill=_mix(colour, T.WIN_BG, alpha), anchor="ls")

    def frame(self, t: float) -> Image.Image:
        img = self.backdrop.copy()
        d = ImageDraw.Draw(img)
        self._window(d, t)
        self._lines(d, t)
        self._card(d, t)
        self._status(d, t)
        return img


def _ui_act(shots: list[Path], *, seconds_each: float, fps: int,
            fade: float = 0.35) -> Iterable[Image.Image]:
    """Real report-UI screenshots, cross-faded, as the closing act."""
    prepared = []
    for shot in shots:
        image = Image.open(shot).convert("RGB")
        canvas = Image.new("RGB", (T.W, T.H), _rgb(T.BG))
        scale = min(T.W / image.width, T.H / image.height)
        resized = image.resize((int(image.width * scale), int(image.height * scale)),
                               Image.LANCZOS)
        canvas.paste(resized, ((T.W - resized.width) // 2,
                               (T.H - resized.height) // 2))
        prepared.append(canvas)
    for index, canvas in enumerate(prepared):
        for frame_no in range(int(seconds_each * fps)):
            t = frame_no / fps
            if index and t < fade:
                yield Image.blend(prepared[index - 1], canvas, t / fade)
            else:
                yield canvas


def _end_card(*, seconds: float, fps: int, fade: float = 0.5) -> Iterable[Image.Image]:
    """Closing title: what it is, and the one line needed to try it."""
    base = _glow(centre=0.34, spread=0.5, strength=0.5)
    d = ImageDraw.Draw(base)
    title = _font("sans", 66, bold=True)
    lede = _font("sans", 22)
    code = _font("mono", 21)
    small = _font("sans", 14)
    cx = T.W / 2
    d.text((cx, 250), "EvalRX", font=title, fill=_rgb(T.BRIGHT), anchor="mm")
    d.text((cx, 320), "Your eval tells you what failed.",
           font=lede, fill=_rgb(T.FG), anchor="mm")
    d.text((cx, 352), "EvalRX investigates why — and tests what fixes it.",
           font=lede, fill=_rgb(T.FG), anchor="mm")
    box = [cx - 205, 410, cx + 205, 462]
    d.rounded_rectangle(box, radius=9, fill=_rgb(T.WIN_BG2), outline=_rgb(T.BORDER))
    d.text((cx - 180, 436), "$", font=code, fill=_rgb(T.BLUE), anchor="lm")
    d.text((cx - 160, 436), "pip install evalrx", font=code, fill=_rgb(T.BRIGHT),
           anchor="lm")
    d.text((cx, 512), "github.com/evalvitals/evalrx   ·   "
                      "evalvitals.github.io/evalrx/demo",
           font=small, fill=_rgb(T.DIM), anchor="mm")
    for frame_no in range(int(seconds * fps)):
        t = frame_no / fps
        if t < fade:
            yield Image.blend(Image.new("RGB", (T.W, T.H), _rgb(T.BG)), base, t / fade)
        else:
            yield base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storyboard", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--command", action="append", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--ui-shot", type=Path, action="append", default=None,
                    help="report-UI screenshot to show after the run (repeatable)")
    ap.add_argument("--ui-seconds", type=float, default=2.6)
    ap.add_argument("--end-card", action="store_true",
                    help="append the closing title frame")
    ap.add_argument("--end-seconds", type=float, default=3.0)
    ap.add_argument("--gif", type=Path, default=None)
    ap.add_argument("--keep-frames", type=Path, default=None)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")

    board = json.loads(args.storyboard.read_text())
    meta = board["meta"]
    commands = args.command or default_commands(meta)
    tl = TL.build(board, commands=commands)
    heading = args.title or f"evalrx · {meta.get('model')} × {meta.get('dataset')}"
    renderer = Renderer(tl, heading=heading)

    work = Path(tempfile.mkdtemp(prefix="evalrx-demo-"))
    frames_dir = args.keep_frames or work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    total_frames = int(tl["total"] * args.fps)
    for frame_no in range(total_frames):
        renderer.frame(frame_no / args.fps).save(frames_dir / f"f{count:05d}.png")
        count += 1
    if args.ui_shot:
        for image in _ui_act(args.ui_shot, seconds_each=args.ui_seconds, fps=args.fps):
            image.save(frames_dir / f"f{count:05d}.png")
            count += 1

    if args.end_card:
        for image in _end_card(seconds=args.end_seconds, fps=args.fps):
            image.save(frames_dir / f"f{count:05d}.png")
            count += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", str(frames_dir / "f%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart", str(args.out),
    ], check=True)
    print(f"{args.out}  ({args.out.stat().st_size / 1024:.0f} KB, {count} frames, "
          f"{count / args.fps:.1f}s)")

    if args.gif:
        palette = work / "palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.out),
                        "-vf", "fps=15,scale=960:-1:flags=lanczos,palettegen",
                        str(palette)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.out),
                        "-i", str(palette), "-lavfi",
                        "fps=15,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse",
                        str(args.gif)], check=True)
        print(f"{args.gif}  ({args.gif.stat().st_size / 1024:.0f} KB)")

    if not args.keep_frames:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
