"""Shared geometry, palette and timing for the demo-video renderers.

One module so the animated SVG (README) and the rasterised MP4 (slides, social)
cannot drift apart: both import these numbers rather than each carrying their
own copy.

Palette follows the repo's own brand accents — the README mermaid theme's blue
``#3D8DFF`` / green ``#39A96B`` / red ``#D45656`` — on the report UI's dark
ground (``#07110f``, the exported page's ``theme-color``).
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- geometry ---
W, H = 1280, 720                 # 16:9, matches the existing Re.mp4 cut
MARGIN = 26                      # backdrop → window gutter
WIN_X, WIN_Y = MARGIN, MARGIN
WIN_W, WIN_H = W - 2 * MARGIN, H - 2 * MARGIN
WIN_R = 12                       # window corner radius
BAR_H = 42                       # title bar height
FOOT_H = 38                      # status bar height
RAIL_W = 0                       # no left gutter: the stage pills live in the title bar

PAD_X = 30                       # body text left inset
PAD_TOP = 22
LINE_H = 23                      # monospace line box
FONT_SIZE = 14.5
MONO = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'DejaVu Sans Mono', monospace")
SANS = ("'Inter', ui-sans-serif, -apple-system, 'Helvetica Neue', "
        "'Segoe UI', sans-serif")
# Advance width of one monospace glyph at FONT_SIZE, used to place coloured
# spans inside a row without measuring text at render time.
CH = FONT_SIZE * 0.6

TEXT_X = WIN_X + RAIL_W + PAD_X
TEXT_Y0 = WIN_Y + BAR_H + PAD_TOP + FONT_SIZE

# ----------------------------------------------------------------- palette ---
BG = "#05100e"                   # page backdrop, just under the window
WIN_BG = "#07110f"               # terminal ground
WIN_BG2 = "#091714"              # title/status bar
BORDER = "#16302b"
GRID = "#0c1c19"

FG = "#d7e4e0"                   # ordinary output
DIM = "#5d726d"                  # leader dots, timings, metadata
BRIGHT = "#f2fbf8"               # typed command, headline numbers
BLUE = "#3D8DFF"                 # stage badge, prompt
CYAN = "#4fd6c0"                 # stage label
GREEN = "#39A96B"                # supported / fixed
RED = "#D45656"                  # refuted / not fixed
AMBER = "#E8B84B"                # recommendation / escalate
VIOLET = "#A78BFA"               # M3, the "idea" stage

# The rail dot colour per stage — a cool→warm ramp that reads as progress.
STAGE_COLOR = {"M1": BLUE, "M2": CYAN, "M3": VIOLET, "M4": GREEN, "M5": AMBER}
STAGE_ORDER = ("M1", "M2", "M3", "M4", "M5")

LIGHTS = ("#ED6A5E", "#F4BF4F", "#61C554")

# ------------------------------------------------------------------ timing ---
LEAD_IN = 0.45                   # before the prompt appears
TYPE_CPS = 26.0                  # typed characters per second
AFTER_ENTER = 0.55               # beat between Enter and the first output
HOLD_END = 2.6                   # freeze on the result strip before looping
CARD_IN = 0.7                    # result strip fade-in


def display_gap(real_seconds: float | None) -> float:
    """Screen time for a stage that really took *real_seconds*.

    Log-compressed: a 190s probe and a 5.4h fix attempt both have to fit in a
    README loop, but their order of magnitude should still be visible, so the
    gap grows with the log of the real duration rather than being uniform.
    """
    if not real_seconds or real_seconds <= 0:
        return 0.34
    return min(1.80, 0.42 + 0.62 * math.log10(1.0 + real_seconds / 10.0))


def clock(seconds: float) -> str:
    """``t+01:23:45`` / ``t+03:11`` for the header's real-elapsed readout."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"t+{h}:{m:02d}:{s:02d}" if h else f"t+{m:02d}:{s:02d}"
