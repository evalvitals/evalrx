"""Storyboard → a timed, pre-coloured list of screen lines.

The renderers are deliberately dumb: this module decides *when* every line
appears and *what colour each span of it is*, so the SVG and the MP4 show the
same cut. A line is a list of spans; a span is (text, fill, weight), placed
left to right by monospace advance — no text measuring at render time.
"""
from __future__ import annotations

from typing import Any

from . import theme as T


def _spans_for_row(beat: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Colour one narrated row the way the terminal narrator does."""
    code = beat["code"]
    stage = code.strip()
    detail = beat["detail"]
    mark = beat.get("mark")
    spans: list[tuple[str, str, str]] = [
        (code + " ", T.STAGE_COLOR.get(stage, T.BLUE), "600"),
        (beat["label"], T.CYAN, "400"),
        (beat["dots"] + " ", T.DIM, "400"),
    ]
    if "{mark}" in detail:
        head, _, tail = detail.partition("{mark}")
        if head:
            spans.append((head, T.FG, "400"))
        spans.append(("✓" if mark == "ok" else "✗",
                      T.GREEN if mark == "ok" else T.RED, "600"))
        detail = tail
    # Split the trailing "(12.3s)" so it can sit back in the dim register.
    if detail.endswith("s)") and " (" in detail:
        body, _, timing = detail.rpartition(" (")
        spans.append((body, T.FG, "400"))
        spans.append((" (" + timing, T.DIM, "400"))
    else:
        spans.append((detail, T.FG, "400"))
    return spans


def _spans_for_plain(beat: dict[str, Any]) -> list[tuple[str, str, str]]:
    cls = beat.get("cls") or "dim"
    fill = {"dim": T.DIM, "banner": T.BRIGHT}.get(cls, T.FG)
    weight = "600" if cls == "banner" else "400"
    return [(beat["text"], fill, weight)]


def build(board: dict[str, Any], *, commands: list[str]) -> dict[str, Any]:
    """Lay the storyboard out on a clock.

    *commands* are the shell lines typed before the output starts — the real
    install and run commands, typed at :data:`theme.TYPE_CPS`.
    """
    meta = board["meta"]
    lines: list[dict[str, Any]] = []
    typed: list[dict[str, Any]] = []
    t = T.LEAD_IN

    for index, command in enumerate(commands):
        typed.append({"t0": t, "text": command, "row": len(lines)})
        lines.append({"t": t, "row": len(lines), "typing": index,
                      "spans": [("$ ", T.BLUE, "600"), (command, T.BRIGHT, "500")]})
        t += len(command) / T.TYPE_CPS + T.AFTER_ENTER
        if index == 0:
            # The install is real but uninteresting: one confirmation line.
            lines.append({"t": t, "row": len(lines), "spans": [
                ("Successfully installed ", T.DIM, "400"),
                ("evalrx-0.1.2", T.FG, "500")]})
            t += 0.55

    for beat in board["beats"]:
        kind = beat["kind"]
        if kind == "rule":
            lines.append({"t": t, "row": len(lines),
                          "spans": [("─" * 66, T.BORDER, "400")]})
            t += 0.14
            continue
        spans = _spans_for_row(beat) if kind == "row" else _spans_for_plain(beat)
        lines.append({
            "t": t,
            "row": len(lines),
            "spans": spans,
            "stage": beat.get("code", "").strip() or None,
            "elapsed_sec": beat.get("elapsed_sec"),
        })
        t += T.display_gap(beat.get("real_sec")) if kind == "row" else 0.22

    card_t = t + 0.35
    total = card_t + T.CARD_IN + T.HOLD_END

    n_cases = meta.get("n_cases") or 0
    n_failed = meta.get("n_failed") or 0
    tiles = [
        {"value": str(n_cases), "label": "cases diagnosed"},
        {"value": str(n_failed), "label": "failures investigated"},
        {"value": f"{meta.get('n_supported', 0)}/{meta.get('n_hypotheses', 0)}",
         "label": "mechanisms held up on unseen cases"},
    ]
    if meta.get("fixed"):
        verdict = ("REPAIR VALIDATED", T.GREEN)
    else:
        verdict = ("NO REPAIR CLEARED THE BAR — REPORTED, NOT HIDDEN", T.AMBER)

    return {
        "meta": meta,
        "lines": lines,
        "typed": typed,
        "card": {"t": card_t, "tiles": tiles, "verdict": verdict},
        "total": total,
        "commands": commands,
    }
