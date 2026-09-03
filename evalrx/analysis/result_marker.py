"""Shared parser for ``<MARKER>=<json>`` result lines printed by agent-written scripts.

The explorer (``EXPLORATORY_RESULT_JSON=``) and the M2 stats-tool generator
(``STATS_RESULT_JSON=``) both ask the generated Python to print its result as
one JSON object after a fixed marker. The original parsers read only the
remainder of the marker LINE, so ``print(f"MARKER={json.dumps(obj, indent=2)}")``
— a perfectly valid object that merely spans several lines — parsed as ``{`` and
failed (qwen3.5-2b / gsm8k, 2026-08-27: three explorer attempts in a row, the
report lost). The marker is the contract; the line break is not.

:func:`extract_marker_json` therefore decodes from the LAST marker to wherever
the JSON value ends (``json.JSONDecoder.raw_decode``), so single-line,
pretty-printed and trailing-noise outputs all read the same object.
"""

from __future__ import annotations

import json
from typing import Any


def extract_marker_json(stdout: str, marker: str) -> tuple[Any, str]:
    """Return ``(value, "")`` for the JSON value that follows the last *marker*
    line in *stdout*, or ``(None, error)`` when there is none / it is invalid.

    The marker must start a line (after whitespace); the value may continue on
    the following lines and may be followed by unrelated output. When the
    marker appears more than once the last occurrence wins, matching the
    line-oriented behaviour these parsers always had.
    """
    start = -1
    pos = 0
    for line in stdout.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(marker):
            start = pos + (len(line) - len(stripped)) + len(marker)
        pos += len(line)
    if start < 0:
        return None, f"no {marker} line in output"
    payload = stdout[start:].lstrip()
    if not payload:
        return None, f"empty {marker} payload"
    try:
        value, _end = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as exc:
        return None, f"unparseable {marker} JSON: {exc}"
    return value, ""
