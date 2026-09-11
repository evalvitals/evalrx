"""Clickable terminal hyperlinks (OSC 8) for the CLI's own "here's your UI /
here's your file" lines.

Most modern terminals (iTerm2, Kitty, WezTerm, GNOME Terminal, Windows
Terminal, VS Code's integrated terminal, ...) render an OSC 8 escape
sequence as an actual clickable link over the visible text; a terminal that
doesn't understand it just ignores the escape bytes and the plain text still
reads fine either way. On a non-tty stream (redirected to a file, piped to
another program) the plain text is used with no escape codes at all, so
nothing downstream ever has to parse them out.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO

_OSC8_START = "\033]8;;"
_OSC8_END = "\033\\"


def _supports_hyperlinks(stream: IO[str]) -> bool:
    if os.environ.get("EVALRX_NO_HYPERLINKS"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def hyperlink(target: str, label: str | None = None, *, stream: IO[str] | None = None) -> str:
    """Return `label` (default: `target`) as a clickable link over `target`.

    `target` is an ``http(s)://`` URL, used as-is, or a local path, turned
    into a ``file://`` URI (resolved relative to the current directory).
    Degrades to plain `label` when the destination stream isn't a terminal
    or `EVALRX_NO_HYPERLINKS` is set.
    """
    stream = stream if stream is not None else sys.stdout
    label = target if label is None else label
    if not _supports_hyperlinks(stream):
        return label
    uri = target if target.startswith(("http://", "https://")) else Path(target).resolve().as_uri()
    return f"{_OSC8_START}{uri}{_OSC8_END}{label}{_OSC8_START}{_OSC8_END}"
