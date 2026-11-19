"""What the CLI judges share: running the binary and deciding whether it answered.

Each judge builds its own command line; this module runs it the same way for
all of them (process-group kill on timeout, prompt on stdin) and applies the
same reading of the result:

* a limit notice (plan / session / spend / usage) printed as if it were an
  answer raises, because the CLIs report one on stdout with exit 0;
* a non-zero exit with no output raises, with the reason the CLI gave;
* an empty answer is returned as ``""`` with a warning, so callers that read
  free text can fall back; typed calls (``ask_judge``) treat it as invalid.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from evalrx.agent_runtime.cli_runtime import ProcessRun, SubprocessRunner

#: Limit notices the coding CLIs print on stdout, exit 0, in place of an answer.
LIMIT_NOTICE = re.compile(
    r"hit your (monthly |weekly |session |daily )?(spend|usage|rate) limit|"
    r"(spend|usage) limit (reached|exceeded)|out of (extra )?usage|"
    r"limit resets", re.I)

#: A real judge answer is never this short AND a limit notice.
_NOTICE_MAX_CHARS = 400


def run_cli(
    name: str, cmd: list[str], *, cwd: Path, timeout_sec: int, stdin_text: str | None = None,
) -> ProcessRun:
    """Run one judge invocation; raise ``RuntimeError`` when it times out."""
    run = SubprocessRunner().run(cmd, cwd, timeout_sec, stdin_text=stdin_text)
    if run.timed_out:
        raise RuntimeError(f"{name} timed out after {timeout_sec}s")
    return run


def read_reply(name: str, run: ProcessRun, output: str, *, failure_reason: str = "") -> str:
    """The judge's answer from *output*, or ``RuntimeError`` when it gave none.

    *failure_reason* replaces stderr/stdout as the explanation of a failed
    exit (agy keeps its reason in a log file).
    """
    output = output.strip()
    if len(output) < _NOTICE_MAX_CHARS and LIMIT_NOTICE.search(output):
        raise RuntimeError(f"{name}: account limit reached, not a judge answer: {output!r}")
    if run.returncode != 0 and not output:
        reason = failure_reason or (run.stderr or run.stdout or "").strip()[-500:]
        raise RuntimeError(f"{name} exited {run.returncode}: {reason}")
    if not output:
        detail = f" -- {failure_reason}" if failure_reason else ""
        warnings.warn(
            f"{name} returned an empty response{detail}; it is likely rate-limited or "
            "out of quota.",
            stacklevel=3,
        )
    return output
