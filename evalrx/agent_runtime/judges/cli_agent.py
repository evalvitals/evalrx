"""A CLI coding agent used as a judge: it writes its answer to a file.

A coding agent's output stream is a whole trajectory (plans, tool calls, drafts),
not an answer, so it is never parsed. The agent is asked to write its final
answer to ``answer.json`` in a fresh directory and that file is the reply. Each
call gets its own directory, so a retry can never read an earlier attempt.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalrx.agent_runtime.cli_types import CliAgentConfig

ANSWER_FILE = "answer.json"

_ANSWER_NOTE = (
    f"\n\nWrite your final answer, and nothing else, to the file {ANSWER_FILE} in "
    "the current directory. Do not write any other files."
)


class CliAgentJudge:
    """``generate(prompt) -> str`` over a CLI coding agent (``claude_code``, ``codex``, ...).

    Returns the text of the answer file, ``""`` when the agent wrote none (so
    :func:`~evalrx.agent_runtime.judge_call.ask_judge` asks again), and raises
    when the agent itself failed (timeout, crash).
    """

    def __init__(self, cli_config: "CliAgentConfig", *, timeout_sec: int = 90) -> None:
        self.cli_config = cli_config
        self.timeout_sec = timeout_sec

    def __repr__(self) -> str:
        return f"CliAgentJudge(provider={self.cli_config.provider!r}, model={self.cli_config.model!r})"

    def generate(self, prompt: str) -> str:
        from evalrx.agent_runtime.codegen import CodegenRunner

        with tempfile.TemporaryDirectory(prefix="evalrx_answer_") as tmp:
            result = CodegenRunner(self.cli_config).run(
                prompt + _ANSWER_NOTE, workdir=Path(tmp), timeout_sec=self.timeout_sec,
            )
            answer = Path(tmp) / ANSWER_FILE
            if answer.is_file():
                return answer.read_text(encoding="utf-8")
        if result.error:
            raise RuntimeError(f"{self.cli_config.provider}: {result.error}")
        return ""
