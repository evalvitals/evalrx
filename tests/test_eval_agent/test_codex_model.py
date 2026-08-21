"""CodexModel uses stdin and an isolated workdir for judge turns."""

from __future__ import annotations

import stat
from pathlib import Path

from evalvitals.eval_agent import CodexModel


def _fake_codex(tmp_path: Path) -> str:
    binary = tmp_path / "codex"
    binary.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = \"--output-last-message\" ]; then out=$2; shift 2; continue; fi\n"
        "  shift\n"
        "done\n"
        "input=$(cat)\n"
        "printf 'JUDGE: %s' \"$input\" > \"$out\"\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return str(binary)


def test_codex_judge_uses_stdin_and_forwards_model(tmp_path):
    judge = CodexModel(binary_path=_fake_codex(tmp_path), model="gpt-5.6-terra")
    assert judge.generate("a long evidence prompt") == "JUDGE: a long evidence prompt"


def test_codex_judge_makes_images_visible_by_name(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"png")
    judge = CodexModel(binary_path=_fake_codex(tmp_path))
    answer = judge.generate("inspect this", images=[image])
    assert "Images available in workspace: chart.png" in answer
