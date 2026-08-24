"""CodexModel uses stdin and an isolated workdir for judge turns."""

from __future__ import annotations

import stat
from pathlib import Path

from evalvitals.eval_agent import CodexModel
from evalvitals.agent_runtime.providers.codex import CodexAgent


def _fake_codex(tmp_path: Path) -> str:
    binary = tmp_path / "codex"
    binary.write_text(
        "#!/bin/sh\n"
        "args=\"$*\"\n"
        "out=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = \"--output-last-message\" ]; then out=$2; shift 2; continue; fi\n"
        "  shift\n"
        "done\n"
        "input=$(cat)\n"
        "printf 'ARGS: %s\\nJUDGE: %s' \"$args\" \"$input\" > \"$out\"\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return str(binary)


def test_codex_judge_uses_stdin_and_forwards_model(tmp_path):
    judge = CodexModel(
        binary_path=_fake_codex(tmp_path), model="gpt-5.6-terra", effort="medium"
    )
    answer = judge.generate("a long evidence prompt")
    assert "--model gpt-5.6-terra" in answer
    assert 'model_reasoning_effort="medium"' in answer
    assert answer.endswith("JUDGE: a long evidence prompt")


def test_codex_judge_makes_images_visible_by_name(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"png")
    judge = CodexModel(binary_path=_fake_codex(tmp_path))
    answer = judge.generate("inspect this", images=[image])
    assert "Images available in workspace: chart.png" in answer


def test_codex_coding_agent_allows_isolated_non_git_workspaces(tmp_path):
    cmd = CodexAgent(binary_path="codex")._build_cmd("write pipeline.py", tmp_path)
    assert "--skip-git-repo-check" in cmd
