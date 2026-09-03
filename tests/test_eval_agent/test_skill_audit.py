"""Regression tests for provider-neutral Agent Skill auditing."""

from __future__ import annotations

import json

from evalrx.agent_runtime.skill_audit import build_agent_audit


def _workdir(tmp_path, *skills: str):
    for skill in skills:
        path = tmp_path / ".claude" / "skills" / skill
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(f"# {skill}\n", encoding="utf-8")
    return tmp_path


def test_claude_skill_audit_records_native_skill_invocation(tmp_path):
    workdir = _workdir(tmp_path, "outcome-driver-analysis")
    stream = json.dumps({
        "type": "assistant",
        "message": {"content": [{
            "type": "tool_use", "name": "Skill",
            "input": {"skill": "outcome-driver-analysis"},
        }]},
    })

    audit = build_agent_audit(
        provider="claude_code", skills=["/skills/outcome-driver-analysis"], workdir=workdir,
        stdout=stream, stderr="", returncode=0, timed_out=False, elapsed_sec=1.234,
        files={"analysis.py": "print('ok')"}, error=None,
    )

    assert audit["skills_requested"] == ["outcome-driver-analysis"]
    assert audit["skills_installed"] == ["outcome-driver-analysis"]
    assert audit["skills_invoked"] == ["outcome-driver-analysis"]
    assert audit["evidence"][0]["kind"] == "native_skill_invocation"
    assert audit["execution"]["status"] == "completed"


def test_codex_skill_audit_requires_file_use_not_directory_discovery(tmp_path):
    workdir = _workdir(tmp_path, "nature-figure")
    discovery = json.dumps({"type": "item.completed", "item": {
        "type": "command_execution", "command": "rg --files .claude/skills",
    }})
    read = json.dumps({"type": "item.completed", "item": {
        "type": "command_execution",
        "command": "sed -n '1,80p' .claude/skills/nature-figure/SKILL.md",
    }})

    audit = build_agent_audit(
        provider="codex", skills=["/skills/nature-figure"], workdir=workdir,
        stdout=discovery + "\n" + read, stderr="", returncode=0, timed_out=False,
        elapsed_sec=1, files={"analysis.py": "x = 1"}, error=None,
    )

    assert audit["skill_observability"] == "json_command_events"
    assert audit["skills_invoked"] == ["nature-figure"]
    assert len(audit["evidence"]) == 1


def test_antigravity_skill_audit_does_not_infer_use_from_prose(tmp_path):
    workdir = _workdir(tmp_path, "nature-figure")
    audit = build_agent_audit(
        provider="antigravity", skills=["/skills/nature-figure"], workdir=workdir,
        stdout="I will apply nature-figure now.", stderr="", returncode=0,
        timed_out=False, elapsed_sec=1, files={"analysis.py": "x = 1"}, error=None,
    )

    assert audit["skills_installed"] == ["nature-figure"]
    assert audit["skills_invoked"] == []
    assert audit["skill_observability"] == "not_observable"
