"""Provider-neutral, evidence-backed auditing for configured Agent Skills.

The audit deliberately distinguishes a skill being requested or installed from
the provider trace proving it was invoked/read.  A provider without a machine
readable trace is reported as ``not_observable`` rather than guessed from the
assistant's prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = 1


def _skill_names(paths: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(Path(path).name for path in paths if Path(path).name))


def _json_lines(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _claude_evidence(stdout: str, requested: list[str]) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    requested_set = set(requested)
    for event in _json_lines(stdout):
        if event.get("type") != "assistant":
            continue
        content = (event.get("message") or {}).get("content") or []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") != "Skill":
                continue
            inp = block.get("input") or {}
            raw_name = str(inp.get("skill") or "").strip().lstrip("/")
            if raw_name in requested_set:
                evidence.append({
                    "skill": raw_name,
                    "kind": "native_skill_invocation",
                    "detail": "Claude Code Skill tool invocation",
                })
    return evidence


def _codex_evidence(stdout: str, requested: list[str]) -> list[dict[str, str]]:
    """Find conservative evidence that Codex read a vendored skill guide.

    ``codex exec --json`` exposes shell command events, but Codex has no
    ``Skill`` tool.  Merely listing a directory is not evidence of use, so only
    commands that name a skill's ``SKILL.md`` (or a file within its directory)
    are admitted, excluding ``rg --files`` discovery commands.
    """
    evidence: list[dict[str, str]] = []
    for event in _json_lines(stdout):
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        if item.get("type") != "command_execution":
            continue
        command = str(item.get("command") or "")
        if "rg --files" in command:
            continue
        for name in requested:
            marker = f".claude/skills/{name}/"
            if marker in command:
                evidence.append({
                    "skill": name,
                    "kind": "vendored_skill_file_read",
                    "detail": "Codex command referenced a file in the vendored skill directory",
                })
    return evidence


def build_agent_audit(
    *,
    provider: str,
    skills: list[str] | tuple[str, ...],
    workdir: Path,
    stdout: str,
    stderr: str,
    returncode: int,
    timed_out: bool,
    elapsed_sec: float,
    files: dict[str, str],
    error: str | None,
) -> dict[str, Any]:
    """Build one durable audit record for a provider invocation."""
    requested = _skill_names(skills)
    installed = [
        name for name in requested
        if (workdir / ".claude" / "skills" / name / "SKILL.md").is_file()
    ]
    if provider == "claude_code":
        evidence = _claude_evidence(stdout, requested)
        observation = "native_skill_events"
    elif provider == "codex":
        evidence = _codex_evidence(stdout, requested)
        observation = "json_command_events"
    elif provider == "antigravity":
        evidence = []
        observation = "not_observable"
    else:
        evidence = []
        observation = "not_observable"

    invoked = list(dict.fromkeys(item["skill"] for item in evidence))
    if timed_out:
        status = "timed_out"
    elif error:
        status = "failed"
    elif files:
        status = "completed"
    else:
        status = "completed_without_code"

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "provider": provider,
        "skills_requested": requested,
        "skills_installed": installed,
        "skills_invoked": invoked,
        "skill_observability": observation,
        "evidence": evidence,
        "execution": {
            "status": status,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_sec": round(elapsed_sec, 3),
            "produced_files": sorted(files),
            "error": error or "",
            "stderr": stderr[:4000],
        },
    }
