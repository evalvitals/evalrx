"""Resolve bundled and explicitly configured Agent Skills."""

from __future__ import annotations

from evalrx.agent_assets.skills import SKILL_BACKENDS, bundled_skill_paths


def resolve_skill_paths(
    *,
    provider: str,
    explicit: list[str] | tuple[str, ...] = (),
    use_bundled: bool = True,
) -> tuple[str, ...]:
    """Return ordered skill dirs for a provider, de-duplicating explicit repeats."""
    skill_dirs = list(explicit or [])
    if use_bundled and provider in SKILL_BACKENDS:
        bundled = [path for path in bundled_skill_paths() if path not in skill_dirs]
        skill_dirs = bundled + skill_dirs
    return tuple(skill_dirs)
