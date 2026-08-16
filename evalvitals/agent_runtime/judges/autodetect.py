"""Judge liveness-probe autodetection: find a CLI judge that actually responds.

A quota-exhausted CLI model often exits 0 with an empty response rather than
an error, so a tiny generation probe is the only reliable availability check.
Promoted from the ``_pick_agy_model``/``_pick_claude_model``/``_resolve_judge``
helpers duplicated across the ``VLDiagnoseLoop`` examples (``examples/m1_m3/*``,
``examples/m1_m4/*``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from evalvitals.agent_runtime.judges.agy import AgyModel
from evalvitals.agent_runtime.judges.claude import ClaudeModel

_DEFAULT_PROBE_PROMPT = "Reply with exactly the word OK"

# Probe order: cheap/fast candidates first; quotas are per-model and reset on
# independent clocks, so whichever responds first is "the available version".
DEFAULT_AGY_CANDIDATES: tuple[str, ...] = (
    "Gemini 3.1 Pro (Low)",
    "Claude Sonnet 4.6 (Thinking)",
    "GPT-OSS 120B (Medium)",
    "Gemini 3.5 Flash (Low)",
    "Claude Opus 4.6 (Thinking)",
)
# Claude judge candidates: the user's session model first (Fable), then
# cheaper aliases.
DEFAULT_CLAUDE_CANDIDATES: tuple[str, ...] = ("claude-fable-5", "sonnet", "haiku")


def pick_live_model(
    factory: "Callable[[str], Any]",
    candidates: "Sequence[str]",
    *,
    probe_prompt: str = _DEFAULT_PROBE_PROMPT,
) -> str:
    """Return the first candidate name whose model actually answers *probe_prompt*.

    *factory* builds a model from a candidate name (e.g. ``lambda name:
    AgyModel(model=name, timeout_sec=60)``). A quota-exhausted model typically
    returns an empty string rather than raising, so an empty response is
    treated the same as a failed candidate. Raises ``RuntimeError`` (with every
    attempt's failure reason) when all candidates are dead.
    """
    errors: list[str] = []
    for name in candidates:
        try:
            probe = factory(name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # quota warnings are expected here
                out = probe.generate(probe_prompt)
            if str(out).strip():
                return name
            errors.append(f"{name!r}: empty response (likely quota-exhausted)")
        except RuntimeError as exc:
            errors.append(f"{name!r}: {exc}")
    raise RuntimeError(
        f"no model responded to the availability probe ({len(candidates)} tried): "
        + "; ".join(errors)
    )


def pick_agy_model(
    candidates: "Sequence[str]" = DEFAULT_AGY_CANDIDATES, *, timeout_sec: int = 60,
) -> str:
    """Return the first agy model name that answers a probe prompt."""
    return pick_live_model(
        lambda name: AgyModel(model=name, timeout_sec=timeout_sec), candidates,
    )


def pick_claude_model(
    candidates: "Sequence[str]" = DEFAULT_CLAUDE_CANDIDATES, *, timeout_sec: int = 90,
) -> str:
    """Return the first claude model name that answers a probe prompt."""
    return pick_live_model(
        lambda name: ClaudeModel(model=name, timeout_sec=timeout_sec), candidates,
    )


@dataclass(frozen=True)
class ResolvedJudge:
    """A CLI judge resolved by :func:`resolve_cli_judge`, with its provenance."""

    judge: Any
    provider: str  # "agy" | "claude"
    model: str


def resolve_cli_judge(
    provider: str = "auto",
    model: str = "auto",
    *,
    effort: str = "",
    timeout_sec: int = 240,
) -> ResolvedJudge:
    """Resolve a live CLI judge.

    ``provider="auto"`` tries agy first (its quota errors fail fast), then
    claude. ``model="auto"`` probes :data:`DEFAULT_AGY_CANDIDATES` /
    :data:`DEFAULT_CLAUDE_CANDIDATES` for a live one; an explicit model name
    is used directly (constructed, not probed). Raises ``RuntimeError``
    (aggregating every attempt's failure) when nothing is available; raises
    ``ValueError`` for an unrecognized *provider*.
    """

    def _try_agy() -> ResolvedJudge:
        name = pick_agy_model() if model == "auto" else model
        return ResolvedJudge(AgyModel(model=name, timeout_sec=timeout_sec), "agy", name)

    def _try_claude() -> ResolvedJudge:
        name = pick_claude_model() if model == "auto" else model
        return ResolvedJudge(
            ClaudeModel(model=name, timeout_sec=timeout_sec, effort=effort), "claude", name,
        )

    try:
        chain = {"agy": (_try_agy,), "claude": (_try_claude,),
                 "auto": (_try_agy, _try_claude)}[provider]
    except KeyError:
        raise ValueError(
            f"resolve_cli_judge: unknown provider {provider!r} (agy/claude/auto)"
        ) from None

    errors: list[str] = []
    for attempt in chain:
        try:
            return attempt()
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError("no CLI judge available: " + "; ".join(errors))
