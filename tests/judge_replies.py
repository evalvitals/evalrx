"""JSON replies a scripted judge gives to M3's typed proposal and review calls."""

from __future__ import annotations

import json
from typing import Any


def hypothesis(
    statement: str,
    failure_mode: str,
    *,
    plain: str = "The model gets these cases wrong for the reason stated.",
    test: str = "",
    expected: str | None = None,
) -> dict[str, Any]:
    """One entry of a :class:`~evalrx.eval_agent.stages.diagnosis.Proposal`."""
    entry: dict[str, Any] = {
        "hypothesis": statement, "plain_statement": plain, "failure_mode": failure_mode,
        "test": test,
    }
    if expected is not None:
        entry["expected_association"] = expected
    return entry


def proposal(*hypotheses: dict[str, Any]) -> str:
    """The proposer's reply; no hypotheses means "no problem found"."""
    return json.dumps({"hypotheses": list(hypotheses)})


def review(*decisions: tuple[int, str, str]) -> str:
    """The critic's reply: ``(number, "keep" | "reject", reason)`` per hypothesis."""
    return json.dumps({"decisions": [
        {"number": n, "decision": d, "reason": r} for n, d, r in decisions
    ]})


def keep_all(n: int, reason: str = "evidence directly supports this claim") -> str:
    return review(*((i, "keep", reason) for i in range(1, n + 1)))
