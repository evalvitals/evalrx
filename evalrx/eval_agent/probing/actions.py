"""What one probing action is, what came back, and the identity tying them together.

An action is ``(analyzer, configuration, cases, purpose)``. Its identity is a
content hash over the analyzer, its full configuration, the ordered case ids
and the execution context (model, policy version), so a resumed run recognises
a finished action and a changed configuration never reuses an old result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: Terminal outcomes of an executed action. ``skipped`` (budget) is never
#: persisted as an observation, so a resumed run with more budget runs it.
OK = "ok"
UNSUPPORTED = "unsupported"   # capability / data precondition not met; nothing ran
FAILED = "failed"             # raised during execution; no measurement
UNFAITHFUL = "unfaithful"     # the edit's own self-check failed; no measurement
SKIPPED = "skipped"
TERMINAL = frozenset({OK, UNSUPPORTED, FAILED, UNFAITHFUL})


def canonical(value: Any) -> Any:
    """A JSON-stable rendering of *value* for hashing and logging.

    Callables become their import path (a function's repr carries a memory
    address, which would give every run a new identity); sets are sorted;
    anything else unknown falls back to its type name and repr.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(v) for v in value)
    if callable(value):
        module = getattr(value, "__module__", None) or "?"
        name = getattr(value, "__qualname__", None) or type(value).__qualname__
        return f"callable:{module}.{name}"
    if hasattr(value, "item") and callable(value.item):  # numpy / torch scalars
        try:
            return canonical(value.item())
        except Exception:  # noqa: BLE001 - not a scalar after all
            pass
    return f"{type(value).__name__}:{value!r}"


def digest(payload: Any) -> str:
    """16-hex content hash of :func:`canonical` ``(payload)``."""
    text = json.dumps(canonical(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ProbeAction:
    """One planned experiment, executed exactly as written.

    Attributes:
        analyzer:  registered analyzer name.
        config:    constructor keyword arguments (only the ones the plan sets;
                   the executed analyzer's full ``get_params()`` is recorded on
                   the observation).
        case_ids:  the cases it runs on, in batch order. The analyzer receives
                   exactly these — no cap, no backfill.
        route:     the triage route or method layer it serves (``H``, ``N``, ``N1``).
        purpose:   why the plan contains it, in one line.
        source:    which policy proposed it.
        reserve:   estimated model calls by method, reserved before execution.
    """

    analyzer: str
    config: dict[str, Any]
    case_ids: tuple[str, ...]
    route: str
    purpose: str
    source: str = "mvp0_fixed_order"
    reserve: dict[str, int] = field(default_factory=dict)

    def identity(self, context: dict[str, Any]) -> str:
        return digest({
            "analyzer": self.analyzer, "config": self.config,
            "case_ids": list(self.case_ids), "context": context,
        })

    def to_dict(self) -> dict[str, Any]:
        out = canonical(asdict(self))
        out["n_cases"] = len(self.case_ids)
        return out


@dataclass
class Observation:
    """What one execution attempt of an action produced.

    ``findings`` is ``None`` unless ``status == "ok"``: a failed, unsupported
    or unfaithful action has no measurement, and no zero is written in its place.

    ``fidelity["status"]`` is one of ``passed`` / ``failed`` / ``unchecked``
    (the tool has an implementation check that did not run) /
    ``not_applicable`` (no implementation edit to check).
    """

    action_id: str
    analyzer: str
    attempt: int
    status: str
    params: dict[str, Any]
    planned_ids: list[str]
    measured_ids: list[str] = field(default_factory=list)
    missing: dict[str, str] = field(default_factory=dict)
    findings: "dict[str, Any] | None" = None
    error_type: "str | None" = None
    error: "str | None" = None
    fidelity: dict[str, Any] = field(default_factory=lambda: {"status": "not_applicable"})
    cost: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonical(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in names})

    def per_case(self) -> dict[str, dict[str, Any]]:
        """``{case_id: per_case row}`` for an ``ok`` observation, else empty."""
        rows = (self.findings or {}).get("per_case") or []
        return {str(r.get("sample_id")): r for r in rows if isinstance(r, dict) and r.get("sample_id") is not None}
