"""What one probing action is, what came back, and the identity tying them together.

An action is ``(analyzer, configuration, cases, purpose)``. Its identity is a
content hash over the analyzer, its full configuration, the ordered case ids
and the execution context (model, policy version), so a resumed run recognises
a finished action and a changed configuration never reuses an old result.
A :class:`PlanStep` is the template an action is built from; the cases it
runs on are chosen when it is executed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
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
    address, which would give every run a new identity); a case batch (e.g. a
    donor pool) becomes its ordered case ids; sets are sorted; anything else
    unknown falls back to its type name and repr.
    """
    from evalrx.core.case import CaseBatch, FailureCase

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, CaseBatch):
        return {"case_ids": [str(c.id) for c in value]}
    if isinstance(value, FailureCase):
        return f"case:{value.id}"
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
class PlanStep:
    """What to build, why, and what it may cost: a battery step or a rule's candidate.

    ``key`` names the step when one analyzer serves two steps (default: the
    analyzer); evidence is stored under it, so two routes asking the same
    step share its observations. ``per_case`` runs a one-case analyzer once
    per case; ``keep_artifacts`` keeps those result artifacts with the
    findings; ``task_kinds`` limits the step to those task kinds (empty =
    every kind).

    ``reuse`` says that a case's row does not depend on which other cases share
    the batch, so a larger sample only runs the new cases and keeps the old
    rows. A step whose rows do depend on the batch (words inferred from the
    batch's prompts, a fit on the batch, noise seeded by batch position) has
    ``reuse=False`` and ``reuse_note`` says why: it re-runs on the whole sample.

    ``min_output_tokens``: the step asks for text before the answer (e.g. a
    description), so a known output cap below this leaves it nothing to read.

    ``deferred``: a battery step no route reads (a descriptive reading): the
    rules run it once the sample has settled, and the scored scheduler takes it
    as a candidate of the layer it serves instead of running it up front.
    """

    analyzer: str
    config: dict[str, Any]
    route: str
    purpose: str
    fidelity_check: "str | None" = None          # "edit" = tap/linear self-check, "patch" = patch self-check
    calls_per_case: dict[str, int] = field(default_factory=dict)
    key: str = ""
    per_case: bool = False
    keep_artifacts: tuple[str, ...] = ()
    task_kinds: tuple[str, ...] = ()
    reuse: bool = True
    reuse_note: str = ""
    caveat: str = ""
    min_output_tokens: int = 0
    deferred: bool = False

    @property
    def name(self) -> str:
        return self.key or self.analyzer


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

    def step_identity(self, context: dict[str, Any]) -> str:
        """The identity without the cases: the same experiment, whichever cases it ran on.

        Observations sharing it may be combined case by case only when the
        step's rows do not depend on the batch (``PlanStep.reuse``).
        """
        return digest({"analyzer": self.analyzer, "config": self.config, "context": context})

    def to_dict(self) -> dict[str, Any]:
        # a shallow read: asdict() would deep-copy a donor pool's cases (and media)
        out = canonical({f.name: getattr(self, f.name) for f in fields(self)})
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
