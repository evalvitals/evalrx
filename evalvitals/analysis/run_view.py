"""Normalized, UI-facing representation of an EvalVitals run.

The persisted formats predate the web workbench and intentionally differ:
``exploratory_report.json`` is a finished M2/M3 artifact, while a diagnostic
loop is an event stream.  Rendering either format directly made the UI grow
two incompatible information architectures.  This module is the compatibility
boundary: readers turn either source into the small, stable model the UI uses.

It deliberately does *not* replace the wire contract.  New producers should
write the contract; these adapters keep existing report bundles and historical
``run_log.jsonl`` directories readable during that migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class _StrEnum(str, Enum):
    """``enum.StrEnum`` for the Python floor this package declares.

    ``StrEnum`` is 3.11+, and ``pyproject.toml`` says ``requires-python >=3.10``
    with 3.10 in the CI matrix — so importing it here broke collection of this
    module, and of its tests, on the oldest version we claim to support.

    ``(str, Enum)`` is what the rest of the codebase already uses
    (:class:`evalvitals.core.case.Label`); the explicit ``__str__`` is what
    keeps it interchangeable, since plain ``(str, Enum)`` formats as
    ``StageId.M1`` where ``StrEnum`` formats as ``M1``.
    """

    def __str__(self) -> str:
        return str(self.value)


class StageId(_StrEnum):
    """Pipeline identifiers retained for auditability, not navigation order."""

    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M5 = "M5"
    M4 = "M4"


class StageState(_StrEnum):
    """Reader-visible stage state; absence is distinct from an empty result."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    PARTIAL = "partial"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class StageView:
    id: StageId
    label: str
    state: StageState
    count: int | None = None
    detail: str = ""


@dataclass
class RunView:
    """One source-independent run consumed by the unified web shell."""

    kind: Literal["explore", "diagnostic", "casebench", "empty"]
    root: Path
    title: str
    stages: list[StageView]
    report: dict[str, Any] | None = None
    confirm: dict[str, Any] | None = None
    fix_report: dict[str, Any] | None = None
    story: dict[str, Any] | None = None
    case_studies: list[Any] = field(default_factory=list)

    @property
    def is_diagnostic(self) -> bool:
        return self.kind in {"diagnostic", "casebench"}

    def stage(self, stage_id: StageId) -> StageView:
        return next(s for s in self.stages if s.id == stage_id)


_STAGE_LABELS = {
    StageId.M1: "Measure",
    StageId.M2: "Explore evidence",
    StageId.M3: "Explain hypotheses",
    # M5 precedes M4 in the actual current loop.  The UI uses this action
    # order rather than implying the numeric labels are execution order.
    StageId.M5: "Validate hypotheses",
    StageId.M4: "Intervene & repair",
}


def _stages(states: dict[StageId, tuple[StageState, int | None, str]]) -> list[StageView]:
    return [
        StageView(stage_id, _STAGE_LABELS[stage_id], *states[stage_id])
        for stage_id in StageId
    ]


def _sibling_json(root: Path, name: str) -> dict[str, Any] | None:
    import json

    try:
        value = json.loads((root / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def from_session(session: dict[str, Any]) -> RunView:
    """Adapt the dict returned by :func:`dashboard.load_run`.

    Keeping the adapter dependent on the public loader result, rather than on
    paths or event parsing, makes it usable by both ``dashboard`` and ``web``.
    """

    root = Path(str(session.get("root") or ".")).resolve()
    kind = str(session.get("kind") or "empty")
    if kind == "loop":
        return _from_loop(root, session.get("story") or {})
    if kind == "casebench":
        studies = list(session.get("case_studies") or [])
        title = getattr(studies[0], "name", root.name) if studies else root.name
        ready = StageState.SUCCEEDED if studies else StageState.EMPTY
        return RunView(
            kind="casebench", root=root, title=str(title), case_studies=studies,
            stages=_stages({stage: (ready if stage is StageId.M1 else StageState.UNAVAILABLE, None, "")
                            for stage in StageId}),
        )
    runs = list(session.get("runs") or [])
    if runs:
        turn = runs[0]
        report = turn.get("report") if isinstance(turn, dict) else None
        artifact_dir = Path(str(turn.get("dir") or root)) if isinstance(turn, dict) else root
        return _from_explore(root, artifact_dir, report if isinstance(report, dict) else {})
    return RunView(
        kind="empty", root=root, title=root.name,
        stages=_stages({stage: (StageState.UNAVAILABLE, None, "No readable run artifact.") for stage in StageId}),
    )


def _from_explore(root: Path, artifact_dir: Path, report: dict[str, Any]) -> RunView:
    confirm = _sibling_json(artifact_dir, "confirm_report.json")
    fix_report = _sibling_json(artifact_dir, "fix_report.json")
    ok = bool(report.get("ok", True))
    m2 = StageState.SUCCEEDED if ok else StageState.FAILED
    hypotheses = [h for h in report.get("hypotheses") or [] if isinstance(h, dict)]
    m3 = StageState.SUCCEEDED if hypotheses else StageState.EMPTY
    records = report.get("data_profile") or {}
    has_measurement = bool(records or (artifact_dir / "records.json").exists())
    title = str(report.get("plain_question") or report.get("question") or root.name)
    states = {
        StageId.M1: (StageState.SUCCEEDED if has_measurement else StageState.UNAVAILABLE, None,
                     "Normalized input records" if has_measurement else "No per-case measurement artifact."),
        StageId.M2: (m2, len(report.get("takeaways") or []), "Exploratory report"),
        StageId.M3: (m3, len(hypotheses), "Falsifiable hypotheses"),
        StageId.M5: (StageState.SUCCEEDED if confirm else StageState.NOT_STARTED,
                     len((confirm or {}).get("hypothesis_verdicts") or []) if confirm else None,
                     "Held-out validation" if confirm else "No held-out validation artifact."),
        StageId.M4: (StageState.SUCCEEDED if fix_report else StageState.NOT_STARTED,
                     len(((fix_report or {}).get("fix") or {}).get("attempted") or []) if fix_report else None,
                     "Intervention and repair sweep" if fix_report else "No repair artifact."),
    }
    return RunView("explore", root, title, _stages(states), report, confirm, fix_report)


def _from_loop(root: Path, story: dict[str, Any]) -> RunView:
    probes = list(story.get("probes") or [])
    analyses = list(story.get("analyses") or [])
    diagnoses = list(story.get("diagnoses") or [])
    surgeries = list(story.get("surgeries") or [])
    fixes = list(story.get("fixes") or [])
    lifecycle = story.get("run_start") or {}
    stored_name = root.parent.name if root.name.lower() in {"output", "outputs", "results", "run"} else root.name
    title = str(lifecycle.get("protocol_description") or lifecycle.get("question") or stored_name)

    def outcome(events: list[Any], detail: str) -> tuple[StageState, int | None, str]:
        if not events:
            return StageState.NOT_STARTED, None, detail
        return StageState.SUCCEEDED, len(events), detail

    m5 = [event for event in surgeries if str(event.get("module", "")).lower() == "m5"]
    m4 = [event for event in surgeries if str(event.get("module", "")).lower() == "m4"]
    # A legacy/post-loop ``fix`` event is semantically M4 even when it does
    # not carry an explicit m4 module tag.
    m4_count = len(m4) + len(fixes)
    states = {
        StageId.M1: outcome(probes, "Analyzer measurements"),
        StageId.M2: outcome(analyses, "Evidence screening"),
        StageId.M3: outcome(diagnoses, "Mechanism hypotheses"),
        StageId.M5: outcome(m5, "Hypothesis validation"),
        StageId.M4: (StageState.SUCCEEDED, m4_count, "Interventions and repair attempts")
                    if m4_count else (StageState.NOT_STARTED, None, "No intervention or repair run."),
    }
    return RunView("diagnostic", root, title, _stages(states), story=story)
