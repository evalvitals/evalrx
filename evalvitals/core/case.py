"""FailureCase — the central data object of EvalVitals.

Everything in the system speaks ``FailureCase``:
  - datasets *produce* batches of cases,
  - analyzers *attribute* failures over cases,
  - stats *test* significance across cases,
  - the agent *accumulates* and *evolves* a corpus of cases.

Analyzers accept ``str | FailureCase | list | CaseBatch`` and normalise via
:func:`as_casebatch`, so ``model.call_attention("a prompt")`` stays ergonomic
while the canonical unit of work remains the case.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Optional


class Label(str, Enum):
    """Outcome label for a case."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Source(str, Enum):
    """Where a case came from — important for self-evolution provenance."""

    HUMAN = "human"
    DATASET = "dataset"
    AGENT = "agent"


@dataclass
class Provenance:
    """How a case was created."""

    source: Source = Source.HUMAN
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Inputs:
    """Model inputs for a case.

    ``prompt`` is always present; the multimodal slots are populated per the
    model's modality set — ``image`` for VLMs, ``audio`` / ``video`` for omni
    models (e.g. Qwen3-Omni).  Each slot holds a decoded object (PIL.Image /
    waveform / frames) **or** a path/URL the backend resolves.
    """

    prompt: str
    image: Any = None   # PIL.Image | path/URL | None  (VLM)
    audio: Any = None   # np.ndarray | path/URL | None  (omni)
    video: Any = None   # frames | path/URL | None      (omni)

    def __str__(self) -> str:
        return self.prompt


# ----------------------------------------------------------------------
# Agent / trajectory data model
# ----------------------------------------------------------------------
# A FailureCase can be a single (unit) interaction OR carry a multi-step agent
# trajectory.  The trajectory schema below is the canonical substrate for both
# execution logging and agent failure analysis (loop detection, first-error
# attribution, recovery).  It deliberately mirrors the keys an OTel/OpenInference
# span carries and an existing node-tree conversation already stores, so a real
# engine's trace adapts in via ``Trajectory.from_records`` without re-modelling.

class StepRole(str, Enum):
    """Role of a single trajectory step."""

    USER = "user"
    PLANNER = "planner"
    ACTOR = "actor"
    TOOL = "tool"
    VERIFIER = "verifier"
    MEMORY = "memory"
    SYSTEM = "system"


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of *value* into something ``json.dumps`` accepts.

    Rich objects degrade to descriptors instead of raising: images (anything
    with ``.size`` + ``.mode``, i.e. PIL) become ``"<image WxH>"`` — trajectories
    reference media, they never embed pixels.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "size") and hasattr(value, "mode"):  # PIL.Image, without importing PIL
        try:
            w, h = value.size
            return f"<image {w}x{h}>"
        except Exception:  # pragma: no cover - exotic size attrs
            return "<image>"
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


@dataclass
class Step:
    """One atomic step in an agent trajectory.

    Execution fills the top block; analyzers WRITE the annotation block
    (``is_first_error`` / ``failure_mode`` / ``judge_confidence``).
    """

    idx: int
    role: StepRole = StepRole.ACTOR
    content: Any = None                    # thought / message / final text
    agent_id: str = "main"                 # which agent acted (multi-agent traces)
    tool_call: Optional[dict] = None       # {"name": ..., "args": {...}, "id": ...}
    observation: Any = None                # tool/env result (images by ref)
    span: dict = field(default_factory=dict)   # OTel-ish attrs: tokens, latency_ms, cost, model
    # --- analysis annotations (written by analyzers, not execution) ---
    is_first_error: Optional[bool] = None
    failure_mode: Optional[str] = None     # MAST code, e.g. "FM-2.4"
    judge_confidence: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (images and rich objects degrade to descriptors)."""
        return {
            "idx": self.idx,
            "role": self.role.value,
            "content": _json_safe(self.content),
            "agent_id": self.agent_id,
            "tool_call": _json_safe(self.tool_call),
            "observation": _json_safe(self.observation),
            "span": _json_safe(self.span),
            "is_first_error": self.is_first_error,
            "failure_mode": self.failure_mode,
            "judge_confidence": self.judge_confidence,
        }


@dataclass
class Trajectory:
    """The full sequence of an agent run — the unit of trajectory analysis."""

    sample_id: str
    goal: str = ""
    steps: list[Step] = field(default_factory=list)
    final_answer: Any = None
    ground_truth: Any = None
    outcome: Label = Label.UNKNOWN          # reuse the engine's outcome, don't recompute
    metrics: dict[str, Any] = field(default_factory=dict)  # n_steps, tool_cost, latency, tokens, recovery_rate

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict — the on-disk trajectory format (``trajectory.json``)."""
        return {
            "sample_id": self.sample_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": _json_safe(self.final_answer),
            "ground_truth": _json_safe(self.ground_truth),
            "outcome": self.outcome.value,
            "metrics": _json_safe(self.metrics),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict],
        *,
        sample_id: str,
        goal: str = "",
        final_answer: Any = None,
        ground_truth: Any = None,
        outcome: Label = Label.UNKNOWN,
        metrics: dict | None = None,
    ) -> "Trajectory":
        """Build a Trajectory from a list of plain dict step-records.

        This is the seam for an external engine (e.g. the node-tree
        ``conversation_history`` of an existing API runner): map each turn to a
        ``{role, content, tool_call, observation, span, agent_id}`` dict and
        pass them here.  No engine internals leak into the analysis layer.
        """
        steps: list[Step] = []
        for i, rec in enumerate(records):
            role = rec.get("role", StepRole.ACTOR)
            steps.append(
                Step(
                    idx=i,
                    role=StepRole(role) if not isinstance(role, StepRole) else role,
                    content=rec.get("content"),
                    agent_id=rec.get("agent_id", "main"),
                    tool_call=rec.get("tool_call"),
                    observation=rec.get("observation"),
                    span=rec.get("span", {}),
                )
            )
        return cls(
            sample_id=sample_id,
            goal=goal,
            steps=steps,
            final_answer=final_answer,
            ground_truth=ground_truth,
            outcome=outcome,
            metrics=metrics or {},
        )


@dataclass
class FailureCase:
    """A single unit of failure analysis.

    Attributes:
        inputs:     What was fed to the model (:class:`Inputs`).
        expected:   Gold / expected behaviour, if known.
        observed:   What the model actually produced, if run.
        label:      :class:`Label` — pass / fail / unknown.
        tags:       Free-form failure-taxonomy tags (e.g. ``{"hallucination"}``).
        provenance: How this case was created (:class:`Provenance`).
        id:         Stable identifier (auto-generated if omitted).
        metadata:   Free-form extra fields.
    """

    inputs: Inputs
    expected: Any = None
    observed: Any = None
    trajectory: Optional[Trajectory] = None  # set for multi-step / agent cases
    label: Label = Label.UNKNOWN
    tags: set[str] = field(default_factory=set)
    provenance: Provenance = field(default_factory=Provenance)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prompt(cls, prompt: str, **kwargs) -> "FailureCase":
        """Build a case from a raw prompt string."""
        return cls(inputs=Inputs(prompt=prompt), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "inputs": {
                "prompt": self.inputs.prompt,
                "image": _json_safe(self.inputs.image),
                "audio": _json_safe(self.inputs.audio),
                "video": _json_safe(self.inputs.video),
            },
            "expected": self.expected,
            "observed": self.observed,
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "label": self.label.value,
            "tags": sorted(self.tags),
            "provenance": {
                "source": self.provenance.source.value,
                "metadata": self.provenance.metadata,
            },
            "metadata": self.metadata,
        }


class CaseBatch:
    """An ordered collection of :class:`FailureCase` with convenience helpers."""

    def __init__(self, cases: Iterable[FailureCase] | None = None) -> None:
        self._cases: list[FailureCase] = list(cases) if cases else []

    # -- constructors --------------------------------------------------
    @classmethod
    def from_prompts(cls, prompts: Iterable[str], **kwargs) -> "CaseBatch":
        """Build a batch from raw prompt strings."""
        return cls(FailureCase.from_prompt(p, **kwargs) for p in prompts)

    # -- list-like behaviour -------------------------------------------
    def __iter__(self) -> Iterator[FailureCase]:
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, idx: int) -> FailureCase:
        return self._cases[idx]

    def append(self, case: FailureCase) -> None:
        self._cases.append(case)

    # -- querying (used by the agent / stats) --------------------------
    def filter(
        self,
        label: Label | None = None,
        tags: set[str] | None = None,
    ) -> "CaseBatch":
        """Return a new batch matching *label* and/or containing all *tags*."""
        out = [
            c
            for c in self._cases
            if (label is None or c.label == label)
            and (tags is None or tags.issubset(c.tags))
        ]
        return CaseBatch(out)

    def stratified_head(self, n: int) -> "list[FailureCase]":
        """First *n* cases, but label-balanced: take FAIL and PASS in document
        order and interleave so a capped subsample keeps minority-class (FAIL)
        representation instead of whatever happens to sit at the head.

        White-box analyzers cap how many cases they probe (one forward each);
        on an enriched/curated batch a plain ``[:n]`` head is mostly PASS, which
        starves the FAIL group and leaves downstream group contrasts
        underpowered. Returns all cases unchanged when ``n >= len`` or ``n<=0``.
        """
        if n <= 0 or n >= len(self._cases):
            return list(self._cases)
        fails = [c for c in self._cases if c.label == Label.FAIL]
        rest = [c for c in self._cases if c.label != Label.FAIL]
        # Aim for a balanced split, but never drop available minority cases below
        # what fits: give FAIL up to half the budget, fill the rest with others.
        n_fail = min(len(fails), max(n // 2, n - len(rest)))
        keep = fails[:n_fail] + rest[: n - n_fail]
        # preserve original document order among the kept cases
        kept = set(id(c) for c in keep)
        return [c for c in self._cases if id(c) in kept]

    def __repr__(self) -> str:
        return f"CaseBatch(n={len(self)})"


def as_casebatch(
    data: str | FailureCase | Inputs | Trajectory | Iterable | CaseBatch,
) -> CaseBatch:
    """Normalise common inputs into a :class:`CaseBatch`.

    Accepts:
      - a ``str``                → single-case batch from the prompt,
      - a :class:`FailureCase`   → single-case batch,
      - an :class:`Inputs`       → single-case batch,
      - a :class:`Trajectory`    → single-case batch,
      - a :class:`CaseBatch`     → returned unchanged,
      - any iterable of the above.
    """
    if isinstance(data, CaseBatch):
        return data
    if isinstance(data, str):
        return CaseBatch([FailureCase.from_prompt(data)])
    if isinstance(data, FailureCase):
        return CaseBatch([data])
    if isinstance(data, Inputs):
        return CaseBatch([FailureCase(inputs=data)])
    if isinstance(data, Trajectory):  # agent analyzers accept a Trajectory directly
        return CaseBatch([FailureCase(inputs=Inputs(prompt=data.goal), trajectory=data)])
    if isinstance(data, Iterable):
        batch = CaseBatch()
        for item in data:
            # flatten via single-item normalisation
            for case in as_casebatch(item):
                batch.append(case)
        return batch
    raise TypeError(
        f"Cannot interpret {type(data).__name__} as cases. "
        "Pass a str, FailureCase, Inputs, Trajectory, CaseBatch, or an iterable of these."
    )
