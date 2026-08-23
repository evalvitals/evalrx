"""Shared wire types for the M1-M5 stage contract.

These models describe the **serialized** shape crossing a stage boundary — what
lands on disk and what a frontend reads — not the in-memory Python objects.
The two are deliberately different: ``AnalysisReport.to_dict()`` drops
``raw_results`` and flattens ``findings`` to strings, so a contract written
against the in-memory dataclass would not describe what a reader actually gets.

Design rules enforced here (see ``docs/stage_io.md``):

1. **Envelope strict, payload free.** Fields the pipeline joins/aggregates on are
   required and typed; analyzer- and dataset-specific content stays open
   (``extra="allow"``) so a new analyzer never requires a contract change.
2. **One join key.** Every per-case row carries ``sample_id`` and it equals
   :attr:`FailureCaseWire.id`. A broken join is reported, never silently
   degraded into "no effect found" (see :class:`JoinReport`).
3. **Modalities are slots + sets, never combination enums.** Adding a modality
   must not multiply the type space.
4. **Absent / null / empty / zero / false are five different things.** Every
   optional field is explicitly ``| None`` so a reader can tell them apart; never
   collapse them with a truthiness fallback.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Bumped only when a field's *meaning* changes. Adding fields is additive and
#: does NOT bump it — readers must ignore unknown fields, not fail on them.
#:
#: 4: ``AnalyzerSelection.model_kind`` (a combination enum) replaced by modality
#:    slots + ``is_agent``; ``StepWire.role`` realigned to ``core.case.StepRole``.
SCHEMA_VERSION = 4


class WireModel(BaseModel):
    """Base for every model in this contract: strict envelope, no coercion."""

    model_config = ConfigDict(
        extra="forbid",           # envelope fields are closed by default
        strict=False,             # allow int->float, but not str->int
        validate_assignment=True,
        frozen=False,
    )


class OpenWireModel(WireModel):
    """Base for payload areas whose content is producer-defined.

    Unknown keys are kept (not dropped) so the raw inspector can show them and a
    newer producer never loses data against an older reader.
    """

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Enums — closed value sets the UI branches on
# ---------------------------------------------------------------------------

class Label(str, Enum):
    """Per-case outcome. Not a statistical verdict."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Source(str, Enum):
    HUMAN = "human"
    DATASET = "dataset"
    AGENT = "agent"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class EvidenceGrade(str, Enum):
    """Strength tier of the deciding evidence.

    ``INTERVENTION`` beats ``OBSERVATIONAL``: a paired contrast changed one
    variable and re-ran, so it carries causal weight an association does not.
    """

    INTERVENTION = "intervention"
    OBSERVATIONAL = "observational"
    NONE = "none"


class StageState(str, Enum):
    """Per-stage UI state. A single generic "N/A" loses information the reader
    needs — in particular ``EMPTY`` (ran, produced nothing) and ``FAILED``
    (did not run correctly) must never render the same."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    ABSTAINED = "abstained"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


#: Open string set on purpose — a new modality must be one more member of a set,
#: never a new combination enum (VA / TIA / ... is 2^n and does not compose).
#:
#: LLM / VLM / ALM / AVLM are not four types in this contract; they are four
#: subsets of this set. An AVLM is ``{"text", "image", "audio"}`` and needs no
#: schema change to exist.
Modality = Literal["text", "image", "audio", "video"]

#: Every modality slot other than ``text``. Text is always present (a prompt is
#: mandatory), so it is never a *slot* — it is the floor.
MEDIA_SLOTS: tuple[Modality, ...] = ("image", "audio", "video")


# ---------------------------------------------------------------------------
# References — things that are NOT inline payload
# ---------------------------------------------------------------------------

class ExternalRef(WireModel):
    """Pointer replacing an inline value that exceeded the size budget.

    Detect by shape (``kind == "external_ref"``), lazy-load ``path``, and keep
    ``n_items`` visible while loading. Never render the pointer as one item.
    """

    kind: Literal["external_ref"] = "external_ref"
    path: str = Field(description="Relative to runRoot. Never an absolute host path.")
    n_items: int | None = Field(default=None, description="Item count in the target, when countable.")
    bytes: int = Field(description="Serialized size of the externalized value.")


T = TypeVar("T")
#: An inline value that may have been swapped for a pointer. Readers must handle
#: both arms; the discriminator is ``kind == "external_ref"``.
Externalizable = Union[T, ExternalRef]


class MediaRef(WireModel):
    """A media *reference*, never media bytes.

    ``kind`` tells the reader whether a preview is even possible: an in-memory
    PIL image that was never persisted degrades to ``descriptor`` (e.g.
    ``"<image 640x480>"``), and offering a broken preview for it is a bug.
    """

    kind: Literal["path", "url", "descriptor"]
    value: str
    mime: str | None = None
    n_bytes: int | None = None

    @classmethod
    def coerce(cls, value: Any) -> "MediaRef | None":
        """Build a ref from whatever a producer put in a modality slot.

        ``FailureCase.to_dict()`` runs every slot through ``_json_safe``, which
        yields a bare string: a path/URL when the slot held one, or a descriptor
        like ``"<image 640x480>"`` for a PIL object that was never persisted.
        Both are legitimate producer output, so the contract classifies them here
        rather than rejecting the case — a case is not invalid because its image
        lived in memory.
        """
        if value is None or isinstance(value, MediaRef):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        text = str(value).strip()
        if not text:
            return None
        if text.startswith("<") and text.endswith(">"):
            return cls(kind="descriptor", value=text)
        if text.startswith(("http://", "https://", "data:", "s3://", "gs://")):
            return cls(kind="url", value=text)
        return cls(kind="path", value=text)


class ArtifactRef(WireModel):
    """A heavy artifact (``.npy``, ``.png``, workspace dir) left on disk.

    Heavy arrays never go inline: a browser must not parse a ``.npy``, and a
    JSONL event must not carry an attention map.
    """

    path: str = Field(description="Relative to runRoot.")
    media_type: Literal["npy", "png", "jpg", "json", "text", "dir", "other"] = "other"
    bytes: int | None = None


# ---------------------------------------------------------------------------
# Case data — the raw rows every stage joins back to
# ---------------------------------------------------------------------------

class InputsWire(OpenWireModel):
    """The question, plus optional modality slots.

    Slots are independent and may be filled in any combination. There is
    deliberately no "modality" discriminator field: ``prompt + image`` and
    ``prompt + image + audio`` are the same type with different slots filled.
    """

    prompt: str
    image: MediaRef | None = None
    audio: MediaRef | None = None
    video: MediaRef | None = None

    @field_validator("image", "audio", "video", mode="before")
    @classmethod
    def _coerce_media(cls, v: Any) -> Any:
        return MediaRef.coerce(v)

    def modalities(self) -> set[Modality]:
        """Modalities actually present on this case."""
        present: set[Modality] = {"text"}
        for slot in MEDIA_SLOTS:
            if getattr(self, slot) is not None:
                present.add(slot)
        return present


#: Legacy role names this contract shipped before it was aligned to
#: :class:`evalvitals.core.case.StepRole`. Mapped rather than rejected so a
#: trajectory written by an older producer still validates.
_LEGACY_STEP_ROLES = {"env": "tool", "critic": "verifier"}


class StepWire(OpenWireModel):
    """One step of an agent trajectory.

    ``role`` is exactly :class:`evalvitals.core.case.StepRole`. It was not, and
    the divergence was not cosmetic: ``planner`` / ``verifier`` / ``memory`` /
    ``system`` are values the producer emits every multi-agent run, and every one
    of them failed validation here, while ``env`` / ``critic`` named steps no
    producer can emit.
    """

    idx: int
    role: Literal[
        "user", "planner", "actor", "tool", "verifier", "memory", "system"
    ] = "actor"
    content: Any | None = None
    agent_id: str = "main"
    tool_call: dict[str, Any] | None = None
    observation: Any | None = None
    span: dict[str, Any] = Field(default_factory=dict, description="tokens / latency_ms / cost / model")
    is_first_error: bool | None = None
    failure_mode: str | None = None
    judge_confidence: float | None = None

    @field_validator("role", mode="before")
    @classmethod
    def _map_legacy_role(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _LEGACY_STEP_ROLES.get(v, v)
        return getattr(v, "value", v)


class TrajectoryWire(OpenWireModel):
    """Multi-step agent run. Present only for agent cases.

    A trajectory is a *data* property, not a model capability: trajectory
    analyzers run with ``model=None`` on rows loaded from disk, so eligibility
    is gated on the batch in hand, not on what the model can do.
    """

    sample_id: str = Field(description="Must equal the owning FailureCaseWire.id.")
    goal: str = ""
    steps: list[StepWire] = Field(default_factory=list)
    final_answer: Any | None = None
    ground_truth: Any | None = None
    outcome: Label = Label.UNKNOWN
    metrics: dict[str, Any] = Field(default_factory=dict)


class ProvenanceWire(OpenWireModel):
    source: Source = Source.HUMAN
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureCaseWire(OpenWireModel):
    """One evaluated case. ``id`` is the pipeline's single join key.

    ``expected`` / ``observed`` are intentionally ``Any``: a gold answer may be a
    string, a bounding box, or a structured object depending on the benchmark.
    Render strings directly and objects in a structured/raw view.
    """

    id: str = Field(min_length=1, description="Stable join key. Every per-case row points here.")
    inputs: InputsWire
    expected: Any | None = None
    observed: Any | None = None
    trajectory: TrajectoryWire | None = None
    label: Label = Label.UNKNOWN
    tags: list[str] = Field(default_factory=list, description="Order is not semantically meaningful.")
    provenance: ProvenanceWire = Field(default_factory=ProvenanceWire)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-specific columns. Never assume a fixed schema here.",
    )

    @field_validator("trajectory")
    @classmethod
    def _trajectory_id_matches(cls, v: TrajectoryWire | None, info) -> TrajectoryWire | None:
        # A trajectory whose sample_id differs from the case id gets registered
        # under BOTH keys downstream, which double-counts that case in every
        # group comparison. Reject the divergence at the boundary instead.
        case_id = (info.data or {}).get("id")
        if v is not None and case_id and v.sample_id != case_id:
            raise ValueError(
                f"trajectory.sample_id={v.sample_id!r} != case id={case_id!r}; "
                "a divergent id double-counts this case in stats group splits"
            )
        return v


class CaseBatchWire(WireModel):
    """Serialized ``CaseBatch``. The in-memory class has no ``to_dict()``, so
    this shape is the contract for anything that crosses a process boundary."""

    schema_version: int = SCHEMA_VERSION
    n_cases: int
    cases: list[FailureCaseWire]

    @field_validator("cases")
    @classmethod
    def _ids_unique(cls, v: list[FailureCaseWire]) -> list[FailureCaseWire]:
        ids = [c.id for c in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate case ids break every downstream join: {sorted(dupes)[:5]}")
        return v

    def probed_modalities(self) -> set[Modality]:
        """Slots this batch actually fills — the union over its cases.

        Distinct from what the model *declares*, and it is the distinction that
        decides analyzer routing: an omni model evaluated on an audio benchmark
        declares image too, and routing on the declaration sends image analyzers
        at a batch with no images in it.
        """
        present: set[Modality] = set()
        for case in self.cases:
            present |= case.inputs.modalities()
        return present or {"text"}


class CaseBatchRef(WireModel):
    """A batch referenced by file rather than inlined (the normal case)."""

    path: str
    n_cases: int
    split: Literal["all", "explore", "confirm", "discovery"] | None = None


# ---------------------------------------------------------------------------
# Cross-cutting diagnostics
# ---------------------------------------------------------------------------

class JoinReport(WireModel):
    """Health of one ``sample_id`` <-> ``FailureCase.id`` join.

    This exists because a broken join and a real null result are indistinguishable
    downstream: unmatched ids are skipped, both groups come out empty, and the
    tool reports "no significant association" — the same output as a model with
    no defect. Emitting coverage makes "the pipeline is broken" a separate,
    visible state from "the model is fine".
    """

    left: str = Field(description="e.g. 'attention.findings.per_case'")
    right: str = Field(description="e.g. 'CaseBatch(explore)'")
    n_left: int
    n_right: int
    n_matched: int
    coverage: float = Field(ge=0.0, le=1.0, description="n_matched / n_left")
    unmatched_sample: list[str] = Field(
        default_factory=list, max_length=10, description="First few unmatched ids, for triage."
    )

    @property
    def is_broken(self) -> bool:
        """Zero overlap is a plumbing failure, never a scientific finding."""
        return self.n_left > 0 and self.n_matched == 0


class StageStatus(WireModel):
    """Terminal state of one stage in one cycle.

    ``EMPTY`` and ``ABSTAINED`` are successes with nothing to show; ``FAILED``
    and ``UNAVAILABLE`` are not. Collapsing them loses the only signal a reader
    has for whether to trust the absence of a result.
    """

    stage: Literal["pre_m1", "m1", "explore", "m2", "m3", "m4_surgery", "m4_fix", "m5"]
    state: StageState
    cycle: int = Field(description="Normal cycles start at 0; the post-loop fix uses -1.")
    reason: str | None = Field(
        default=None,
        description="Why, for any state that is not plain success — the skip rule, the error, "
                    "the reason for abstaining. One field, because a reader wants one answer.",
    )
    duration_sec: float | None = None
    joins: list[JoinReport] = Field(default_factory=list)


class StageEnvelope(WireModel):
    """Common header on every persisted stage output.

    Which stage this is comes from ``status.stage`` — carrying it twice invites
    the two copies to disagree, and the reader has no way to tell which is right.
    """

    schema_version: int = SCHEMA_VERSION
    trace_id: str = Field(description="Run-level correlation id.")
    span_id: str | None = Field(default=None, description="e.g. 'c0.m1'.")
    cycle: int = 0
    produced_at: str = Field(description="ISO-8601 UTC.")
    status: StageStatus


__all__ = [
    "SCHEMA_VERSION", "WireModel", "OpenWireModel",
    "Label", "Source", "HypothesisStatus", "EvidenceGrade", "StageState",
    "Modality", "MEDIA_SLOTS",
    "ExternalRef", "Externalizable", "MediaRef", "ArtifactRef",
    "InputsWire", "StepWire", "TrajectoryWire", "ProvenanceWire",
    "FailureCaseWire", "CaseBatchWire", "CaseBatchRef",
    "JoinReport", "StageStatus", "StageEnvelope",
]
