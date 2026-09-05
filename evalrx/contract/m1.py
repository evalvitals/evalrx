"""M1 — probe: run analyzers on cases, emit per-analyzer measurements.

M1 measures; it does not judge. Its output carries no notion of pass/fail — the
outcome lives on the cases. Nothing downstream can do anything with M1's numbers
until they are joined back to labels by ``sample_id``, which is why that field is
the only hard requirement in an otherwise open payload.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from evalrx.contract.common import (
    ArtifactRef,
    CaseBatchRef,
    Modality,
    OpenWireModel,
    StageEnvelope,
    WireModel,
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class ProtocolWire(OpenWireModel):
    """The user's natural-language statement of what to investigate.

    Carried through to M2 (tool selection), M3 (prompt) and M4 (the
    protocol-consistency gate) — a hypothesis can be statistically real and still
    be off-topic, and only this makes that judgeable.
    """

    description: str
    task_domain: str | None = None
    probe_hints: list[str] = Field(default_factory=list)


#: A default Python repr: ``<pkg.Class object at 0x7f...>``. Not a model name.
_PYTHON_REPR = re.compile(r"^<[\w.]+ object at 0x[0-9a-f]+>$")


class ModelRef(WireModel):
    """Identity + declared capabilities of the model under diagnosis.

    ``modalities`` is a SET, matched by subset against each analyzer's
    ``applies_to_modalities``. Adding a modality adds one member here, not a new
    combination type.
    """

    name: str = Field(
        min_length=1,
        description="What to call this model on a screen — a spec key or a product "
                    "name, e.g. 'qwen3.5-2b' or 'VideoLLaMA2.1-7B-AV'.",
    )
    backend: str | None = None
    modalities: list[Modality] = Field(default_factory=lambda: ["text"])
    capabilities: list[str] = Field(
        default_factory=list,
        description="e.g. GENERATE / LOGPROBS / ATTENTION / HIDDEN_STATES.",
    )
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _not_a_repr(cls, v: str) -> str:
        """Reject a default Python repr as a model name.

        ``repr(model)`` was what the pipeline recorded, so the UI showed
        ``<videollama2_model.MockAVModel object at 0x795d025e8590>`` — unreadable,
        and worse, the address changes every run, so two runs of the SAME model
        record two different identities and nothing can be compared across them.
        A model that cannot state its own name gets its class name, which is at
        least stable; see ``contract.emit.model_ref``.
        """
        if _PYTHON_REPR.match(v.strip()):
            raise ValueError(
                f"model name is a Python repr, not a name: {v!r}. Give the model a "
                "`display_name` (or a spec key); an address is not an identity."
            )
        return v


class ProbeInput(WireModel):
    """Everything M1 needs. Cases are referenced, not inlined — a batch is large
    and is already persisted."""

    model: ModelRef
    data: CaseBatchRef
    protocol: ProtocolWire | None = None
    prior_hypotheses: list[str] = Field(
        default_factory=list, description="Hypothesis ids from earlier cycles, for focused re-probing."
    )
    hint_failure_modes: list[str] = Field(
        default_factory=list, description="Failure-mode tags used by the no-judge fallback selector."
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

#: Keys in a per-case row that identify the case rather than measure it.
ID_KEYS = frozenset({"sample_id", "case_id", "id"})


class PerCaseRow(OpenWireModel):
    """One measurement row. THE contract of the whole pipeline.

    Only ``sample_id`` is fixed. Every other key is analyzer-defined; any key
    whose value is a finite number or bool is automatically harvested as a signal
    named ``"<analyzer>.<key>"``. Non-numeric values are carried for display and
    ignored by the statistics.
    """

    sample_id: str = Field(min_length=1, description="Must equal FailureCaseWire.id. No fuzzy matching.")

    @model_validator(mode="after")
    def _numerics_must_be_flat(self) -> "PerCaseRow":
        # A nested dict of numbers looks like a signal to a human and is invisible
        # to the harvester, which scans one level only. Silently dropping it means
        # an analyzer's whole contribution vanishes with no error anywhere — so
        # reject it at the boundary and make the producer flatten.
        for key, value in (self.__pydantic_extra__ or {}).items():
            nested = None
            if isinstance(value, dict) and any(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values()
            ):
                nested = "dict"
            elif isinstance(value, (list, tuple)) and value and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
            ):
                # Observed live: step_rollout_value emits `step_values: [1.0, ...]`,
                # which reads as a signal and reaches no statistic.
                nested = "sequence"
            if nested:
                raise ValueError(
                    f"per_case[{key!r}] holds a numeric {nested}; the signal harvester scans one "
                    f"level of scalars only, so this is silently dropped. Reduce it to scalars "
                    f"({key}_mean / {key}_min / ...) or move the vector to "
                    f"artifacts['per_case_maps'], which the tensor-level tools do read."
                )
        return self

    def signals(self) -> dict[str, float]:
        """Numeric/bool keys that will become per-case signals."""
        return {
            k: float(v)
            for k, v in (self.__pydantic_extra__ or {}).items()
            if k not in ID_KEYS and isinstance(v, (int, float, bool))
        }


class FindingsWire(OpenWireModel):
    """Analyzer-specific light payload. JSON-safe by construction.

    Two shapes are read by the pipeline and therefore named here; everything else
    is free-form and passes through to the raw inspector.
    """

    per_case: list[PerCaseRow] = Field(
        default_factory=list, description="One row per case. Empty is valid (a scalar-only analyzer)."
    )
    by_strategy: dict[str, dict[str, float]] | None = Field(
        default=None,
        description="{strategy -> {case_id -> success}} for paired contrasts. "
                    "This is what makes an INTERVENTION-grade verdict possible downstream.",
    )

    def scalars(self) -> dict[str, float]:
        """Top-level numeric keys, harvested as aggregate signals."""
        return {
            k: float(v)
            for k, v in (self.__pydantic_extra__ or {}).items()
            if isinstance(v, (int, float, bool))
        }


class ResultWire(WireModel):
    """One analyzer's serialized output — the content of
    ``artifacts/c<cycle>_<analyzer>.result.json``.

    Note what is NOT here: ``Result.artifacts`` (tensors, attention maps) never
    goes inline. It is referenced through ``artifact_paths`` so a browser is
    never asked to parse a ``.npy``.
    """

    analyzer: str = Field(min_length=1, description="Registered analyzer name. Becomes the signal prefix.")
    model: str
    n_cases: int = Field(ge=0)
    findings: FindingsWire = Field(default_factory=FindingsWire)
    signal_docs: dict[str, str] = Field(
        default_factory=dict,
        description="metric -> what it measures, in one plain sentence, from the analyzer "
                    "that produced it. The analyzer is the only thing that knows; every "
                    "reader downstream was otherwise guessing from the identifier, and "
                    "guessing produced chart labels no one could read. A metric absent "
                    "here is undocumented — report it as such, do not paraphrase the name.",
    )
    artifact_paths: dict[str, ArtifactRef] = Field(
        default_factory=dict, description="Heavy outputs, by artifact key."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("analyzer")
    @classmethod
    def _no_dots(cls, v: str) -> str:
        # Signals are named "<analyzer>.<metric>"; a dot in the analyzer name makes
        # that name ambiguous to split.
        if "." in v:
            raise ValueError(f"analyzer name must not contain '.': {v!r}")
        return v


class AnalyzerSelection(WireModel):
    """Which analyzers ran, and how they were chosen.

    Kept separate from the results so a reader can tell "not selected" from
    "selected and produced nothing". The judge's per-analyzer reasoning is not
    here — it is the response half of ``prompts/c<cycle>_m1_selection.*``.

    Modality is recorded as three related sets rather than one ``model_kind``
    label. The label was a combination enum (``vlm`` / ``omni`` / ...), which is
    what design rule 3 forbids: it has no member for an audio-visual model, and
    "omni" collapses the one distinction routing depends on — an omni model
    evaluated on an audio benchmark must be routed as audio, not as everything
    it is capable of.
    """

    model_modalities: list[Modality] = Field(
        default_factory=lambda: ["text"],
        description="What the model DECLARES it can consume (from its spec).",
    )
    probed_modalities: list[Modality] = Field(
        default_factory=lambda: ["text"],
        description="What the case batch actually FILLS. Text is the floor, not a slot.",
    )
    routed_on: list[Modality] = Field(
        default_factory=lambda: ["text"],
        description="The slots routing actually used. Normally model ∩ probed; equal to "
                    "model_modalities when the batch filled no media slot at all, because "
                    "an empty batch is no evidence about what is under test. A reader "
                    "comparing this with probed_modalities can see that fallback happened.",
    )
    is_agent: bool = Field(
        default=False,
        description="The BATCH carries agent trajectories. Orthogonal to modality — a VLM "
                    "can drive a tool loop. NOT 'the model supports tool calls': every "
                    "chat model served over an OpenAI-compatible endpoint declares that, "
                    "so reading the capability here labelled a plain single-turn text run "
                    "'llm+agent' and told the reader trajectories were analysed when none "
                    "existed. Ranking makes the same distinction — a declared capability "
                    "never outranks the data.",
    )
    selector: Literal["llm_judge", "static_strategy", "explicit"] = "static_strategy"
    generated: list[str] = Field(
        default_factory=list, description="Bespoke probes written when no standard analyzer covered the mode."
    )

    @property
    def profile(self) -> str:
        """Display label (``llm`` / ``vlm`` / ``alm`` / ``avlm`` / ``av`` ...).

        Derived, never stored: a label is a lossy rendering of ``routed_on`` and
        a stored copy would be free to disagree with the set it summarises.
        Compose it for a heading; branch on the set, never on this string.
        """
        slots = set(self.routed_on)
        tag = "".join(
            letter for slot, letter in (("audio", "a"), ("video", "v"), ("image", "v"))
            if slot in slots
        )
        # image and video both render as the visual "v"; dedupe while keeping a<v order
        tag = "".join(dict.fromkeys(tag))
        base = f"{tag}lm" if tag else "llm"
        return f"{base}+agent" if self.is_agent else base


class ProbeOutput(StageEnvelope):
    """M1's serialized output.

    Keyed by analyzer name because downstream routes BY name: M3 writes
    ``test_design="attention.image_token_ratio"`` and M4 resolves it against this
    map. A list would make that a scan and would not enforce uniqueness.
    """

    model: ModelRef | None = Field(
        default=None,
        description="Who was diagnosed. Carried on the OUTPUT, not only on ProbeInput: "
                    "a reader opens this file with no input beside it, and the only "
                    "identity here used to be `ResultWire.model`, a repr — so the "
                    "answer to 'which model is this report about' was a memory address.",
    )
    results: dict[str, ResultWire] = Field(default_factory=dict)
    selection: AnalyzerSelection = Field(default_factory=AnalyzerSelection)
    failed_analyzers: dict[str, str] = Field(
        default_factory=dict,
        description="analyzer -> error. Some failed + some succeeded = PARTIAL, not FAILED.",
    )

    @model_validator(mode="after")
    def _keys_match_analyzer_field(self) -> "ProbeOutput":
        for key, res in self.results.items():
            if res.analyzer != key:
                raise ValueError(f"results[{key!r}].analyzer == {res.analyzer!r}; keys must match")
        return self

    def signal_names(self) -> list[str]:
        """Every ``"<analyzer>.<metric>"`` this output exposes to M2/M4 routing."""
        names: set[str] = set()
        for aname, res in self.results.items():
            names |= {f"{aname}.{k}" for k in res.findings.scalars()}
            for row in res.findings.per_case:
                names |= {f"{aname}.{k}" for k in row.signals()}
        return sorted(names)


__all__ = [
    "ProtocolWire", "ModelRef", "ProbeInput",
    "ID_KEYS", "PerCaseRow", "FindingsWire", "ResultWire",
    "AnalyzerSelection", "ProbeOutput",
]
