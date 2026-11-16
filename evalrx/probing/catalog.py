"""Probe tools — every registered analyzer described as a callable tool.

An :class:`~evalrx.core.analyzer.Analyzer` already carries what a tool needs:
a registered name, constructor hyper-parameters, required capabilities, the
modalities it reads, and what its numbers mean. This module reads that off the
class and presents it in one uniform shape, so a caller (a person, the M1
selector, or an LLM judge) can ask three questions without reading source:

* what probes exist, and what does each one answer?  (:func:`probe_specs`)
* which of them can run on THIS model and THIS batch, and if one cannot,
  what exactly is missing?                               (:func:`describe_probes`)
* what arguments does a probe take?        (:meth:`ProbeToolSpec.parameters_schema`)

The availability report follows the convention of VLMProbe's
``Prober.describe()``: every method is listed, an unavailable one says why, and
"cannot run here" is never confused with "ran and measured zero".

Nothing here imports torch, and nothing here runs a model.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from evalrx.core.capability import Capability
from evalrx.core.registry import registry
from evalrx.core.tool import Tool

if TYPE_CHECKING:
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.model import Model

# ----------------------------------------------------------------------
# Families
# ----------------------------------------------------------------------

#: ``analyzers/<package>`` -> family name. Packages not listed keep their name.
_PACKAGE_FAMILY = {"patching": "causal", "weights": "weight_edit"}

#: What each family is for, in one sentence a judge can route on.
FAMILY_QUESTIONS: dict[str, str] = {
    "perturbation": "Does the answer change when the input is changed in a controlled way?",
    "uncertainty": "Does the model know when it is wrong?",
    "hallucination": "Does the output assert things the input does not support?",
    "reasoning": "Is the failure in extraction, termination, arithmetic, knowledge or the chain?",
    "attention": "Where does the model look?",
    "attribution": "Which input regions does the answer depend on?",
    "lens": "What do intermediate states predict, layer by layer?",
    "causal": "Which internal route carries the effect, measured by intervening on it?",
    "geometry": "How are cases arranged in representation space?",
    "weight_edit": "What does a direct weight edit change, and is the change specific?",
    "agent": "Where does a multi-step agent run go wrong?",
}

#: Families whose probes modify the forward pass or the parameters while they run.
_INTERVENING_FAMILIES = frozenset({"causal", "weight_edit"})

#: Capabilities a black-box (API) model can provide.
_BLACKBOX_CAPS = frozenset({Capability.GENERATE, Capability.LOGPROBS, Capability.TOOL_CALLS})

_JSON_TYPES: dict[type, str] = {
    bool: "boolean", int: "integer", float: "number", str: "string",
    list: "array", tuple: "array", dict: "object",
}
_ANNOTATION_TYPES = {
    "bool": "boolean", "int": "integer", "float": "number", "str": "string",
    "list": "array", "tuple": "array", "dict": "object",
    "sequence": "array", "iterable": "array", "mapping": "object",
}


def family_of(cls: type) -> str:
    """Family of an analyzer class, from its ``analyzers/<package>`` location.

    A class attribute ``probe_family`` overrides the location, for analyzers
    registered from outside ``evalrx.analyzers``.
    """
    explicit = getattr(cls, "probe_family", None)
    if explicit:
        return str(explicit)
    parts = (cls.__module__ or "").split(".")
    if "analyzers" in parts:
        idx = parts.index("analyzers")
        if len(parts) > idx + 2:
            package = parts[idx + 1]
            return _PACKAGE_FAMILY.get(package, package)
    return "custom"


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeParam:
    """One constructor argument of a probe.

    ``injected`` marks an argument that cannot be written as JSON (a model, a
    callable, a weight edit): the host supplies it, a judge never does.
    """

    name: str
    json_type: "str | None" = None
    default: Any = None
    required: bool = False
    injected: bool = False
    nullable: bool = False

    def schema(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.json_type:
            out["type"] = [self.json_type, "null"] if self.nullable else self.json_type
        if not self.required and _is_jsonable(self.default):
            out["default"] = list(self.default) if isinstance(self.default, tuple) else self.default
        return out


def _is_jsonable(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_jsonable(v) for k, v in value.items())
    return False


def _annotation_type(annotation: Any) -> "tuple[str | None, bool, bool]":
    """``(json_type, nullable, annotated)`` read from a constructor annotation.

    ``annotated`` is False when there is nothing to read; a present annotation
    with no JSON counterpart (a callable, a model) returns ``(None, _, True)``.
    """
    if annotation is inspect.Parameter.empty:
        return None, False, False
    if isinstance(annotation, type):
        return _JSON_TYPES.get(annotation), False, True
    text = str(annotation).strip().strip("\"'")
    nullable = False
    wrapped = re.fullmatch(r"(?:typing\.)?(Optional|Union)\[(.*)\]", text, re.DOTALL)
    if wrapped:
        nullable = wrapped.group(1) == "Optional"
        text = wrapped.group(2) if nullable else wrapped.group(2).replace(",", "|")
    if text in {"Any", "typing.Any", "object"}:
        return None, True, False
    members = [m.strip() for m in _split_top_level(text)]
    nullable = nullable or any(m in {"None", "NoneType"} for m in members)
    kinds = {
        _ANNOTATION_TYPES.get(re.split(r"[\[\s]", m, maxsplit=1)[0].split(".")[-1].lower())
        for m in members if m not in {"None", "NoneType"}
    }
    if None in kinds:
        return None, nullable, True  # at least one member has no JSON form
    if kinds == {"integer", "number"}:
        kinds = {"number"}
    return (next(iter(kinds)) if len(kinds) == 1 else None), nullable, True


def _split_top_level(text: str) -> list[str]:
    """Split a union on ``|`` outside brackets (``dict[str, int | None] | None``)."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == "|" and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _params_of(cls: type) -> "tuple[ProbeParam, ...]":
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return ()
    params: list[ProbeParam] = []
    for name, p in signature.parameters.items():
        if name == "self" or p.kind in (inspect.Parameter.VAR_POSITIONAL,
                                        inspect.Parameter.VAR_KEYWORD):
            continue
        required = p.default is inspect.Parameter.empty
        json_type, nullable, annotated = _annotation_type(p.annotation)
        default = None if required else p.default
        if json_type is None and not required and default is not None:
            json_type = _JSON_TYPES.get(type(default))
        if not required and default is None:
            nullable = True
        # Host-only arguments: a value that is not JSON, or an annotation that
        # names something with no JSON form (a callable, a model, an edit).
        injected = (
            not _is_jsonable(default)
            or (json_type is None and (required or annotated))
        )
        params.append(ProbeParam(name=name, json_type=json_type, default=default,
                                 required=required, injected=injected, nullable=nullable))
    return tuple(params)


# ----------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------

def _first_line(text: "str | None") -> str:
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def is_stub(cls: type) -> bool:
    """True when ``_run`` only raises ``NotImplementedError`` (a registered placeholder)."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls._run)))
    except (OSError, TypeError, SyntaxError, AttributeError):
        return False
    body = [n for n in tree.body[0].body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def _failure_modes_by_probe() -> dict[str, tuple[str, ...]]:
    from evalrx.eval_agent.stages.probe import _FAILURE_MODE_TO_ANALYZERS

    inverse: dict[str, list[str]] = {}
    for mode, names in _FAILURE_MODE_TO_ANALYZERS.items():
        for name in names:
            inverse.setdefault(name, []).append(mode)
    return {name: tuple(modes) for name, modes in inverse.items()}


@dataclass(frozen=True)
class ProbeToolSpec:
    """A registered analyzer, described as a tool.

    Attributes:
        name:          Registered analyzer key; also the tool name.
        family:        Functional family (``lens``, ``causal``, ``uncertainty``, ...).
        summary:       One line saying what the probe does.
        question:      The question its result answers.
        access:        ``"black_box"`` (generation / logprobs only), ``"white_box"``
                       (needs internals) or ``"data_only"`` (reads recorded data;
                       the model may be ``None``).
        requires:      Capability values the model must provide.
        modalities:    Modalities the probe applies to.
        data_slots:    Modality slots the batch must actually fill.
        needs_trajectories: Whether it reads multi-step agent runs.
        intervenes:    Whether it changes the forward pass or weights while it
                       runs (always restored afterwards).
        cost:          ``"light"`` | ``"moderate"`` | ``"heavy"`` — a coarse hint
                       derived from access and family unless the analyzer class
                       sets ``probe_cost``.
        params:        Constructor arguments.
        signals:       ``metric -> meaning`` from the analyzer's ``signal_docs``.
        failure_modes: Failure-mode tags this probe is routed to.
        implemented:   False for a registered placeholder.
    """

    name: str
    family: str
    summary: str
    question: str
    access: str
    requires: tuple[str, ...]
    modalities: tuple[str, ...]
    data_slots: tuple[str, ...]
    needs_trajectories: bool
    intervenes: bool
    cost: str
    params: tuple[ProbeParam, ...]
    signals: Mapping[str, str] = field(default_factory=dict)
    failure_modes: tuple[str, ...] = ()
    implemented: bool = True
    analyzer_cls: "type[Analyzer] | None" = field(default=None, repr=False, compare=False)

    # -- arguments ------------------------------------------------------
    @property
    def injected_params(self) -> tuple[str, ...]:
        """Required arguments only the host can supply."""
        return tuple(p.name for p in self.params if p.required and p.injected)

    @property
    def callable_params(self) -> tuple[ProbeParam, ...]:
        """Arguments a caller may pass as JSON."""
        return tuple(p for p in self.params if not p.injected)

    def parameters_schema(self) -> dict[str, Any]:
        """JSON schema of the arguments a caller may pass."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {p.name: p.schema() for p in self.callable_params},
            "additionalProperties": False,
        }
        required = [p.name for p in self.callable_params if p.required]
        if required:
            schema["required"] = required
        return schema

    # -- views ----------------------------------------------------------
    def to_tool(self, fn: Any = None) -> Tool:
        """The probe as a :class:`~evalrx.core.tool.Tool` (OpenAI / HF tool schema)."""
        description = self.summary
        if self.question and self.question != self.summary:
            description = f"{self.summary} Answers: {self.question}"
        return Tool(name=self.name, description=description,
                    parameters=self.parameters_schema(), fn=fn)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "summary": self.summary,
            "question": self.question,
            "access": self.access,
            "requires": list(self.requires),
            "modalities": list(self.modalities),
            "data_slots": list(self.data_slots),
            "needs_trajectories": self.needs_trajectories,
            "intervenes": self.intervenes,
            "cost": self.cost,
            "parameters": self.parameters_schema(),
            "injected_params": list(self.injected_params),
            "signals": dict(self.signals),
            "failure_modes": list(self.failure_modes),
            "implemented": self.implemented,
        }


def _signals_of(cls: type) -> dict[str, str]:
    out: dict[str, str] = {}
    for metric, doc in (getattr(cls, "signal_docs", None) or {}).items():
        out[str(metric)] = str(doc[1] if isinstance(doc, tuple) and len(doc) == 2 else doc)
    return out


def _access_of(cls: type) -> str:
    requires = frozenset(getattr(cls, "requires", frozenset()))
    if not requires:
        return "data_only"
    return "black_box" if requires <= _BLACKBOX_CAPS else "white_box"


def _cost_of(cls: type, access: str, intervenes: bool) -> str:
    explicit = getattr(cls, "probe_cost", None)
    if explicit:
        return str(explicit)
    if access == "data_only":
        return "light"
    if intervenes or Capability.GRADIENTS in getattr(cls, "requires", frozenset()):
        return "heavy"
    return "moderate"


def spec_for(name: str, cls: "type[Analyzer] | None" = None,
             failure_modes: "Mapping[str, tuple[str, ...]] | None" = None) -> ProbeToolSpec:
    """Build the :class:`ProbeToolSpec` of one registered analyzer."""
    cls = cls or registry.analyzers.get(name)
    family = family_of(cls)
    summary = str(getattr(cls, "description", "") or _first_line(cls.__doc__) or name)
    access = _access_of(cls)
    intervenes = bool(getattr(cls, "probe_intervenes", family in _INTERVENING_FAMILIES))
    modes = failure_modes if failure_modes is not None else _failure_modes_by_probe()
    return ProbeToolSpec(
        name=name,
        family=family,
        summary=summary,
        question=str(getattr(cls, "question", "") or FAMILY_QUESTIONS.get(family, summary)),
        access=access,
        requires=tuple(sorted(c.value for c in getattr(cls, "requires", frozenset()))),
        modalities=tuple(sorted(getattr(cls, "applies_to_modalities", frozenset()))),
        data_slots=tuple(sorted(getattr(cls, "requires_modalities", frozenset()) or ())),
        needs_trajectories=bool(getattr(cls, "requires_trajectories", False)),
        intervenes=intervenes,
        cost=_cost_of(cls, access, intervenes),
        params=_params_of(cls),
        signals=_signals_of(cls),
        failure_modes=tuple(modes.get(name, ())),
        implemented=not is_stub(cls),
        analyzer_cls=cls,
    )


def probe_specs(names: "Iterable[str] | None" = None) -> dict[str, ProbeToolSpec]:
    """``{name: ProbeToolSpec}`` for every registered analyzer (or just *names*)."""
    modes = _failure_modes_by_probe()
    chosen = sorted(registry.analyzers.list()) if names is None else list(names)
    return {name: spec_for(name, failure_modes=modes) for name in chosen}


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeAvailability:
    """Whether a probe can run here, and every reason it cannot.

    ``codes`` are stable identifiers for programmatic use: ``not_implemented``,
    ``capability``, ``modality``, ``data_slot``, ``trajectories``,
    ``data_precondition``, ``injection``.
    """

    available: bool
    missing: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "missing": list(self.missing),
                "codes": list(self.codes)}


def availability(spec: ProbeToolSpec, model: "Model | None", data: Any = None, *,
                 overrides: "Iterable[str]" = ()) -> ProbeAvailability:
    """Check *spec* against a model and, when given, the batch about to be probed.

    Args:
        spec:      The probe.
        model:     The model under test (``None`` is valid for data-only probes).
        data:      The case batch, when known. Without it the data-shape checks
                   are skipped: no evidence is not evidence of absence.
        overrides: Names for which the host already holds a constructed
                   analyzer; those never report a missing injected argument.
    """
    missing: list[str] = []
    codes: list[str] = []

    def fail(code: str, reason: str) -> None:
        codes.append(code)
        missing.append(reason)

    if not spec.implemented:
        fail("not_implemented", "registered placeholder; the analysis is not implemented yet")

    provided = {c.value if isinstance(c, Capability) else str(c)
                for c in (getattr(model, "capabilities", None) or ())}
    lacking = [c for c in spec.requires if c not in provided]
    if lacking:
        where = "a local white-box model" if spec.access == "white_box" else "the model"
        fail("capability", f"needs {where} that provides: {', '.join(lacking)}")

    if model is not None or spec.requires:
        model_modalities = set(getattr(model, "modalities", None) or {"text"})
        if spec.modalities and not (set(spec.modalities) & model_modalities):
            fail("modality", f"applies to {', '.join(spec.modalities)} models; "
                             f"this model declares {', '.join(sorted(model_modalities))}")

    if data is not None:
        from evalrx.eval_agent.stages.probe import StrategyProbe, _carries_trajectories

        if spec.needs_trajectories and not _carries_trajectories(data):
            fail("trajectories", "reads agent trajectories; the batch has none")
        if spec.data_slots and spec.name in StrategyProbe.slot_starved({spec.name}, data, model):
            fail("data_slot", f"the batch fills none of: {', '.join(spec.data_slots)}")
        if not codes:
            from evalrx.eval_agent.stages.probe_agent import _analyzer_data_preconditions_met

            if not _analyzer_data_preconditions_met(spec.name, data, model):
                fail("data_precondition", "the cases lack the labels or metadata this probe reads")

    if spec.injected_params and spec.name not in set(overrides):
        fail("injection", "the host must supply: " + ", ".join(spec.injected_params)
             + " (pass a constructed analyzer via analyzer_overrides)")

    return ProbeAvailability(available=not codes, missing=tuple(missing), codes=tuple(codes))


def describe_probes(model: "Model | None", data: Any = None, *,
                    overrides: "Iterable[str]" = (),
                    families: "Iterable[str] | None" = None) -> dict[str, dict[str, Any]]:
    """Report every probe with its availability on *model* (and *data*).

    Returns ``{name: {...spec fields..., "available": bool, "missing": [...],
    "codes": [...]}}``. No forward pass is made.
    """
    wanted = None if families is None else {str(f) for f in families}
    overrides = tuple(overrides)
    report: dict[str, dict[str, Any]] = {}
    for name, spec in probe_specs().items():
        if wanted is not None and spec.family not in wanted:
            continue
        report[name] = {**spec.to_dict(),
                        **availability(spec, model, data, overrides=overrides).to_dict()}
    return report
