"""L4 fix executor — a training-free weight edit (weight-space programming).

Fine-tuning is not the only way to change a model's parameters. A weight edit
writes a small, structured change straight into a few tensors: here, the
rank-one rule ``w' = w - alpha * u (u^T w)`` on selected columns of some MLP
output projections. No gradient step is taken and no data is fitted beyond two
group means.

What this executor does, and what keeps it honest:

* The edit is built by :func:`evalrx.wsp.direction.contrast_edit` from a
  DIAGNOSIS pool of labelled cases (failing vs passing), or loaded from a
  directory the host supplies. The pool must be disjoint from the batch being
  validated; an overlap is refused, because deriving the direction from the
  cases it is then scored on is leakage, not repair.
* The edit is written in place for the duration of the generation pass and
  restored exactly afterwards (:func:`evalrx.wsp.edit.applied`), also when a
  case raises.
* The scores go back to the caller's paired validation unchanged, so a weight
  edit is judged by the same test as every other tier, including the cases it
  breaks.
* Unless disabled, the same batch is also generated under a matched
  ``random_direction`` control. The control's pass count is recorded beside
  the edit's. A repair that a random change of the same size reproduces is a
  perturbation effect, and the record lets a reader see that.

The judge never writes the edit or any hook code; it can only be given the
few numeric settings in :class:`WeightEditSpec`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.eval_agent.stages.fix.tools import score_to_bool

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model

logger = logging.getLogger(__name__)

ScoreFn = Callable[["FailureCase", str], Optional[bool]]

CANDIDATE_NAME = "rank_one_weight_edit"
CANDIDATE_KIND = "weight_edit"


@dataclass
class WeightEditSpec:
    """Settings of one weight-edit repair candidate.

    Attributes:
        alpha:      Strength of ``w' = w - alpha u (u^T w)``. 1 removes the
                    selected neurons' write along the failure direction; values
                    above 1 reverse it.
        n_columns:  Neurons edited per layer.
        n_layers:   Layers edited when *layers* is not given (the ones whose
                    writes separate failing from passing cases most).
        layers:     Explicit decoder-layer indices.
        edit_dir:   Directory of a saved edit (``WeightEdit.save`` or the Astra
                    layout). When set, that edit is applied as written and the
                    contrast settings above are ignored.
        verify_base: With *edit_dir*: require the live tensors to match the
                    hashes recorded in the edit.
        control:    Also generate under a matched random-direction control.
        max_relative_change: Refuse an edit that changes any tensor by more
                    than this fraction of its norm. ``None`` = no limit.
        max_pool_cases: Cap on failing and on passing pool cases read to build
                    the direction.
        seed:       Seed of the control draw.
    """

    alpha: float = 1.0
    n_columns: int = 64
    n_layers: int = 2
    layers: "list[int] | None" = None
    edit_dir: str = ""
    verify_base: bool = False
    control: bool = True
    max_relative_change: "float | None" = None
    max_pool_cases: int = 64
    seed: int = 0

    @classmethod
    def from_payload(cls, payload: "dict[str, Any] | None") -> "WeightEditSpec":
        """Build from a candidate payload, ignoring keys that are not settings."""
        known = {k: v for k, v in (payload or {}).items() if k in cls.__dataclass_fields__}
        spec = cls(**known)
        spec.alpha = float(spec.alpha)
        spec.n_columns = int(spec.n_columns)
        spec.n_layers = int(spec.n_layers)
        if spec.layers is not None:
            spec.layers = [int(i) for i in spec.layers]
        if spec.n_columns < 1 or spec.n_layers < 1:
            raise ValueError("n_columns and n_layers must be positive")
        return spec

    def to_dict(self) -> "dict[str, Any]":
        return asdict(self)


@dataclass
class WeightEditRepairResult:
    """Outcome of one weight-edit generate-and-score pass.

    ``scores``/``ok``/``error`` mirror :class:`~.internals.LoraRepairResult`.
    ``audit`` records what was applied and how the control fared.
    """

    scores: "dict[str, Optional[bool]]" = field(default_factory=dict)
    outputs: "dict[str, str]" = field(default_factory=dict)
    ok: bool = False
    error: str = ""
    audit: "dict[str, Any]" = field(default_factory=dict)


def weight_edit_available(model: "Model") -> bool:
    """Whether *model* exposes a torch module with editable MLP output projections."""
    try:
        from evalrx.wsp.direction import write_sites

        return bool(write_sites(model))
    except Exception:  # noqa: BLE001 - no torch, no module, unknown architecture: not available
        return False


def _generate_all(model: "Model", cases: "CaseBatch", score_fn: ScoreFn,
                  ) -> "tuple[dict[str, Optional[bool]], dict[str, str]]":
    scores: "dict[str, Optional[bool]]" = {}
    outputs: "dict[str, str]" = {}
    for case in cases:
        try:
            out = str(model.generate(case.inputs))
        except Exception as exc:  # noqa: BLE001 - one case must not abort the pass
            logger.debug("weight-edit generate failed on %s: %s", case.id, exc)
            scores[case.id] = None
            continue
        outputs[case.id] = out
        scores[case.id] = score_to_bool(score_fn(case, out))
    return scores, outputs


def _pass_count(scores: "dict[str, Optional[bool]]") -> "dict[str, int]":
    return {"n_pass": sum(1 for s in scores.values() if s is True),
            "n_scored": sum(1 for s in scores.values() if s is not None)}


def run_weight_edit_repair(
    model: "Model",
    pool: "CaseBatch | None",
    val_cases: "CaseBatch",
    payload: "dict[str, Any] | None",
    score_fn: ScoreFn,
) -> WeightEditRepairResult:
    """Build (or load) a weight edit, generate *val_cases* under it, and score them.

    Never raises for an expected refusal: the reason comes back in ``error``
    with ``ok=False`` and the model untouched, so one candidate cannot abort a
    fix run.
    """
    try:
        spec = WeightEditSpec.from_payload(payload)
    except (TypeError, ValueError) as exc:
        return WeightEditRepairResult(error=f"invalid weight-edit settings: {exc}")
    if not weight_edit_available(model):
        return WeightEditRepairResult(
            error="needs a local white-box model with linear MLP output projections"
        )

    from evalrx.core.case import Label
    from evalrx.wsp.controls import build_control
    from evalrx.wsp.direction import contrast_edit
    from evalrx.wsp.edit import WeightEdit, applied
    from evalrx.wsp.structure import edit_structure

    if spec.edit_dir:
        try:
            edit = WeightEdit.load(spec.edit_dir)
        except Exception as exc:  # noqa: BLE001
            return WeightEditRepairResult(error=f"could not load edit from {spec.edit_dir!r}: {exc}")
    else:
        if pool is None or len(pool) == 0:
            return WeightEditRepairResult(
                error="no diagnosis pool configured — FixAgent(finetune_pool=...) supplies "
                "the labelled cases the edit direction is computed from, never the "
                "validation split"
            )
        overlap = {c.id for c in pool} & {c.id for c in val_cases}
        if overlap:
            return WeightEditRepairResult(
                error=f"the diagnosis pool shares {len(overlap)} case(s) with the batch "
                "being validated; the edit direction must come from disjoint cases"
            )
        fails = [c for c in pool if c.label == Label.FAIL][: spec.max_pool_cases]
        passes = [c for c in pool if c.label == Label.PASS][: spec.max_pool_cases]
        if not fails or not passes:
            return WeightEditRepairResult(
                error="the diagnosis pool needs both failing and passing cases "
                f"(has {len(fails)} failing, {len(passes)} passing)"
            )
        try:
            edit = contrast_edit(model, fails, passes, layers=spec.layers,
                                 n_layers=spec.n_layers, n_columns=spec.n_columns,
                                 alpha=spec.alpha, name=CANDIDATE_NAME)
        except Exception as exc:  # noqa: BLE001
            return WeightEditRepairResult(error=f"could not build the edit: {exc}")

    audit: "dict[str, Any]" = {"edit": edit.describe(), "settings": spec.to_dict()}
    try:
        audit["structure"] = edit_structure(model, edit)["tensors"]
        with applied(model, edit, verify_base=spec.verify_base,
                     max_relative_change=spec.max_relative_change) as report:
            audit["applied"] = report.to_dict()
            scores, outputs = _generate_all(model, val_cases, score_fn)
    except Exception as exc:  # noqa: BLE001 - refused before or while writing; weights restored
        return WeightEditRepairResult(error=f"the edit could not be applied: {exc}", audit=audit)
    audit["edit_result"] = _pass_count(scores)

    if spec.control:
        try:
            control = build_control(model, edit, "random_direction", seed=spec.seed)
            with applied(model, control):
                control_scores, _ = _generate_all(model, val_cases, score_fn)
            audit["control_random_direction"] = _pass_count(control_scores)
        except Exception as exc:  # noqa: BLE001 - a missing control is reported, not fatal
            audit["control_random_direction"] = {"error": str(exc)}
    return WeightEditRepairResult(scores=scores, outputs=outputs, ok=True, audit=audit)
