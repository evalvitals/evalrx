"""Route a diagnostic question to the probes that can answer it.

:class:`ProbeRouter` joins three things that were decided in separate places:

* what can run       — :func:`~evalrx.probing.catalog.availability`;
* what is worth running first — the slot-composed priority and failure-mode
  boosts of :class:`~evalrx.eval_agent.stages.probe.StrategyProbe`;
* what the caller asked for — failure modes, families, access level.

The result keeps the reason for every decision: each selected probe says why it
was ranked where it was, and each rejected probe that the request pointed at
says what it is missing. A caller that cannot get the evidence it wanted learns
that, instead of receiving a shorter list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from evalrx.probing.catalog import (
    FAMILY_QUESTIONS,
    ProbeToolSpec,
    availability,
    probe_specs,
)

if TYPE_CHECKING:
    from evalrx.core.model import Model


def normalize_mode(mode: str) -> str:
    """Failure-mode tag in the form the routing table is keyed by."""
    return str(mode).strip().lower().replace(" ", "_").replace("-", "_")


def _short(value: Any, limit: int = 32) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class RoutedProbe:
    """One selected probe and why it was selected."""

    name: str
    spec: ProbeToolSpec
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.spec.family, "access": self.spec.access,
                "cost": self.spec.cost, "question": self.spec.question,
                "reasons": list(self.reasons)}


@dataclass
class ProbeRoute:
    """The outcome of one routing request."""

    selected: list[RoutedProbe] = field(default_factory=list)
    #: ``name -> reasons`` for probes the request pointed at that cannot run.
    unavailable: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Failure-mode tags in the request that no probe is mapped to.
    unknown_modes: tuple[str, ...] = ()

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.selected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [p.to_dict() for p in self.selected],
            "unavailable": {k: list(v) for k, v in self.unavailable.items()},
            "unknown_modes": list(self.unknown_modes),
        }

    def to_prompt(self, *, with_params: bool = True) -> str:
        """Render the route as a catalog an LLM judge can pick from."""
        lines: list[str] = []
        family = None
        for probe in sorted(self.selected, key=lambda p: (p.spec.family,)):
            if probe.spec.family != family:
                family = probe.spec.family
                lines.append(f"[{family}] {FAMILY_QUESTIONS.get(family, '')}".rstrip())
            bits = [f"  - {probe.name} ({probe.spec.access}, {probe.spec.cost}): "
                    f"{probe.spec.summary}"]
            if with_params and probe.spec.callable_params:
                args = ", ".join(
                    p.name if p.required else f"{p.name}={_short(p.default)}"
                    for p in probe.spec.callable_params
                )
                bits.append(f"params: {args}")
            if probe.reasons:
                bits.append("why: " + "; ".join(probe.reasons))
            lines.append(" | ".join(bits))
        if self.unavailable:
            lines.append("UNAVAILABLE:")
            for name, reasons in sorted(self.unavailable.items()):
                lines.append(f"  - {name}: {'; '.join(reasons)}")
        if self.unknown_modes:
            lines.append("No probe is mapped to: " + ", ".join(self.unknown_modes))
        return "\n".join(lines)


class ProbeRouter:
    """Select and order probes for a model, a batch and a question.

    Args:
        selector:  The priority source. Defaults to a fresh
                   :class:`~evalrx.eval_agent.stages.probe.StrategyProbe`.
        overrides: Names for which the host holds a constructed analyzer; they
                   are never rejected for a missing injected argument.
    """

    def __init__(self, selector: Any = None, overrides: "Iterable[str]" = ()) -> None:
        if selector is None:
            from evalrx.eval_agent.stages.probe import StrategyProbe

            selector = StrategyProbe()
        self.selector = selector
        self.overrides = tuple(overrides)

    def route(
        self,
        model: "Model | None",
        data: Any = None,
        *,
        failure_modes: "Iterable[str] | None" = None,
        families: "Iterable[str] | None" = None,
        access: "str | None" = None,
        allow_interventions: bool = True,
        only_matching: bool = False,
        max_probes: "int | None" = None,
    ) -> ProbeRoute:
        """Rank the probes that can run and answer the request.

        Args:
            model:               The model under test.
            data:                The batch about to be probed, when known.
            failure_modes:       Failure-mode tags (for example from M3
                                 hypotheses). Probes mapped to them lead.
            families:            Keep only these families.
            access:              ``"black_box"`` keeps probes an API model can
                                 run; ``"white_box"`` keeps the ones that need
                                 internals. ``None`` keeps both.
            allow_interventions: When False, drop probes that modify the
                                 forward pass or weights.
            only_matching:       When True and *failure_modes* is given, return
                                 only probes mapped to those modes.
            max_probes:          Cap on the number of selected probes.
        """
        from evalrx.eval_agent.stages.probe import _FAILURE_MODE_TO_ANALYZERS

        specs = probe_specs()
        modes = [normalize_mode(m) for m in (failure_modes or ()) if str(m).strip()]
        wanted_families = None if families is None else {str(f) for f in families}

        targeted: dict[str, list[str]] = {}
        unknown: list[str] = []
        for mode in dict.fromkeys(modes):
            mapped = _FAILURE_MODE_TO_ANALYZERS.get(mode)
            if not mapped:
                unknown.append(mode)
                continue
            for name in mapped:
                targeted.setdefault(name, []).append(mode)

        def in_scope(spec: ProbeToolSpec) -> bool:
            if wanted_families is not None and spec.family not in wanted_families:
                return False
            if access == "black_box" and spec.access == "white_box":
                return False
            if access == "white_box" and spec.access != "white_box":
                return False
            if not allow_interventions and spec.intervenes:
                return False
            return not (only_matching and modes and spec.name not in targeted)

        runnable: dict[str, ProbeToolSpec] = {}
        route = ProbeRoute(unknown_modes=tuple(unknown))
        for name, spec in specs.items():
            if not in_scope(spec):
                continue
            verdict = availability(spec, model, data, overrides=self.overrides)
            if verdict.available:
                runnable[name] = spec
            elif name in targeted or wanted_families is not None:
                # Only report rejections the request actually pointed at; an
                # unfiltered request would otherwise list half the registry.
                route.unavailable[name] = verdict.missing

        order = self._priority(model, data, modes)
        rank = {name: i for i, name in enumerate(order)}
        slots = sorted(self._slots(model, data))
        ordered = sorted(runnable, key=lambda n: (rank.get(n, len(rank)), n))
        for name in ordered:
            reasons: list[str] = []
            if name in targeted:
                reasons.append("targets " + ", ".join(targeted[name]))
            if name in rank:
                reasons.append(f"priority {rank[name] + 1} for a {'+'.join(slots)} run")
            else:
                reasons.append("runnable, not in the priority list")
            route.selected.append(RoutedProbe(name, runnable[name], tuple(reasons)))
        if max_probes is not None:
            route.selected = route.selected[: max(0, int(max_probes))]
        return route

    # ------------------------------------------------------------------
    def _priority(self, model: Any, data: Any, modes: list[str]) -> list[str]:
        try:
            return list(self.selector.select(model, hint_failure_modes=modes or None, data=data))
        except Exception:  # noqa: BLE001 - a model the selector cannot read still gets a route
            return []

    def _slots(self, model: Any, data: Any) -> set[str]:
        try:
            return set(self.selector.routed_slots(model, data))
        except Exception:  # noqa: BLE001
            return {"text"}
