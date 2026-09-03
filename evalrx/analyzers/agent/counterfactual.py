"""Counterfactual replay — causal step attribution for agent trajectories.

For each decision step, re-run the trajectory forked at that step and see whether
the outcome flips: a high flip-rate ⇒ the step is causally influential.  The actual
re-run is INJECTED as ``rerun_fn(trajectory, step_idx, seed) -> bool`` (it wraps a
live Agent + verifier), keeping this analyzer decoupled and testable.

Caveat (built into the design): with temperature > 0, run ``n_replays`` per step
and read the flip-RATE — a single replay conflates "this step was causal" with
model noise.  Prefer temperature=0 replay or a larger ``n_replays``.

Causal framework:
  "Causality: Models, Reasoning and Inference"
  Pearl (2000/2009), Cambridge University Press — do-calculus, interventions

Applied to neural NLP (causal mediation / interchange intervention):
  "Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias"
  Vig et al., 2020 — https://arxiv.org/abs/2004.12265
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.case import Label
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("counterfactual")
class CounterfactualReplay(Analyzer):
    """Rank a trajectory's tool-call steps by how often re-running flips the outcome."""

    name = "counterfactual"
    requires = frozenset({Capability.TOOL_CALLS})
    #: TOOL_CALLS says the model CAN call tools; this says the cases ARE agent
    #: runs. A chat model whose template renders tools declares the former.
    requires_trajectories = True
    applies_to_modalities = frozenset({"text", "image"})  # trajectory analysis is modality-agnostic

    def __init__(self, rerun_fn: Callable, n_replays: int = 3) -> None:
        super().__init__(rerun_fn=rerun_fn, n_replays=n_replays)
        self.rerun_fn = rerun_fn

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case = []
        for case in cases:
            traj = case.trajectory
            if traj is None:
                continue
            original_success = traj.outcome == Label.PASS
            candidates = [s for s in traj.steps if s.tool_call]
            step_scores = []
            for s in candidates:
                flips = sum(
                    1 for seed in range(self.n_replays)
                    if bool(self.rerun_fn(traj, s.idx, seed)) != original_success
                )
                step_scores.append({
                    "step": s.idx,
                    "action": s.tool_call.get("name"),
                    "flip_rate": round(flips / self.n_replays, 4) if self.n_replays else 0.0,
                })
            step_scores.sort(key=lambda d: -d["flip_rate"])
            per_case.append({
                "sample_id": traj.sample_id,
                "original_success": original_success,
                "most_influential_step": step_scores[0] if step_scores else None,
                "steps": step_scores,
            })
        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            findings={
                "n_trajectories": len(per_case),
                "per_case": per_case,
                "_caveat": "flip-rate ≈ causal influence; use temperature=0 or larger n_replays to separate from noise.",
            },
        )
