"""Pre-M1 — probe search: synthesize new test cases instead of analyzing old ones.

Optional and structurally different from every other stage: its output is DATA,
not a measurement or a verdict, so it is the one stage whose result can be fed
back in as another stage's input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from evalvitals.contract.common import CaseBatchRef, StageEnvelope, WireModel
from evalvitals.contract.m1 import ModelRef, ProtocolWire


class ProbeSearchInput(WireModel):
    """Seeds plus a budget. The search anchors both regime trees on the seeds."""

    model: ModelRef
    seed_pool: CaseBatchRef
    protocol: ProtocolWire | None = None
    budget: int = Field(default=20, ge=1, description="Total simulations (T_max).")
    beta: float = Field(default=1.0, description="UCB exploration constant.")
    w_max: int = Field(default=3, ge=1, description="Max children before progressive widening.")


class ProbeSearchOutput(StageEnvelope):
    """Discovered cases plus search accounting.

    ``all_cases`` — not ``failure_cases`` — is what M1 should normally receive:
    a batch of nothing but FAILs has no control group, and every downstream
    comparison degenerates.

    The in-memory result also holds ``macro_root`` / ``micro_root``, two trees
    whose nodes point at both parent and children. That graph is cyclic and
    cannot be serialized naively, which is why the search trajectory is carried
    here as an explicit flat edge list rather than a nested object.
    """

    n_simulations: int = Field(ge=0)
    n_macro: int = Field(ge=0)
    n_micro: int = Field(ge=0)

    all_cases: CaseBatchRef = Field(description="Every evaluated case, PASS and FAIL. Feed this to M1.")
    failure_cases: CaseBatchRef = Field(description="The FAIL subset. Feed this to cluster_failures.")

    @property
    def error_rate(self) -> float:
        return self.failure_cases.n_cases / self.n_simulations if self.n_simulations else 0.0

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> "ProbeSearchOutput":
        if self.failure_cases.n_cases > self.all_cases.n_cases:
            raise ValueError("failure_cases cannot exceed all_cases")
        return self


__all__ = ["ProbeSearchInput", "ProbeSearchOutput"]
