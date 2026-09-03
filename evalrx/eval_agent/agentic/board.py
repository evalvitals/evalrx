"""The evidence board: everything the judge sees when deciding the next action.

The host reconstructs the full decision context from this board on every
turn — there is no CLI session state carried between judge calls, only what
is serialized here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BudgetState:
    """Guardrails on how long an agentic run may keep deciding actions."""

    max_actions: int = 12
    token_budget: int = 0
    time_budget_sec: float = 0.0
    actions_taken: int = 0
    tokens_used: int = 0
    _started_monotonic: float = 0.0

    def start(self) -> None:
        self._started_monotonic = time.monotonic()

    def exhausted(self) -> str | None:
        """Return which budget tripped (for ``stopped_by``), or ``None``."""
        if self.max_actions > 0 and self.actions_taken >= self.max_actions:
            return "max_actions"
        if self.token_budget > 0 and self.tokens_used >= self.token_budget:
            return "budget"
        if (
            self.time_budget_sec > 0
            and self._started_monotonic > 0
            and (time.monotonic() - self._started_monotonic) >= self.time_budget_sec
        ):
            return "time_budget"
        return None


@dataclass
class EvidenceBoard:
    """Accumulated evidence the judge reasons over, one decision turn at a time."""

    protocol_summary: str = ""
    data_summary: dict[str, Any] = field(default_factory=dict)
    probe_findings: list[dict[str, Any]] = field(default_factory=list)
    stats_findings: list[dict[str, Any]] = field(default_factory=list)
    stats_confirmatory: bool = False
    explore_takeaways: list[dict[str, Any]] = field(default_factory=list)
    failure_modes: list[dict[str, Any]] = field(default_factory=list)
    probe_search_findings: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    action_log: list[dict[str, Any]] = field(default_factory=list)
    budget: BudgetState = field(default_factory=BudgetState)

    def has_supported_hypothesis(self) -> bool:
        return any(
            h.get("status") == "supported" and h.get("is_consistent_with_protocol", True)
            for h in self.hypotheses
        )

    def calls_made(self, tool: str) -> int:
        return sum(1 for a in self.action_log if a.get("tool") == tool)

    def to_prompt(self, registry: "Any") -> str:
        """Render the board as the decision prompt's evidence section."""
        lines: list[str] = []
        if self.protocol_summary:
            lines.append(f"PROTOCOL:\n{self.protocol_summary}\n")
        if self.data_summary:
            lines.append(f"DATA: {self.data_summary}\n")

        lines.append(f"PROBE FINDINGS (M1): {self.probe_findings or 'none yet'}")
        lines.append(
            f"STATS FINDINGS (M2, confirmatory={self.stats_confirmatory}): "
            f"{self.stats_findings or 'none yet'}"
        )
        if self.explore_takeaways:
            lines.append(f"EXPLORE TAKEAWAYS: {self.explore_takeaways}")
        if self.failure_modes:
            lines.append(f"FAILURE MODES (clustered): {self.failure_modes}")
        if self.probe_search_findings:
            lines.append(f"PROBE SEARCH (newly synthesized failures): {self.probe_search_findings}")
        lines.append(f"HYPOTHESES: {self.hypotheses or 'none yet'}")

        if self.action_log:
            lines.append("\nACTIONS TAKEN SO FAR:")
            for a in self.action_log:
                lines.append(
                    f"  [{a['step']}] {a['tool']}({a.get('params', {})}) "
                    f"-> ok={a['ok']}: {a['summary']}"
                )
        else:
            lines.append("\nACTIONS TAKEN SO FAR: none")

        remaining = (
            self.budget.max_actions - self.budget.actions_taken
            if self.budget.max_actions > 0
            else "unlimited"
        )
        lines.append(f"\nACTIONS REMAINING: {remaining}")
        lines.append(f"\nAVAILABLE TOOLS:\n{registry.catalog_for_prompt(self)}")
        return "\n".join(lines)
