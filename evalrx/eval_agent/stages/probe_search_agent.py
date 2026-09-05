"""ProbeSearchAgent — wires ProbeLLM's hierarchical MCTS
(:mod:`evalrx.analysis.probe_search`) to a real target model + judge.

``analysis.probe_search`` stays standalone (a generic tree search over
injected callables); this eval_agent-layer module supplies those callables
from real components:

- verifier V              -> :class:`~evalrx.eval_agent.stages.case_discovery.CaseDiscoveryAgent`
- Macro/Micro generators   -> :class:`~evalrx.eval_agent.stages.probe_candidate_generator.VLMProbeCandidateGenerator`

The discovered failure cases (``ProbeSearchResult.failure_cases``) are plain
``FailureCase`` objects and feed directly into
:func:`evalrx.analysis.failure_modes.cluster_failures` for failure-mode
synthesis, or into M1-M4 like any other labeled batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evalrx.analysis.probe_search import ProbeSearch, ProbeSearchResult
from evalrx.eval_agent.stages.case_discovery import CaseDiscoveryAgent
from evalrx.eval_agent.stages.probe_candidate_generator import VLMProbeCandidateGenerator

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol

logger = logging.getLogger(__name__)


@dataclass
class ProbeSearchAgent:
    """Run a hierarchical Macro/Micro MCTS probe search against a target model.

    Args:
        judge:    Text-only judge used both to score PASS/FAIL (via
                  ``CaseDiscoveryAgent``) and to paraphrase Macro/Micro
                  candidates (via ``VLMProbeCandidateGenerator``). Required —
                  without it, generation is unavailable and the search finds
                  nothing (``ProbeSearchResult.n_simulations == 0``).
        protocol: Optional experiment protocol passed to the discovery judge
                  for scoring context.
        budget:   Total simulations (T_max in the paper's Eq.4).
        beta:     UCB exploration constant (Eq.7).
        w_max:    Max children per search-tree node before progressive
                  widening forces a deeper descent instead of a new sibling.
    """

    judge: "Model"
    protocol: "ExperimentProtocol | None" = None
    budget: int = 20
    beta: float = 1.0
    w_max: int = 3

    def __post_init__(self) -> None:
        if self.judge is None:
            raise ValueError(
                "ProbeSearchAgent requires a judge (e.g. ClaudeModel() or AgyModel()) "
                "— without one, candidate generation is unavailable and the search "
                "would silently discover nothing."
            )

    def run(self, model: "Model", seed_pool: "CaseBatch") -> ProbeSearchResult:
        discovery = CaseDiscoveryAgent(judge=self.judge)
        generator = VLMProbeCandidateGenerator(seed_pool=seed_pool, judge=self.judge)

        def verify(case: "FailureCase") -> "FailureCase":
            report = discovery.discover(model, [case], protocol=self.protocol)
            cases = list(report.cases)
            return cases[0] if cases else case

        search = ProbeSearch(
            generate_macro=generator.macro,
            generate_micro=generator.micro,
            verify=verify,
            seeds=seed_pool,
            budget=self.budget,
            beta=self.beta,
            w_max=self.w_max,
        )
        result = search.run()
        logger.info(
            "ProbeSearchAgent: %d simulation(s) (macro=%d micro=%d), error_rate=%.2f",
            result.n_simulations, result.n_macro, result.n_micro, result.error_rate,
        )
        return result
