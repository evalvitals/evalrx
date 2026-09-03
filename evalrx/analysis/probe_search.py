"""Hierarchical MCTS probing search (ProbeLLM-style adaptive test-case discovery).

Standalone tree-search skeleton: it knows nothing about models, judges, or
datasets — those are injected as three callables (``generate_macro``,
``generate_micro``, ``verify``). This mirrors the paper's two-level design
(arXiv 2602.12966, "ProbeLLM"): a global choice between Macro (broad
coverage — diversify topics) and Micro (local refinement — densify evidence
around a seed) regimes, each running its own UCB-guided MCTS to pick which
node to expand next, and a shared node/backup bookkeeping (Eqs. 5-10 in the
paper).

Failure cases discovered here are plain ``FailureCase`` objects — the result
feeds directly into :func:`evalrx.analysis.failure_modes.cluster_failures`
for failure-mode synthesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from evalrx.core.case import CaseBatch, FailureCase, Label

Regime = str  # "macro" | "micro"

GenerateMacroFn = Callable[["ProbeNode", "list[ProbeNode]"], "FailureCase | None"]
GenerateMicroFn = Callable[["ProbeNode"], "FailureCase | None"]
VerifyFn = Callable[[FailureCase], FailureCase]


@dataclass
class ProbeNode:
    """One evaluated test case in the search tree (paper Eq.5: (x, y*, y, fail)).

    ``N``/``E`` are *this node's* expansion stats (paper Eq.6): how many times
    a child was proposed+evaluated from here, and how many of those children
    were failures. They are updated on nodes along the *selection path*
    (root..this node), not on the freshly created child itself — a brand-new
    node always starts at N=E=0 until it is later expanded.
    """

    case: FailureCase
    regime: Regime
    parent: "ProbeNode | None" = None
    children: list["ProbeNode"] = field(default_factory=list)
    N: int = 0
    E: int = 0

    def p_hat(self) -> float:
        """Empirical failure rate of expansions from this node (Eq.6)."""
        return self.E / max(1, self.N)

    def is_fail(self) -> bool:
        return self.case.label == Label.FAIL


@dataclass
class ProbeSearchResult:
    """Output of one :meth:`ProbeSearch.run` call."""

    macro_root: ProbeNode
    micro_root: ProbeNode
    n_simulations: int
    n_macro: int
    n_micro: int
    failure_cases: CaseBatch
    all_cases: CaseBatch

    @property
    def error_rate(self) -> float:
        return len(self.failure_cases) / self.n_simulations if self.n_simulations else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "n_macro": self.n_macro,
            "n_micro": self.n_micro,
            "error_rate": self.error_rate,
            "n_failures": len(self.failure_cases),
        }


def _ucb(node: ProbeNode, parent_n: int, beta: float) -> float:
    """UCB(v|u) = p_hat(v) + beta*sqrt(log(max(1,N(u))) / max(1,N(v)))  (Eq.7)."""
    return node.p_hat() + beta * math.sqrt(math.log(max(1, parent_n)) / max(1, node.N))


def _select_expandable(root: ProbeNode, *, beta: float, w_max: int) -> ProbeNode:
    """Descend via UCB while the current node is already at max width,
    stopping at the first node with room for a new child (paper Eq.7-8, the
    "continue this selection step until reaching an expandable node")."""
    u = root
    while u.children and len(u.children) >= w_max:
        u = max(u.children, key=lambda v: _ucb(v, u.N, beta))
    return u


def _choose_regime(macro_root: ProbeNode, micro_root: ProbeNode, beta: float) -> Regime:
    """Top-level Macro/Micro choice, via the *same* UCB rule applied to the
    two regime roots — the global root's two children in the paper's
    hierarchy (Fig.2/Sec.3.2), rather than a fixed or alternating split."""
    total_n = macro_root.N + micro_root.N
    ucb_macro = _ucb(macro_root, total_n, beta)
    ucb_micro = _ucb(micro_root, total_n, beta)
    return "macro" if ucb_macro >= ucb_micro else "micro"


class ProbeSearch:
    """Hierarchical MCTS driver.

    Args:
        generate_macro:  ``(node, explored_macro_nodes) -> FailureCase | None``.
                          Proposes one new, *unevaluated* candidate case aimed
                          at broad topical coverage; receives every macro-tree
                          node explored so far so it can pick its own
                          diversity/frontier heuristic (e.g. embed + cluster
                          medoids, paper Eq.11) — this module stays agnostic
                          to how that is done.
        generate_micro:  ``(node) -> FailureCase | None``. Proposes one local
                          perturbation of ``node.case`` (paper's controlled
                          entity/attribute substitutions).
        verify:          ``(case) -> case``. Runs the target model + scorer,
                          returning the same case with ``.observed``/``.label``
                          populated (paper Eq.1's verifier V). A thin wrapper
                          around ``CaseDiscoveryAgent.discover`` in practice.
        seeds:           Initial seed cases (from an existing benchmark) used
                          to anchor both regime roots.
        budget:          Total number of simulations (T_max in Eq.4).
        beta:            UCB exploration constant (Eq.7).
        w_max:           Max children per node before progressive widening
                          forces a deeper UCB descent instead of a new sibling.
    """

    def __init__(
        self,
        *,
        generate_macro: GenerateMacroFn,
        generate_micro: GenerateMicroFn,
        verify: VerifyFn,
        seeds: "list[FailureCase] | CaseBatch",
        budget: int = 20,
        beta: float = 1.0,
        w_max: int = 3,
    ) -> None:
        seeds = list(seeds)
        if not seeds:
            raise ValueError("ProbeSearch requires at least one seed case")
        self._generate_macro = generate_macro
        self._generate_micro = generate_micro
        self._verify = verify
        self.budget = budget
        self.beta = beta
        self.w_max = max(1, w_max)

        self.macro_root = ProbeNode(case=seeds[0], regime="macro")
        self.micro_root = ProbeNode(case=seeds[0], regime="micro")
        self._macro_nodes: list[ProbeNode] = [self.macro_root]
        self._evaluated: list[FailureCase] = []
        self._n_macro = 0
        self._n_micro = 0

    def run(self) -> ProbeSearchResult:
        for _ in range(self.budget):
            regime = _choose_regime(self.macro_root, self.micro_root, self.beta)
            root = self.macro_root if regime == "macro" else self.micro_root
            u = _select_expandable(root, beta=self.beta, w_max=self.w_max)

            new_case = (
                self._generate_macro(u, self._macro_nodes)
                if regime == "macro"
                else self._generate_micro(u)
            )
            if new_case is None:
                continue  # generator exhausted for this node; budget step still spent

            evaluated = self._verify(new_case)
            child = ProbeNode(case=evaluated, regime=regime, parent=u)
            u.children.append(child)
            if regime == "macro":
                self._macro_nodes.append(child)
                self._n_macro += 1
            else:
                self._n_micro += 1
            self._evaluated.append(evaluated)

            fail = 1 if evaluated.label == Label.FAIL else 0
            node: "ProbeNode | None" = u
            while node is not None:
                node.N += 1
                node.E += fail
                node = node.parent

        failures = CaseBatch([c for c in self._evaluated if c.label == Label.FAIL])
        return ProbeSearchResult(
            macro_root=self.macro_root,
            micro_root=self.micro_root,
            n_simulations=len(self._evaluated),
            n_macro=self._n_macro,
            n_micro=self._n_micro,
            failure_cases=failures,
            all_cases=CaseBatch(self._evaluated),
        )
