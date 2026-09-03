"""The agent's action space over the EvalRX package.

The self-evolving agent doesn't reach into internals — it acts through this
thin tool layer, which wraps the registry, the analyzer/run machinery, and the
store.  Each function is one "skill" the agent can call.  Functions that only
forward to the core are implemented now; the ones needing the Stage-2 store or
case-generation raise ``NotImplementedError`` but already have their final
signatures, so the loop can be wired against them.

Design intent: this module *is* the contract between the agent and the package.
Keeping it small and uniform is what makes the agent able to drive the package
without bespoke glue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalrx.core.registry import registry

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model
    from evalrx.core.result import Result


# ----------------------------------------------------------------------
# Discovery — "what can I do?"  (implemented: pure registry reads)
# ----------------------------------------------------------------------

def list_models() -> list[str]:
    """All known model spec keys (the agent's model-selection surface)."""
    from evalrx.specs import list_specs

    return list_specs()


def list_analyses() -> list[str]:
    """All registered analyzer names."""
    return registry.analyzers.list()


def compatible_analyses(model: "Model") -> list[str]:
    """Analyzer names whose capability requirements *model* satisfies.

    This is the agent's key planning primitive: given a model, what analyses
    are even runnable on it?
    """
    return registry.analyzers.names_compatible_with(model)


# ----------------------------------------------------------------------
# Action — "run an analysis"  (implemented: forwards to the analyzer)
# ----------------------------------------------------------------------

def run_analysis(model: "Model", analysis: str, data: Any, **params) -> "Result":
    """Instantiate analyzer *analysis* with *params* and run it on *model*/*data*."""
    analyzer_cls = registry.analyzers.get(analysis)
    return analyzer_cls(**params).run(model, data)


# ----------------------------------------------------------------------
# Case construction & memory  (Stage 2 — needs generators + store)
# ----------------------------------------------------------------------

def make_cases(spec: Any) -> "CaseBatch":
    """Construct/sample a batch of candidate failure cases from a spec.

    Stage 2: backed by dataset samplers and agent-driven case synthesis.
    """
    raise NotImplementedError("make_cases is planned for Stage 2.")


def record(result: "Result") -> None:
    """Persist a result (and its cases) into the store for self-evolution."""
    raise NotImplementedError("record is planned for Stage 2 (needs Store).")


def query_store(query: Any) -> Any:
    """Query accumulated cases/results/hypotheses from the store."""
    raise NotImplementedError("query_store is planned for Stage 2 (needs Store).")
