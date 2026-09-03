"""Pipeline — sklearn-style composition of analyzers.

Run several analyzers over the same ``(model, cases)`` and collect their
results under their step names::

    pipe = Pipeline([
        ("attention", AttentionAnalyzer(layer=-1)),
        # ("saliency", SaliencyAnalyzer()),   # Stage 2
    ])
    results = pipe.run(qwen, "The capital of France is")
    results["attention"].summary()

Each analyzer's capabilities are validated independently by its own ``run``.
Cross-step data flow (e.g. perturb → analyze) is a Stage-2 extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalrx.core.case import as_casebatch

if TYPE_CHECKING:
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.model import Model
    from evalrx.core.result import Result


class Pipeline:
    """Ordered collection of named analyzers run over shared inputs."""

    def __init__(self, steps: list[tuple[str, "Analyzer"]]) -> None:
        self.steps = steps

    def run(self, model: "Model", data: Any) -> dict[str, "Result"]:
        """Run every step over the same normalised case batch."""
        cases = as_casebatch(data)
        return {name: analyzer.run(model, cases) for name, analyzer in self.steps}

    def __repr__(self) -> str:
        names = ", ".join(name for name, _ in self.steps)
        return f"Pipeline([{names}])"
