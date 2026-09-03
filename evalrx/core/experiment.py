"""Experiment — a declarative spec that bridges the agent and the core.

The self-evolving agent does not call analyzers directly; it emits *experiments*
("run analyzer X on model Y over cases Z"), which an :class:`ExperimentRunner`
executes, caches, and turns into results.  Making the experiment a first-class
object lets the agent queue, log, replay, and reason about its own actions.

This is intentionally minimal in Stage 1: enough structure for the agent layer
to target, with caching as the one piece of real behaviour.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.model import Model
    from evalrx.core.result import Result


def _data_fingerprint(data: Any) -> Any:
    """Deterministic content fingerprint for cache keys.

    Crucially does NOT call ``as_casebatch`` (which mints fresh uuids for raw
    inputs) — that would make the same data hash differently each call.  Uses
    stable case ids when present, otherwise the prompt/content text.
    """
    from evalrx.core.case import CaseBatch, FailureCase, Inputs

    if isinstance(data, CaseBatch):
        return [c.id for c in data]
    if isinstance(data, FailureCase):
        return [data.id]
    if isinstance(data, Inputs):
        return [f"inputs:{data.prompt}"]
    if isinstance(data, str):
        return [f"str:{data}"]
    if isinstance(data, (list, tuple)):
        return [_data_fingerprint(x) for x in data]
    return [repr(data)]


@dataclass
class Experiment:
    """A declarative analysis request.

    Attributes:
        model:    The model to analyse.
        analyzer: The configured analyzer to run.
        data:     Cases (or anything :func:`as_casebatch` accepts).
        metadata: Free-form context (e.g. the hypothesis this tests).
    """

    model: "Model"
    analyzer: "Analyzer"
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """A stable CONTENT hash of (model, analyzer+params, data).

        Used as the cache key so that two *equivalent* experiments dedupe (and a
        rerun after restart hits cache) — unlike ``id()``, which is a transient
        memory address.  The data fingerprint uses case ids when available.
        """
        try:
            params = self.analyzer.get_params()
        except Exception:  # non-estimator analyzer
            params = {}
        data_fp = _data_fingerprint(self.data)
        payload = json.dumps(
            {
                "model": repr(self.model),
                "analyzer": getattr(self.analyzer, "name", type(self.analyzer).__name__),
                "params": params,
                "data": data_fp,
            },
            sort_keys=True,
            default=repr,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


class ExperimentRunner:
    """Execute :class:`Experiment` specs, caching results by experiment CONTENT."""

    def __init__(self) -> None:
        self._cache: dict[str, "Result"] = {}

    def run(self, experiment: Experiment, use_cache: bool = True) -> "Result":
        """Run an experiment, returning (and caching) its :class:`Result`."""
        key = experiment.fingerprint()
        if use_cache and key in self._cache:
            return self._cache[key]
        result = experiment.analyzer.run(experiment.model, experiment.data)
        result.metadata.setdefault("experiment", experiment.metadata)
        result.metadata.setdefault("fingerprint", key)
        self._cache[key] = result
        return result

    def run_many(self, experiments: list[Experiment]) -> list["Result"]:
        return [self.run(exp) for exp in experiments]
