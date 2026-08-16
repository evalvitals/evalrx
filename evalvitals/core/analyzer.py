"""Analyzer — the sklearn-like estimator at the heart of EvalVitals.

Consistent contract, every analyzer:
  - is configured with hyper-parameters in ``__init__`` (stored, introspectable
    via ``get_params``/``set_params`` — exactly like a scikit-learn estimator),
  - declares the capabilities it ``requires``,
  - runs via ``run(model, data) -> Result``.

``run`` normalises ``data`` into a :class:`CaseBatch`, verifies the model
provides the required capabilities (clear :class:`CapabilityError` otherwise),
then delegates to the subclass's :meth:`_run`.  Subclasses implement only
``_run`` and never repeat the boilerplate.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import CaseBatch, as_casebatch

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result


class Analyzer(ABC):
    """Abstract base for every analysis (sklearn-estimator style).

    Class attributes (set by subclasses):
        name:     Registered short name (e.g. ``"attention"``).
        requires: Capabilities the analysed model must provide.

    Example::

        result = AttentionAnalyzer(layer=-1).run(qwen, "The capital of France is")
    """

    name: str = "analyzer"
    requires: frozenset[Capability] = frozenset()
    #: Modalities this analysis applies to; matched against ``model.modalities``.
    #: ``{"text"}`` runs on any text-capable model; ``{"image"}`` only on VLMs.
    applies_to_modalities: frozenset[str] = frozenset({"text"})
    #: True when the analysis reads multi-step agent runs
    #: (:class:`~evalvitals.core.case.Trajectory`) rather than single turns.
    #:
    #: A requirement on the DATA, which is why it is not a
    #: :class:`~evalvitals.core.capability.Capability`: these analyzers run on
    #: trajectories loaded from disk with ``model=None``, so nothing about the
    #: model can express it. ``requires``/``applies_to_modalities`` therefore let
    #: every one of them match a plain LLM or VLM, and a single-turn QA batch got
    #: offered trajectory analyzers — the ones that built with default args then
    #: ran and reported ``n_trajectories: 0``. Selection gates on this against
    #: the batch actually in hand.
    requires_trajectories: bool = False

    def __init__(self, **params: Any) -> None:
        # Store hyper-parameters sklearn-style for introspection / reproduction.
        self._params: dict[str, Any] = dict(params)
        for key, value in params.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # sklearn-style introspection
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, Any]:
        """Return the analyzer's hyper-parameters.

        Derived from the concrete subclass's ``__init__`` signature so that
        typed subclasses never silently omit a parameter by forgetting to
        forward it to ``super().__init__``.
        """
        sig = inspect.signature(type(self).__init__)
        typed_params = {
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        if typed_params:
            return {name: getattr(self, name) for name in typed_params}
        # No typed params — base **params pattern or zero-param analyzer.
        return dict(self._params)

    def set_params(self, **params: Any) -> "Analyzer":
        """Update hyper-parameters in place and return self (chainable)."""
        self._params.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, model: "Model", data: Any) -> "Result":
        """Validate capabilities, normalise *data*, and run the analysis.

        Args:
            model: Any :class:`~evalvitals.core.model.Model`.
            data:  ``str | FailureCase | Inputs | CaseBatch | iterable`` — normalised
                   via :func:`~evalvitals.core.case.as_casebatch`.

        Returns:
            A :class:`~evalvitals.core.result.Result` (subclass) instance.

        Raises:
            CapabilityError: if *model* lacks a required capability.
        """
        self._check_capabilities(model)
        cases = as_casebatch(data)
        return self._run(model, cases)

    @abstractmethod
    def _run(self, model: "Model", cases: CaseBatch) -> "Result":
        """Subclass hook: perform the analysis over an already-normalised batch."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_capabilities(self, model: "Model") -> None:
        if not self.requires:  # capability-free (e.g. trajectory heuristics) — model may be None
            return
        missing = set(self.requires) - set(getattr(model, "capabilities", frozenset()))
        if missing:
            raise CapabilityError(
                analyzer=self.name,
                model=repr(model),
                missing=missing,
            )

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"
