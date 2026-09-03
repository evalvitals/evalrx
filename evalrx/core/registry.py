"""Registries — the agent's discovery surface.

Analyzers register themselves by name.  An agent (or the framework) enumerates
them and matches them to a model by capability::

    registry.analyzers.list()                  # ["attention", "token_entropy", ...]
    registry.analyzers.compatible_with(model)  # [AttentionAnalyzer, ...]

Model *identity* lives in :mod:`evalrx.specs` (``ModelSpec``), not here;
``registry.models`` is a deprecated shim that delegates to it.  This is what
makes the package programmatically explorable rather than something you have to
read source code to use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.model import Model

T = TypeVar("T")


class _Registry(Generic[T]):
    """Name → class mapping with a registration decorator."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator: ``@registry.register("qwen")``."""

        def decorator(cls: type[T]) -> type[T]:
            key = name.lower()
            if key in self._items:
                raise ValueError(f"{self._kind} '{name}' is already registered.")
            self._items[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"Unknown {self._kind} '{name}'. Registered: {self.list()}"
            )
        return self._items[key]

    def has(self, name: str) -> bool:
        return name.lower() in self._items

    def list(self) -> list[str]:
        return sorted(self._items)

    def all(self) -> dict[str, type[T]]:
        return dict(self._items)


class AnalyzerRegistry(_Registry["Analyzer"]):
    """Analyzer registry with capability-based matching."""

    @staticmethod
    def _matches(cls: type["Analyzer"], model: "Model") -> bool:
        """Analyzer runs on *model* iff capabilities are satisfied AND modalities overlap."""
        if not model.supports(cls.requires):
            return False
        model_modalities = getattr(model, "modalities", frozenset({"text"}))
        return bool(cls.applies_to_modalities & model_modalities)

    def compatible_with(self, model: "Model") -> list[type["Analyzer"]]:
        """Return analyzer classes runnable on *model* (capability + modality match)."""
        return [cls for cls in self._items.values() if self._matches(cls, model)]

    def names_compatible_with(self, model: "Model") -> list[str]:
        """Same as :meth:`compatible_with` but returns registered names."""
        return sorted(name for name, cls in self._items.items() if self._matches(cls, model))


class _DeprecatedModelRegistry(_Registry["Model"]):
    """Deprecated shim: model identity now lives in :mod:`evalrx.specs`.

    ``registry.models`` is kept so older code keeps working, but it delegates to
    the spec registry and warns.  Use ``evalrx.list_specs()`` /
    ``evalrx.get_spec()`` (or ``evalrx.load`` / ``compose``) instead.
    """

    @staticmethod
    def _warn() -> None:
        import warnings

        warnings.warn(
            "registry.models is deprecated; model identity now lives in "
            "evalrx.specs (use evalrx.list_specs() / get_spec(), or "
            "evalrx.load / compose to build a model).",
            DeprecationWarning,
            stacklevel=3,
        )

    def list(self) -> list[str]:
        self._warn()
        from evalrx.specs import list_specs

        return list_specs()

    def has(self, name: str) -> bool:
        self._warn()
        from evalrx.specs import REGISTRY

        return name.lower() in REGISTRY

    def get(self, name: str):  # returns a ModelSpec, not a class
        self._warn()
        from evalrx.specs import get_spec

        return get_spec(name.lower())


class Registry:
    """Top-level namespace bundling the model and analyzer registries."""

    def __init__(self) -> None:
        # Model identity is the spec registry's job now; this is a back-compat shim.
        self.models: _DeprecatedModelRegistry = _DeprecatedModelRegistry("model")
        self.analyzers: AnalyzerRegistry = AnalyzerRegistry("analyzer")


# Singleton used across the package.
registry = Registry()


def register_model(name: str):
    """Class decorator registering a model under *name*."""
    return registry.models.register(name)


def register_analyzer(name: str):
    """Class decorator registering an analyzer under *name*."""
    return registry.analyzers.register(name)
