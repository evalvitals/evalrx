"""compose() — combine a ModelSpec with a Backend into a Model.

This is where the orthogonal decomposition becomes a single object:

    handle = compose(spec, "hf_local", want={Capability.ATTENTION})

Capability negotiation happens HERE (before any weights load): if you ask for
``ATTENTION`` against the ``api`` backend, you get a clear :class:`CapabilityError`
up front rather than a crash deep inside a hook.  ``api_only`` specs (closed
weights) refuse every backend except ``api``.
"""

from __future__ import annotations

from typing import Iterable

from evalrx.core.capability import Capability, CapabilityError
from evalrx.models.backends import BACKENDS, Backend, RuntimeConfig


def negotiate(handle, want: Iterable[Capability], model_id: str, where: str):
    """Raise ``CapabilityError`` if *handle* cannot provide every wanted capability.

    Shared by :func:`compose` (spec/registry path) and :func:`evalrx.wrap`
    (live-model path) so both negotiate capabilities identically.
    """
    missing = set(want) - set(handle.capabilities)
    if missing:
        raise CapabilityError(analyzer=where, model=model_id, missing=missing)
    return handle


def compose(
    spec,
    backend: "str | Backend" = "hf_local",
    runtime: RuntimeConfig | None = None,
    want: Iterable[Capability] = (),
):
    """Build a Model for *spec* on *backend*, negotiating capabilities first.

    Args:
        spec:    a :class:`~evalrx.core.spec.ModelSpec` (or a registry key str).
        backend: backend name (``"api"`` / ``"hf_local"`` / ``"vllm_offline"``) or instance.
        runtime: :class:`RuntimeConfig`; defaults applied if omitted.
        want:    capabilities the caller needs; raises if the backend can't provide them.
    """
    if isinstance(spec, str):
        from evalrx.specs import get_spec

        spec = get_spec(spec)
    if isinstance(backend, str):
        if backend not in BACKENDS:
            raise KeyError(f"Unknown backend {backend!r}. Known: {sorted(BACKENDS)}")
        backend = BACKENDS[backend]()
    runtime = runtime or RuntimeConfig()

    if spec.api_only and backend.kind != "api":
        raise ValueError(
            f"{spec.key!r} is api-only (closed weights); cannot use backend {backend.kind!r}."
        )

    # Build first (lazy — no weights load), then negotiate against the ACTUAL
    # handle capabilities.  This is precise for conditional caps like TOOL_CALLS,
    # which depend on the model (chat template), not just the backend.
    handle = backend.build(spec, runtime)
    return negotiate(handle, want, model_id=spec.key, where=f"request@{backend.kind}")
