"""JAX model adapters for the ``jax_local`` backend.

The backend (``evalrx.models.backends.jax_local``) never imports a JAX
framework itself; it drives the :class:`~evalrx.models.jax.protocol.JaxModelAdapter`
contract defined here. ``evalrx.models.jax.gemma`` is the reference adapter
(Google DeepMind's ``gemma`` library, Gemma 4 E2B / E4B). Everything in this
package imports without jax or torch installed; the heavy imports live inside
``load()``. Design notes: ``docs/design_jax_backend.md``.
"""

from evalrx.models.jax.protocol import (
    CAPTURE_ATTN,
    CAPTURE_HIDDEN,
    CAPTURE_LOGITS,
    Encoding,
    ForwardOut,
    GenerateOut,
    JaxModelAdapter,
    NormParams,
    SamplingParams,
)

__all__ = [
    "CAPTURE_ATTN",
    "CAPTURE_HIDDEN",
    "CAPTURE_LOGITS",
    "Encoding",
    "ForwardOut",
    "GenerateOut",
    "JaxModelAdapter",
    "NormParams",
    "SamplingParams",
]
