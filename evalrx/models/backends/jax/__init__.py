"""The ``jax_local`` backend: JAX models (TPU, GPU, CPU) behind an adapter.

``backend`` holds :class:`JaxLocalModel` / :class:`JaxLocalBackend`;
``protocol`` is the :class:`JaxModelAdapter` contract a framework implements;
``boundary`` converts captured arrays to torch at the ``Trace`` boundary;
``adapters.gemma`` is the reference adapter (Google DeepMind's ``gemma``
library). Nothing here imports jax or torch at module load. Design notes:
``docs/design_jax_backend.md``.
"""

from evalrx.models.backends.jax.backend import JaxLocalBackend, JaxLocalModel
from evalrx.models.backends.jax.protocol import (
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
    "JaxLocalBackend",
    "JaxLocalModel",
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
