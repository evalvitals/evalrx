"""The ``hf_local`` backend: Hugging Face transformers on PyTorch (CPU / GPU).

``model`` holds :class:`HFLocalModel` / :class:`HFLocalBackend`; ``discover``
finds decoder layers, norms and towers on a loaded model; ``inference`` infers
a :class:`~evalrx.core.spec.ModelSpec` from a live model for ``evalrx.wrap``.
torch and transformers load lazily inside ``load()``.
"""

from evalrx.models.backends.hf.model import HFLocalBackend, HFLocalModel

__all__ = ["HFLocalBackend", "HFLocalModel"]
