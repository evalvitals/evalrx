"""Import aliases for modules that moved into ``evalrx.models.backends`` (0.1.2).

Backends are grouped by framework (``api``, ``hf``, ``jax``); the old flat
paths keep importing, each with a ``DeprecationWarning``, and resolve to the
SAME module object (so attribute access and monkeypatching behave as before).
A meta-path finder does this instead of stub files at the old paths, which
would hide the moves from git's rename tracking.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

MOVED: dict[str, str] = {
    "evalrx.models.backends.hf_local": "evalrx.models.backends.hf.model",
    "evalrx.models.backends.jax_local": "evalrx.models.backends.jax.backend",
    "evalrx.models.backends.openai_compat": "evalrx.models.backends.api.openai",
    "evalrx.models.backends.gemini_compat": "evalrx.models.backends.api.gemini",
    "evalrx.models.blackbox.gemini": "evalrx.models.backends.api.gemini_model",
    "evalrx.models._discover": "evalrx.models.backends.hf.discover",
    "evalrx.models.inference": "evalrx.models.backends.hf.inference",
    "evalrx.models.jax": "evalrx.models.backends.jax",
    "evalrx.models.jax.gemma": "evalrx.models.backends.jax.adapters.gemma",
    "evalrx.models.jax.protocol": "evalrx.models.backends.jax.protocol",
    "evalrx.models.jax._boundary": "evalrx.models.backends.jax.boundary",
}


class _MovedModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname not in MOVED:
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=fullname == "evalrx.models.jax")

    def create_module(self, spec):
        new = MOVED[spec.name]
        warnings.warn(f"{spec.name} moved to {new}; the old path will be removed in a future release",
                      DeprecationWarning, stacklevel=3)
        return importlib.import_module(new)   # importlib keeps its __name__ / __spec__

    def exec_module(self, module):
        pass


def install() -> None:
    if not any(isinstance(f, _MovedModuleFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MovedModuleFinder())


install()
