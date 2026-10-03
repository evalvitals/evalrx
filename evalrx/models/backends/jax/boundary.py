"""JAX -> torch conversion at the ``Trace`` boundary (CPU path).

Phase 1 of ``docs/design_jax_backend.md`` keeps the analyzers untouched: the
backend hands them CPU torch tensors with the SAME dtypes ``hf_local`` would
(bf16 stays bf16 through a bit-exact int16 view, since numpy has no bfloat16
and ``torch.from_numpy`` rejects the ``ml_dtypes`` one). torch is imported
lazily so the module loads on the light install.
"""

from __future__ import annotations

from typing import Any

from evalrx.models.backends.jax.protocol import NormParams


def to_numpy(x: Any):
    """Device-to-host copy as a numpy array (jax arrays, numpy arrays, lists)."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    try:
        import jax

        if isinstance(x, jax.Array):
            x = jax.device_get(x)
    except ImportError:  # pragma: no cover - light install
        pass
    return np.asarray(x)


def to_torch(x: Any):
    """A CPU torch tensor holding *x*, dtype preserved (bfloat16 included)."""
    import numpy as np
    import torch

    a = to_numpy(x)
    if a.dtype.name == "bfloat16":
        bits = np.ascontiguousarray(a).view(np.int16)
        if not bits.flags.writeable:
            bits = bits.copy()
        return torch.from_numpy(bits).view(torch.bfloat16)
    a = np.ascontiguousarray(a)
    if not a.flags.writeable:
        a = a.copy()
    return torch.from_numpy(a)


def make_torch_rmsnorm(norm: NormParams):
    """A tiny ``torch.nn.Module`` reproducing the adapter's head-side RMSNorm.

    ``LogitLensAnalyzer`` reads ``next(norm.parameters()).dtype`` and calls the
    module on intermediate hidden states, exactly as it does with the HF norm
    ``_discover.get_final_norm`` returns.
    """
    import torch

    class _RMSNorm(torch.nn.Module):
        def __init__(self, scale, eps: float, plus_one: bool) -> None:
            super().__init__()
            self.eps = float(eps)
            self.plus_one = bool(plus_one)
            self.weight = torch.nn.Parameter(scale, requires_grad=False)

        def forward(self, x):  # noqa: D401 - torch convention
            xf = x.float()
            y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            w = self.weight.float()
            if self.plus_one:
                w = 1.0 + w
            return (y * w).to(x.dtype)

        def extra_repr(self) -> str:
            return f"dim={tuple(self.weight.shape)}, eps={self.eps}, plus_one={self.plus_one}"

    scale = to_torch(norm.scale) if norm.scale is not None else torch.ones(1)
    return _RMSNorm(scale, norm.eps, norm.plus_one)
