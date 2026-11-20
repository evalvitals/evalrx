"""Control edits — what an edit must be compared against before it is explained.

An edit that changes behaviour proves little on its own: a change of the same
size in a random direction, or the same rule applied to other neurons, may do
the same. Each control here keeps everything about the edit except the one
property under test.

``random_direction``
    Same columns, same per-column change norm, same sign pattern, but along a
    random unit direction instead of the edit's own. Tests whether the
    DIRECTION matters.
``random_columns``
    The same change moved to an equal number of other, randomly chosen columns.
    For a rank-one rule the rule is re-applied to those columns; otherwise each
    changed column's delta is moved. Tests whether the choice of NEURONS matters.

Controls are built against the live weights and returned as dense-delta
:class:`~evalrx.wsp.edit.WeightEdit` objects, so they are applied by the same
code path as the edit they control for.
"""

from __future__ import annotations

from typing import Any

from evalrx.wsp.edit import TensorEdit, WeightEdit, module_of, resolve_parameters

CONTROL_KINDS = ("random_direction", "random_columns")


def _generator(seed: int) -> Any:
    import torch

    return torch.Generator().manual_seed(int(seed))


def _random_direction(d: Any, gen: Any) -> Any:
    """Norm- and sign-matched change along one random unit direction."""
    import torch

    cols = torch.nonzero((d != 0).any(dim=0)).flatten()
    if cols.numel() == 0:
        return torch.zeros_like(d)
    dc = d[:, cols]
    u = torch.linalg.svd(dc, full_matrices=False)[0][:, 0]
    signs = torch.sign(u @ dc)
    signs[signs == 0] = 1.0
    r = torch.randn(d.shape[0], generator=gen, dtype=torch.float64)
    r = r / r.norm()
    out = torch.zeros_like(d)
    out[:, cols] = torch.outer(r, dc.norm(dim=0) * signs)
    return out


def _random_columns(w0: Any, d: Any, tensor_edit: TensorEdit, gen: Any) -> "Any | None":
    """The edit's change on an equal number of other columns; ``None`` when there are too few."""
    import torch

    cols = torch.nonzero((d != 0).any(dim=0)).flatten()
    free = torch.ones(d.shape[1], dtype=torch.bool)
    free[cols] = False
    pool = torch.nonzero(free).flatten()
    if cols.numel() == 0 or pool.numel() < cols.numel():
        return None
    picked = pool[torch.randperm(pool.numel(), generator=gen)[: cols.numel()]]
    out = torch.zeros_like(d)
    pure_rank_one = (tensor_edit.u is not None and tensor_edit.a is None
                     and tensor_edit.delta is None and tensor_edit.index is None)
    if pure_rank_one:
        u = torch.from_numpy(tensor_edit.u).double()
        out[:, picked] = -tensor_edit.alpha * torch.outer(u, u @ w0[:, picked])
    else:
        out[:, picked] = d[:, cols]
    return out


def build_control(model: Any, edit: WeightEdit, kind: str, seed: int = 0) -> WeightEdit:
    """Build one control for *edit* against the model's current weights.

    Raises ``ValueError`` for an unknown *kind*, a non-2-D edited tensor, or a
    ``random_columns`` request on a tensor with no unchanged columns left.
    """
    import torch

    if kind not in CONTROL_KINDS:
        raise ValueError(f"unknown control {kind!r}; expected one of {CONTROL_KINDS}")
    module = module_of(model)
    gen = _generator(seed)
    tensors: list[TensorEdit] = []
    with torch.no_grad():
        for tensor_edit, (key, param) in zip(edit.tensors, resolve_parameters(module, edit)):
            if param.ndim != 2:
                raise ValueError(f"{key}: controls are defined for 2-D weights")
            w0 = param.detach().double().cpu()
            d = tensor_edit.apply_to(param).double().cpu() - w0
            if kind == "random_direction":
                control = _random_direction(d, gen)
            else:
                control = _random_columns(w0, d, tensor_edit, gen)
                if control is None:
                    raise ValueError(
                        f"{key}: no unchanged columns left for a random_columns control"
                    )
            tensors.append(TensorEdit(name=key, form="delta", delta=control.float(),
                                      shape=tuple(param.shape)))
    return WeightEdit(tensors, name=f"{edit.name}:{kind}:{seed}",
                      metadata={"control_for": edit.name, "kind": kind, "seed": int(seed)})
