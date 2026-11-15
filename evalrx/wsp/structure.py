"""What a weight edit changed — read from the weights alone.

The first question about any edit is structural, and it needs no forward pass:
how many entries moved, in which rows and columns, and is the change low rank?
For a rank-one edit ``dW = u a^T`` the per-column coefficient ``a_j`` can be
compared with the column's old projection ``p_j = u^T w_j``: a constant ratio
``a_j / p_j = -alpha`` means the edit is the rule ``w' = w - alpha u (u^T w)``
applied to those columns, and ``alpha`` can be read off.
"""

from __future__ import annotations

from typing import Any

from evalrx.wsp.edit import WeightEdit, module_of, resolve_parameters


def delta_structure(w0: Any, w1: Any, *, top_singular: int = 8) -> dict[str, Any]:
    """Describe ``w1 - w0`` for one 2-D weight.

    Returns (all values JSON-serialisable):

    ``changed_entries`` / ``changed_rows`` / ``changed_columns``
        Where the tensor moved.
    ``relative_change``
        ``||dW||_F / ||W0||_F``.
    ``singular_values``
        Leading singular values of the change restricted to its changed columns.
    ``effective_rank``
        Participation ratio ``(sum s)^2 / sum s^2`` of the singular values.
    ``rank_one_residual``
        ``||dW - u a^T||_F / ||dW||_F`` for the top singular pair; 0 for an
        exactly rank-one change.
    ``alpha_estimate`` / ``alpha_spread``
        Mean and standard deviation of ``-a_j / p_j`` over the changed columns
        (only columns whose old projection is not negligible).
    ``projection_selectivity``
        Mean ``|u^T w_j|`` over changed columns divided by the mean over all
        columns; above 1 means the edit chose columns that write along ``u``.
    """
    import torch

    w0 = w0.detach().double()
    w1 = w1.detach().double()
    if w0.shape != w1.shape:
        raise ValueError(f"shape mismatch: {tuple(w0.shape)} vs {tuple(w1.shape)}")
    d = w1 - w0
    changed = d != 0
    out: dict[str, Any] = {
        "shape": list(w0.shape),
        "changed_entries": int(changed.sum()),
        "relative_change": float(d.norm() / w0.norm()) if float(w0.norm()) > 0 else None,
    }
    if d.ndim != 2 or out["changed_entries"] == 0:
        return out
    cols = torch.nonzero(changed.any(dim=0)).flatten()
    out["changed_rows"] = int(changed.any(dim=1).sum())
    out["changed_columns"] = int(cols.numel())
    dc = d[:, cols]
    u_all, s, _ = torch.linalg.svd(dc, full_matrices=False)
    u = u_all[:, 0]
    a = u @ dc
    out["singular_values"] = [float(v) for v in s[:top_singular]]
    out["effective_rank"] = float(s.sum() ** 2 / (s ** 2).sum()) if float(s.sum()) > 0 else 0.0
    out["rank_one_residual"] = float((dc - torch.outer(u, a)).norm() / dc.norm())
    p_all = u @ w0
    p = p_all[cols]
    scale = float(p_all.abs().mean())
    solid = p.abs() > 1e-3 * max(scale, 1e-30)
    if bool(solid.any()):
        ratio = -(a[solid] / p[solid])
        out["alpha_estimate"] = float(ratio.mean())
        out["alpha_spread"] = float(ratio.std()) if int(solid.sum()) > 1 else 0.0
    if scale > 0:
        out["projection_selectivity"] = float(p.abs().mean() / scale)
    return out


def edit_structure(model: Any, edit: WeightEdit, **kwargs: Any) -> dict[str, Any]:
    """:func:`delta_structure` for every tensor of *edit* against the live weights.

    Also reports ``direction_cosines``: ``|cos|`` between the top left singular
    vectors of each pair of edited tensors of equal row count, which says
    whether the edit writes the same direction at several layers.
    """
    import torch

    module = module_of(model)
    rows: list[dict[str, Any]] = []
    directions: list[tuple[str, Any]] = []
    with torch.no_grad():
        for tensor_edit, (key, param) in zip(edit.tensors, resolve_parameters(module, edit)):
            new = tensor_edit.apply_to(param)
            row = {"name": key, "form": tensor_edit.form, **delta_structure(param, new, **kwargs)}
            rows.append(row)
            if row.get("changed_columns"):
                d = new.double() - param.double()
                directions.append((key, torch.linalg.svd(d, full_matrices=False)[0][:, 0]))
    cosines = {
        f"{a_key} ~ {b_key}": float((ua @ ub).abs())
        for i, (a_key, ua) in enumerate(directions)
        for b_key, ub in directions[i + 1:]
        if ua.numel() == ub.numel()
    }
    return {"edit": edit.name, "tensors": rows, "direction_cosines": cosines}
