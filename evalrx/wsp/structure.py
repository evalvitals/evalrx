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

from evalrx.wsp.edit import TensorEdit, WeightEdit, module_of, resolve_parameters


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


def _delta_and_direction(param: Any, tensor_edit: TensorEdit) -> "tuple[Any, Any, Any]":
    """``(delta, changed columns, top left singular vector)`` of one tensor edit."""
    import torch

    d = tensor_edit.apply_to(param).double().cpu() - param.detach().double().cpu()
    cols = torch.nonzero((d != 0).any(dim=0)).flatten() if d.ndim == 2 else torch.zeros(0)
    if d.ndim != 2 or cols.numel() == 0:
        return d, cols, None
    return d, cols, torch.linalg.svd(d[:, cols], full_matrices=False)[0][:, 0]


def fit_rank_one(model: Any, edit: WeightEdit, *, name: "str | None" = None) -> WeightEdit:
    """Re-express *edit* as the rank-one rule that best reproduces it on the live weights.

    For every edited 2-D tensor: ``u`` is the top left singular vector of the
    change, the columns are the changed ones, and ``alpha`` is the mean of
    ``-a_j / p_j`` (see :func:`delta_structure`). The result can be scaled
    (:meth:`WeightEdit.scaled`), which an edit stored as overwritten entries
    cannot. ``metadata["fit"]`` records, per tensor, the fitted ``alpha`` and the
    share of the original change the rule leaves unexplained
    (``rank_one_residual``); check it before treating the fit as the edit.
    """
    import torch

    module = module_of(model)
    tensors: list[TensorEdit] = []
    fit: list[dict[str, Any]] = []
    with torch.no_grad():
        for tensor_edit, (key, param) in zip(edit.tensors, resolve_parameters(module, edit)):
            d, cols, u = _delta_and_direction(param, tensor_edit)
            if u is None:
                raise ValueError(f"{key}: no 2-D change to fit a rank-one rule to")
            row = delta_structure(param.detach().cpu(), (param.detach().double().cpu() + d))
            if "alpha_estimate" not in row:
                raise ValueError(f"{key}: the changed columns have no component along the "
                                 "change direction, so no alpha can be read off")
            tensors.append(TensorEdit.rank_one(key, u, row["alpha_estimate"], cols=cols))
            fit.append({"name": key, "alpha": row["alpha_estimate"],
                        "alpha_spread": row.get("alpha_spread"),
                        "rank_one_residual": row["rank_one_residual"],
                        "n_columns": int(cols.numel())})
    return WeightEdit(tensors, name=name or f"{edit.name}_rank_one",
                      metadata={"source": "rank_one_fit", "fit_of": edit.name, "fit": fit})


def edit_agreement(model: Any, edit: WeightEdit, reference: WeightEdit) -> dict[str, Any]:
    """How closely *edit* matches *reference*: same tensors, direction, neurons.

    For every tensor both edits change, reports ``direction_cosine`` (absolute
    cosine between the top left singular vectors of the two changes; a
    direction's sign is arbitrary), and the overlap of the changed columns
    (``n_shared``, ``jaccard``, and ``expected_shared_by_chance`` for two random
    column sets of those sizes). ``only_in_edit`` / ``only_in_reference`` list
    tensors just one of them touches.
    """
    import torch

    module = module_of(model)
    mine = {key: (param, t) for t, (key, param) in zip(edit.tensors,
                                                       resolve_parameters(module, edit))}
    theirs = {key: (param, t) for t, (key, param) in zip(reference.tensors,
                                                         resolve_parameters(module, reference))}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for key in sorted(set(mine) & set(theirs)):
            param = mine[key][0]
            _, cols_a, u_a = _delta_and_direction(param, mine[key][1])
            _, cols_b, u_b = _delta_and_direction(param, theirs[key][1])
            a, b = set(cols_a.tolist()), set(cols_b.tolist())
            row: dict[str, Any] = {
                "name": key, "n_columns_edit": len(a), "n_columns_reference": len(b),
                "n_shared": len(a & b),
                "jaccard": round(len(a & b) / len(a | b), 4) if a | b else None,
            }
            if param.ndim == 2 and param.shape[1]:
                row["expected_shared_by_chance"] = round(len(a) * len(b) / param.shape[1], 3)
            if u_a is not None and u_b is not None:
                row["direction_cosine"] = float((u_a @ u_b).abs())
            rows.append(row)
    return {"edit": edit.name, "reference": reference.name, "tensors": rows,
            "only_in_edit": sorted(set(mine) - set(theirs)),
            "only_in_reference": sorted(set(theirs) - set(mine))}
