"""Weight edits — direct, training-free changes to a model's parameters.

Weight-space programming (WSP) changes a model by writing new values into a few
weight tensors instead of training. The edits studied so far are small and
structured: a rank-one change to some columns of one MLP output projection,
``w' = w - alpha * u (u^T w)``, or a sparse set of overwritten entries.

This module is the data model and the one place such an edit touches a model:

* :class:`TensorEdit` — the change to one tensor, stored in the form it was
  written in (rank-one rule, low-rank factors, dense delta, sparse overwrite);
* :class:`WeightEdit` — a named set of tensor edits with provenance;
* :func:`applied` — a context manager that writes the edit into a live module
  and restores the exact original values on exit, including after an error.

Reading the edit files published with *Sparks of Weight-Space Programming*
(``edit.npz`` + ``manifest.json``) is supported by :meth:`WeightEdit.from_astra`.

Torch is imported lazily; building, saving and loading edits needs numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import numpy as np

FORMAT_VERSION = 1


def _f32(x: Any) -> "np.ndarray | None":
    if x is None:
        return None
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().float().cpu().numpy()
    return np.ascontiguousarray(x, dtype=np.float32)


def _index(x: Any) -> "np.ndarray | None":
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x, dtype=np.int64).reshape(-1)


def bf16_bits_to_f32(bits: Any) -> np.ndarray:
    """bfloat16 bit patterns (uint16) -> float32 values."""
    return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)


@dataclass
class TensorEdit:
    """The change to one weight tensor.

    The components are applied in this order, each one optional:

    1. rank-one rule   ``W[:, cols] -= alpha * u (u^T W[:, cols])``
       (``u`` is normalised to unit length; ``cols=None`` means every column);
    2. low-rank update ``W += A @ B`` (``A``: ``[rows, k]``, ``B``: ``[k, cols]``);
    3. dense delta     ``W += delta``;
    4. sparse overwrite ``W.flat[index] = values``.

    The rank-one rule is defined for a 2-D weight whose ROWS index the space
    ``u`` lives in. For an MLP output projection (``[d_model, d_mlp]``) that is
    the residual stream, and each column is one neuron's write vector.

    Attributes:
        name:  Parameter name inside the module, or a unique suffix of it.
        form:  Label of the leading component: ``rank_one`` | ``low_rank`` |
               ``delta`` | ``sparse_set``. Informational; application is driven
               by which components are present.
        base_sha256: When known, the hash of the tensor the edit was made on.
    """

    name: str
    form: str = "delta"
    u: "np.ndarray | None" = None
    alpha: float = 0.0
    cols: "np.ndarray | None" = None
    a: "np.ndarray | None" = None
    b: "np.ndarray | None" = None
    delta: "np.ndarray | None" = None
    index: "np.ndarray | None" = None
    values: "np.ndarray | None" = None
    shape: "tuple[int, ...] | None" = None
    base_sha256: str = ""
    edited_sha256: str = ""

    def __post_init__(self) -> None:
        self.u, self.a, self.b = _f32(self.u), _f32(self.a), _f32(self.b)
        self.delta, self.values = _f32(self.delta), _f32(self.values)
        self.cols, self.index = _index(self.cols), _index(self.index)
        self.alpha = float(self.alpha)
        if self.shape is not None:
            self.shape = tuple(int(s) for s in self.shape)
        if self.u is not None:
            self.u = self.u.reshape(-1)
            norm = float(np.linalg.norm(self.u))
            if not np.isfinite(norm) or norm == 0.0:
                raise ValueError(f"{self.name}: the edit direction u must be finite and non-zero")
            self.u = self.u / norm
        if self.values is not None:
            self.values = self.values.reshape(-1)
        if (self.index is None) != (self.values is None):
            raise ValueError(f"{self.name}: index and values must be given together")
        if self.index is not None and len(self.index) != len(self.values):
            raise ValueError(f"{self.name}: index and values differ in length")
        if (self.a is None) != (self.b is None):
            raise ValueError(f"{self.name}: low-rank factors a and b must be given together")

    # -- constructors ---------------------------------------------------
    @classmethod
    def rank_one(cls, name: str, u: Any, alpha: float, cols: Any = None) -> "TensorEdit":
        """``W[:, cols] -= alpha * u (u^T W[:, cols])``.

        ``alpha=1`` removes the component along ``u``; ``alpha=2`` reflects it;
        larger values reflect and amplify (``alpha=6``: flipped, five times larger).
        """
        return cls(name=name, form="rank_one", u=u, alpha=alpha, cols=cols)

    # -- application ----------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return (self.u is None and self.a is None and self.delta is None
                and self.index is None)

    def apply_to(self, weight: Any) -> Any:
        """Return the edited tensor (same dtype and device as *weight*; *weight* is untouched)."""
        import torch

        if self.shape is not None and tuple(weight.shape) != self.shape:
            raise ValueError(
                f"{self.name}: the edit was made for shape {self.shape}, "
                f"the tensor has shape {tuple(weight.shape)}"
            )
        # float64 so that composing components does not accumulate rounding
        # before the single cast back to the tensor's own dtype.
        w = weight.detach().to(torch.float64).clone()
        dev = w.device

        def t(x: np.ndarray) -> Any:
            return torch.from_numpy(x).to(device=dev, dtype=torch.float64)

        if self.u is not None and self.alpha != 0.0:
            if w.ndim != 2:
                raise ValueError(f"{self.name}: the rank-one rule needs a 2-D weight")
            u = t(self.u)
            if u.numel() != w.shape[0]:
                raise ValueError(
                    f"{self.name}: u has {u.numel()} entries, the weight has {w.shape[0]} rows"
                )
            if self.cols is None:
                w = w - self.alpha * torch.outer(u, u @ w)
            else:
                cols = torch.from_numpy(self.cols).to(dev)
                if int(cols.max()) >= w.shape[1] or int(cols.min()) < 0:
                    raise ValueError(f"{self.name}: column index out of range")
                sub = w[:, cols]
                w[:, cols] = sub - self.alpha * torch.outer(u, u @ sub)
        if self.a is not None:
            w = w + (t(self.a) @ t(self.b)).reshape(w.shape)
        if self.delta is not None:
            w = w + t(self.delta).reshape(w.shape)
        if self.index is not None:
            flat = w.reshape(-1).clone()
            idx = torch.from_numpy(self.index).to(dev)
            if int(idx.max()) >= flat.numel() or int(idx.min()) < 0:
                raise ValueError(f"{self.name}: flat index out of range")
            flat[idx] = t(self.values)
            w = flat.reshape(w.shape)
        # Through float32 on the way down: edits are written as float32 values
        # and then cast, and going straight from float64 to a 16-bit type can
        # round a tie differently from that convention.
        return w.to(torch.float32).to(weight.dtype)

    def scaled(self, factor: float) -> "TensorEdit":
        """The same edit at *factor* times its strength.

        Exact for the rank-one, low-rank and dense components. A sparse
        overwrite stores absolute values, not a change, so it cannot be scaled
        without the base tensor and is refused.
        """
        if self.index is not None:
            raise ValueError(
                f"{self.name}: a sparse overwrite cannot be scaled; convert it to a "
                "dense delta against the base weights first (WeightEdit.as_delta)"
            )
        return TensorEdit(
            name=self.name, form=self.form, u=self.u, alpha=self.alpha * factor, cols=self.cols,
            a=None if self.a is None else self.a * factor, b=self.b,
            delta=None if self.delta is None else self.delta * factor, shape=self.shape,
        )

    # -- description ----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "form": self.form}
        if self.shape is not None:
            out["shape"] = list(self.shape)
        if self.u is not None:
            out["alpha"] = self.alpha
            out["n_cols"] = None if self.cols is None else int(len(self.cols))
        if self.a is not None:
            out["rank"] = int(self.a.shape[1]) if self.a.ndim == 2 else 1
        if self.index is not None:
            out["n_overwritten"] = int(len(self.index))
        if self.base_sha256:
            out["base_sha256"] = self.base_sha256
        return out

    _ARRAYS = ("u", "cols", "a", "b", "delta", "index", "values")


@dataclass
class WeightEdit:
    """A set of tensor edits that together make one edit to a model.

    Attributes:
        tensors:  The per-tensor changes.
        name:     Short identifier for reports.
        metadata: Free-form provenance (who wrote the edit, from what evidence).
    """

    tensors: list[TensorEdit] = field(default_factory=list)
    name: str = "weight_edit"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [t.name for t in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: a tensor is edited more than once: {sorted(names)}")

    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def tensor_names(self) -> list[str]:
        return [t.name for t in self.tensors]

    def scaled(self, factor: float) -> "WeightEdit":
        return WeightEdit([t.scaled(factor) for t in self.tensors],
                          name=f"{self.name}_x{factor:g}",
                          metadata={**self.metadata, "scaled_by": factor})

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable summary (no arrays)."""
        return {"name": self.name, "n_tensors": len(self.tensors),
                "tensors": [t.describe() for t in self.tensors], "metadata": self.metadata}

    # -- building from weights -------------------------------------------
    @classmethod
    def from_state_dicts(cls, base: Mapping[str, Any], edited: Mapping[str, Any], *,
                         name: str = "weight_edit", atol: float = 0.0) -> "WeightEdit":
        """The edit that turns *base* into *edited*, as one dense delta per changed tensor."""
        tensors: list[TensorEdit] = []
        for key, w0 in base.items():
            if key not in edited:
                continue
            d = _f32(edited[key]).astype(np.float64) - _f32(w0).astype(np.float64)
            if d.shape and float(np.abs(d).max(initial=0.0)) > atol:
                tensors.append(TensorEdit(name=key, form="delta", delta=d.astype(np.float32),
                                          shape=tuple(d.shape)))
        return cls(tensors, name=name, metadata={"source": "state_dict_difference"})

    def as_delta(self, module: Any) -> "WeightEdit":
        """This edit as dense deltas against the CURRENT weights of *module*.

        *module* may be a ``torch.nn.Module`` or a white-box model handle.
        """
        module = module_of(module)
        tensors = []
        for tensor_edit, (key, param) in zip(self.tensors, resolve_parameters(module, self)):
            delta = (tensor_edit.apply_to(param).double() - param.detach().double()).float()
            tensors.append(TensorEdit(name=key, form="delta", delta=delta,
                                      shape=tuple(param.shape)))
        return WeightEdit(tensors, name=self.name, metadata={**self.metadata, "as_delta": True})

    # -- persistence -----------------------------------------------------
    def save(self, directory: str) -> str:
        """Write ``manifest.json`` + ``edit.npz`` to *directory*; return the directory."""
        os.makedirs(directory, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        manifest: dict[str, Any] = {
            "format": "evalrx.weight_edit", "version": FORMAT_VERSION,
            "name": self.name, "metadata": self.metadata, "tensors": [],
        }
        for i, t in enumerate(self.tensors):
            entry: dict[str, Any] = {
                "name": t.name, "form": t.form, "alpha": t.alpha,
                "shape": None if t.shape is None else list(t.shape),
                "base_sha256": t.base_sha256, "edited_sha256": t.edited_sha256, "arrays": [],
            }
            for attr in TensorEdit._ARRAYS:
                value = getattr(t, attr)
                if value is not None:
                    arrays[f"t{i}_{attr}"] = value
                    entry["arrays"].append(attr)
            manifest["tensors"].append(entry)
        np.savez_compressed(os.path.join(directory, "edit.npz"), **arrays)
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, default=str)
        return directory

    @classmethod
    def load(cls, directory: str) -> "WeightEdit":
        """Read an edit written by :meth:`save` (or an Astra edit directory)."""
        with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("format") != "evalrx.weight_edit":
            return cls.from_astra(directory)
        if int(manifest.get("version", 0)) > FORMAT_VERSION:
            raise ValueError(f"{directory}: written by a newer format version")
        with np.load(os.path.join(directory, "edit.npz"), allow_pickle=False) as z:
            tensors = []
            for i, entry in enumerate(manifest["tensors"]):
                arrays = {attr: z[f"t{i}_{attr}"] for attr in entry.get("arrays", [])}
                tensors.append(TensorEdit(
                    name=entry["name"], form=entry.get("form", "delta"),
                    alpha=entry.get("alpha", 0.0), shape=entry.get("shape"),
                    base_sha256=entry.get("base_sha256", ""),
                    edited_sha256=entry.get("edited_sha256", ""), **arrays,
                ))
        return cls(tensors, name=manifest.get("name", "weight_edit"),
                   metadata=dict(manifest.get("metadata") or {}))

    @classmethod
    def from_astra(cls, directory: str, *, name: "str | None" = None) -> "WeightEdit":
        """Read an edit in the layout of the *Sparks of Weight-Space Programming* logs.

        ``manifest.json`` lists the edited tensors; ``edit.npz`` holds, per tensor
        ``i``: for ``form == "lowrank"`` the factors ``t{i}_a`` (``[rows, k]``) and
        ``t{i}_b`` (``[k, cols]``) added to the base; and for every form the flat
        indices ``t{i}_idx`` whose final values are the bfloat16 bit patterns
        ``t{i}_val``. Those overwritten entries are absolute values, so the edit
        reproduces the evaluated checkpoint only on the base it was made on; the
        recorded ``base_sha256`` is kept for :func:`applied` to check.
        """
        with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        tensors = []
        with np.load(os.path.join(directory, "edit.npz"), allow_pickle=False) as z:
            for i, entry in enumerate(manifest["tensors"]):
                lowrank = entry.get("form") == "lowrank"
                tensors.append(TensorEdit(
                    name=entry["name"],
                    form="low_rank" if lowrank else "sparse_set",
                    a=z[f"t{i}_a"] if lowrank else None,
                    b=z[f"t{i}_b"] if lowrank else None,
                    index=z[f"t{i}_idx"] if f"t{i}_idx" in z.files else None,
                    values=(bf16_bits_to_f32(z[f"t{i}_val"])
                            if f"t{i}_val" in z.files else None),
                    shape=entry.get("shape"),
                    base_sha256=entry.get("base_sha256", ""),
                    edited_sha256=entry.get("edited_sha256", ""),
                ))
        label = name or os.path.basename(os.path.dirname(os.path.abspath(directory))) or "astra"
        return cls(tensors, name=label,
                   metadata={"source": "astra", "dtype": "bfloat16",
                             "changed_entries_total": manifest.get("changed_entries_total")})


# ----------------------------------------------------------------------
# Touching a live module
# ----------------------------------------------------------------------

def module_of(model: Any) -> Any:
    """The ``torch.nn.Module`` to edit: *model* itself, or the one behind a white-box handle."""
    import torch.nn as nn

    if isinstance(model, nn.Module):
        return model
    from evalrx.analyzers._internals.access import torch_module

    return torch_module(model, "a weight edit")


def _anchored_suffix(name: str) -> str:
    """The part of a parameter name that must match: from the indexed container on.

    ``model.language_model.layers.15.mlp.down_proj.weight`` anchors at
    ``layers.15.mlp.down_proj.weight``. Wrapper prefixes before the container
    differ between exports of the same model and may be dropped; the layer
    index and everything after it may not, or an edit could land on another
    layer. A name with no index has no droppable prefix.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[max(i - 1, 0):])
    return name


def resolve_parameters(module: Any, edit: WeightEdit) -> "list[tuple[str, Any]]":
    """``[(full parameter name, parameter), ...]`` for the tensors of *edit*, in order.

    A tensor name matches a parameter exactly, or — when the two differ only in
    wrapper prefixes — by the suffix that starts at the indexed container (see
    :func:`_anchored_suffix`). That lets an edit written against
    ``model.language_model.layers.15.mlp.down_proj.weight`` apply to a module
    that names the same tensor ``model.layers.15.mlp.down_proj.weight``. A name
    that matches nothing, or more than one parameter, is an error.
    """
    params = dict(module.named_parameters())
    out: list[tuple[str, Any]] = []
    for tensor_edit in edit.tensors:
        if tensor_edit.name in params:
            out.append((tensor_edit.name, params[tensor_edit.name]))
            continue
        suffix = _anchored_suffix(tensor_edit.name)
        match = [k for k in params if k == suffix or k.endswith("." + suffix)]
        if len(match) != 1:
            problem = "matches no parameter" if not match else f"is ambiguous ({match[:3]})"
            raise KeyError(f"weight edit tensor {tensor_edit.name!r} {problem} in "
                           f"{type(module).__name__}")
        out.append((match[0], params[match[0]]))
    if len({key for key, _ in out}) != len(out):
        raise KeyError(f"{edit.name}: two tensor edits resolve to the same parameter")
    return out


def tensor_sha256(tensor: Any) -> str:
    """sha256 of a tensor's raw bytes in its own dtype (the hash edit manifests record)."""
    import torch

    data = tensor.detach().cpu().contiguous()
    if data.dtype == torch.bfloat16:
        data = data.view(torch.int16)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


@dataclass
class AppliedEdit:
    """What :func:`applied` did: one row per edited tensor."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def relative_change(self) -> float:
        """Largest ``||dW|| / ||W||`` over the edited tensors."""
        return max((r["relative_change"] for r in self.rows), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"tensors": self.rows, "max_relative_change": self.relative_change}


@contextmanager
def applied(model: Any, edit: WeightEdit, *, verify_base: bool = False,
            max_relative_change: "float | None" = None) -> Iterator[AppliedEdit]:
    """Write *edit* into the model for the duration of the block, then restore it.

    Every edited tensor is computed first; nothing is written if any of them
    fails (a bad name, a shape mismatch, a change above *max_relative_change*).
    On exit, normal or not, the original values are copied back and compared,
    so a caller never continues with a half-edited or half-restored model.

    Args:
        model:               A ``torch.nn.Module`` or a white-box model handle.
        edit:                The edit to apply.
        verify_base:         Require every tensor's recorded ``base_sha256`` to
                             match the live tensor. A tensor with no recorded
                             hash is an error under this flag.
        max_relative_change: Refuse an edit whose ``||dW|| / ||W||`` exceeds
                             this on any tensor.

    Yields:
        :class:`AppliedEdit` with, per tensor, the resolved name, relative
        change, and counts of changed entries, rows and columns.
    """
    import torch

    module = module_of(model)
    resolved = resolve_parameters(module, edit)
    staged: list[tuple[Any, Any, Any]] = []
    report = AppliedEdit()
    with torch.no_grad():
        for tensor_edit, (key, param) in zip(edit.tensors, resolved):
            if verify_base:
                if not tensor_edit.base_sha256:
                    raise ValueError(f"{tensor_edit.name}: verify_base needs a recorded base hash")
                live = tensor_sha256(param)
                if live != tensor_edit.base_sha256:
                    raise ValueError(
                        f"{key}: the live tensor differs from the one the edit was made on "
                        f"(sha256 {live[:12]} != {tensor_edit.base_sha256[:12]})"
                    )
            new = tensor_edit.apply_to(param)
            if not bool(torch.isfinite(new).all()):
                raise ValueError(f"{key}: the edit produces non-finite weights")
            diff = (new.double() - param.double())
            base_norm = float(param.double().norm())
            rel = float(diff.norm()) / base_norm if base_norm > 0 else float("inf")
            if max_relative_change is not None and rel > max_relative_change:
                raise ValueError(
                    f"{key}: relative change {rel:.3g} exceeds the limit {max_relative_change:g}"
                )
            changed = diff != 0
            row = {"name": key, "form": tensor_edit.form, "relative_change": rel,
                   "changed_entries": int(changed.sum())}
            if changed.ndim == 2:
                row["changed_rows"] = int(changed.any(dim=1).sum())
                row["changed_columns"] = int(changed.any(dim=0).sum())
            report.rows.append(row)
            staged.append((param, param.detach().clone(), new))
        for param, _, new in staged:
            param.copy_(new)
    try:
        yield report
    finally:
        with torch.no_grad():
            for param, backup, _ in staged:
                param.copy_(backup)
            broken = [i for i, (param, backup, _) in enumerate(staged)
                      if not torch.equal(param, backup)]
        if broken:  # pragma: no cover - copy_ of a clone is exact; this guards the contract
            raise RuntimeError(f"{edit.name}: failed to restore {len(broken)} edited tensor(s)")
