"""Derive a rank-one weight edit from a contrast between two groups of cases.

A weight edit needs a direction and a set of neurons. When no one has written
the edit by hand, both can be read from the model's own activations on labelled
cases, without any gradient step:

* the DIRECTION at a layer is the difference between the mean MLP output on
  failing cases and on passing cases (unit length) — the component of the
  layer's write that separates the two groups;
* the NEURONS are the columns of the MLP output projection that push hardest
  along that direction on failing cases relative to passing ones:
  ``score_j = (mean_fail_j - mean_pass_j) * (u^T w_j)``.

The resulting edit ``w_j' = w_j - alpha * u (u^T w_j)`` on those columns removes
(``alpha = 1``) or reverses (``alpha > 1``) exactly that push.

This is a heuristic for PROPOSING an edit. It says nothing about whether the
edit helps: that is decided by validating it on cases the direction was not
computed from, against controls (:mod:`evalrx.wsp.controls`).

Activations are read at the last prompt position, where the first answer token
is predicted. A failure that only develops during generation (for example a
repetition loop) is not visible there.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from evalrx.wsp.edit import TensorEdit, WeightEdit, module_of

if TYPE_CHECKING:
    from evalrx.core.model import Model

#: Attribute paths of the MLP output projection inside a decoder layer, most
#: common first. The projection must be an ``nn.Linear`` whose weight is
#: ``[d_model, d_mlp]``.
_WRITE_PATHS = ("mlp.down_proj", "mlp.fc2", "mlp.c_proj", "mlp.fc_out", "feed_forward.w2",
                "feed_forward.down_proj", "ffn.down_proj")


@dataclass(frozen=True)
class WriteSite:
    """The MLP output projection of one decoder layer."""

    layer: int
    module: Any = field(repr=False, compare=False)
    name: str = ""  # full parameter name of the weight


def write_sites(model: Any) -> "list[WriteSite]":
    """MLP output projections of the language decoder, one per layer that has one.

    Layers whose MLP is not a single linear write (for example a
    mixture-of-experts block) are skipped rather than guessed at.
    """
    import torch.nn as nn

    from evalrx.analyzers._internals.access import decoder_stack

    module = module_of(model)
    stack = decoder_stack(module)
    names = {id(p): n for n, p in module.named_parameters()}
    sites: list[WriteSite] = []
    for index, layer in enumerate(stack.layers):
        for path in _WRITE_PATHS:
            obj = layer
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, nn.Linear) and id(obj.weight) in names:
                sites.append(WriteSite(layer=index, module=obj, name=names[id(obj.weight)]))
                break
    return sites


@contextmanager
def _capture_last(sites: "list[WriteSite]") -> "Iterator[dict[int, dict[str, Any]]]":
    """Record each site's input and output at the last sequence position."""
    captured: dict[int, dict[str, Any]] = {s.layer: {} for s in sites}

    def make(layer: int):
        def hook(_module: Any, args: Any, output: Any) -> None:
            captured[layer]["in"] = args[0].detach()[0, -1].double().cpu()
            captured[layer]["out"] = output.detach()[0, -1].double().cpu()
        return hook

    handles = [s.module.register_forward_hook(make(s.layer)) for s in sites]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def collect_write_stats(model: "Model", cases: Iterable[Any], *,
                        layers: "Iterable[int] | None" = None) -> "dict[int, dict[str, Any]]":
    """Mean MLP input (neuron activations) and output at the last position.

    Returns ``{layer: {"in": mean activations [d_mlp], "out": mean write
    [d_model], "n": cases}}``. Cases whose forward pass fails are skipped.
    """
    from evalrx.core.capability import Capability

    sites = write_sites(model)
    if layers is not None:
        wanted = {int(i) for i in layers}
        sites = [s for s in sites if s.layer in wanted]
    sums: dict[int, dict[str, Any]] = {s.layer: {"in": None, "out": None, "n": 0} for s in sites}
    for case in cases:
        with _capture_last(sites) as captured:
            try:
                model.forward(getattr(case, "inputs", case), capture={Capability.LOGITS})
            except Exception:  # noqa: BLE001 - one unreadable case must not end the pass
                continue
        for layer, got in captured.items():
            if "in" not in got:
                continue
            acc = sums[layer]
            acc["in"] = got["in"] if acc["in"] is None else acc["in"] + got["in"]
            acc["out"] = got["out"] if acc["out"] is None else acc["out"] + got["out"]
            acc["n"] += 1
    return {
        layer: {"in": acc["in"] / acc["n"], "out": acc["out"] / acc["n"], "n": acc["n"]}
        for layer, acc in sums.items() if acc["n"]
    }


def rank_columns(stats_a: "dict[str, Any]", stats_b: "dict[str, Any]", weight: Any,
                 ) -> "dict[str, Any]":
    """Direction and per-neuron scores for one layer from two groups' mean activity.

    ``stats_a`` is the group to move away from (failing cases, loop positions),
    ``stats_b`` the reference (passing cases, productive positions). Returns

    ``u``           unit difference of the mean writes, ``a - b``;
    ``push``        ``(act_a - act_b) * (u^T w_j)`` — how much more neuron j
                    writes along ``u`` in group a than in group b;
    ``projection``  ``u^T w_j`` — what neuron j writes along ``u`` when active;
    ``activation``  ``act_a - act_b`` — how much more neuron j fires in group a;
    ``separation``  ``||a - b||`` relative to the mean size of the two writes.

    ``u`` is ``None`` when the two groups write the same thing.
    """
    gap = stats_a["out"] - stats_b["out"]
    scale = 0.5 * (stats_a["out"].norm() + stats_b["out"].norm())
    separation = float(gap.norm() / scale) if float(scale) > 0 else 0.0
    if float(gap.norm()) == 0.0:
        return {"u": None, "separation": separation}
    u = gap / gap.norm()
    projection = u @ weight.detach().double().cpu()
    activation = stats_a["in"] - stats_b["in"]
    return {"u": u, "push": activation * projection, "projection": projection,
            "activation": activation, "separation": separation}


def edit_from_stats(
    model: Any,
    stats_a: "dict[int, dict[str, Any]]",
    stats_b: "dict[int, dict[str, Any]]",
    *,
    layers: "Iterable[int] | None" = None,
    n_layers: int = 2,
    n_columns: int = 64,
    alpha: float = 1.0,
    name: str = "contrast_rank_one",
    metadata: "dict[str, Any] | None" = None,
) -> WeightEdit:
    """Build the rank-one edit that removes what group a writes beyond group b.

    The shared core of :func:`contrast_edit` and
    :func:`continuation_contrast_edit`; *stats_a* / *stats_b* are
    ``{layer: {"in", "out", "n"}}`` as the collectors return them.
    """
    import torch

    sites = {s.layer: s for s in write_sites(model)}
    if not sites:
        raise ValueError("no MLP output projection found in the decoder layers")
    chosen = None if layers is None else [int(i) for i in layers]
    if chosen is not None:
        unknown = sorted(set(chosen) - set(sites))
        if unknown:
            raise ValueError(f"layers {unknown} have no MLP output projection "
                             f"(available: {sorted(sites)})")
    shared = sorted(set(stats_a) & set(stats_b) & set(sites))
    if not shared:
        raise ValueError("need at least one readable FAIL case and one readable PASS case")
    ranked = {layer: rank_columns(stats_a[layer], stats_b[layer], sites[layer].module.weight)
              for layer in shared}
    separation = {layer: ranked[layer]["separation"] for layer in shared}
    if chosen is None:
        chosen = sorted(sorted(shared, key=lambda i: -separation[i])[: max(1, int(n_layers))])
    else:
        chosen = [i for i in chosen if i in separation]

    tensors: list[TensorEdit] = []
    picked: dict[str, Any] = {}
    for layer in chosen:
        row = ranked[layer]
        if row["u"] is None:
            continue
        push = row["push"]
        k = min(int(n_columns), push.numel())
        cols = torch.topk(push, k).indices
        cols = cols[push[cols] > 0]  # only neurons that push TOWARD group a
        if cols.numel() == 0:
            continue
        tensors.append(TensorEdit.rank_one(sites[layer].name, row["u"], alpha,
                                           cols=cols.sort().values))
        picked[str(layer)] = {"n_columns": int(cols.numel()),
                              "separation": round(separation[layer], 6)}
    if not tensors:
        raise ValueError("the FAIL and PASS groups do not differ at any candidate layer")
    return WeightEdit(tensors, name=name, metadata={
        "source": "contrast", "alpha": float(alpha), "layers": picked,
        "layer_separation": {str(i): round(separation[i], 6) for i in shared},
        **(metadata or {}),
    })


def contrast_edit(
    model: "Model",
    fail_cases: Iterable[Any],
    pass_cases: Iterable[Any],
    *,
    layers: "Iterable[int] | None" = None,
    n_layers: int = 2,
    n_columns: int = 64,
    alpha: float = 1.0,
    name: str = "contrast_rank_one",
) -> WeightEdit:
    """Build a rank-one edit from the FAIL-minus-PASS contrast (see module docstring).

    Args:
        model:      White-box model handle.
        fail_cases: Cases the model gets wrong.
        pass_cases: Cases the model gets right.
        layers:     Decoder layers to edit. Default: the *n_layers* layers
                    whose mean writes differ most between the groups, relative
                    to the size of the write.
        n_layers:   How many layers to pick when *layers* is not given.
        n_columns:  Neurons edited per layer (capped at the layer's width).
        alpha:      Strength of ``w' = w - alpha u (u^T w)``.

    Raises:
        ValueError: when either group is empty after the forward passes, or the
                    model has no recognisable MLP output projection.
    """
    chosen = None if layers is None else [int(i) for i in layers]
    available = {s.layer for s in write_sites(model)}
    if not available:
        raise ValueError("no MLP output projection found in the decoder layers")
    if chosen is not None and set(chosen) - available:
        raise ValueError(f"layers {sorted(set(chosen) - available)} have no MLP output "
                         f"projection (available: {sorted(available)})")
    fail = collect_write_stats(model, fail_cases, layers=chosen)
    passed = collect_write_stats(model, pass_cases, layers=chosen)
    shared = sorted(set(fail) & set(passed))
    return edit_from_stats(
        model, fail, passed, layers=chosen, n_layers=n_layers, n_columns=n_columns,
        alpha=alpha, name=name,
        metadata={"n_fail": min((fail[i]["n"] for i in shared), default=0),
                  "n_pass": min((passed[i]["n"] for i in shared), default=0),
                  "position": "last_prompt_token"},
    )


def collect_continuation_stats(
    model: "Model",
    cases: Iterable[Any],
    *,
    groups: str = "repetition",
    repeat_ngram: int = 32,
    max_tokens: int = 4096,
    layers: "Iterable[int] | None" = None,
) -> Any:
    """Mean MLP input and output per position group along recorded generations.

    Each case's recorded output is appended to its prompt
    (:mod:`evalrx.wsp.continuation`) and run once; at every layer the MLP
    output projection's input and output are averaged over the positions of
    each group (``loop`` / ``productive`` with ``groups="repetition"``).

    Returns a :class:`~evalrx.wsp.continuation.ContinuationStats`.
    """
    import torch

    from evalrx.wsp.continuation import ContinuationStats, encode_continuations, group_names

    names = group_names(groups)
    sites = write_sites(model)
    if layers is not None:
        wanted = {int(i) for i in layers}
        sites = [s for s in sites if s.layer in wanted]
    module = module_of(model)
    device = next(module.parameters()).device
    sequences, skipped = encode_continuations(
        model, cases, groups=groups, repeat_ngram=repeat_ngram, max_tokens=max_tokens)

    sums = {g: {s.layer: {"in": None, "out": None, "n": 0} for s in sites} for g in names}
    current: dict[str, Any] = {}

    def make(layer: int):
        def hook(_module: Any, args: Any, output: Any) -> None:
            x = args[0].detach()[0].double()
            y = output.detach()[0].double()
            for g, index in current["index"].items():
                if not len(index):
                    continue
                acc = sums[g][layer]
                xin, yout = x[index].sum(dim=0).cpu(), y[index].sum(dim=0).cpu()
                acc["in"] = xin if acc["in"] is None else acc["in"] + xin
                acc["out"] = yout if acc["out"] is None else acc["out"] + yout
                acc["n"] += int(len(index))
        return hook

    handles = [s.module.register_forward_hook(make(s.layer)) for s in sites]
    try:
        for seq in sequences:
            current["index"] = {g: torch.tensor(seq.positions(g), dtype=torch.long, device=device)
                                for g in names}
            with torch.no_grad():
                module(input_ids=torch.tensor([seq.ids], device=device))
    finally:
        for handle in handles:
            handle.remove()
    return ContinuationStats(
        groups={g: {layer: {"in": acc["in"] / acc["n"], "out": acc["out"] / acc["n"],
                            "n": acc["n"]}
                    for layer, acc in per_layer.items() if acc["n"]}
                for g, per_layer in sums.items()},
        n_cases=len(sequences), skipped=dict(skipped),
    )


def continuation_contrast_edit(
    model: "Model",
    cases: Iterable[Any],
    *,
    repeat_ngram: int = 32,
    max_tokens: int = 4096,
    layers: "Iterable[int] | None" = None,
    n_layers: int = 2,
    n_columns: int = 64,
    alpha: float = 1.0,
    name: str = "loop_contrast_rank_one",
    stats: Any = None,
) -> WeightEdit:
    """Build a rank-one edit from the contrast between loop and productive positions.

    The counterpart of :func:`contrast_edit` for a failure that only exists
    during generation. Instead of failing vs passing cases at the end of the
    prompt, it contrasts positions INSIDE a repeated stretch with positions
    outside one, along the recorded generations of *cases*
    (``case.observed``): the direction is the mean MLP write in loops minus the
    mean write elsewhere, and the neurons are the ones that write along it more
    in loops.

    Pass *stats* (from :func:`collect_continuation_stats`) to reuse one
    collection pass for several settings.

    Like :func:`contrast_edit` this proposes an edit; it does not show the edit
    helps. Validate it on generations it was not derived from.
    """
    if stats is None:
        stats = collect_continuation_stats(model, cases, groups="repetition",
                                           repeat_ngram=repeat_ngram, max_tokens=max_tokens,
                                           layers=layers)
    loop, productive = stats.groups.get("loop") or {}, stats.groups.get("productive") or {}
    if not loop or not productive:
        raise ValueError(
            "need both loop and productive positions; found "
            f"{stats.n_positions('loop')} loop and {stats.n_positions('productive')} productive "
            f"in {stats.n_cases} generation(s) (skipped: {stats.skipped})"
        )
    return edit_from_stats(
        model, loop, productive, layers=layers, n_layers=n_layers, n_columns=n_columns,
        alpha=alpha, name=name,
        metadata={"n_loop_positions": stats.n_positions("loop"),
                  "n_productive_positions": stats.n_positions("productive"),
                  "n_generations": stats.n_cases, "repeat_ngram": int(repeat_ngram),
                  "position": "continuation"},
    )


@contextmanager
def output_shift(model: Any, shifts: "dict[str, Any]") -> Iterator[None]:
    """Add a constant vector to the output of chosen linear layers, weights untouched.

    *shifts* maps a weight's full parameter name (as in
    :attr:`WriteSite.name`) to a vector of the layer's output size. This is the
    activation-steering counterpart of a weight edit: if a constant shift
    reproduces the edit's effect, the edit did not need to depend on the input.
    """
    import torch

    module = module_of(model)
    owners = {f"{n}.weight" if n else "weight": m for n, m in module.named_modules()
              if isinstance(m, torch.nn.Linear)}
    handles = []
    try:
        for key, vector in shifts.items():
            if key not in owners:
                raise KeyError(f"no linear layer owns parameter {key!r}")

            def hook(_module: Any, _args: Any, output: Any, vector: Any = vector) -> Any:
                return output + vector.to(device=output.device, dtype=output.dtype)

            handles.append(owners[key].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
