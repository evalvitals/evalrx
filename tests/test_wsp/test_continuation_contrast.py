"""Deriving an edit from loop vs productive positions, fitting a rank-one rule to
an edit, and comparing two edits — on the model with a known planted defect."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evalrx.core.case import FailureCase, Inputs, Label  # noqa: E402
from evalrx.wsp import (  # noqa: E402
    TensorEdit,
    WeightEdit,
    applied,
    collect_continuation_stats,
    continuation_contrast_edit,
    edit_agreement,
    encode_continuation,
    encode_continuations,
    fit_rank_one,
    rank_columns,
    write_sites,
)
from tests.toy_wsp import (  # noqa: E402
    DEFECT_LAYER,
    DEFECT_NEURON,
    N_LAYERS,
    WRONG_TOKEN,
    D,
    ToyModel,
    snapshot,
    unchanged,
)

_PLAIN = "3 4 5 6 2 5 3 6 4 2 6 3"
_PROMPT = "2 2"
_KEY = f"layers.{DEFECT_LAYER}.mlp.down_proj.weight"


def _gen(continuation, prompt: str = _PROMPT) -> FailureCase:
    return FailureCase(inputs=Inputs(prompt=prompt), observed=continuation, label=Label.FAIL)


def _looping(trigger: int = 0, cycles: int = 8) -> FailureCase:
    return _gen(f"{_PLAIN} {' '.join([f'{trigger} {WRONG_TOKEN}'] * cycles)}")


def _reference(alpha: float = 6.0) -> WeightEdit:
    return WeightEdit([TensorEdit.rank_one(_KEY, np.eye(D)[WRONG_TOKEN], alpha,
                                           cols=[DEFECT_NEURON])], name="reference")


# ----------------------------------------------------------------------
# Sequences
# ----------------------------------------------------------------------

def test_sequence_has_prompt_then_continuation_with_grouped_targets():
    model = ToyModel()
    seq = encode_continuation(model, _gen("0 7 0 7 0 7 3"), repeat_ngram=2)
    assert seq.ids == [2, 2, 0, 7, 0, 7, 0, 7, 3] and seq.start == 2
    assert seq.targets == [2, 3, 4, 5, 6, 7, 8]
    assert seq.labels == ["productive"] * 4 + ["loop"] * 3
    # A target is predicted from the state one position earlier.
    assert seq.positions() == [1, 2, 3, 4, 5, 6, 7]
    assert seq.positions("loop") == [5, 6, 7]


def test_sequence_reasons_and_truncation():
    model = ToyModel()
    assert encode_continuation(model, _gen(None)) == "no_continuation"
    assert encode_continuation(model, _gen("0 7"), max_tokens=2) == (
        "no_continuation_tokens_after_truncation")
    assert len(encode_continuation(model, _gen("0 7 0 7"), max_tokens=4).ids) == 4
    assert encode_continuation(model, _gen("0 7"), groups="none").labels == ["all", "all"]
    with pytest.raises(ValueError, match="groups must be one of"):
        encode_continuation(model, _gen("0 7"), groups="paragraph")
    sequences, skipped = encode_continuations(model, [_gen("0 7"), _gen(None), _gen(" ")])
    assert len(sequences) == 1 and dict(skipped) == {"no_continuation": 2}


# ----------------------------------------------------------------------
# Statistics and the derived edit
# ----------------------------------------------------------------------

def test_stats_cover_every_layer_and_count_positions():
    model = ToyModel()
    stats = collect_continuation_stats(model, [_looping(0), _looping(1)], repeat_ngram=2)
    assert stats.n_cases == 2
    assert stats.n_positions("loop") == 24 and stats.n_positions("productive") == 32
    for group in ("loop", "productive"):
        assert sorted(stats.groups[group]) == list(range(N_LAYERS))
        for row in stats.groups[group].values():
            assert row["in"].shape == (D,) and row["out"].shape == (D,)
    for site in write_sites(model):
        assert not site.module._forward_hooks


def test_stats_match_a_direct_computation():
    model = ToyModel()
    case = _gen("0 7 0 7 0 7 3")
    stats = collect_continuation_stats(model, [case], repeat_ngram=2)
    captured = {}
    site = write_sites(model)[DEFECT_LAYER]
    handle = site.module.register_forward_hook(
        lambda m, args, out: captured.update(x=args[0].detach()[0].double(),
                                             y=out.detach()[0].double()))
    with torch.no_grad():
        model.module(input_ids=torch.tensor([[2, 2, 0, 7, 0, 7, 0, 7, 3]]))
    handle.remove()
    loop = stats.groups["loop"][DEFECT_LAYER]
    assert torch.allclose(loop["in"], captured["x"][[5, 6, 7]].mean(dim=0))
    assert torch.allclose(loop["out"], captured["y"][[5, 6, 7]].mean(dim=0))
    assert loop["n"] == 3


def test_the_planted_neuron_fires_more_inside_loops():
    model = ToyModel()
    stats = collect_continuation_stats(model, [_looping(0), _looping(1)], repeat_ngram=2)
    inside = float(stats.groups["loop"][DEFECT_LAYER]["in"][DEFECT_NEURON])
    outside = float(stats.groups["productive"][DEFECT_LAYER]["in"][DEFECT_NEURON])
    assert inside > 3 * outside


def test_loop_contrast_recovers_the_planted_layer_neuron_and_direction():
    model = ToyModel()
    snap = snapshot(model)
    edit = continuation_contrast_edit(model, [_looping(0), _looping(1)], repeat_ngram=2,
                                      n_layers=1, n_columns=1, alpha=6.0)
    assert edit.tensor_names == [_KEY]
    assert edit.tensors[0].cols.tolist() == [DEFECT_NEURON]
    assert edit.metadata["position"] == "continuation"
    assert edit.metadata["n_loop_positions"] == 24
    # The planted layer separates loops from ordinary text most clearly.
    separation = edit.metadata["layer_separation"]
    assert max(separation, key=separation.get) == str(DEFECT_LAYER)
    agreement = edit_agreement(model, edit, _reference())["tensors"][0]
    assert agreement["direction_cosine"] > 0.99
    assert agreement["n_shared"] == 1 and agreement["jaccard"] == 1.0
    assert unchanged(model, snap)


def test_precomputed_stats_give_the_same_edit():
    model = ToyModel()
    cases = [_looping(0), _looping(1)]
    stats = collect_continuation_stats(model, cases, repeat_ngram=2)
    direct = continuation_contrast_edit(model, cases, repeat_ngram=2, n_layers=1, n_columns=2)
    reused = continuation_contrast_edit(model, [], n_layers=1, n_columns=2, stats=stats)
    assert reused.tensor_names == direct.tensor_names
    assert np.allclose(reused.tensors[0].u, direct.tensors[0].u)
    assert reused.tensors[0].cols.tolist() == direct.tensors[0].cols.tolist()


def test_both_position_groups_are_required():
    model = ToyModel()
    with pytest.raises(ValueError, match="need both loop and productive positions"):
        continuation_contrast_edit(model, [_gen(_PLAIN)], repeat_ngram=2)
    with pytest.raises(ValueError, match="need both loop and productive positions"):
        continuation_contrast_edit(model, [_gen(None)], repeat_ngram=2)


def test_rank_columns_reports_each_signal():
    model = ToyModel()
    stats = collect_continuation_stats(model, [_looping(0), _looping(1)], repeat_ngram=2)
    weight = write_sites(model)[DEFECT_LAYER].module.weight
    row = rank_columns(stats.groups["loop"][DEFECT_LAYER],
                       stats.groups["productive"][DEFECT_LAYER], weight)
    assert int(row["push"].argmax()) == DEFECT_NEURON
    assert int(row["projection"].abs().argmax()) == DEFECT_NEURON
    assert int(row["activation"].argmax()) == DEFECT_NEURON
    assert torch.allclose(row["push"], row["activation"] * row["projection"])
    same = rank_columns(stats.groups["loop"][0], stats.groups["loop"][0], weight)
    assert same["u"] is None and same["separation"] == 0.0


# ----------------------------------------------------------------------
# Fitting a rank-one rule, comparing edits
# ----------------------------------------------------------------------

def test_fit_rank_one_recovers_the_rule_behind_a_dense_edit():
    model = ToyModel()
    snap = snapshot(model)
    dense = _reference(alpha=6.0).as_delta(model)
    assert dense.tensors[0].form == "delta"
    fit = fit_rank_one(model, dense)
    row = fit.metadata["fit"][0]
    assert row["alpha"] == pytest.approx(6.0, abs=1e-4)
    assert row["rank_one_residual"] < 1e-6 and row["n_columns"] == 1
    assert fit.tensors[0].form == "rank_one" and fit.tensors[0].cols.tolist() == [DEFECT_NEURON]
    with applied(model, dense):
        target = snapshot(model)
    with applied(model, fit):
        assert all(torch.allclose(v, target[k], atol=1e-5)
                   for k, v in model.module.state_dict().items())
    assert unchanged(model, snap)


def test_a_fitted_edit_can_be_scaled_where_the_original_cannot():
    model = ToyModel()
    weight = write_sites(model)[DEFECT_LAYER].module.weight.detach()
    with applied(model, _reference(6.0)):
        edited = write_sites(model)[DEFECT_LAYER].module.weight.detach().clone()
    changed = torch.nonzero((edited != weight).reshape(-1)).flatten()
    overwrite = WeightEdit([TensorEdit(name=_KEY, form="sparse_set", index=changed,
                                       values=edited.reshape(-1)[changed])])
    with pytest.raises(ValueError, match="cannot be scaled"):
        overwrite.scaled(0.5)
    half = fit_rank_one(model, overwrite).scaled(0.5)
    assert half.tensors[0].alpha == pytest.approx(3.0, abs=1e-4)
    with applied(model, half) as small, applied(model, _reference(3.0)):
        pass
    assert small.relative_change > 0


def test_fit_refuses_an_edit_with_nothing_to_fit():
    model = ToyModel()
    null = WeightEdit([TensorEdit(name=_KEY, delta=np.zeros((D, D)))])
    with pytest.raises(ValueError, match="no 2-D change"):
        fit_rank_one(model, null)


def test_agreement_separates_direction_from_neuron_overlap():
    model = ToyModel()
    reference = _reference()
    same_direction_other_neuron = WeightEdit([TensorEdit.rank_one(
        _KEY, np.eye(D)[WRONG_TOKEN], 6.0, cols=[DEFECT_NEURON, 0])])
    row = edit_agreement(model, same_direction_other_neuron, reference)["tensors"][0]
    assert row["direction_cosine"] > 0.99
    assert row["n_shared"] == 1 and row["jaccard"] == 0.5
    assert row["expected_shared_by_chance"] == pytest.approx(2 * 1 / D)

    tilted = np.eye(D)[WRONG_TOKEN] + np.eye(D)[2]          # 45 degrees off the reference
    other_direction = WeightEdit([TensorEdit.rank_one(_KEY, tilted, 6.0, cols=[DEFECT_NEURON])])
    row = edit_agreement(model, other_direction, reference)["tensors"][0]
    assert row["n_shared"] == 1 and row["jaccard"] == 1.0
    assert row["direction_cosine"] == pytest.approx(2 ** -0.5, abs=1e-3)


def test_agreement_lists_tensors_only_one_edit_touches():
    model = ToyModel()
    elsewhere = WeightEdit([TensorEdit.rank_one("layers.0.mlp.down_proj.weight",
                                                np.eye(D)[1], 1.0, cols=[1])])
    report = edit_agreement(model, elsewhere, _reference())
    assert report["tensors"] == []
    assert report["only_in_edit"] == ["layers.0.mlp.down_proj.weight"]
    assert report["only_in_reference"] == [_KEY]
