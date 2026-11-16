"""Deriving an edit from a FAIL/PASS contrast, on a model with a known planted defect."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evalrx.core.capability import Capability  # noqa: E402
from evalrx.core.case import Label  # noqa: E402
from evalrx.wsp import (  # noqa: E402
    applied,
    build_control,
    collect_write_stats,
    contrast_edit,
    edit_structure,
    output_shift,
    write_sites,
)
from tests.toy_wsp import (  # noqa: E402
    DEFECT_LAYER,
    DEFECT_NEURON,
    N_LAYERS,
    WRONG_TOKEN,
    D,
    ToyModel,
    echo_batch,
    exact_score,
    snapshot,
    unchanged,
)


def _groups(batch):
    return ([c for c in batch if c.label == Label.FAIL],
            [c for c in batch if c.label == Label.PASS])


def _accuracy(model, batch) -> float:
    return sum(exact_score(c, model.generate(c.inputs)) for c in batch) / len(batch)


def test_write_sites_finds_one_projection_per_layer():
    model = ToyModel()
    sites = write_sites(model)
    assert [s.layer for s in sites] == list(range(N_LAYERS))
    params = dict(model.module.named_parameters())
    for site in sites:
        assert site.name == f"layers.{site.layer}.mlp.down_proj.weight"
        assert params[site.name] is site.module.weight


def test_write_stats_have_the_layer_shapes_and_remove_their_hooks():
    model = ToyModel()
    fails, _ = _groups(echo_batch())
    stats = collect_write_stats(model, fails)
    assert sorted(stats) == list(range(N_LAYERS))
    for row in stats.values():
        assert row["in"].shape == (D,) and row["out"].shape == (D,) and row["n"] == len(fails)
    for site in write_sites(model):
        assert not site.module._forward_hooks
    only = collect_write_stats(model, fails, layers=[DEFECT_LAYER])
    assert list(only) == [DEFECT_LAYER]


def test_the_planted_neuron_fires_on_failing_cases_only():
    model = ToyModel()
    fails, passes = _groups(echo_batch())
    on = collect_write_stats(model, fails)[DEFECT_LAYER]["in"][DEFECT_NEURON]
    off = collect_write_stats(model, passes)[DEFECT_LAYER]["in"][DEFECT_NEURON]
    assert float(on) > 1.0 and abs(float(off)) < 0.5


def test_contrast_edit_recovers_the_planted_layer_neuron_and_direction():
    model = ToyModel()
    fails, passes = _groups(echo_batch())
    edit = contrast_edit(model, fails, passes, n_layers=1, n_columns=1)
    assert len(edit) == 1
    tensor = edit.tensors[0]
    assert tensor.name == f"layers.{DEFECT_LAYER}.mlp.down_proj.weight"
    assert tensor.cols.tolist() == [DEFECT_NEURON]
    expected = np.zeros(D)
    expected[WRONG_TOKEN] = 1.0
    assert np.allclose(tensor.u, expected, atol=0.05)
    assert edit.metadata["n_fail"] == len(fails) and edit.metadata["n_pass"] == len(passes)


def test_the_edit_repairs_unseen_cases_and_breaks_none():
    model = ToyModel()
    pool, held_out = echo_batch(4, 6), echo_batch(6, 8)
    snap = snapshot(model)
    assert _accuracy(model, held_out) == pytest.approx(8 / 14)
    edit = contrast_edit(model, *_groups(pool), n_layers=1, n_columns=1)
    with applied(model, edit):
        assert _accuracy(model, held_out) == 1.0
    assert unchanged(model, snap)
    assert _accuracy(model, held_out) == pytest.approx(8 / 14)


def test_a_matched_random_direction_does_not_repair():
    model = ToyModel()
    pool, held_out = echo_batch(4, 6), echo_batch(6, 8)
    edit = contrast_edit(model, *_groups(pool), n_layers=1, n_columns=1)
    for seed in range(5):
        with applied(model, build_control(model, edit, "random_direction", seed=seed)):
            assert _accuracy(model, held_out) < 1.0


def test_a_wider_edit_still_repairs_without_side_effects():
    model = ToyModel()
    pool, held_out = echo_batch(4, 6), echo_batch(6, 8)
    edit = contrast_edit(model, *_groups(pool), n_layers=2, n_columns=4)
    assert len(edit) == 2
    for tensor in edit.tensors:
        assert 1 <= len(tensor.cols) <= 4
    with applied(model, edit):
        assert _accuracy(model, held_out) == 1.0


def test_structure_of_the_derived_edit_reads_back_as_rank_one():
    model = ToyModel()
    edit = contrast_edit(model, *_groups(echo_batch()), n_layers=1, n_columns=1, alpha=1.0)
    row = edit_structure(model, edit)["tensors"][0]
    assert row["changed_columns"] == 1
    assert row["rank_one_residual"] < 1e-5
    assert row["alpha_estimate"] == pytest.approx(1.0, abs=1e-3)


def test_explicit_layers_are_respected_and_checked():
    model = ToyModel()
    fails, passes = _groups(echo_batch())
    edit = contrast_edit(model, fails, passes, layers=[DEFECT_LAYER], n_columns=2)
    assert edit.tensor_names == [f"layers.{DEFECT_LAYER}.mlp.down_proj.weight"]
    with pytest.raises(ValueError, match="no MLP output projection"):
        contrast_edit(model, fails, passes, layers=[99])


def test_both_groups_are_required():
    model = ToyModel()
    fails, passes = _groups(echo_batch())
    with pytest.raises(ValueError, match="FAIL case and one readable PASS"):
        contrast_edit(model, [], passes)
    with pytest.raises(ValueError, match="FAIL case and one readable PASS"):
        contrast_edit(model, fails, [])


def test_identical_groups_give_no_edit():
    model = ToyModel()
    _, passes = _groups(echo_batch())
    with pytest.raises(ValueError, match="do not differ"):
        contrast_edit(model, passes[:1], passes[:1])


def test_output_shift_changes_the_forward_pass_only_inside_the_block():
    model = ToyModel()
    case = list(echo_batch())[4]
    key = f"layers.{DEFECT_LAYER}.mlp.down_proj.weight"
    clean = model.forward(case.inputs, capture={Capability.LOGITS}).logits[-1]
    bump = torch.zeros(D)
    bump[WRONG_TOKEN] = 5.0
    with output_shift(model, {key: bump}):
        shifted = model.forward(case.inputs, capture={Capability.LOGITS}).logits[-1]
    after = model.forward(case.inputs, capture={Capability.LOGITS}).logits[-1]
    assert float(shifted[WRONG_TOKEN] - clean[WRONG_TOKEN]) > 4.0
    assert torch.equal(after, clean)
    with pytest.raises(KeyError):
        with output_shift(model, {"layers.9.mlp.down_proj.weight": bump}):
            pass


def test_write_sites_on_a_real_architecture():
    pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    sites = write_sites(model)
    assert len(sites) == 6
    params = dict(model._loaded[0].named_parameters())
    for site in sites:
        assert site.name.endswith(f"layers.{site.layer}.mlp.down_proj.weight")
        assert params[site.name].shape == (32, 64)
