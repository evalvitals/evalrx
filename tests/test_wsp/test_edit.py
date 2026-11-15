"""Weight edits: the arithmetic of each form, and the apply/restore contract."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evalrx.wsp import (  # noqa: E402
    TensorEdit,
    WeightEdit,
    applied,
    bf16_bits_to_f32,
    resolve_parameters,
    tensor_sha256,
)


class _Block(torch.nn.Module):
    def __init__(self, d: int = 6, h: int = 10) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(d, h, bias=False)
        self.down_proj = torch.nn.Linear(h, d, bias=False)


class _Net(torch.nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList([_Block() for _ in range(n)])


def _snapshot(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _same(net, snap) -> bool:
    return all(torch.equal(v, snap[k]) for k, v in net.state_dict().items())


# ----------------------------------------------------------------------
# Arithmetic
# ----------------------------------------------------------------------

def test_rank_one_alpha_one_removes_the_projection_on_chosen_columns():
    torch.manual_seed(1)
    w = torch.randn(6, 10)
    u = torch.randn(6)
    cols = [1, 4, 7]
    out = TensorEdit.rank_one("w", u, alpha=1.0, cols=cols).apply_to(w)
    unit = u / u.norm()
    assert torch.allclose(unit @ out[:, cols], torch.zeros(3), atol=1e-6)
    untouched = [c for c in range(10) if c not in cols]
    assert torch.equal(out[:, untouched], w[:, untouched])


def test_rank_one_alpha_two_is_a_reflection():
    torch.manual_seed(2)
    w = torch.randn(6, 10)
    u = torch.randn(6)
    out = TensorEdit.rank_one("w", u, alpha=2.0).apply_to(w)
    unit = u / u.norm()
    assert torch.allclose(out.norm(dim=0), w.norm(dim=0), atol=1e-5)
    assert torch.allclose(unit @ out, -(unit @ w), atol=1e-5)


def test_rank_one_alpha_six_flips_and_scales_by_five():
    torch.manual_seed(3)
    w = torch.randn(6, 10)
    u = torch.randn(6)
    out = TensorEdit.rank_one("w", u, alpha=6.0).apply_to(w)
    unit = u / u.norm()
    assert torch.allclose(unit @ out, -5.0 * (unit @ w), atol=1e-4)


def test_direction_is_normalised_and_must_be_usable():
    edit = TensorEdit.rank_one("w", [3.0, 4.0], alpha=1.0)
    assert np.allclose(edit.u, [0.6, 0.8])
    with pytest.raises(ValueError):
        TensorEdit.rank_one("w", [0.0, 0.0], alpha=1.0)
    with pytest.raises(ValueError):
        TensorEdit.rank_one("w", [float("nan"), 1.0], alpha=1.0)


def test_low_rank_dense_and_sparse_components_compose_in_order():
    w = torch.zeros(2, 3)
    edit = TensorEdit(
        name="w", form="low_rank",
        a=np.array([[1.0], [2.0]]), b=np.array([[1.0, 0.0, 1.0]]),   # [[1,0,1],[2,0,2]]
        delta=np.full((2, 3), 0.5),                                   # +0.5 everywhere
        index=[0, 5], values=[9.0, -9.0],                             # overwrite last
    )
    out = edit.apply_to(w)
    assert out.tolist() == [[9.0, 0.5, 1.5], [2.5, 0.5, -9.0]]


def test_apply_to_does_not_touch_its_input_and_keeps_dtype():
    w = torch.randn(4, 5, dtype=torch.float64)
    before = w.clone()
    out = TensorEdit.rank_one("w", torch.randn(4), 3.0).apply_to(w)
    assert torch.equal(w, before)
    assert out.dtype == torch.float64 and out is not w


@pytest.mark.parametrize("edit,message", [
    (TensorEdit.rank_one("w", [1.0, 0.0, 0.0], 1.0), "rows"),
    (TensorEdit.rank_one("w", [1.0, 0.0, 0.0, 0.0], 1.0, cols=[99]), "column index"),
    (TensorEdit(name="w", index=[999], values=[1.0]), "flat index"),
    (TensorEdit(name="w", delta=np.zeros((4, 5)), shape=(9, 9)), "shape"),
])
def test_edits_that_do_not_fit_are_refused(edit, message):
    with pytest.raises(ValueError, match=message):
        edit.apply_to(torch.zeros(4, 5))


def test_malformed_components_are_refused_at_construction():
    with pytest.raises(ValueError):
        TensorEdit(name="w", index=[0, 1], values=[1.0])
    with pytest.raises(ValueError):
        TensorEdit(name="w", index=[0])
    with pytest.raises(ValueError):
        TensorEdit(name="w", a=np.zeros((2, 1)))
    with pytest.raises(ValueError):
        WeightEdit([TensorEdit(name="w", delta=np.zeros(1)),
                    TensorEdit(name="w", delta=np.zeros(1))])


def test_scaling_is_exact_for_rules_and_refused_for_overwrites():
    torch.manual_seed(4)
    w = torch.randn(6, 10)
    u = torch.randn(6)
    half = TensorEdit.rank_one("w", u, 6.0).scaled(0.5).apply_to(w)
    assert torch.allclose(half, TensorEdit.rank_one("w", u, 3.0).apply_to(w))
    dense = TensorEdit(name="w", delta=np.ones((6, 10))).scaled(2.0).apply_to(w)
    assert torch.allclose(dense, w + 2.0)
    with pytest.raises(ValueError, match="cannot be scaled"):
        TensorEdit(name="w", index=[0], values=[1.0]).scaled(2.0)


# ----------------------------------------------------------------------
# Resolving names
# ----------------------------------------------------------------------

def test_names_resolve_exactly_or_by_unique_suffix():
    net = _Net()
    exact = WeightEdit([TensorEdit(name="layers.1.down_proj.weight", delta=np.zeros((6, 10)))])
    assert resolve_parameters(net, exact)[0][0] == "layers.1.down_proj.weight"
    # An edit written against a differently rooted model still finds its tensor.
    foreign = WeightEdit([TensorEdit(name="model.language_model.layers.2.down_proj.weight",
                                     delta=np.zeros((6, 10)))])
    assert resolve_parameters(net, foreign)[0][0] == "layers.2.down_proj.weight"


def test_a_wrong_layer_index_never_lands_on_another_layer():
    net = _Net(n=1)  # only layers.0 exists
    edit = WeightEdit([TensorEdit(name="model.layers.9.down_proj.weight",
                                  delta=np.zeros((6, 10)))])
    with pytest.raises(KeyError, match="no parameter"):
        resolve_parameters(net, edit)


def test_names_without_an_index_have_no_droppable_prefix():
    net = _Net()
    with pytest.raises(KeyError, match="no parameter"):
        resolve_parameters(net, WeightEdit([TensorEdit(name="x.down_proj.weight",
                                                       delta=np.zeros((6, 10)))]))


def test_a_name_that_fits_two_towers_is_ambiguous():
    class Two(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = _Net(n=1)
            self.text = _Net(n=1)

    edit = WeightEdit([TensorEdit(name="model.layers.0.down_proj.weight",
                                  delta=np.zeros((6, 10)))])
    with pytest.raises(KeyError, match="ambiguous"):
        resolve_parameters(Two(), edit)


# ----------------------------------------------------------------------
# apply / restore
# ----------------------------------------------------------------------

def _edit(alpha: float = 6.0) -> WeightEdit:
    torch.manual_seed(5)
    return WeightEdit([
        TensorEdit.rank_one("layers.0.down_proj.weight", torch.randn(6), alpha, cols=[0, 3]),
        TensorEdit.rank_one("layers.2.down_proj.weight", torch.randn(6), alpha, cols=[1]),
    ], name="t")


def test_applied_writes_then_restores_exactly():
    net = _Net()
    snap = _snapshot(net)
    with applied(net, _edit()) as report:
        assert not _same(net, snap)
        rows = {r["name"]: r for r in report.rows}
        assert rows["layers.0.down_proj.weight"]["changed_columns"] == 2
        assert rows["layers.2.down_proj.weight"]["changed_columns"] == 1
        assert report.relative_change > 0
    assert _same(net, snap)


def test_applied_restores_when_the_block_raises():
    net = _Net()
    snap = _snapshot(net)
    with pytest.raises(RuntimeError, match="boom"):
        with applied(net, _edit()):
            raise RuntimeError("boom")
    assert _same(net, snap)


def test_nothing_is_written_when_any_tensor_fails():
    net = _Net()
    snap = _snapshot(net)
    bad = WeightEdit([
        TensorEdit.rank_one("layers.0.down_proj.weight", torch.randn(6), 6.0),
        TensorEdit.rank_one("layers.1.down_proj.weight", torch.randn(7), 6.0),  # wrong size
    ])
    entered = False
    with pytest.raises(ValueError):
        with applied(net, bad):
            entered = True
    assert entered is False
    assert _same(net, snap)


def test_size_limit_refuses_a_large_edit():
    net = _Net()
    snap = _snapshot(net)
    with pytest.raises(ValueError, match="exceeds the limit"):
        with applied(net, _edit(alpha=50.0), max_relative_change=0.01):
            pass
    assert _same(net, snap)
    with applied(net, _edit(alpha=1e-6), max_relative_change=0.01):
        pass


def test_non_finite_results_are_refused():
    net = _Net()
    snap = _snapshot(net)
    edit = WeightEdit([TensorEdit(name="layers.0.down_proj.weight",
                                  delta=np.full((6, 10), np.inf, dtype=np.float32))])
    with pytest.raises(ValueError, match="non-finite"):
        with applied(net, edit):
            pass
    assert _same(net, snap)


def test_verify_base_checks_the_recorded_hash():
    net = _Net()
    param = net.layers[0].down_proj.weight
    good = WeightEdit([TensorEdit(name="layers.0.down_proj.weight", delta=np.ones((6, 10)),
                                  base_sha256=tensor_sha256(param))])
    with applied(net, good, verify_base=True):
        pass
    stale = WeightEdit([TensorEdit(name="layers.0.down_proj.weight", delta=np.ones((6, 10)),
                                   base_sha256="0" * 64)])
    with pytest.raises(ValueError, match="differs from the one the edit was made on"):
        with applied(net, stale, verify_base=True):
            pass
    unhashed = WeightEdit([TensorEdit(name="layers.0.down_proj.weight", delta=np.ones((6, 10)))])
    with pytest.raises(ValueError, match="recorded base hash"):
        with applied(net, unhashed, verify_base=True):
            pass


# ----------------------------------------------------------------------
# Building from weights, persistence
# ----------------------------------------------------------------------

def test_from_state_dicts_reproduces_the_edited_model():
    net = _Net()
    base = _snapshot(net)
    with applied(net, _edit()):
        edited = _snapshot(net)
    diff = WeightEdit.from_state_dicts(base, edited)
    assert sorted(diff.tensor_names) == ["layers.0.down_proj.weight", "layers.2.down_proj.weight"]
    with applied(net, diff):
        assert all(torch.allclose(v, edited[k], atol=1e-6) for k, v in net.state_dict().items())


def test_as_delta_is_equivalent_on_the_weights_it_was_built_from():
    net = _Net()
    edit = _edit()
    with applied(net, edit):
        target = _snapshot(net)
    with applied(net, edit.as_delta(net)):
        assert all(torch.allclose(v, target[k], atol=1e-6) for k, v in net.state_dict().items())


def test_save_and_load_round_trip(tmp_path):
    edit = _edit()
    edit.metadata["note"] = "round trip"
    loaded = WeightEdit.load(edit.save(str(tmp_path / "e")))
    assert loaded.name == "t" and loaded.metadata["note"] == "round trip"
    assert loaded.tensor_names == edit.tensor_names
    for a, b in zip(edit.tensors, loaded.tensors):
        assert np.array_equal(a.u, b.u) and np.array_equal(a.cols, b.cols)
        assert a.alpha == b.alpha and a.form == b.form
    net = _Net()
    with applied(net, edit):
        target = _snapshot(net)
    with applied(net, loaded):
        assert _same(net, target)


def test_describe_is_json_serialisable_and_carries_no_arrays():
    text = json.dumps(_edit().describe())
    assert "alpha" in text and "n_cols" in text


# ----------------------------------------------------------------------
# The published edit layout
# ----------------------------------------------------------------------

def _bf16_bits(x32: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bits, round to nearest even (reference implementation)."""
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def _sha(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


def _write_astra_dir(tmp_path, base_bits, form):
    """Write an edit directory in the published layout; return (dir, expected bits)."""
    rng = np.random.default_rng(0)
    rows, cols = base_bits.shape
    arrays = {}
    expected = base_bits.copy().reshape(-1)
    if form == "lowrank":
        a = rng.normal(size=(rows, 2)).astype(np.float32)
        b = rng.normal(size=(2, cols)).astype(np.float32)
        w = bf16_bits_to_f32(base_bits).astype(np.float64)
        for k in range(2):
            w = w + np.outer(a[:, k], b[k])
        expected = _bf16_bits(w.astype(np.float32)).reshape(-1)
        arrays["t0_a"], arrays["t0_b"] = a, b
    idx = np.array([0, 7, 13], dtype=np.int32)
    val = _bf16_bits(np.array([0.25, -1.5, 3.0], dtype=np.float32))
    expected[idx] = val
    arrays["t0_idx"], arrays["t0_val"] = idx, val
    directory = tmp_path / form / "edit"
    directory.mkdir(parents=True)
    np.savez(directory / "edit.npz", **arrays)
    manifest = {"tensors": [{
        "name": "model.language_model.layers.0.down_proj.weight",
        "shape": [rows, cols], "dtype": "bfloat16", "form": form,
        "base_sha256": _sha(base_bits),
        "edited_sha256": _sha(expected.reshape(rows, cols)),
    }], "changed_entries_total": 3}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return str(directory), expected.reshape(rows, cols)


@pytest.mark.parametrize("form", ["sparse", "lowrank"])
def test_published_layout_rebuilds_the_checkpoint_bit_for_bit(tmp_path, form):
    net = _Net().to(torch.bfloat16)
    param = net.layers[0].down_proj.weight
    base_bits = param.detach().view(torch.int16).numpy().astype(np.uint16)
    directory, expected_bits = _write_astra_dir(tmp_path, base_bits, form)

    edit = WeightEdit.from_astra(directory)
    assert edit.tensors[0].form == ("low_rank" if form == "lowrank" else "sparse_set")
    assert edit.metadata["source"] == "astra"
    with applied(net, edit, verify_base=True):
        assert tensor_sha256(param) == _sha(expected_bits)
    assert tensor_sha256(param) == _sha(base_bits)
    # WeightEdit.load recognises the layout without being told.
    assert WeightEdit.load(directory).tensors[0].base_sha256 == _sha(base_bits)


def test_published_edit_is_refused_on_a_different_base(tmp_path):
    net = _Net().to(torch.bfloat16)
    base_bits = net.layers[0].down_proj.weight.detach().view(torch.int16).numpy().astype(np.uint16)
    directory, _ = _write_astra_dir(tmp_path, base_bits, "sparse")
    other = _Net()
    with torch.no_grad():
        other.layers[0].down_proj.weight.add_(1.0)
    other = other.to(torch.bfloat16)
    with pytest.raises(ValueError, match="differs from the one the edit was made on"):
        with applied(other, WeightEdit.from_astra(directory), verify_base=True):
            pass
