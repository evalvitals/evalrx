"""Reading an edit from the weights, and building the controls it is compared with."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evalrx.wsp import (  # noqa: E402
    CONTROL_KINDS,
    TensorEdit,
    WeightEdit,
    applied,
    build_control,
    delta_structure,
    edit_structure,
)


class _Net(torch.nn.Module):
    def __init__(self, rows: int = 12, cols: int = 40) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.a = torch.nn.Linear(cols, rows, bias=False)
        self.b = torch.nn.Linear(cols, rows, bias=False)


def _rank_one(alpha: float = 6.0, cols=(2, 5, 9, 17)):
    torch.manual_seed(1)
    w0 = torch.randn(12, 40)
    u = torch.randn(12)
    w1 = TensorEdit.rank_one("w", u, alpha, cols=list(cols)).apply_to(w0)
    return w0, w1, u / u.norm()


def test_structure_recovers_a_rank_one_rule():
    w0, w1, _ = _rank_one(alpha=6.0)
    s = delta_structure(w0, w1)
    assert s["changed_columns"] == 4 and s["changed_rows"] == 12
    assert s["rank_one_residual"] < 1e-5
    assert s["effective_rank"] == pytest.approx(1.0, abs=1e-4)
    assert s["alpha_estimate"] == pytest.approx(6.0, abs=1e-3)
    assert s["alpha_spread"] < 1e-3
    assert 0 < s["relative_change"] < 5


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.5])
def test_alpha_is_read_off_for_other_strengths(alpha):
    w0, w1, _ = _rank_one(alpha=alpha)
    assert delta_structure(w0, w1)["alpha_estimate"] == pytest.approx(alpha, abs=1e-3)


def test_structure_flags_a_change_that_is_not_rank_one():
    torch.manual_seed(2)
    w0 = torch.randn(12, 40)
    w1 = w0.clone()
    w1[:, :6] += torch.randn(12, 6)
    s = delta_structure(w0, w1)
    assert s["rank_one_residual"] > 0.3
    assert s["effective_rank"] > 2.0
    assert len(s["singular_values"]) == 6


def test_structure_of_no_change_and_of_a_vector():
    w = torch.randn(4, 4)
    assert delta_structure(w, w.clone())["changed_entries"] == 0
    bias = delta_structure(torch.ones(5), torch.ones(5) * 2)
    assert bias["changed_entries"] == 5 and "singular_values" not in bias
    with pytest.raises(ValueError):
        delta_structure(torch.ones(2, 2), torch.ones(3, 3))


def test_selectivity_is_high_when_columns_were_chosen_by_projection():
    torch.manual_seed(3)
    w0 = torch.randn(12, 40)
    u = torch.randn(12)
    unit = u / u.norm()
    top = torch.topk((unit @ w0).abs(), 4).indices.tolist()
    w1 = TensorEdit.rank_one("w", u, 6.0, cols=top).apply_to(w0)
    assert delta_structure(w0, w1)["projection_selectivity"] > 1.5


def test_edit_structure_reports_shared_directions_across_tensors():
    net = _Net()
    u = torch.randn(12)
    shared = WeightEdit([TensorEdit.rank_one("a.weight", u, 6.0, cols=[1, 2]),
                         TensorEdit.rank_one("b.weight", u, 6.0, cols=[3, 4])])
    report = edit_structure(net, shared)
    assert len(report["tensors"]) == 2
    assert list(report["direction_cosines"].values())[0] == pytest.approx(1.0, abs=1e-5)


def _edit(cols=(2, 5, 9, 17)) -> WeightEdit:
    torch.manual_seed(4)
    return WeightEdit([TensorEdit.rank_one("a.weight", torch.randn(12), 6.0, cols=list(cols))],
                      name="e")


def _delta_of(net, edit):
    before = net.a.weight.detach().clone()
    with applied(net, edit):
        return (net.a.weight.detach() - before).double()


def test_random_direction_keeps_columns_and_norms_but_not_the_direction():
    net = _Net()
    edit = _edit()
    d = _delta_of(net, edit)
    c = _delta_of(net, build_control(net, edit, "random_direction", seed=0))
    assert torch.equal((c != 0).any(dim=0), (d != 0).any(dim=0))
    assert torch.allclose(c.norm(dim=0), d.norm(dim=0), atol=1e-5)
    u_edit = torch.linalg.svd(d, full_matrices=False)[0][:, 0]
    u_ctrl = torch.linalg.svd(c, full_matrices=False)[0][:, 0]
    assert float((u_edit @ u_ctrl).abs()) < 0.95
    assert float(torch.linalg.svd(c, full_matrices=False)[1][1]) < 1e-6  # still rank one


def test_random_columns_applies_the_same_rule_elsewhere():
    net = _Net()
    edit = _edit()
    d = _delta_of(net, edit)
    c = _delta_of(net, build_control(net, edit, "random_columns", seed=0))
    edit_cols = set(torch.nonzero((d != 0).any(dim=0)).flatten().tolist())
    ctrl_cols = set(torch.nonzero((c != 0).any(dim=0)).flatten().tolist())
    assert len(ctrl_cols) == len(edit_cols) and not (ctrl_cols & edit_cols)
    # Same rule: the control is the alpha = 6 edit along the same u on those columns.
    w0 = net.a.weight.detach().double()
    u = torch.from_numpy(edit.tensors[0].u).double()
    cols = sorted(ctrl_cols)
    assert torch.allclose(c[:, cols], -6.0 * torch.outer(u, u @ w0[:, cols]), atol=1e-5)


def test_random_columns_moves_the_change_for_other_edit_forms():
    net = _Net()
    delta = np.zeros((12, 40), dtype=np.float32)
    delta[:, [0, 1]] = 1.0
    edit = WeightEdit([TensorEdit(name="a.weight", delta=delta)])
    c = _delta_of(net, build_control(net, edit, "random_columns", seed=3))
    cols = torch.nonzero((c != 0).any(dim=0)).flatten().tolist()
    assert len(cols) == 2 and not ({0, 1} & set(cols))
    assert torch.allclose(c[:, cols], torch.ones(12, 2, dtype=torch.float64), atol=1e-6)


def test_controls_are_seeded():
    net = _Net()
    edit = _edit()
    for kind in CONTROL_KINDS:
        a = _delta_of(net, build_control(net, edit, kind, seed=7))
        b = _delta_of(net, build_control(net, edit, kind, seed=7))
        c = _delta_of(net, build_control(net, edit, kind, seed=8))
        assert torch.equal(a, b) and not torch.equal(a, c)


def test_control_errors_are_explicit():
    net = _Net()
    with pytest.raises(ValueError, match="unknown control"):
        build_control(net, _edit(), "shuffle")
    everything = _edit(cols=range(40))
    with pytest.raises(ValueError, match="no unchanged columns"):
        build_control(net, everything, "random_columns")


def test_building_a_control_leaves_the_model_untouched():
    net = _Net()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    for kind in CONTROL_KINDS:
        build_control(net, _edit(), kind)
    assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())
