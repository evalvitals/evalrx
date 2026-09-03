"""Hierarchical MCTS probing search (analysis.probe_search) — pure tree-search
math and orchestration, exercised entirely with fake generator/verify
callables (no model, no GPU, no judge)."""

from __future__ import annotations

import math

import pytest

from evalrx.analysis.probe_search import (
    ProbeNode,
    ProbeSearch,
    _choose_regime,
    _select_expandable,
    _ucb,
)
from evalrx.core.case import FailureCase, Inputs, Label


def _case(prompt: str, label: "Label | None" = None) -> FailureCase:
    c = FailureCase(inputs=Inputs(prompt=prompt))
    if label is not None:
        c.label = label
    return c


# ---------------------------------------------------------------------------
# _ucb / _select_expandable / _choose_regime — pure math
# ---------------------------------------------------------------------------

def test_ucb_unvisited_node_has_zero_p_hat_plus_exploration_bonus():
    parent = ProbeNode(case=_case("p"), regime="macro", N=4)
    child = ProbeNode(case=_case("c"), regime="macro", N=0, E=0)
    value = _ucb(child, parent.N, beta=2.0)
    assert value == pytest.approx(2.0 * math.sqrt(math.log(4) / 1))


def test_ucb_rewards_higher_empirical_failure_rate():
    parent_n = 10
    high_fail = ProbeNode(case=_case("a"), regime="macro", N=4, E=4)  # p_hat=1.0
    low_fail = ProbeNode(case=_case("b"), regime="macro", N=4, E=0)   # p_hat=0.0
    assert _ucb(high_fail, parent_n, beta=0.1) > _ucb(low_fail, parent_n, beta=0.1)


def test_select_expandable_stays_at_root_when_under_width_cap():
    root = ProbeNode(case=_case("root"), regime="macro")
    root.children.append(ProbeNode(case=_case("c0"), regime="macro", parent=root))
    u = _select_expandable(root, beta=1.0, w_max=3)
    assert u is root  # only 1 child, cap is 3 -> still expandable


def test_select_expandable_descends_via_ucb_when_at_width_cap():
    root = ProbeNode(case=_case("root"), regime="macro", N=20)
    weak = ProbeNode(case=_case("weak"), regime="macro", parent=root, N=5, E=0)
    strong = ProbeNode(case=_case("strong"), regime="macro", parent=root, N=5, E=5)
    root.children = [weak, strong]  # at width cap (2 == w_max)
    u = _select_expandable(root, beta=0.01, w_max=2)  # small beta -> exploitation dominates
    assert u is strong  # higher p_hat, low beta means it wins


def test_choose_regime_prefers_higher_ucb_root():
    macro_root = ProbeNode(case=_case("m"), regime="macro", N=8, E=8)  # p_hat=1.0
    micro_root = ProbeNode(case=_case("u"), regime="micro", N=8, E=0)  # p_hat=0.0
    assert _choose_regime(macro_root, micro_root, beta=0.01) == "macro"


def test_choose_regime_explores_less_visited_root_with_high_beta():
    macro_root = ProbeNode(case=_case("m"), regime="macro", N=100, E=0)
    micro_root = ProbeNode(case=_case("u"), regime="micro", N=1, E=0)
    # Same p_hat (0.0) for both; a large beta should favor the far-less-visited one.
    assert _choose_regime(macro_root, micro_root, beta=10.0) == "micro"


# ---------------------------------------------------------------------------
# ProbeSearch.run — end-to-end with fake generator/verify callables
# ---------------------------------------------------------------------------

def test_probe_search_requires_at_least_one_seed():
    with pytest.raises(ValueError):
        ProbeSearch(
            generate_macro=lambda node, explored: None,
            generate_micro=lambda node: None,
            verify=lambda c: c,
            seeds=[],
        )


def _counting_generators():
    """Deterministic fake Macro/Micro generators + verify: every 3rd evaluated
    case fails, generators always produce a fresh distinct candidate."""
    counter = {"n": 0}

    def generate_macro(node, explored):
        counter["n"] += 1
        return _case(f"macro-{counter['n']}")

    def generate_micro(node):
        counter["n"] += 1
        return _case(f"micro-{counter['n']}")

    def verify(case: FailureCase) -> FailureCase:
        case.label = Label.FAIL if counter["n"] % 3 == 0 else Label.PASS
        return case

    return generate_macro, generate_micro, verify


def test_probe_search_runs_full_budget_and_splits_macro_micro():
    generate_macro, generate_micro, verify = _counting_generators()
    search = ProbeSearch(
        generate_macro=generate_macro, generate_micro=generate_micro, verify=verify,
        seeds=[_case("seed", Label.PASS)], budget=15, beta=1.0, w_max=2,
    )
    result = search.run()
    assert result.n_simulations == 15
    assert result.n_macro + result.n_micro == 15
    assert result.n_macro > 0 and result.n_micro > 0  # both regimes get exercised
    assert len(result.all_cases) == 15
    assert all(c.label == Label.FAIL for c in result.failure_cases)
    assert 0.0 < result.error_rate < 1.0


def test_probe_search_backup_updates_ancestor_stats_along_selection_path():
    generate_macro, generate_micro, verify = _counting_generators()
    search = ProbeSearch(
        generate_macro=generate_macro, generate_micro=generate_micro, verify=verify,
        seeds=[_case("seed", Label.PASS)], budget=1, beta=1.0, w_max=3,
    )
    result = search.run()
    root = result.macro_root if result.n_macro == 1 else result.micro_root
    # Exactly one simulation ran through this regime's root -> root.N==1,
    # and E reflects whether that one child failed.
    assert root.N == 1
    assert root.E in (0, 1)
    assert len(root.children) == 1
    assert root.children[0].N == 0  # freshly created leaf, not yet expanded itself


def test_probe_search_generator_returning_none_is_skipped_without_crashing():
    def generate_macro(node, explored):
        return None  # simulate an exhausted/failed generation attempt

    def generate_micro(node):
        return _case("micro-only", Label.PASS)

    search = ProbeSearch(
        generate_macro=generate_macro, generate_micro=generate_micro,
        verify=lambda c: c, seeds=[_case("seed", Label.PASS)], budget=5, w_max=2,
    )
    result = search.run()
    # Every macro attempt returns None -> only micro simulations ever land.
    assert result.n_macro == 0
    assert result.n_simulations == result.n_micro
    assert result.n_simulations <= 5


def test_probe_search_failure_cases_is_subset_of_all_cases():
    generate_macro, generate_micro, verify = _counting_generators()
    search = ProbeSearch(
        generate_macro=generate_macro, generate_micro=generate_micro, verify=verify,
        seeds=[_case("seed", Label.PASS)], budget=9, w_max=2,
    )
    result = search.run()
    all_ids = {c.id for c in result.all_cases}
    fail_ids = {c.id for c in result.failure_cases}
    assert fail_ids <= all_ids


def test_result_to_dict_has_expected_keys():
    generate_macro, generate_micro, verify = _counting_generators()
    search = ProbeSearch(
        generate_macro=generate_macro, generate_micro=generate_micro, verify=verify,
        seeds=[_case("seed", Label.PASS)], budget=3, w_max=2,
    )
    result = search.run()
    assert set(result.to_dict()) == {
        "n_simulations", "n_macro", "n_micro", "error_rate", "n_failures",
    }
