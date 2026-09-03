"""Phase B1 — operationalization bridge: SignalRecipe + compile_recipe (expr DSL)."""

from __future__ import annotations

import pytest

from evalrx.analysis.operationalize import (
    RecipeError,
    SignalRecipe,
    compile_recipe,
    compile_recipes,
    per_case_finding,
)
from evalrx.analysis.stats_tools import build_stats_input


def _records() -> list[dict]:
    return [
        {"case_id": "c0", "obj_size": 20, "attention": 0.1},   # small + peripheral
        {"case_id": "c1", "obj_size": 80, "attention": 0.1},   # large
        {"case_id": "c2", "obj_size": 20, "attention": 0.9},   # focused
        {"case_id": "c3", "obj_size": 20},                     # missing 'attention'
    ]


# ---------------------------------------------------------------------------
# expr DSL: interactions / thresholds / composites reduce to one signal column
# ---------------------------------------------------------------------------

def test_expr_composite_predicate_threshold_and_interaction():
    r = SignalRecipe(
        name="explored.small_and_peripheral",
        kind="expr",
        expr="(obj_size < 40) and (attention < 0.3)",
    )
    out = compile_recipe(r, _records())

    assert out == {"c0": 1.0, "c1": 0.0, "c2": 0.0}  # c3 skipped (missing column)


def test_expr_continuous_arithmetic_and_funcs():
    rows = [{"case_id": "c0", "a": 5, "b": 2}, {"case_id": "c1", "a": 1, "b": 4}]
    out = compile_recipe(SignalRecipe(name="gap", kind="expr", expr="abs(a - b)"), rows)
    assert out == {"c0": 3.0, "c1": 3.0}


def test_expr_chained_comparison():
    rows = [{"case_id": "c0", "x": 0.5}, {"case_id": "c1", "x": 1.5}]
    out = compile_recipe(SignalRecipe(name="inrange", kind="expr", expr="0 <= x <= 1"), rows)
    assert out == {"c0": 1.0, "c1": 0.0}


def test_missing_column_skips_case_never_defaults():
    out = compile_recipe(
        SignalRecipe(name="s", kind="expr", expr="attention < 0.5"), _records()
    )
    assert "c3" not in out                   # c3 has no 'attention' -> skipped
    assert set(out) == {"c0", "c1", "c2"}


def test_non_numeric_result_skips_case():
    rows = [{"case_id": "c0", "x": 1}]
    # ternary returns a string for the True branch -> not a signal value -> skipped
    out = compile_recipe(SignalRecipe(name="s", kind="expr", expr="'big' if x else 0"), rows)
    assert out == {}


# ---------------------------------------------------------------------------
# Safety: the DSL rejects anything outside the pure numeric subset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",  # call to non-whitelisted name
        "obj_size.bit_length()",                # attribute access
        "obj_size[0]",                          # subscript
        "open('x')",                            # non-whitelisted call
        "(lambda: 1)()",                        # lambda
        "obj_size <",                           # syntax error
    ],
)
def test_disallowed_constructs_raise_recipe_error(expr):
    with pytest.raises(RecipeError):
        compile_recipe(SignalRecipe(name="bad", kind="expr", expr=expr), _records())


def test_whitelisted_funcs_are_allowed():
    rows = [{"case_id": "c0", "a": 3, "b": 9}]
    out = compile_recipe(SignalRecipe(name="m", kind="expr", expr="max(a, b)"), rows)
    assert out == {"c0": 9.0}


def test_empty_expr_raises():
    with pytest.raises(RecipeError):
        compile_recipe(SignalRecipe(name="x", kind="expr", expr="   "), _records())


def test_code_kind_is_deferred():
    with pytest.raises(NotImplementedError):
        compile_recipe(SignalRecipe(name="x", kind="code", code="..."), _records())


def test_unknown_kind_raises():
    with pytest.raises(RecipeError):
        compile_recipe(SignalRecipe(name="x", kind="mystery"), _records())


# ---------------------------------------------------------------------------
# from_dict round-trip + batch compile drops failing recipes
# ---------------------------------------------------------------------------

def test_signal_recipe_from_dict():
    r = SignalRecipe.from_dict(
        {"name": "s", "kind": "expr", "expr": "x > 0", "suggested_test": "signal_label_assoc"}
    )
    assert r.name == "s" and r.kind == "expr" and r.suggested_test == "signal_label_assoc"


def test_compile_recipes_keeps_good_drops_bad():
    good = SignalRecipe(name="good", kind="expr", expr="obj_size < 40")
    bad = SignalRecipe(name="bad", kind="expr", expr="obj_size <")  # syntax error
    empty = SignalRecipe(name="empty", kind="expr", expr="missing_col > 0")  # all skipped
    out = compile_recipes([good, bad, empty], _records())
    assert set(out) == {"good"}


# ---------------------------------------------------------------------------
# End-to-end: compiled signal lands in StatsInput.per_case via a synthetic analyzer
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, findings):
        self.findings = findings


def test_compiled_signal_flows_into_stats_input_per_case():
    r = SignalRecipe(name="small_and_peripheral", kind="expr",
                     expr="(obj_size < 40) and (attention < 0.3)")
    values = compile_recipe(r, _records())
    entries = per_case_finding({r.name: values})

    results = {"explored": _FakeResult({"per_case": entries})}
    inp = build_stats_input(results, data=None)

    # build_stats_input keys it as "<analyzer>.<signal>" like any per-case finding.
    assert "explored.small_and_peripheral" in inp.per_case
    assert inp.per_case["explored.small_and_peripheral"]["c0"] == 1.0
