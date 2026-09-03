"""WS4 — attention_decoding: tensor-level omnibus over per-case attention maps.

A cross-validated linear decoder's out-of-fold AUC, calibrated by a label-
permutation null. Detects "do FAIL and PASS attend differently?" from the full
map, not a scalar reduction.
"""

from __future__ import annotations

import numpy as np

from evalrx.analysis.stats_tools import (
    StatsInput,
    build_stats_input,
    default_plan,
    run_stats_tool,
)
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.result import Result


def _maps(n_fail=15, n_pass=15, *, structured: bool, seed=0, g=6):
    """Per-case 2-D maps. structured=True → FAIL maps carry a corner hotspot."""
    rng = np.random.default_rng(seed)
    labels: dict[str, bool] = {}
    vec: dict[str, np.ndarray] = {}
    for i in range(n_fail):
        cid = f"f{i}"
        m = rng.normal(0.0, 0.1, size=(g, g))
        if structured:
            m[0, 0] += 2.0  # FAIL attends to the corner
        labels[cid] = True
        vec[cid] = m
    for i in range(n_pass):
        cid = f"p{i}"
        m = rng.normal(0.0, 0.1, size=(g, g))  # PASS: no hotspot (same noise both ways)
        labels[cid] = False
        vec[cid] = m
    return StatsInput(labels=labels, per_case_vectors={"relative_attention.map": vec})


class TestAttentionDecoding:
    def test_detects_structured_difference(self):
        inp = _maps(structured=True, seed=1)
        r = run_stats_tool("attention_decoding", inp, {"n_perm": 200, "seed": 0})
        assert r.ok
        assert r.details["method"] == "energy_distance"
        assert r.effect > 0.0                       # positive energy distance
        assert r.reject is True                     # permutation-significant
        assert r.p_value is not None and r.p_value < 0.05
        assert r.details["cv_auc"] > 0.8            # companion decoder also separates
        assert r.details["n"] == 30 and r.details["n_fail"] == 15

    def test_null_on_unstructured_maps(self):
        inp = _maps(structured=False, seed=2)
        r = run_stats_tool("attention_decoding", inp, {"n_perm": 200, "seed": 0})
        assert r.ok
        # No class structure → neither energy distance nor the decoder beats the null.
        assert r.reject is False
        assert r.p_value is not None and r.p_value >= 0.05

    def test_underpowered_when_too_few_maps(self):
        inp = _maps(n_fail=2, n_pass=2, structured=True)
        r = run_stats_tool("attention_decoding", inp, {"n_perm": 50})
        assert r.ok is False
        assert "insufficient" in (r.error or "")

    def test_handles_variable_map_shapes(self):
        # Maps of different native shapes are resized to a common grid.
        rng = np.random.default_rng(3)
        labels, vec = {}, {}
        for i in range(14):
            cid = f"f{i}"
            labels[cid] = True
            m = rng.normal(0, 0.1, size=(4, 4))
            m[0, 0] += 2.0
            vec[cid] = m
        for i in range(14):
            cid = f"p{i}"
            labels[cid] = False
            vec[cid] = rng.normal(0, 0.1, size=(8, 8))  # different shape
        inp = StatsInput(labels=labels, per_case_vectors={"attn.map": vec})
        r = run_stats_tool("attention_decoding", inp, {"n_perm": 60, "grid": 6, "seed": 0})
        assert r.ok and r.details["n_features"] == 36


class TestBuildStatsInputMaps:
    def test_per_case_maps_become_vectors(self):
        cases, per_case = [], []
        maps = {}
        for i in range(6):
            cid = f"f{i}" if i < 3 else f"p{i}"
            cases.append(FailureCase(id=cid, inputs=Inputs(prompt="q"),
                                     label=Label.FAIL if i < 3 else Label.PASS))
            per_case.append({"sample_id": cid, "max_relative_weight": 1.0 + i})
            maps[cid] = np.ones((4, 4), dtype=np.float16) * i
        res = Result(analyzer="relative_attention", model="m", cases=CaseBatch(cases),
                     findings={"per_case": per_case},
                     artifacts={"per_case_maps": maps})
        inp = build_stats_input({"relative_attention": res}, CaseBatch(cases))
        assert "relative_attention.map" in inp.per_case_vectors
        assert set(inp.per_case_vectors["relative_attention.map"]) == set(maps)

    def test_default_plan_includes_decoding_when_maps_present(self):
        inp = _maps(structured=True)
        tools = [t for t, _, _ in default_plan(inp)]
        assert "attention_decoding" in tools
