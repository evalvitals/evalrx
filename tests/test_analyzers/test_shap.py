"""MM-SHAP / VL-SHAP + the shared permutation-sampling Shapley core."""

from __future__ import annotations

from evalrx.analyzers.perturbation._shapley import shapley_values
from evalrx.analyzers.perturbation.mm_shap import MMShapAnalyzer
from evalrx.analyzers.perturbation.vl_shap import VLShapAnalyzer
from evalrx.core.capability import Capability
from evalrx.core.case import FailureCase, Inputs
from tests.conftest import FakeModel


def test_shapley_recovers_additive_weights():
    # additive game: phi[p] == weight[p] exactly, for any permutation sample
    w = {"a": 1.0, "b": 2.0, "c": 0.0}
    phi = shapley_values(list(w), lambda kept: sum(w[p] for p in kept), n_samples=8, seed=0)
    assert abs(phi["a"] - 1.0) < 1e-9 and abs(phi["b"] - 2.0) < 1e-9 and abs(phi["c"]) < 1e-9


def test_mm_shap_modality_split():
    # injected additive scorer: word "key" worth 1.0, image worth 2.0
    def score_fn(inp: Inputs) -> float:
        s = 0.0
        if "key" in inp.prompt.split():
            s += 1.0
        if inp.image is not None:
            s += 2.0
        return s

    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS})
    case = FailureCase(inputs=Inputs(prompt="the key word", image="IMG"))
    res = MMShapAnalyzer(score_fn=score_fn, n_samples=32).run(model, case)
    f = res.findings
    assert abs(f["image_contribution"] - 2.0) < 1e-6
    assert abs(f["text_contribution"] - 1.0) < 1e-6
    assert abs(f["mm_score"] - 2.0 / 3.0) < 1e-3   # image-reliant
    assert f["top_text_tokens"][0]["token"] == "key"


def test_vl_shap_region_attribution():
    # injected region scorer: region 3 carries all the signal
    def region_score_fn(kept: set) -> float:
        return 5.0 if 3 in kept else 0.0

    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS})
    case = FailureCase(inputs=Inputs(prompt="describe", image="IMG"))
    res = VLShapAnalyzer(n_regions=4, region_score_fn=region_score_fn, n_samples=32).run(model, case)
    top = res.findings["top_regions"][0]
    assert top["region"] == 3 and abs(top["shapley"] - 5.0) < 1e-6


def test_shap_requires_logprobs():
    import pytest

    from evalrx.core.capability import CapabilityError
    model = FakeModel(capabilities={Capability.GENERATE})  # no LOGPROBS
    with pytest.raises(CapabilityError):
        MMShapAnalyzer(score_fn=lambda i: 0.0).run(model, FailureCase(inputs=Inputs(prompt="x")))
