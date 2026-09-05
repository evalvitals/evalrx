"""M4 label-free fallback: Bonferroni multiplicity control over the family.

When M2 supplies no stats_results, M4 falls back to an unpaired clustered-bootstrap
compare() on the extracted per-case signal. The whole cycle shares one report, so
every hypothesis is tested on that fallback — a best-of-N family. A directional
verdict must now be significant at the family-corrected level (alpha / family),
mirroring the primary path's e-BH, instead of each hypothesis self-promoting.
"""

from __future__ import annotations

from evalrx.analysis.stats_agent import StatsAnalysisReport
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.result import Result
from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus
from evalrx.eval_agent.stages.hypothesis_tester import HypothesisTester


def _hyp(s: str) -> Hypothesis:
    return Hypothesis(statement=s, target_model="m", predicted_failure_mode="")


def _borderline_fallback():
    """24 cases whose signal/label split is rigged to a borderline effect
    (eff=0.417): the unpaired CI excludes 0 at alpha=0.05 but INCLUDES 0 at
    alpha/4. signal-present = 12 cases (6 FAIL, 6 PASS); signal-absent = 12
    cases (1 FAIL, 11 PASS)."""
    cases, per_case = [], []
    for i in range(12):  # signal-present (metric=1): first 6 FAIL, next 6 PASS
        cid = f"sig{i}"
        cases.append(FailureCase(id=cid, inputs=Inputs(prompt="q"),
                                 label=Label.FAIL if i < 6 else Label.PASS))
        per_case.append({"sample_id": cid, "metric": 1})
    for i in range(12):  # signal-absent (metric=0): 1 FAIL, 11 PASS
        cid = f"ctl{i}"
        cases.append(FailureCase(id=cid, inputs=Inputs(prompt="q"),
                                 label=Label.FAIL if i == 0 else Label.PASS))
        per_case.append({"sample_id": cid, "metric": 0})
    result = Result(analyzer="probe", model="m", findings={"per_case": per_case})
    report = StatsAnalysisReport(model_name="m", raw_results={"probe": result})
    return CaseBatch(cases), report


def test_single_fallback_hypothesis_is_supported():
    """family=1: the borderline effect is significant at alpha -> SUPPORTED."""
    data, report = _borderline_fallback()
    res = HypothesisTester(min_effect=0.05).test([_hyp("signal drives failure")],
                                                 report, data, protocol=None)[0]
    assert res.status == HypothesisStatus.SUPPORTED
    assert res.evidence["family_size"] == 1
    assert res.evidence["significant"] is True


def test_fallback_family_is_bonferroni_culled():
    """family=4: the same borderline effect is NOT significant at alpha/4, so the
    best-of-N over the fallback no longer manufactures a SUPPORTED verdict."""
    data, report = _borderline_fallback()
    hyps = [_hyp(f"hypothesis {i}") for i in range(4)]
    results = HypothesisTester(min_effect=0.05).test(hyps, report, data, protocol=None)

    assert all(r.status == HypothesisStatus.INCONCLUSIVE for r in results)
    top = results[0]
    assert top.evidence["family_size"] == 4
    assert top.evidence["significant"] is False
    assert "multiplicity" in top.verdict
    # the corrected level is alpha/family, not the raw alpha
    assert abs(top.evidence["corrected_alpha"] - 0.05 / 4) < 1e-9
