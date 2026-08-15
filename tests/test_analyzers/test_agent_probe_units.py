"""Unit edges for the intervention probes: grader, Shapley paths, rubric parsing."""

from __future__ import annotations

from evalvitals.analyzers.agent.reliability import ReliabilityProbe, default_grader
from evalvitals.analyzers.agent.tool_shap import ToolShap, _sampled_shapley
from evalvitals.analyzers.agent.trajectory_rubric import TrajectoryRubricJudge
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label, Step, StepRole, Trajectory


def _traj(answer, sample_id="s"):
    return Trajectory(sample_id=sample_id, goal="g",
                      steps=[Step(idx=0, role=StepRole.ACTOR, content=answer)],
                      final_answer=answer)


# ── default_grader ────────────────────────────────────────────────────────────
def test_grader_yes_no_uses_first_token():
    case = FailureCase(inputs=Inputs(prompt="q"), expected="no")
    assert default_grader(_traj("No, there is not. Yes really."), case) is True
    assert default_grader(_traj("Yes — although no bottle."), case) is False


def test_grader_substring_and_ungradable():
    case = FailureCase(inputs=Inputs(prompt="q"), expected="pizza")
    assert default_grader(_traj("The object is a PIZZA on a tray."), case) is True
    assert default_grader(_traj("a burger"), case) is False
    assert default_grader(_traj("anything"), FailureCase(inputs=Inputs(prompt="q"))) is None


def test_reliability_ungraded_cases_report_consistency_only():
    probe = ReliabilityProbe(runs_fn=lambda case, k: [_traj("a"), _traj("a")], k=2)
    result = probe.run(None, CaseBatch([FailureCase(inputs=Inputs(prompt="q"))]))
    entry = result.findings["per_case"][0]
    assert "success_rate" not in entry
    assert entry["answer_consistent"] == 1
    assert result.findings["mean_success_rate"] is None
    assert "deterministic" in result.findings["_caveat"]


# ── ToolShap subset scheduling + Monte-Carlo path ─────────────────────────────
def test_subset_schedule_full_enumeration_for_small_kits():
    probe = ToolShap(run_with_tools=lambda c, t: None, tool_names=["a", "b", "c"])
    subsets, exact = probe._subsets()
    assert exact and len(subsets) == 8  # 2^3


def test_subset_schedule_samples_beyond_budget():
    names = list("abcde")  # 2^5 = 32 > budget
    probe = ToolShap(run_with_tools=lambda c, t: None, tool_names=names,
                     max_combinations=12, seed=1)
    subsets, exact = probe._subsets()
    assert not exact
    assert len(subsets) == 12
    assert frozenset(names) in subsets and frozenset() in subsets
    for t in names:  # every leave-one-out is guaranteed
        assert frozenset(names) - {t} in subsets


def test_sampled_shapley_uses_only_observed_pairs():
    values = {
        frozenset(): 0.0,
        frozenset({"a"}): 1.0,
        frozenset({"a", "b"}): 1.0,
    }
    phi = _sampled_shapley(values, ["a", "b"])
    assert phi["a"] == 1.0        # ({} -> {a}) is the only observed pair
    assert phi["b"] == 0.0        # ({a} -> {a,b}) marginal is 0


def test_tool_shap_ungradable_case_still_reports_answer_shapley():
    def runner(case, names):
        return _traj("with zoom" if "zoom" in names else "bare")

    probe = ToolShap(run_with_tools=runner, tool_names=["zoom"])
    result = probe.run(None, CaseBatch([FailureCase(inputs=Inputs(prompt="q"))]))
    entry = result.findings["per_case"][0]
    assert entry["baseline_pass"] is None and "shap_outcome_zoom" not in entry
    assert entry["shap_answer_zoom"] > 0


# ── rubric judge robustness ───────────────────────────────────────────────────
class _Judge:
    def __init__(self, reply):
        self._reply = reply

    def generate(self, prompt):
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


def _rubric_batch():
    traj = Trajectory(sample_id="s0", goal="g", outcome=Label.FAIL,
                      steps=[Step(idx=0, role=StepRole.USER, content="g")])
    return CaseBatch([FailureCase(inputs=Inputs(prompt="g"), trajectory=traj, label=Label.FAIL)])


def test_rubric_garbage_reply_marks_judge_not_ok():
    result = TrajectoryRubricJudge(judge=_Judge("not json at all")).run(None, _rubric_batch())
    entry = result.findings["per_case"][0]
    assert entry["judge_ok"] == 0 and result.findings["n_judged"] == 0


def test_rubric_judge_exception_is_a_finding_not_a_crash():
    result = TrajectoryRubricJudge(judge=_Judge(RuntimeError("quota"))).run(None, _rubric_batch())
    entry = result.findings["per_case"][0]
    assert entry["judge_ok"] == 0 and "quota" in entry["judge_error"]


def test_rubric_unknown_mode_falls_back_by_label():
    reply = '{"failure_mode": "FM-WEIRD", "rubric": {}, "first_error_step": -1}'
    result = TrajectoryRubricJudge(judge=_Judge(reply)).run(None, _rubric_batch())
    assert result.findings["per_case"][0]["failure_mode"] == "FM-REASONING"  # FAIL-labelled case
