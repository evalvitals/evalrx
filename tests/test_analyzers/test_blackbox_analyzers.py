"""Newly-implemented black-box analyzers: POPE, CHAIR, counterfactual replay."""

from __future__ import annotations

from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay
from evalvitals.analyzers.hallucination.chair import CHAIRAnalyzer, chair_score, extract_objects
from evalvitals.analyzers.hallucination.pope import POPEAnalyzer, parse_yes_no
from evalvitals.core.capability import Capability
from evalvitals.core.case import FailureCase, Inputs, Label, Step, StepRole, Trajectory
from evalvitals.core.model import Model
from evalvitals.datasets import cases_from_records


class ScriptModel(Model):
    """Returns a scripted answer per generate() call (no weights)."""

    capabilities = frozenset({Capability.GENERATE, Capability.TOOL_CALLS})
    modalities = frozenset({"text", "image"})

    def __init__(self, answers):
        self._answers = list(answers)
        self._i = 0

    def generate(self, inputs, **kwargs):
        a = self._answers[min(self._i, len(self._answers) - 1)]
        self._i += 1
        return a

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


# ---------------- POPE ----------------
def test_parse_yes_no():
    assert parse_yes_no("Yes, there is a dog.") == "yes"
    assert parse_yes_no("No.") == "no"
    assert parse_yes_no("I am not sure") is None


def test_pope_metrics():
    recs = [
        {"question": "Is there a dog?", "pope_label": "yes"},
        {"question": "Is there a cat?", "pope_label": "yes"},
        {"question": "Is there a car?", "pope_label": "no"},
        {"question": "Is there a tree?", "pope_label": "no"},
    ]
    cases = cases_from_records(recs)
    model = ScriptModel(["yes", "no", "yes", "no"])  # tp, fn, fp, tn
    res = POPEAnalyzer().run(model, cases)
    f = res.findings
    assert f["n"] == 4 and f["accuracy"] == 0.5
    assert f["precision"] == 0.5 and f["recall"] == 0.5 and f["f1"] == 0.5
    assert f["per_case"][0]["has_gold"] is True
    assert f["per_case"][0]["unparsed"] is False
    assert f["per_case"][0]["is_correct"] is True


def test_pope_mechanism_signals():
    """false_positive (hallucinated presence) vs false_negative (missed finding)."""
    recs = [
        {"question": "Is there a dog?", "pope_label": "yes"},   # pred yes → TP
        {"question": "Is there a cat?", "pope_label": "yes"},   # pred no  → FN (miss)
        {"question": "Is there a car?", "pope_label": "no"},    # pred yes → FP (hallucination)
        {"question": "Is there a tree?", "pope_label": "no"},   # pred no  → TN
    ]
    cases = cases_from_records(recs)
    model = ScriptModel(["yes", "no", "yes", "no"])
    f = POPEAnalyzer().run(model, cases).findings
    flags = [(e["false_positive"], e["false_negative"]) for e in f["per_case"]]
    assert flags == [(False, False), (False, True), (True, False), (False, False)]
    # the marginals those flags are built from -- the columns M2/M5 may test
    assert [(e["answered_yes"], e["gold_yes"]) for e in f["per_case"]] == [
        (True, True), (False, True), (True, False), (False, False)]
    assert f["false_positive_rate"] == 0.5  # fp=1 of (fp+tn)=2
    # Cases without a gold label never flag a mechanism signal.
    no_gold = cases_from_records([{"question": "What modality is this?"}])
    f2 = POPEAnalyzer().run(ScriptModel(["yes"]), no_gold).findings
    assert f2["per_case"][0]["false_positive"] is False
    assert f2["per_case"][0]["false_negative"] is False
    assert "gold_yes" not in f2["per_case"][0] and f2["per_case"][0]["answered_yes"] is True


# ---------------- CHAIR ----------------
def test_chair_score_and_extract():
    assert chair_score(["dog", "cat"], ["dog"])["chair_i"] == 0.5
    assert extract_objects("a dog and a cat", ["dog", "cat", "car"]) == ["dog", "cat"]


def test_chair_analyzer_aggregates():
    recs = [
        {"caption_goal": "describe", "gt_objects": ["dog"]},
        {"caption_goal": "describe", "gt_objects": ["car"]},
    ]
    cases = cases_from_records(recs, prompt_key="caption_goal")
    model = ScriptModel(["a dog and a cat", "a car"])  # case1 hallucinates 'cat'; case2 clean
    res = CHAIRAnalyzer(object_vocab=["dog", "cat", "car"]).run(model, cases)
    assert res.findings["chair_i"] == 0.25   # mean of 0.5 and 0.0
    assert res.findings["chair_s"] == 0.5    # 1 of 2 captions has a hallucination


# ---------------- counterfactual ----------------
def test_counterfactual_ranks_influential_step():
    traj = Trajectory(
        sample_id="s", goal="g", outcome=Label.FAIL,
        steps=[
            Step(idx=0, role=StepRole.USER, content="g"),
            Step(idx=1, role=StepRole.ACTOR, tool_call={"name": "search", "args": {}}),
            Step(idx=2, role=StepRole.TOOL, observation="none"),
            Step(idx=3, role=StepRole.ACTOR, tool_call={"name": "open", "args": {}}),
        ],
    )
    case = FailureCase(inputs=Inputs(prompt="g"), trajectory=traj, label=Label.FAIL)

    # re-running forked at step 3 always succeeds (flips the FAIL outcome); step 1 never does.
    def rerun_fn(trajectory, step_idx, seed):
        return step_idx == 3

    model = ScriptModel(["x"])  # needs TOOL_CALLS capability (it has it)
    res = CounterfactualReplay(rerun_fn=rerun_fn, n_replays=2).run(model, case)
    pc = res.findings["per_case"][0]
    assert pc["original_success"] is False
    assert pc["most_influential_step"]["step"] == 3
    assert pc["most_influential_step"]["flip_rate"] == 1.0
