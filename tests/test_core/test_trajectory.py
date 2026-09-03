"""Agent-trace data model: Step / Trajectory / FailureCase.trajectory."""

from __future__ import annotations

from evalrx.core.case import (
    FailureCase,
    Inputs,
    Label,
    Step,
    StepRole,
    Trajectory,
)


def test_trajectory_from_records():
    records = [
        {"role": "user", "content": "book a flight"},
        {"role": "actor", "tool_call": {"name": "search", "args": {"q": "flights"}}},
        {"role": "tool", "observation": "no results"},
        {"role": "actor", "content": "sorry, none found"},
    ]
    traj = Trajectory.from_records(
        records, sample_id="s1", goal="book a flight", outcome=Label.FAIL
    )
    assert len(traj) == 4
    assert traj.steps[0].role is StepRole.USER
    assert traj.steps[1].tool_call["name"] == "search"
    assert traj.steps[1].idx == 1
    assert traj.outcome is Label.FAIL


def test_step_annotation_fields_default_none():
    s = Step(idx=0)
    assert s.is_first_error is None
    assert s.failure_mode is None
    # analyzers write these:
    s.is_first_error = True
    s.failure_mode = "FM-2.4"
    assert s.is_first_error and s.failure_mode == "FM-2.4"


def test_failurecase_carries_trajectory():
    traj = Trajectory.from_records([{"role": "actor", "content": "hi"}], sample_id="s2")
    case = FailureCase(inputs=Inputs(prompt="goal"), trajectory=traj, label=Label.FAIL)
    assert case.trajectory is traj
    assert len(case.trajectory) == 1


def test_unit_case_has_no_trajectory():
    case = FailureCase.from_prompt("just a prompt")
    assert case.trajectory is None


# ----------------------------------------------------------------------
# Serialization — the on-disk trajectory format
# ----------------------------------------------------------------------
class _FakeImage:
    """PIL-like: has .size and .mode, must degrade to a descriptor, not pixels."""

    size, mode = (640, 480), "RGB"


def _traj_with_rich_steps() -> Trajectory:
    return Trajectory(
        sample_id="s3",
        goal="what is this?",
        steps=[
            Step(idx=0, role=StepRole.USER, content="what is this?", span={"has_image": True}),
            Step(
                idx=1,
                role=StepRole.ACTOR,
                content="<tool_call>...</tool_call>",
                tool_call={"name": "zoom", "args": {"bbox": [0, 0, 1, 1]}, "id": None},
                span={"turn": 1},
            ),
            Step(
                idx=2,
                role=StepRole.TOOL,
                content="zoom",
                observation={"text": "zoomed", "n_images": 1, "image": _FakeImage()},
            ),
        ],
        final_answer="a cat",
        outcome=Label.UNKNOWN,
        metrics={"n_steps": 3},
    )


def test_trajectory_to_dict_is_json_safe():
    import json

    d = _traj_with_rich_steps().to_dict()
    dumped = json.dumps(d)  # would raise on PIL-like objects / enums
    assert d["outcome"] == "unknown"
    assert d["steps"][0]["role"] == "user"
    assert d["steps"][1]["tool_call"]["name"] == "zoom"
    assert d["steps"][2]["observation"]["image"] == "<image 640x480>"
    assert json.loads(dumped)["final_answer"] == "a cat"


def test_failurecase_to_dict_includes_trajectory_and_safe_image():
    import json

    case = FailureCase(
        inputs=Inputs(prompt="q", image=_FakeImage()),
        trajectory=_traj_with_rich_steps(),
        label=Label.FAIL,
    )
    d = case.to_dict()
    json.dumps(d)
    assert d["inputs"]["image"] == "<image 640x480>"
    assert d["trajectory"]["sample_id"] == "s3"
    assert len(d["trajectory"]["steps"]) == 3


def test_unit_case_to_dict_has_null_trajectory():
    d = FailureCase.from_prompt("p").to_dict()
    assert d["trajectory"] is None
