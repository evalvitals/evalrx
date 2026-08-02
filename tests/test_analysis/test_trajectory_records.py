"""Trajectory -> flat records: the M1->M2 bridge for agent runs."""

from __future__ import annotations

from evalvitals.analysis.stats_tools import build_stats_input_from_records
from evalvitals.analysis.trajectory_records import (
    trajectories_to_records,
    trajectory_features,
)
from evalvitals.core.case import (
    FailureCase,
    Inputs,
    Label,
    Step,
    StepRole,
    Trajectory,
)


def _traj(sample_id="t1", outcome=Label.UNKNOWN) -> Trajectory:
    zoom = {"name": "image_zoom_in", "args": {"bbox": [0, 0, 1000, 456]}, "id": None}
    return Trajectory(
        sample_id=sample_id,
        goal="what?",
        steps=[
            Step(idx=0, role=StepRole.USER, content="what?"),
            Step(idx=1, role=StepRole.ACTOR, tool_call=zoom, span={"turn": 1}),
            Step(idx=2, role=StepRole.TOOL, content="image_zoom_in",
                 observation={"text": "Zoomed ...", "n_images": 1}),
            Step(idx=3, role=StepRole.ACTOR, tool_call=zoom, span={"turn": 2}),
            Step(idx=4, role=StepRole.TOOL, content="image_zoom_in",
                 observation={"text": "Zoomed ...", "n_images": 1}),
            Step(idx=5, role=StepRole.ACTOR,
                 tool_call={"name": "image_ocr", "args": {}, "id": None}, span={"turn": 3}),
            Step(idx=6, role=StepRole.TOOL, content="image_ocr",
                 observation="[tool error in 'image_ocr': boom]"),
            Step(idx=7, role=StepRole.ACTOR, content="a dog"),
        ],
        final_answer="a dog",
        outcome=outcome,
        metrics={"n_steps": 8, "n_turns": 4, "n_tool_calls": 3, "terminated": "final"},
    )


def test_features_capture_loops_errors_and_images():
    f = trajectory_features(_traj())
    assert f["n_tool_calls"] == 3
    assert f["distinct_tools"] == 2
    assert f["n_calls_image_zoom_in"] == 2 and f["n_calls_image_ocr"] == 1
    assert f["n_repeated_calls"] == 1 and f["repeated_call_frac"] == round(1 / 3, 4)
    assert f["max_consecutive_repeat"] == 2  # zoom, zoom back-to-back
    assert f["n_tool_errors"] == 1 and f["tool_error_rate"] == round(1 / 3, 4)
    assert f["n_images_returned"] == 2
    assert f["first_call_turn"] == 1
    assert f["terminated_final"] == 1
    assert f["answer_len_chars"] == 5


def test_features_recompute_when_metrics_missing():
    t = _traj()
    t.metrics = {}
    f = trajectory_features(t)
    assert f["n_steps"] == 8 and f["n_turns"] == 4  # ACTOR steps


def test_records_merge_case_label_and_scalar_metadata():
    fail = FailureCase(
        inputs=Inputs(prompt="what?"), trajectory=_traj("a"), label=Label.FAIL,
        metadata={"model": "qwen3-vl-2b", "seed": 0, "raw": {"nested": 1}},
    )
    ok = FailureCase(inputs=Inputs(prompt="what?"), trajectory=_traj("b"), label=Label.PASS)
    skipped = FailureCase.from_prompt("no trajectory")
    rows = trajectories_to_records([fail, ok, skipped])
    assert len(rows) == 2
    assert rows[0]["case_id"] == fail.id and rows[0]["label"] == "fail"
    assert rows[0]["model"] == "qwen3-vl-2b" and rows[0]["seed"] == 0
    assert "raw" not in rows[0]  # non-scalar metadata dropped
    assert rows[1]["label"] == "pass"


def test_bare_trajectories_use_outcome_and_sample_id():
    rows = trajectories_to_records([_traj("x", outcome=Label.FAIL)])
    assert rows[0]["case_id"] == "x" and rows[0]["label"] == "fail"


def test_per_tool_columns_are_zero_filled_across_the_batch():
    a = _traj("a")
    b = Trajectory(sample_id="b", goal="g", steps=[
        Step(idx=0, role=StepRole.USER, content="g"),
        Step(idx=1, role=StepRole.ACTOR,
             tool_call={"name": "image_detect", "args": {"query": "dog"}}, span={"turn": 1}),
        Step(idx=2, role=StepRole.TOOL, content="image_detect", observation="ok"),
        Step(idx=3, role=StepRole.ACTOR, content="done"),
    ], final_answer="done", metrics={"terminated": "final"})
    rows = trajectories_to_records([a, b])
    for row in rows:
        assert {"n_calls_image_zoom_in", "n_calls_image_ocr", "n_calls_image_detect"} <= set(row)
    assert rows[1]["n_calls_image_zoom_in"] == 0 and rows[1]["n_calls_image_detect"] == 1


def test_records_feed_build_stats_input_from_records():
    rows = trajectories_to_records([
        FailureCase(inputs=Inputs(prompt="q"), trajectory=_traj("a"), label=Label.FAIL),
        FailureCase(inputs=Inputs(prompt="q"), trajectory=_traj("b"), label=Label.PASS),
    ])
    si = build_stats_input_from_records(rows)
    assert si.labels == {"a": True, "b": False} or set(si.labels) == {rows[0]["case_id"], rows[1]["case_id"]}
    assert "n_tool_calls" in si.per_case
    assert "max_consecutive_repeat" in si.per_case
    assert set(si.per_case["n_tool_calls"].values()) == {3.0}


def test_cost_columns_aggregate_from_actor_spans():
    t = _traj()
    t.steps[1].span.update({"latency_ms": 100.0, "prompt_tokens": 900, "completion_tokens": 40})
    t.steps[3].span.update({"latency_ms": 300.0, "prompt_tokens": 1500, "completion_tokens": 60})
    f = trajectory_features(t)
    assert f["total_latency_ms"] == 400.0
    assert f["total_prompt_tokens"] == 2400 and f["total_completion_tokens"] == 100
    assert f["mean_turn_latency_ms"] == 200.0  # mean over the turns that recorded latency


def test_cost_columns_default_to_zero_without_usage():
    f = trajectory_features(_traj())
    assert f["total_latency_ms"] == 0.0 and f["total_prompt_tokens"] == 0


def test_agent_question_template_names_real_column_families():
    from evalvitals.analysis.trajectory_records import AGENT_QUESTION_TEMPLATE

    feature_cols = set(trajectory_features(_traj()))
    for named in ("n_tool_calls", "max_consecutive_repeat", "repeated_call_frac",
                  "tool_error_rate", "total_completion_tokens", "total_latency_ms"):
        assert named in AGENT_QUESTION_TEMPLATE and named in feature_cols
    # probe-produced families + the fixability framing
    for phrase in ("shap_outcome_", "success_rate", "failure_mode",
                   "not ground truth", "fixable causes"):
        assert phrase in AGENT_QUESTION_TEMPLATE


def test_serialization_roundtrip_preserves_features():
    t = _traj()
    reloaded = Trajectory.from_dict(t.to_dict())
    assert trajectory_features(reloaded) == trajectory_features(t)
    assert reloaded.steps[1].role is StepRole.ACTOR
    assert reloaded.outcome is Label.UNKNOWN
