"""Compatibility adapters behind the unified EvalVitals web shell."""

from __future__ import annotations

import json

from evalvitals.analysis.run_view import StageId, StageState, from_session


def test_explore_adapter_keeps_validation_and_repair_distinct(tmp_path):
    report = {
        "ok": True,
        "plain_question": "What drives failures?",
        "data_profile": {"n_rows": 12},
        "takeaways": [{"title": "A signal"}],
        "hypotheses": [{"statement": "A mechanism"}],
    }
    (tmp_path / "records.json").write_text("[]")
    (tmp_path / "confirm_report.json").write_text(json.dumps({"hypothesis_verdicts": [{}]}))
    session = {"kind": "explore", "root": str(tmp_path), "runs": [
        {"dir": str(tmp_path), "report": report},
    ]}

    view = from_session(session)

    assert view.kind == "explore"
    assert view.stage(StageId.M1).state is StageState.SUCCEEDED
    assert view.stage(StageId.M2).state is StageState.SUCCEEDED
    assert view.stage(StageId.M3).count == 1
    assert view.stage(StageId.M5).state is StageState.SUCCEEDED
    assert view.stage(StageId.M4).state is StageState.NOT_STARTED


def test_loop_adapter_uses_action_order_and_maps_fix_event_to_m4(tmp_path):
    session = {
        "kind": "loop",
        "root": str(tmp_path),
        "story": {
            "run_start": {"protocol_description": "Find the failure mechanism"},
            "probes": [{"event": "probe"}],
            "analyses": [{"event": "analysis"}],
            "diagnoses": [{"event": "diagnosis"}],
            "surgeries": [{"event": "surgery", "module": "m5"}],
            "fixes": [{"event": "fix"}],
        },
    }

    view = from_session(session)

    assert [stage.id for stage in view.stages] == [
        StageId.M1, StageId.M2, StageId.M3, StageId.M5, StageId.M4,
    ]
    assert view.stage(StageId.M5).state is StageState.SUCCEEDED
    assert view.stage(StageId.M4).state is StageState.SUCCEEDED
    assert view.stage(StageId.M4).count == 1
