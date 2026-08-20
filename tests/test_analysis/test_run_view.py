"""Compatibility adapters behind the unified EvalVitals web shell."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def test_unified_shell_opens_attached_diagnostic_in_its_own_workspace(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import evalvitals.analysis.workbench_app as workbench_app

    run = tmp_path / "diagnostic"
    run.mkdir()
    events = [
        {"event": "run_start", "protocol_description": "Find the failure mechanism"},
        {"event": "probe", "cycle": 0, "analyzers": ["attention"]},
        {"event": "analysis", "cycle": 0},
        {"event": "diagnosis", "cycle": 0, "hypotheses": []},
        {"event": "surgery", "cycle": 0, "module": "m5", "status": "supported"},
        {"event": "fix", "cycle": 0},
    ]
    (run / "run_log.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    sys.argv = ["workbench_app.py", str(tmp_path / "workspace"), "--attach", str(run)]
    at = AppTest.from_file(Path(workbench_app.__file__), default_timeout=30)
    at.run()

    assert not at.exception
    assert at.sidebar.selectbox[0].value == "Diagnostic Runs"
    assert [tab.label for tab in at.tabs] == [
        "Overview", "M1 Measure", "M2 Evidence", "M3 Hypotheses",
        "M5 Validate", "M4 Intervene & repair", "Cases & artifacts",
    ]
