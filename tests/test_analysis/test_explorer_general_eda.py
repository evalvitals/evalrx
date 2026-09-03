"""ExploratoryAnalysisAgent must adapt its framing to the data's actual outcome shape:
binary (the old M1 FAIL/PASS behavior), multi-class categorical, continuous,
or no outcome at all (pure unsupervised EDA). These tests guard against the
prompt silently reverting to a hardcoded FAIL/PASS story for data that isn't
shaped that way, and confirm M1's binary contract still works unchanged.
"""

from __future__ import annotations

import json

from evalrx.agent_runtime.sandbox import ExperimentSandbox
from evalrx.analysis.explorer import ExploratoryAnalysisAgent, _framing_block, _profile_rows
from evalrx.analysis.profile import describe_outcome, profile_records

_MINIMAL_PAYLOAD = {
    "observations": ["ok"],
    "visual_plan": [
        {
            "name": "v1",
            "question": "q",
            "data_shape": "many-numeric",
            "plot_kind": "bar",
            "fallback_kind": "bar",
            "required_columns": [],
            "rationale": "r",
        }
    ],
    "chart_readings": [],
    "dashboard_storyboard": [],
    "claims": [],
    "candidate_signals": [],
    "plots": [],
    "tables": {},
    "charts": [],
    "caveats": [],
    "critique": ["c"],
    "recommended_confirmatory_tests": [],
}
_MINIMAL_CODE = f'print("EXPLORATORY_RESULT_JSON=" + {json.dumps(json.dumps(_MINIMAL_PAYLOAD))})'


class _RecordingJudge:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return f"```python\n{_MINIMAL_CODE}\n```"


def _run(rows, *, outcome_col=None, tmp_path):
    judge = _RecordingJudge()
    agent = ExploratoryAnalysisAgent(
        judge=judge, sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False)
    )
    report = agent.explore_records(rows, question="q", outcome_col=outcome_col)
    assert report.ok, report.error
    return judge.prompts[0], report


# ---------------------------------------------------------------------------
# describe_outcome / profile_records outcome-kind classification
# ---------------------------------------------------------------------------

def test_describe_outcome_binary_from_pass_fail_strings():
    rows = [{"case_id": f"c{i}", "label": "fail" if i < 2 else "pass"} for i in range(4)]
    outcome = describe_outcome(profile_records(rows))
    assert outcome == {"present": True, "column": "label", "kind": "binary", "unique": 2}


def test_describe_outcome_continuous_numeric():
    rows = [{"case_id": f"c{i}", "outcome": float(i)} for i in range(20)]
    outcome = describe_outcome(profile_records(rows))
    assert outcome["present"] is True
    assert outcome["kind"] == "continuous"


def test_describe_outcome_categorical_multiclass():
    rows = [
        {"case_id": f"c{i}", "outcome": ["red", "green", "blue"][i % 3]}
        for i in range(9)
    ]
    outcome = describe_outcome(profile_records(rows))
    assert outcome["kind"] == "categorical"
    assert outcome["unique"] == 3


def test_describe_outcome_none_when_no_recognizable_column():
    rows = [{"id": f"c{i}", "x": float(i), "y": float(i) * 2} for i in range(10)]
    outcome = describe_outcome(profile_records(rows))
    assert outcome == {"present": False, "column": None, "kind": "none", "unique": 0}


def test_outcome_col_override_finds_arbitrary_target_name():
    rows = [{"case_id": f"c{i}", "revenue": float(i)} for i in range(20)]
    # "revenue" matches no name heuristic, so auto-detect misses it ...
    assert describe_outcome(profile_records(rows))["present"] is False
    # ... but an explicit override finds it and classifies it correctly.
    outcome = describe_outcome(profile_records(rows, outcome_col="revenue"))
    assert outcome["present"] is True
    assert outcome["column"] == "revenue"
    assert outcome["kind"] == "continuous"


# ---------------------------------------------------------------------------
# _framing_block branches
# ---------------------------------------------------------------------------

def test_framing_block_binary_uses_fail_pass_language():
    block = _framing_block({"present": True, "column": "label", "kind": "binary", "unique": 2})
    assert "FAIL and PASS" in block
    assert "fail_rate" in block


def test_framing_block_continuous_has_no_fail_pass_language():
    block = _framing_block({"present": True, "column": "score", "kind": "continuous", "unique": 50})
    assert "FAIL and PASS" not in block
    assert "fail_rate" not in block
    assert "CONTINUOUS outcome" in block


def test_framing_block_categorical_rejects_binary_collapse():
    block = _framing_block({"present": True, "column": "verdict", "kind": "categorical", "unique": 4})
    assert "do NOT collapse it into a binary FAIL/PASS" in block


def test_framing_block_none_is_unsupervised_and_label_free():
    block = _framing_block({"present": False, "column": None, "kind": "none", "unique": 0})
    assert "UNSUPERVISED" in block
    assert "FAIL and PASS" not in block
    assert "fail_rate" not in block


# ---------------------------------------------------------------------------
# End-to-end: the prompt actually sent to the coding agent adapts per dataset
# ---------------------------------------------------------------------------

def test_m1_style_binary_records_still_get_fail_pass_prompt(tmp_path):
    rows = [{"case_id": f"c{i}", "label": "fail" if i < 3 else "pass"} for i in range(6)]
    prompt, _ = _run(rows, tmp_path=tmp_path)
    assert "FAIL and PASS" in prompt
    assert '"kind": "binary"' in prompt


def test_arbitrary_data_with_no_outcome_gets_unsupervised_prompt(tmp_path):
    rows = [{"id": f"r{i}", "x": float(i), "y": float(i) ** 2} for i in range(10)]
    prompt, report = _run(rows, tmp_path=tmp_path)
    assert "UNSUPERVISED" in prompt
    assert "Call the two groups FAIL and PASS" not in prompt
    assert report.data_profile["outcome"]["present"] is False


def test_arbitrary_continuous_target_gets_regression_style_prompt(tmp_path):
    rows = [{"id": f"r{i}", "temperature": float(i), "yield_pct": 90.0 + i * 0.3} for i in range(20)]
    prompt, _ = _run(rows, outcome_col="yield_pct", tmp_path=tmp_path)
    assert "CONTINUOUS outcome" in prompt
    assert "Call the two groups FAIL and PASS" not in prompt


def test_data_profile_exposes_outcome_and_column_roles(tmp_path):
    rows = [{"case_id": f"c{i}", "model": "m1", "score": float(i), "label": "fail" if i < 2 else "pass"} for i in range(4)]
    profile = _profile_rows(rows)
    assert profile["outcome"]["column"] == "label"
    assert profile["outcome"]["kind"] == "binary"
    assert "score" in profile["numeric_columns"]
    assert profile["group_columns"] == ["model"]
